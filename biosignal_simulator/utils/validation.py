"""
BioSignal Simulator Validation and Quality Assessment Engine.

This module provides physiological feasibility checks and deep signal integrity metrics
to evaluate if simulated or recorded biopotentials are physiologically valid or contain
artifacts like flatlines, clipping, powerline interference, or sudden motion bursts.

Mathematical Formulations:
1. Pan-Tompkins QRS Detection:
   - Differentiator: $y[n] = \\frac{1}{8} (2x[n] + x[n-1] - x[n-3] - 2x[n-4])$
   - Squaring function: $y[n] = x^2[n]$
   - Moving Window Integration: $y[n] = \\frac{1}{N} \\sum_{i=0}^{N-1} x[n-i]$

2. EEG Band Power Integration:
   - Relative band power:
     $$P_{\\text{rel}}(f_1, f_2) = \\frac{\\int_{f_1}^{f_2} P_{xx}(f) df}{\\int_{0.5}^{f_{\\text{Nyquist}}} P_{xx}(f) df}$$

3. EMG Mean and Median Frequencies:
   - Mean Frequency (MNF):
     $$f_{\\text{mean}} = \\frac{\\sum_{i=1}^{K} f_i P(f_i)}{\\sum_{i=1}^{K} P(f_i)}$$
   - Median Frequency (MDF):
     $$\\sum_{i=1}^{\\text{MDF}} P(f_i) = \\sum_{i=\\text{MDF}}^{K} P(f_i) = \\frac{1}{2} \\sum_{i=1}^{K} P(f_i)$$
"""

import os
import json
import datetime
from typing import Any, Dict, List, Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from biosignal_simulator.core.record import SignalRecord
import numpy as np
from scipy import signal as sp_signal

_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

class ValidationReport:
    """
    Structured validation report container containing flags, warnings, and extracted physiological metrics.
    """
    def __init__(self, is_valid: bool, warnings: List[str], metrics: Dict[str, Any]):
        self.is_valid = is_valid
        self.warnings = warnings
        self.metrics = metrics

    def __repr__(self) -> str:
        return (
            f"ValidationReport(is_valid={self.is_valid}, "
            f"warnings_count={len(self.warnings)}, "
            f"metrics={list(self.metrics.keys())})"
        )

    def to_json(self) -> str:
        """Serialize report to a JSON string."""
        return json.dumps({
            'is_valid': self.is_valid,
            'warnings': self.warnings,
            'metrics': self.metrics
        }, indent=4)


class SignalIntegrityChecker:
    """
    Utility to evaluate engineering signal quality characteristics.
    """
    
    @staticmethod
    def detect_flatline(signal: np.ndarray, fs: float, tolerance: float = 1e-6, min_duration_s: float = 0.5) -> List[Tuple[float, float]]:
        """
        Detect flatline segments (constant values) in the signal.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (start_time_s, end_time_s) for flatline segments.
        """
        n_samples = len(signal)
        min_samples = int(np.round(min_duration_s * fs))
        if min_samples <= 1:
            min_samples = 2
            
        flatline_segments = []
        
        # Calculate differences
        diffs = np.abs(np.diff(signal))
        is_flat = diffs < tolerance
        
        # Find contiguous segments
        in_flat = False
        start_idx = 0
        
        for idx in range(len(is_flat)):
            if is_flat[idx]:
                if not in_flat:
                    in_flat = True
                    start_idx = idx
            else:
                if in_flat:
                    in_flat = False
                    segment_len = idx - start_idx + 1
                    if segment_len >= min_samples:
                        flatline_segments.append((start_idx / fs, idx / fs))
                        
        if in_flat:
            segment_len = len(is_flat) - start_idx
            if segment_len >= min_samples:
                flatline_segments.append((start_idx / fs, len(is_flat) / fs))
                
        return flatline_segments

    @staticmethod
    def detect_clipping(signal: np.ndarray, threshold_ratio: float = 0.99) -> Tuple[bool, float]:
        """
        Detect if signal amplitude is clipped/saturated.
        
        Returns
        -------
        Tuple[bool, float]
            (is_clipped, clipping_fraction)
        """
        if len(signal) == 0:
            return False, 0.0
            
        sig_min = np.min(signal)
        sig_max = np.max(signal)
        
        if np.isclose(sig_max, sig_min):
            return False, 0.0
            
        median_val = np.median(signal)
        span = sig_max - sig_min
        tol = 1e-4 * span
        
        # Check if consecutive samples are flat (diff close to 0)
        is_flat = np.abs(np.diff(signal)) < 1e-10
        # To distinguish symmetry peaks from real clipping, require segments of length >= 3
        has_consecutive_flat = is_flat[:-1] & is_flat[1:]
        is_flat_3 = np.zeros(len(signal), dtype=bool)
        is_flat_3[:-2] |= has_consecutive_flat
        is_flat_3[1:-1] |= has_consecutive_flat
        is_flat_3[2:] |= has_consecutive_flat
        
        clip_count = 0
        # Check top clipping if sig_max is significantly above median
        if (sig_max - median_val) > 0.1 * span:
            max_count = np.sum((signal >= (sig_max - tol)) & is_flat_3)
            clip_count += max_count
            
        # Check bottom clipping if sig_min is significantly below median
        if (median_val - sig_min) > 0.1 * span:
            min_count = np.sum((signal <= (sig_min + tol)) & is_flat_3)
            clip_count += min_count
            
        ratio = clip_count / len(signal)
        is_clipped = ratio > (1.0 - threshold_ratio)
        return bool(is_clipped), float(ratio)

    @staticmethod
    def detect_dc_offset(signal: np.ndarray, max_offset_ratio: float = 0.8) -> Tuple[bool, float]:
        """
        Evaluate if signal has a significant DC offset relative to its RMS value.
        """
        mean_val = float(np.mean(signal))
        rms_val = float(np.sqrt(np.mean(signal ** 2)))
        
        if rms_val <= 1e-12:
            return False, 0.0
            
        ratio = np.abs(mean_val) / rms_val
        is_high = ratio > max_offset_ratio
        return bool(is_high), mean_val

    @staticmethod
    def detect_powerline_interference(signal: np.ndarray, fs: float, powerline_freq: float = 50.0, tolerance_hz: float = 2.0) -> Tuple[bool, float]:
        """
        Analyze spectral density to detect significant powerline noise leakage (50 or 60 Hz).
        
        Returns
        -------
        Tuple[bool, float]
            (has_interference, noise_to_signal_spectral_ratio)
        """
        nperseg = min(1024, len(signal))
        if nperseg < 32:
            return False, 0.0
            
        # Compute Welch PSD
        f, psd = sp_signal.welch(signal - np.mean(signal), fs=fs, nperseg=nperseg)
        
        # Identify frequency bins matching powerline
        mask_powerline = (f >= (powerline_freq - tolerance_hz)) & (f <= (powerline_freq + tolerance_hz))
        mask_other = ~mask_powerline & (f > 0.5) # Exclude DC
        
        if not np.any(mask_powerline) or not np.any(mask_other):
            return False, 0.0
            
        powerline_energy = np.sum(psd[mask_powerline])
        other_energy = np.sum(psd[mask_other])
        
        if other_energy <= 1e-15:
            return powerline_energy > 1e-15, 1.0
            
        ratio = powerline_energy / (powerline_energy + other_energy)
        # If powerline contains more than 5% of total signal energy, flag it
        return bool(ratio > 0.05), float(ratio)

    @staticmethod
    def detect_motion_bursts(signal: np.ndarray, fs: float, window_s: float = 0.5, threshold_std: float = 4.0) -> List[Tuple[float, float]]:
        """
        Detect sudden high-amplitude motion bursts using sliding standard deviation.
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (start_time_s, end_time_s) for identified motion segments.
        """
        n_samples = len(signal)
        win_len = int(np.round(window_s * fs))
        if win_len <= 5:
            win_len = 10
            
        # Calculate sliding standard deviation
        kernel = np.ones(win_len) / win_len
        mean_conv = np.convolve(signal, kernel, mode='same')
        squared_diff = np.square(signal - mean_conv)
        var_conv = np.convolve(squared_diff, kernel, mode='same')
        std_profile = np.sqrt(np.clip(var_conv, 0.0, None))
        
        # Define baseline threshold as median of profile
        med_std = np.median(std_profile)
        overall_std = np.std(signal)
        burst_threshold = max(med_std * threshold_std, 1.8 * overall_std)
        
        is_burst = std_profile > burst_threshold
        
        burst_segments = []
        in_burst = False
        start_idx = 0
        
        for idx in range(len(is_burst)):
            if is_burst[idx]:
                if not in_burst:
                    in_burst = True
                    start_idx = idx
            else:
                if in_burst:
                    in_burst = False
                    if (idx - start_idx) >= int(0.1 * fs): # minimum 100 ms burst
                        burst_segments.append((start_idx / fs, idx / fs))
                        
        if in_burst:
            burst_segments.append((start_idx / fs, len(is_burst) / fs))
            
        return burst_segments


class PhysiologicalValidator:
    """
    Extracts and validates physiological parameters for ECG, EEG, EMG, PPG, EDA, and Respiration signals.
    """

    @staticmethod
    def pan_tompkins_qrs_detector(signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Pan-Tompkins QRS Detection algorithm implementation.
        
        Parameters
        ----------
        signal : np.ndarray
            1-D raw ECG signal array.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            Indices of detected R-peaks.
        """
        # Center signal
        x = signal - np.mean(signal)
        nyq = 0.5 * fs
        
        # 1. Bandpass filter: 5 - 15 Hz
        b_band, a_band = sp_signal.butter(3, [5.0 / nyq, 15.0 / nyq], btype='bandpass')
        x_filt = sp_signal.filtfilt(b_band, a_band, x)
        
        # 2. Derivative Filter
        # Transfer function: H(z) = 0.1*(2 + z^-1 - z^-3 - 2z^-4)
        h_diff = np.array([2.0, 1.0, 0.0, -1.0, -2.0]) / 8.0
        x_diff = np.convolve(x_filt, h_diff, mode='same')
        
        # 3. Squaring Function
        x_sq = x_diff ** 2
        
        # 4. Moving Window Integration (typically 150 ms)
        win_samples = int(np.round(0.150 * fs))
        if win_samples <= 0:
            win_samples = 1
        h_integ = np.ones(win_samples) / win_samples
        x_integ = np.convolve(x_sq, h_integ, mode='same')
        
        # 5. Adaptive Threshold Peak Detection
        max_val = np.max(x_integ)
        prom = 0.1 * max_val if max_val > 1e-9 else None
        peaks, _ = sp_signal.find_peaks(x_integ, distance=int(np.round(0.3 * fs)), prominence=prom)
        if len(peaks) == 0:
            return np.array([], dtype=np.int32)
            
        # Refine peaks to find actual R peaks (local max in original signal)
        r_peaks = []
        search_win = int(np.round(0.1 * fs)) # Search +/- 100 ms around integrated peak
        
        for peak in peaks:
            start = max(0, peak - search_win)
            end = min(len(signal), peak + search_win)
            if start < end:
                r_idx = start + np.argmax(signal[start:end])
                # Double check to prevent duplicate peaks
                if not r_peaks or (r_idx - r_peaks[-1]) > int(0.2 * fs):
                    r_peaks.append(r_idx)
                    
        return np.array(r_peaks, dtype=np.int32)

    @staticmethod
    def validate_ecg(signal: np.ndarray, fs: float) -> Tuple[List[str], Dict[str, Any]]:
        """Extract ECG heart rate, RR interval statistics, and check bounds."""
        warnings_list = []
        metrics = {}
        
        # Support multi-channel: take Lead II (usually channel 1) or channel 0
        if signal.ndim > 1:
            # Prefer second channel if multi-channel (like Lead II in standard ECG)
            ecg_data = signal[1] if signal.shape[0] > 1 else signal[0]
        else:
            ecg_data = signal
            
        r_peaks = PhysiologicalValidator.pan_tompkins_qrs_detector(ecg_data, fs)
        metrics['detected_peaks_count'] = int(len(r_peaks))
        
        if len(r_peaks) < 3:
            warnings_list.append("Insufficient R-peaks detected for heart rate evaluation.")
            metrics['heart_rate_bpm'] = 0.0
            metrics['rr_mean_ms'] = 0.0
            metrics['rr_sdnn_ms'] = 0.0
            return warnings_list, metrics
            
        # Calculate RR intervals
        rr_intervals_s = np.diff(r_peaks) / fs
        rr_intervals_ms = rr_intervals_s * 1000.0
        
        rr_mean = float(np.mean(rr_intervals_ms))
        rr_sdnn = float(np.std(rr_intervals_ms))
        hr = float(60.0 / (rr_mean / 1000.0))
        
        metrics['heart_rate_bpm'] = hr
        metrics['rr_mean_ms'] = rr_mean
        metrics['rr_sdnn_ms'] = rr_sdnn
        
        # Physiological warnings
        if hr < 40.0:
            warnings_list.append(f"Bradycardia detected: Heart rate ({hr:.1f} bpm) is below physiological threshold.")
        elif hr > 180.0:
            warnings_list.append(f"Extreme Tachycardia detected: Heart rate ({hr:.1f} bpm) is critically elevated.")
            
        if rr_sdnn > 100.0:
            warnings_list.append(f"Abnormal RR variability (SDNN: {rr_sdnn:.1f} ms) indicates potential arrhythmia (e.g. AFib).")
            
        return warnings_list, metrics

    @staticmethod
    def validate_eeg(signal: np.ndarray, fs: float) -> Tuple[List[str], Dict[str, Any]]:
        """Evaluate EEG spectral band power ratios."""
        warnings_list = []
        metrics = {}
        
        # Take first channel for analysis
        eeg_data = signal[0] if signal.ndim > 1 else signal
        
        nperseg = min(1024, len(eeg_data))
        if nperseg < 64:
            warnings_list.append("Signal is too short for EEG spectral band calculation.")
            return warnings_list, metrics
            
        f, psd = sp_signal.welch(eeg_data - np.mean(eeg_data), fs=fs, nperseg=nperseg)
        
        # Bands definition
        bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 12.0),
            'beta': (12.0, 30.0),
            'gamma': (30.0, 50.0)
        }
        
        total_power = 1e-15
        band_powers = {}
        
        for name, (fmin, fmax) in bands.items():
            mask = (f >= fmin) & (f <= fmax)
            if np.any(mask):
                # Trapezoidal integration
                power = float(_trapz(psd[mask], f[mask]))
            else:
                power = 0.0
            band_powers[name] = power
            total_power += power
            
        # Relative band power
        rel_powers = {}
        for name, power in band_powers.items():
            rel_powers[f'rel_{name}'] = power / total_power
            
        metrics.update(rel_powers)
        metrics['total_spectral_power'] = total_power
        
        # Check abnormal ratios
        # e.g., if awake, alpha or beta should be present; sleep delta is high.
        # If delta relative power is > 0.8, raise deep sleep / coma alert
        if rel_powers['rel_delta'] > 0.8:
            warnings_list.append(f"Dominant Delta activity ({rel_powers['rel_delta']*100:.1f}%) suggests deep slow-wave sleep or pathological suppression.")
        if rel_powers['rel_beta'] > 0.6:
            warnings_list.append(f"Elevated Beta power ({rel_powers['rel_beta']*100:.1f}%) suggests significant cognitive activation, tension, or drug influence.")
            
        return warnings_list, metrics

    @staticmethod
    def validate_emg(signal: np.ndarray, fs: float) -> Tuple[List[str], Dict[str, Any]]:
        """Calculate EMG spectral indices (Mean and Median frequency)."""
        warnings_list = []
        metrics = {}
        
        emg_data = signal[0] if signal.ndim > 1 else signal
        
        nperseg = min(1024, len(emg_data))
        if nperseg < 64:
            warnings_list.append("EMG signal is too short for mean/median frequency tracking.")
            return warnings_list, metrics
            
        f, psd = sp_signal.welch(emg_data - np.mean(emg_data), fs=fs, nperseg=nperseg)
        
        # Standard active EMG frequency range is 10 - 250 Hz
        mask = (f >= 10.0) & (f <= min(250.0, 0.5 * fs))
        f_masked = f[mask]
        psd_masked = psd[mask]
        
        if len(psd_masked) == 0:
            metrics['mean_frequency_hz'] = 0.0
            metrics['median_frequency_hz'] = 0.0
            return warnings_list, metrics
            
        # 1. Mean Frequency (MNF)
        mnf = float(np.sum(f_masked * psd_masked) / np.sum(psd_masked))
        
        # 2. Median Frequency (MDF)
        cumulative_power = np.cumsum(psd_masked)
        total_half = cumulative_power[-1] / 2.0
        mdf_idx = np.where(cumulative_power >= total_half)[0][0]
        mdf = float(f_masked[mdf_idx])
        
        metrics['mean_frequency_hz'] = mnf
        metrics['median_frequency_hz'] = mdf
        
        # Check for muscle fatigue indicators
        # Healthy active EMG has median frequency in range [70, 150] Hz
        if mdf < 60.0:
            warnings_list.append(f"Low EMG median frequency ({mdf:.1f} Hz) suggests significant muscle fatigue or structural pathology.")
            
        return warnings_list, metrics

    @staticmethod
    def validate_ppg(signal: np.ndarray, fs: float) -> Tuple[List[str], Dict[str, Any]]:
        """Evaluate PPG pulse rate and verify waveform features."""
        warnings_list = []
        metrics = {}
        
        ppg_data = signal[0] if signal.ndim > 1 else signal
        
        # Filter PPG between 0.5 and 5 Hz to remove DC drift and high frequencies
        nyq = 0.5 * fs
        b, a = sp_signal.butter(3, [0.5 / nyq, 5.0 / nyq], btype='bandpass')
        ppg_filt = sp_signal.filtfilt(b, a, ppg_data)
        
        # Detect systolic peaks
        peaks, _ = sp_signal.find_peaks(ppg_filt, distance=int(np.round(0.4 * fs)), prominence=0.1 * np.std(ppg_filt))
        
        metrics['detected_pulses_count'] = int(len(peaks))
        
        if len(peaks) < 3:
            warnings_list.append("Insufficient PPG systolic peaks for pulse rate estimation.")
            metrics['pulse_rate_bpm'] = 0.0
            return warnings_list, metrics
            
        intervals = np.diff(peaks) / fs
        mean_interval = float(np.mean(intervals))
        pr = 60.0 / mean_interval
        
        metrics['pulse_rate_bpm'] = pr
        
        if pr < 40.0:
            warnings_list.append(f"Low pulse rate ({pr:.1f} bpm) detected in PPG.")
        elif pr > 160.0:
            warnings_list.append(f"Elevated pulse rate ({pr:.1f} bpm) detected in PPG.")
            
        return warnings_list, metrics

    @staticmethod
    def validate_eda(signal: np.ndarray, fs: float) -> Tuple[List[str], Dict[str, Any]]:
        """Evaluate Electrodermal Activity (EDA) characteristics."""
        warnings_list = []
        metrics = {}
        
        eda_data = signal[0] if signal.ndim > 1 else signal
        
        mean_val = float(np.mean(eda_data))
        min_val = float(np.min(eda_data))
        max_val = float(np.max(eda_data))
        
        metrics['eda_mean_microsiemens'] = mean_val
        metrics['eda_range'] = max_val - min_val
        
        # Standard physiological values: 0.1 to 30 uS (up to 50 in high stress)
        if mean_val < 0.01:
            warnings_list.append(f"EDA skin conductance ({mean_val:.3f} uS) is abnormally close to zero (potential electrode detachment).")
        elif mean_val > 60.0:
            warnings_list.append(f"EDA skin conductance ({mean_val:.1f} uS) exceeds typical maximum physiological levels.")
            
        return warnings_list, metrics

    @staticmethod
    def validate_resp(signal: np.ndarray, fs: float) -> Tuple[List[str], Dict[str, Any]]:
        """Validate respiratory rate and amplitude breathing patterns."""
        warnings_list = []
        metrics = {}
        
        resp_data = signal[0] if signal.ndim > 1 else signal
        
        # Filter: lowpass at 1 Hz
        nyq = 0.5 * fs
        b, a = sp_signal.butter(3, 1.0 / nyq, btype='lowpass')
        resp_filt = sp_signal.filtfilt(b, a, resp_data)
        
        # Peak detection for breathing cycles
        peaks, _ = sp_signal.find_peaks(resp_filt, distance=int(np.round(0.5 * fs)), prominence=0.1 * np.std(resp_filt))
        
        metrics['breathing_cycles_count'] = int(len(peaks))
        
        if len(peaks) < 2:
            warnings_list.append("Insufficient breathing cycles to determine respiration rate.")
            metrics['respiration_rate_cpm'] = 0.0
            return warnings_list, metrics
            
        intervals = np.diff(peaks) / fs
        mean_int = float(np.mean(intervals))
        rr = 60.0 / mean_int # Breaths per minute (CPM)
        
        metrics['respiration_rate_cpm'] = rr
        
        if rr < 6.0:
            warnings_list.append(f"Bradypnea detected: Respiration rate ({rr:.1f} cpm) is below normal limits.")
        elif rr > 40.0:
            warnings_list.append(f"Tachypnea detected: Respiration rate ({rr:.1f} cpm) is critically high.")
            
        return warnings_list, metrics


def validate_config(config: Any) -> None:
    """
    Validate parameter ranges in configuration model schemas.
    """
    if hasattr(config, '__post_init__'):
        config.__post_init__()
        
    if hasattr(config, 'fs') and config.fs is not None:
        if config.fs <= 0:
            raise ValueError("fs must be greater than 0")
    if hasattr(config, 'duration_s') and config.duration_s is not None:
        if config.duration_s <= 0:
            raise ValueError("duration_s must be greater than 0")


def validate_physiological_bounds(signal: np.ndarray, fs: float, signal_type: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Perform biological band/feature boundaries validation.
    """
    sig_t = signal_type.lower()
    if sig_t == 'ecg':
        return PhysiologicalValidator.validate_ecg(signal, fs)
    elif sig_t == 'eeg':
        return PhysiologicalValidator.validate_eeg(signal, fs)
    elif sig_t == 'emg':
        return PhysiologicalValidator.validate_emg(signal, fs)
    elif sig_t == 'ppg':
        return PhysiologicalValidator.validate_ppg(signal, fs)
    elif sig_t == 'eda':
        return PhysiologicalValidator.validate_eda(signal, fs)
    elif sig_t == 'resp':
        return PhysiologicalValidator.validate_resp(signal, fs)
    else:
        return [], {}


def validate_signal(signal: np.ndarray, fs: float, expected_type: str) -> ValidationReport:
    """
    Analyze signal for engineering integrity and physiological feasibility.
    
    Parameters
    ----------
    signal : np.ndarray
        The input biopotential signal.
    fs : float
        Sampling frequency in Hz.
    expected_type : str
        The expected biological type (e.g. 'ecg', 'eeg', 'emg', 'ppg', 'eda', 'resp').
        
    Returns
    -------
    ValidationReport
        A report containing warning lists and extracted physiological metrics.
    """
    warnings = []
    metrics = {}
    
    # 1. Engineering Integrity Checks
    if np.any(np.isnan(signal)):
        warnings.append("Signal contains NaN values.")
    if np.any(np.isinf(signal)):
        warnings.append("Signal contains Infinite values.")
        
    # Check clipping
    is_clipped, clip_ratio = SignalIntegrityChecker.detect_clipping(signal)
    metrics['clipping_ratio'] = clip_ratio
    if is_clipped:
        warnings.append(f"Signal is clipped (Saturated values ratio: {clip_ratio*100:.2f}%).")
        
    # Check DC Offset
    if expected_type.lower() not in ['eda', 'scl']:
        is_dc, dc_val = SignalIntegrityChecker.detect_dc_offset(signal)
        metrics['dc_offset'] = dc_val
        if is_dc:
            warnings.append(f"Significant DC offset detected: {dc_val:.4f} mV.")
            
    # Check Flatline
    flat_min_duration = 1.0 if expected_type.lower() in ['ecg', 'ppg', 'resp'] else 0.5
    flatlines = SignalIntegrityChecker.detect_flatline(signal, fs, min_duration_s=flat_min_duration)
    metrics['flatline_segments'] = flatlines
    if len(flatlines) > 0:
        warnings.append(f"Flatline segments detected: {len(flatlines)} instances found.")
        
    # Check Powerline Interference
    has_interference, pl_ratio = SignalIntegrityChecker.detect_powerline_interference(signal, fs)
    metrics['powerline_interference_ratio'] = pl_ratio
    if has_interference:
        warnings.append(f"Significant powerline interference detected (Spectral ratio: {pl_ratio*100:.1f}%).")
        
    # Check Motion Bursts
    motion_bursts = SignalIntegrityChecker.detect_motion_bursts(signal, fs)
    metrics['motion_bursts_count'] = len(motion_bursts)
    if len(motion_bursts) > 0:
        warnings.append(f"Sudden motion artifact bursts detected: {len(motion_bursts)} segments identified.")

    # 2. Physiological Validation
    sig_t = expected_type.lower()
    phys_warnings = []
    phys_metrics = {}
    
    if sig_t == 'ecg':
        phys_warnings, phys_metrics = PhysiologicalValidator.validate_ecg(signal, fs)
    elif sig_t == 'eeg':
        phys_warnings, phys_metrics = PhysiologicalValidator.validate_eeg(signal, fs)
    elif sig_t == 'emg':
        phys_warnings, phys_metrics = PhysiologicalValidator.validate_emg(signal, fs)
    elif sig_t == 'ppg':
        phys_warnings, phys_metrics = PhysiologicalValidator.validate_ppg(signal, fs)
    elif sig_t == 'eda':
        phys_warnings, phys_metrics = PhysiologicalValidator.validate_eda(signal, fs)
    elif sig_t == 'resp':
        phys_warnings, phys_metrics = PhysiologicalValidator.validate_resp(signal, fs)
        
    warnings.extend(phys_warnings)
    metrics.update(phys_metrics)
    
    # Re-evaluate validity
    is_valid = len(warnings) == 0
    
    return ValidationReport(is_valid=is_valid, warnings=warnings, metrics=metrics)


def generate_validation_report_html(record: "SignalRecord", report: ValidationReport, path: str) -> None:
    """
    Generate a beautifully styled HTML quality validation report.
    
    Parameters
    ----------
    record : SignalRecord
        The SignalRecord being validated.
    report : ValidationReport
        The corresponding validation report.
    path : str
        Target file path.
    """
    status_class = "valid" if report.is_valid else "invalid"
    status_text = "PASSED (Physiologically Valid)" if report.is_valid else "WARNING (Quality Issues Identified)"
    
    warnings_list_html = ""
    if report.warnings:
        for w in report.warnings:
            warnings_list_html += f"<li>{w}</li>\n"
    else:
        warnings_list_html = "<li>No active warnings. Signal is clean and physiologically coherent.</li>"
        
    metrics_rows_html = ""
    for k, v in report.metrics.items():
        if isinstance(v, float):
            v_str = f"{v:.4f}"
        else:
            v_str = str(v)
        metrics_rows_html += f"<tr><td><strong>{k}</strong></td><td>{v_str}</td></tr>\n"
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BioSignal Quality Validation Report</title>
    <style>
        body {{
            font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
            background-color: #0d0f12;
            color: #e2e8f0;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: rgba(25, 30, 40, 0.65);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            backdrop-filter: blur(12px);
            padding: 40px;
        }}
        h1 {{
            color: #ffffff;
            font-size: 28px;
            margin-top: 0;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 15px;
        }}
        .badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 30px;
            font-weight: bold;
            font-size: 14px;
            margin-bottom: 25px;
            text-transform: uppercase;
        }}
        .badge.valid {{
            background-color: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
        }}
        .badge.invalid {{
            background-color: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
            border: 1px solid #f59e0b;
        }}
        .section-title {{
            color: #6366f1;
            font-size: 18px;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .warnings-box {{
            background: rgba(245, 158, 11, 0.05);
            border-left: 4px solid #f59e0b;
            padding: 20px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 30px;
        }}
        .warnings-box ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .warnings-box li {{
            margin-bottom: 8px;
            color: #cbd5e1;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}
        th {{
            background-color: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .meta-item {{
            background: rgba(255, 255, 255, 0.02);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }}
        .meta-item span {{
            display: block;
            font-size: 12px;
            color: #64748b;
            margin-bottom: 4px;
        }}
        .meta-item strong {{
            font-size: 16px;
            color: #cbd5e1;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #475569;
            margin-top: 40px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BioSignal Simulator — Validation & Integrity Report</h1>
        
        <div class="badge {status_class}">{status_text}</div>
        
        <div class="section-title">Signal Specifications</div>
        <div class="meta-grid">
            <div class="meta-item">
                <span>Signal Type</span>
                <strong>{record.signal_type.upper()}</strong>
            </div>
            <div class="meta-item">
                <span>Sampling Rate</span>
                <strong>{record.fs:.1f} Hz</strong>
            </div>
            <div class="meta-item">
                <span>Duration</span>
                <strong>{record.t[-1]:.2f} seconds</strong>
            </div>
            <div class="meta-item">
                <span>Total Samples</span>
                <strong>{len(record.t)} points</strong>
            </div>
        </div>
        
        <div class="section-title">Active Validation Warnings</div>
        <div class="warnings-box">
            <ul>
                {warnings_list_html}
            </ul>
        </div>
        
        <div class="section-title">Extracted Physiological & Quality Metrics</div>
        <table>
            <thead>
                <tr>
                    <th>Metric Name</th>
                    <th>Measured Value</th>
                </tr>
            </thead>
            <tbody>
                {metrics_rows_html}
            </tbody>
        </table>
        
        <div class="footer">
            Generated by BioSignal Simulator Validation Engine &bull; {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
    </div>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_content)
