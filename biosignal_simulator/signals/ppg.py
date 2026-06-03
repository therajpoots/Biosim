"""
High-fidelity Photoplethysmogram (PPG) simulator.

This module provides a physiologically accurate simulation of the photoplethysmogram
(PPG) waveform, as recorded by optical sensors placed on the skin surface.

Physiological Basis
-------------------
PPG measures changes in blood volume in the microvascular bed of tissue.
An LED illuminates the tissue; the photodetector measures variations in light
absorption that are synchronous with the cardiac cycle.

The PPG waveform has three characteristic features per cardiac cycle:

1. **Systolic Peak**: Rapid increase in blood volume following left ventricular
   ejection. Rise time is brief (~100-150 ms), amplitude is maximum.
2. **Dicrotic Notch**: Brief dip following the systolic peak, caused by the
   closure of the aortic valve (end of systole).
3. **Diastolic Peak** (if present): A secondary, lower peak from the reflected
   pressure wave returning from the periphery.

Respiratory Modulation
----------------------
The PPG signal is modulated by respiration in two ways:
1. **Amplitude Modulation (AM)**: Stroke volume varies with intrathoracic pressure
   during breathing (pulsus paradoxus). Peak amplitude varies by ±10-30%.
2. **Baseline Wander**: DC shift of the PPG baseline following respiration frequency.

Pathological Variations
------------------------
- **SpO2 Desaturation**: Affects waveform morphology through peripheral vasoconstriction
- **Atrial Fibrillation**: Irregular beat-to-beat intervals in PPG
- **Hypertension**: Increased systolic peak amplitude, earlier reflected wave
- **Arterial Stiffness (Aging)**: Earlier dicrotic notch, higher augmentation index

Signal Variants
---------------
1. **Transmission PPG**: Measured through tissue (fingertip), high amplitude
2. **Reflection PPG**: Measured via back-reflection (wrist, forehead), different morphology
3. **First Derivative (VPG)**: Velocity Plethysmogram for vascular assessment
4. **Second Derivative (APG)**: Acceleration PPG, used for aging assessment

References
----------
- Allen, J. (2007). Photoplethysmography and its application in clinical physiological
  measurement. Physiol. Meas.
- Elgendi, M. (2012). On the Analysis of Fingertip Photoplethysmogram Signals. Curr.
  Cardiol. Rev.
- Charlton, P.H. et al. (2018). Wearable Photoplethysmography for Cardiovascular
  Monitoring. Proceedings of the IEEE.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

from biosignal_simulator.core.base import BaseSignal
from biosignal_simulator.core.config import PPGConfig
from biosignal_simulator.core.math_utils import normalize_to_rms
from biosignal_simulator.utils.validation import validate_config


# ──────────────────────────────────────────────────────────────────────────────
# PPG Waveform Template Functions
# ──────────────────────────────────────────────────────────────────────────────

def ppg_systolic_peak(
    t_local: np.ndarray,
    peak_time: float,
    amplitude: float,
    rise_sigma: float,
    fall_sigma: float,
) -> np.ndarray:
    """
    Generate the asymmetric systolic peak of the PPG waveform.

    The systolic peak rises steeply (as blood is ejected from the left ventricle)
    and falls more slowly (as blood is distributed to peripheral capillaries).
    This asymmetry is captured using a split-Gaussian model.

    Parameters
    ----------
    t_local : np.ndarray
        Local time within the cardiac cycle.
    peak_time : float
        Time of the systolic peak within the cycle.
    amplitude : float
        Peak amplitude (normalized).
    rise_sigma : float
        Width of the rising limb (standard deviation, seconds).
    fall_sigma : float
        Width of the falling limb (standard deviation, seconds).

    Returns
    -------
    np.ndarray
        Systolic peak waveform values.

    Notes
    -----
    Uses a split-Gaussian shape:
    - For t < peak_time: Gaussian with rise_sigma
    - For t >= peak_time: Gaussian with fall_sigma
    """
    result = np.zeros_like(t_local, dtype=float)
    pre_mask = t_local < peak_time
    post_mask = ~pre_mask
    result[pre_mask] = amplitude * np.exp(
        -0.5 * ((t_local[pre_mask] - peak_time) / max(rise_sigma, 1e-9)) ** 2
    )
    result[post_mask] = amplitude * np.exp(
        -0.5 * ((t_local[post_mask] - peak_time) / max(fall_sigma, 1e-9)) ** 2
    )
    return result


def ppg_dicrotic_notch(
    t_local: np.ndarray,
    notch_time: float,
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    """
    Generate the dicrotic notch (inverted Gaussian).

    The dicrotic notch is a transient dip in the PPG waveform corresponding to
    aortic valve closure. Its depth and timing vary with arterial compliance.

    Parameters
    ----------
    t_local : np.ndarray
        Local time within the cardiac cycle.
    notch_time : float
        Time of the notch center.
    amplitude : float
        Notch depth (positive value, will be subtracted).
    sigma : float
        Width of the notch in seconds.

    Returns
    -------
    np.ndarray
        Dicrotic notch contribution (negative deflection).
    """
    return -amplitude * np.exp(
        -0.5 * ((t_local - notch_time) / max(sigma, 1e-9)) ** 2
    )


def ppg_diastolic_peak(
    t_local: np.ndarray,
    dia_time: float,
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    """
    Generate the diastolic peak (reflected pressure wave).

    The diastolic hump represents the return of the pressure pulse wave reflected
    from the periphery. It is more prominent in young, compliant arteries and
    diminishes with age and arterial stiffness.

    Parameters
    ----------
    t_local : np.ndarray
        Local time within cardiac cycle.
    dia_time : float
        Time of the diastolic peak.
    amplitude : float
        Diastolic peak amplitude.
    sigma : float
        Width of the diastolic peak.

    Returns
    -------
    np.ndarray
        Diastolic peak waveform.
    """
    return amplitude * np.exp(
        -0.5 * ((t_local - dia_time) / max(sigma, 1e-9)) ** 2
    )


def compute_beat_ppg_waveform(
    t_local: np.ndarray,
    rr_interval: float,
    config: "PPGConfig",
    amplitude_scale: float = 1.0,
    stiffness_factor: float = 0.0,
) -> np.ndarray:
    """
    Compute the PPG waveform for a single cardiac beat.

    Parameters
    ----------
    t_local : np.ndarray
        Time array local to the beat cycle (0 to rr_interval).
    rr_interval : float
        Duration of the cardiac cycle in seconds.
    config : PPGConfig
        PPG configuration.
    amplitude_scale : float
        Amplitude scaling for this beat (respiratory modulation).
    stiffness_factor : float
        Arterial stiffness factor (0=young/compliant, 1=stiff/elderly).
        Higher stiffness → earlier notch, higher augmentation index.

    Returns
    -------
    np.ndarray
        PPG waveform for this beat.
    """
    T = rr_interval

    # Peak timings (relative to cycle start)
    sys_peak = config.systolic_fraction * T
    # Dicrotic notch shifts earlier with arterial stiffness
    dic_frac = config.dicrotic_fraction - 0.07 * stiffness_factor
    dic_frac = np.clip(dic_frac, config.systolic_fraction + 0.05, 0.75)
    notch_time = dic_frac * T
    dia_time = (dic_frac + 0.10 + 0.05 * (1.0 - stiffness_factor)) * T
    dia_time = np.clip(dia_time, notch_time + 0.02 * T, 0.92 * T)

    # Widths (scale with cycle duration)
    rise_sigma = 0.04 * T
    fall_sigma = 0.08 * T
    notch_sigma = 0.025 * T
    dia_sigma = 0.07 * T

    # Amplitudes
    # With stiffness: diastolic amplitude increases (reflected wave arrives earlier, larger)
    dia_amp_base = 0.22 + 0.12 * stiffness_factor

    sys_wave = ppg_systolic_peak(
        t_local, sys_peak, amplitude_scale, rise_sigma, fall_sigma
    )
    notch_wave = ppg_dicrotic_notch(t_local, notch_time, 0.14 * amplitude_scale, notch_sigma)
    dia_wave = ppg_diastolic_peak(t_local, dia_time, dia_amp_base * amplitude_scale, dia_sigma)

    return sys_wave + notch_wave + dia_wave


# ──────────────────────────────────────────────────────────────────────────────
# PPG Derivative Signals
# ──────────────────────────────────────────────────────────────────────────────

def compute_vpg(ppg: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute the Velocity Plethysmogram (VPG = dPPG/dt).

    The first derivative of the PPG reflects blood flow velocity and is used
    for analysis of vascular tone and waveform features.

    Parameters
    ----------
    ppg : np.ndarray
        PPG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        VPG signal (same length as PPG).
    """
    return np.gradient(ppg, 1.0 / fs)


def compute_apg(ppg: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute the Acceleration Plethysmogram (APG = d²PPG/dt²).

    The second derivative of the PPG contains features (a, b, c, d, e waves)
    used to assess arterial stiffness and aging.

    Parameters
    ----------
    ppg : np.ndarray
        PPG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        APG signal.
    """
    dt = 1.0 / fs
    return np.gradient(np.gradient(ppg, dt), dt)


def compute_ppg_features(ppg: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract standard PPG waveform features.

    Computes time-domain and spectral features from the PPG signal that
    are used in clinical analysis and wearable health monitoring.

    Parameters
    ----------
    ppg : np.ndarray
        PPG signal (single channel).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'peak_amplitude': Mean systolic peak amplitude
        - 'trough_amplitude': Mean diastolic trough amplitude
        - 'pulse_amplitude': Mean (peak - trough)
        - 'mean_hr_bpm': Estimated heart rate (bpm)
        - 'perfusion_index': Peak amplitude / mean amplitude (%)
        - 'spectral_entropy': Entropy of normalized PSD
        - 'pth_ratio': Peak-to-trough ratio
    """
    # Find peaks (systolic peaks)
    min_dist = int(0.4 * fs)  # Minimum 400 ms between peaks
    ppg_norm = ppg / (np.max(np.abs(ppg)) + 1e-12)

    peaks_idx, _ = sp_signal.find_peaks(ppg_norm, distance=min_dist, prominence=0.2)
    troughs_idx, _ = sp_signal.find_peaks(-ppg_norm, distance=min_dist, prominence=0.1)

    if len(peaks_idx) < 2:
        return {
            "peak_amplitude": 0.0, "trough_amplitude": 0.0,
            "pulse_amplitude": 0.0, "mean_hr_bpm": 0.0,
            "perfusion_index": 0.0, "spectral_entropy": 0.0, "pth_ratio": 0.0,
        }

    peak_amps = ppg[peaks_idx]
    trough_amps = ppg[troughs_idx] if len(troughs_idx) > 0 else np.array([np.min(ppg)])

    mean_peak = float(np.mean(peak_amps))
    mean_trough = float(np.mean(trough_amps))
    pulse_amp = mean_peak - mean_trough

    # Heart rate from peak-to-peak intervals
    rr_samples = np.diff(peaks_idx)
    rr_s = rr_samples / fs
    mean_hr = float(60.0 / np.mean(rr_s)) if len(rr_s) > 0 else 0.0

    # Perfusion index: (peak_amp / mean_amp) * 100
    mean_abs_amp = float(np.mean(np.abs(ppg)))
    pi = (pulse_amp / max(mean_abs_amp, 1e-12)) * 100.0

    # Spectral entropy
    nperseg = min(256, len(ppg) // 4)
    _, psd = sp_signal.welch(ppg, fs=fs, nperseg=nperseg)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))

    return {
        "peak_amplitude": mean_peak,
        "trough_amplitude": mean_trough,
        "pulse_amplitude": pulse_amp,
        "mean_hr_bpm": mean_hr,
        "perfusion_index": pi,
        "spectral_entropy": spectral_entropy,
        "pth_ratio": mean_peak / max(abs(mean_trough), 1e-12),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main PPG Generator
# ──────────────────────────────────────────────────────────────────────────────

class PPGGenerator(BaseSignal):
    """
    High-fidelity Photoplethysmogram (PPG) generator.

    Generates physiologically realistic PPG waveforms with:
    - Beat-by-beat synthesis using split-Gaussian templates
    - Heart rate variability (HRV) using lognormal RR intervals
    - Respiratory amplitude modulation and baseline drift
    - Dicrotic notch and diastolic peak
    - Arterial stiffness and aging effects
    - Rhythm variations (AFib irregular intervals)

    Parameters
    ----------
    config : PPGConfig
        PPG simulation configuration.

    Examples
    --------
    >>> from biosignal_simulator.core.config import PPGConfig
    >>> config = PPGConfig(fs=125.0, duration_s=30.0, heart_rate=72.0)
    >>> gen = PPGGenerator(config)
    >>> ppg = gen.generate()
    >>> ppg.shape
    (3750,)
    """

    def __init__(self, config: PPGConfig) -> None:
        validate_config(config)
        super().__init__(fs=config.fs, duration_s=config.duration_s, seed=config.seed)
        self.config = config

    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate PPG configuration parameters.

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message).
        """
        try:
            self.config.__post_init__()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def generate(self) -> np.ndarray:
        """
        Generate the PPG signal.

        Returns
        -------
        np.ndarray
            PPG signal of shape (n_samples,), normalized to unit RMS.

        Notes
        -----
        Generation pipeline:
        1. Generate RR intervals with HRV (lognormal distribution)
        2. Compute beat onset times
        3. For each sample, compute PPG value from the template of its containing beat
        4. Apply respiratory amplitude modulation
        5. Apply baseline wander (respiratory drift)
        6. Apply stiffness effects if configured
        7. Normalize to unit RMS
        """
        t = self.t
        fs = self.fs
        n_samples = self.n_samples
        config = self.config

        hr = config.heart_rate
        rr_mean = 60.0 / hr

        # HRV parameters
        hrv_cv = 0.04  # Coefficient of variation for normal HRV
        sigma_ln = np.sqrt(np.log(1.0 + hrv_cv ** 2))
        mu_ln = np.log(rr_mean) - 0.5 * sigma_ln ** 2

        # Generate enough beats to cover the duration
        n_beats = int(self.duration_s / rr_mean) + 15

        # Rhythm-dependent RR generation
        if hasattr(config, 'rhythm') and config.rhythm == 'afib':
            # AFib: gamma-distributed RR intervals (highly irregular)
            k_shape = 2.0
            rr_intervals = self.rng.gamma(shape=k_shape, scale=rr_mean / k_shape, size=n_beats)
            rr_intervals = np.clip(rr_intervals, 0.3, 2.5)
        else:
            # Normal HRV: lognormal RR
            rr_intervals = self.rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_beats)
            rr_intervals = np.clip(rr_intervals, 0.35, 2.0)

        # Beat onset times
        beat_onsets = np.concatenate([[0.0], np.cumsum(rr_intervals[:-1])])

        # Respiratory modulation parameters
        m_resp = config.resp_modulation
        f_resp = config.resp_rate
        resp_phase = 2.0 * np.pi * f_resp * t
        resp_envelope = 1.0 + m_resp * np.sin(resp_phase)
        resp_drift = 0.08 * m_resp * np.sin(resp_phase + np.pi * 0.5)  # Phase-shifted drift

        # Arterial stiffness factor (0=young, 1=elderly)
        stiffness = getattr(config, 'stiffness_factor', 0.0)

        # Build PPG sample-by-sample
        ppg_raw = np.zeros(n_samples)
        beat_idx_arr = np.searchsorted(beat_onsets, t) - 1
        beat_idx_arr = np.clip(beat_idx_arr, 0, len(beat_onsets) - 2)

        for sample_i in range(n_samples):
            bi = beat_idx_arr[sample_i]
            t_start = beat_onsets[bi]
            t_end = beat_onsets[bi + 1] if bi + 1 < len(beat_onsets) else t_start + rr_mean
            T_beat = max(t_end - t_start, 0.3)
            t_local_i = t[sample_i] - t_start
            if t_local_i < 0 or t_local_i > T_beat:
                continue

            # Build a single-sample local window (compute at sample level)
            t_local_arr = np.array([t_local_i])
            amp_scale = float(resp_envelope[sample_i])
            ppg_raw[sample_i] = compute_beat_ppg_waveform(
                t_local=t_local_arr,
                rr_interval=T_beat,
                config=config,
                amplitude_scale=amp_scale,
                stiffness_factor=stiffness,
            )[0]

        # Add respiratory baseline drift
        ppg_signal = ppg_raw + resp_drift

        # Center and normalize
        ppg_signal -= np.mean(ppg_signal)
        return normalize_to_rms(ppg_signal, 1.0)

    def generate_fast(self) -> np.ndarray:
        """
        Generate PPG signal using vectorized beat rendering (faster than sample loop).

        Renders full beats as waveform arrays and stitches them together.
        Less accurate than per-sample generation for non-uniform RR intervals,
        but significantly faster.

        Returns
        -------
        np.ndarray
            PPG signal.
        """
        t = self.t
        n_samples = self.n_samples
        config = self.config

        hr = config.heart_rate
        rr_mean = 60.0 / hr

        hrv_cv = 0.04
        sigma_ln = np.sqrt(np.log(1.0 + hrv_cv ** 2))
        mu_ln = np.log(rr_mean) - 0.5 * sigma_ln ** 2

        n_beats = int(self.duration_s / rr_mean) + 10
        rr_intervals = self.rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_beats)
        rr_intervals = np.clip(rr_intervals, 0.35, 2.0)

        beat_onsets = np.cumsum(np.concatenate([[0.0], rr_intervals]))
        beat_onsets = beat_onsets[beat_onsets < self.duration_s + 1.0]

        ppg_raw = np.zeros(n_samples)
        fs = self.fs
        stiffness = getattr(config, 'stiffness_factor', 0.0)

        for bi, t_start in enumerate(beat_onsets[:-1]):
            t_end = beat_onsets[bi + 1]
            T_beat = max(t_end - t_start, 0.3)

            start_idx = max(0, int(t_start * fs))
            end_idx = min(n_samples, int(t_end * fs) + 1)
            if start_idx >= end_idx:
                continue

            t_local = t[start_idx:end_idx] - t_start
            beat_wave = compute_beat_ppg_waveform(
                t_local=t_local,
                rr_interval=T_beat,
                config=config,
                amplitude_scale=1.0,
                stiffness_factor=stiffness,
            )
            ppg_raw[start_idx:end_idx] += beat_wave

        # Respiratory modulation
        m_resp = config.resp_modulation
        f_resp = config.resp_rate
        resp_envelope = 1.0 + m_resp * np.sin(2.0 * np.pi * f_resp * t)
        resp_drift = 0.08 * m_resp * np.sin(2.0 * np.pi * f_resp * t + np.pi * 0.5)

        ppg_signal = ppg_raw * resp_envelope + resp_drift
        ppg_signal -= np.mean(ppg_signal)
        return normalize_to_rms(ppg_signal, 1.0)

    def get_vpg(self) -> np.ndarray:
        """
        Get the Velocity Plethysmogram (first derivative).

        Returns
        -------
        np.ndarray
            VPG signal.
        """
        return compute_vpg(self.generate(), self.fs)

    def get_apg(self) -> np.ndarray:
        """
        Get the Acceleration Plethysmogram (second derivative).

        Returns
        -------
        np.ndarray
            APG signal.
        """
        return compute_apg(self.generate(), self.fs)

    def compute_features(self) -> Dict[str, float]:
        """
        Generate signal and compute PPG features.

        Returns
        -------
        Dict[str, float]
            PPG feature dictionary.
        """
        return compute_ppg_features(self.generate_fast(), self.fs)

    def summary(self) -> Dict[str, Any]:
        """
        Return configuration summary.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary.
        """
        return {
            "heart_rate_bpm": self.config.heart_rate,
            "resp_rate_hz": self.config.resp_rate,
            "resp_modulation": self.config.resp_modulation,
            "systolic_fraction": self.config.systolic_fraction,
            "dicrotic_fraction": self.config.dicrotic_fraction,
            "fs_hz": self.fs,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Factory Functions
# ──────────────────────────────────────────────────────────────────────────────

def make_ppg_normal(
    duration_s: float = 30.0,
    heart_rate: float = 72.0,
    fs: float = 125.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a normal PPG signal.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    heart_rate : float
        Heart rate in bpm.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Normal PPG signal.
    """
    config = PPGConfig(
        fs=fs, duration_s=duration_s, heart_rate=heart_rate, seed=seed,
    )
    return PPGGenerator(config).generate_fast()


def make_ppg_tachycardia(
    duration_s: float = 30.0,
    heart_rate: float = 110.0,
    fs: float = 125.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a tachycardic PPG signal.

    Parameters
    ----------
    duration_s : float
        Duration.
    heart_rate : float
        Heart rate (should be > 100 bpm for tachycardia).
    fs : float
        Sampling frequency.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Tachycardic PPG.
    """
    config = PPGConfig(
        fs=fs, duration_s=duration_s, heart_rate=heart_rate, seed=seed,
    )
    return PPGGenerator(config).generate_fast()


def make_ppg_bradycardia(
    duration_s: float = 30.0,
    heart_rate: float = 45.0,
    fs: float = 125.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a bradycardic PPG signal.

    Parameters
    ----------
    duration_s : float
        Duration.
    heart_rate : float
        Heart rate (< 60 bpm).
    fs : float
        Sampling frequency.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Bradycardic PPG.
    """
    config = PPGConfig(
        fs=fs, duration_s=duration_s, heart_rate=heart_rate, seed=seed,
    )
    return PPGGenerator(config).generate_fast()


def extract_respiratory_rate_from_ppg(ppg: np.ndarray, fs: float) -> float:
    """
    Estimate respiratory rate from PPG amplitude modulation.

    The respiratory rate is estimated from the frequency of amplitude
    envelope fluctuations in the PPG signal.

    Parameters
    ----------
    ppg : np.ndarray
        PPG signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Estimated respiratory rate in Hz.
    """
    # Extract amplitude envelope via Hilbert transform
    analytic = sp_signal.hilbert(ppg)
    envelope = np.abs(analytic)

    # Smooth the envelope
    nyq = 0.5 * fs
    fc = min(0.5 / nyq, 0.95)  # 0.5 Hz lowpass to isolate respiratory freq
    b, a = sp_signal.butter(2, fc, btype="lowpass")
    smooth_env = sp_signal.filtfilt(b, a, envelope)

    # Find dominant frequency via FFT
    n = len(smooth_env)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(smooth_env))

    # Look for peak in 0.1-0.5 Hz (6-30 breaths/min)
    resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(resp_mask):
        return 0.25  # Default 15 breaths/min

    resp_freqs = freqs[resp_mask]
    resp_spectrum = spectrum[resp_mask]
    peak_idx = np.argmax(resp_spectrum)

    return float(resp_freqs[peak_idx])
