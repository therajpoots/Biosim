"""
Signal-to-Noise Ratio (SNR) Characterization and Quality Assessment.

This module provides metrics to evaluate noise contamination levels in physiological signals.

Mathematical Formulations:
    1. Wideband SNR (Time Domain):
       $$SNR_{\\text{wideband}} = 10 \\log_{10}\\left( \\frac{\\sum_{n=1}^{N} x_{\\text{clean}}^2[n]}{\\sum_{n=1}^{N} (x_{\\text{noisy}}[n] - x_{\\text{clean}}[n])^2} \\right)$$

    2. Segmental SNR:
       Calculates the average SNR over short, non-overlapping windows of duration $W$:
       $$SNR_{\\text{seg}} = \\frac{1}{M} \\sum_{m=0}^{M-1} SNR(m) \\quad \\text{where } SNR(m) \\text{ is wideband SNR of block } m$$

    3. Spectral SNR (Frequency Domain):
       Integrates the Power Spectral Density (PSD) of the clean signal $P_{xx}(f)$ and the noise $P_{ee}(f)$
       over a target band $[f_1, f_2]$:
       $$SNR_{\\text{spectral}} = 10 \\log_{10}\\left( \\frac{\\int_{f_1}^{f_2} P_{xx}(f) df}{\\int_{f_1}^{f_2} P_{ee}(f) df} \\right)$$

    4. Wavelet Subband SNR:
       Decomposes the signal into $J$ approximation and detail subbands using a discrete wavelet filter bank:
       $$x(t) = A_J(t) + \\sum_{j=1}^{J} D_j(t)$$
       SNR is evaluated independently at each subband level to characterize band-specific noise.
"""

from typing import Optional, Union, List, Dict
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.core.math_utils import bandpower

def compute_snr_wideband(clean: np.ndarray, noisy: np.ndarray, fs: float) -> float:
    """
    Compute the wideband Signal-to-Noise Ratio (SNR) in decibels (dB).
    
    Parameters
    ----------
    clean : np.ndarray
        Clean reference signal array (1-D or 2-D).
    noisy : np.ndarray
        Noisy signal array matching the shape of `clean`.
    fs : float
        Sampling frequency in Hz.
        
    Returns
    -------
    float
        Achieved SNR in dB. Returns 100.0 if virtually noise-free, -100.0 if clean power is zero.
    """
    p_sig = np.mean(clean ** 2)
    p_noise = np.mean((noisy - clean) ** 2)
    
    if p_noise <= 1e-15:
        return 100.0
    if p_sig <= 1e-15:
        return -100.0
        
    return float(10.0 * np.log10(p_sig / p_noise))


def compute_snr_segmental(
    clean: np.ndarray,
    noisy: np.ndarray,
    fs: float,
    segment_s: float = 1.0
) -> np.ndarray:
    """
    Compute wideband SNR in dB for consecutive non-overlapping windows.
    
    Parameters
    ----------
    clean : np.ndarray
        Clean reference signal array. Supports 1-D and 2-D.
    noisy : np.ndarray
        Noisy signal array matching the shape of `clean`.
    fs : float
        Sampling frequency in Hz.
    segment_s : float
        Duration of each segment in seconds. Default is 1.0 s.
        
    Returns
    -------
    np.ndarray
        1-D array containing SNR values for each segment. If clean is 2-D,
        returns the average segmental SNR across channels.
    """
    if clean.ndim == 2:
        n_ch = clean.shape[0]
        results = []
        for c in range(n_ch):
            results.append(compute_snr_segmental(clean[c], noisy[c], fs, segment_s))
        return np.mean(results, axis=0)
        
    seg_samples = int(np.round(segment_s * fs))
    if seg_samples <= 0:
        raise ValueError("segment_s is too small for the given sampling frequency.")
        
    n_samples = len(clean)
    n_segs = n_samples // seg_samples
    if n_segs == 0:
        return np.array([compute_snr_wideband(clean, noisy, fs)])
        
    snrs = np.zeros(n_segs)
    for i in range(n_segs):
        start = i * seg_samples
        end = start + seg_samples
        snrs[i] = compute_snr_wideband(clean[start:end], noisy[start:end], fs)
        
    return snrs


def compute_snr_narrowband(
    clean: np.ndarray,
    noisy: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float
) -> float:
    """
    Compute SNR restricted to a specific frequency band [fmin, fmax] using bandpass filtering.
    
    Parameters
    ----------
    clean : np.ndarray
        Clean reference signal.
    noisy : np.ndarray
        Noisy signal.
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Low cutoff frequency in Hz.
    fmax : float
        High cutoff frequency in Hz.
        
    Returns
    -------
    float
        Narrowband SNR in dB.
    """
    if clean.ndim == 2:
        n_ch = clean.shape[0]
        results = [compute_snr_narrowband(clean[c], noisy[c], fs, fmin, fmax) for c in range(n_ch)]
        return float(np.mean(results))
        
    nyq = 0.5 * fs
    low = max(0.01, fmin)
    high = min(fmax, nyq - 0.1)
    
    if low >= high:
        raise ValueError("fmin must be less than fmax and both must be in range (0, fs/2).")
        
    b, a = sp_signal.butter(4, [low / nyq, high / nyq], btype='bandpass')
    clean_filt = sp_signal.filtfilt(b, a, clean)
    noisy_filt = sp_signal.filtfilt(b, a, noisy)
    
    return compute_snr_wideband(clean_filt, noisy_filt, fs)


def compute_snr_spectral(
    clean: np.ndarray,
    noisy: np.ndarray,
    fs: float,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None
) -> float:
    """
    Compute SNR in the frequency domain by integrating Power Spectral Densities (PSD).
    
    Parameters
    ----------
    clean : np.ndarray
        Clean reference signal.
    noisy : np.ndarray
        Noisy signal.
    fs : float
        Sampling frequency in Hz.
    fmin : Optional[float]
        Low cutoff for integration. If None, starts at 0 Hz (excluding DC).
    fmax : Optional[float]
        High cutoff for integration. If None, integrates up to Nyquist.
        
    Returns
    -------
    float
        Spectral SNR in dB.
    """
    if clean.ndim == 2:
        n_ch = clean.shape[0]
        results = [compute_snr_spectral(clean[c], noisy[c], fs, fmin, fmax) for c in range(n_ch)]
        return float(np.mean(results))
        
    # Calculate PSD of clean signal and error/noise signal
    nperseg = min(512, len(clean))
    f, psd_clean = sp_signal.welch(clean, fs=fs, nperseg=nperseg)
    f, psd_noise = sp_signal.welch(noisy - clean, fs=fs, nperseg=nperseg)
    
    low = fmin if fmin is not None else 0.5  # default to exclude DC
    high = fmax if fmax is not None else (0.5 * fs)
    
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return compute_snr_wideband(clean, noisy, fs)
        
    # Integrate PSDs using trapezoidal rule
    from scipy.integrate import trapezoid
    p_sig_spectral = trapezoid(psd_clean[mask], f[mask])
    p_noise_spectral = trapezoid(psd_noise[mask], f[mask])
    
    if p_noise_spectral <= 1e-15:
        return 100.0
    if p_sig_spectral <= 1e-15:
        return -100.0
        
    return float(10.0 * np.log10(p_sig_spectral / p_noise_spectral))


def compute_snr_adaptive(noisy: np.ndarray, fs: float, window_s: float = 2.0) -> float:
    """
    Estimate the SNR from the noisy signal alone without a clean reference (blind estimation).
    
    Uses running minimum statistics over a sliding window to isolate the noise floor floor.
    
    Parameters
    ----------
    noisy : np.ndarray
        Noisy biopotential signal array.
    fs : float
        Sampling frequency in Hz.
    window_s : float
        Sliding window size in seconds for noise floor tracking. Default is 2.0 s.
        
    Returns
    -------
    float
        Estimated SNR in dB.
    """
    if noisy.ndim == 2:
        n_ch = noisy.shape[0]
        results = [compute_snr_adaptive(noisy[c], fs, window_s) for c in range(n_ch)]
        return float(np.mean(results))
        
    n_samples = len(noisy)
    win_len = int(np.round(window_s * fs))
    if win_len <= 10:
        win_len = min(100, n_samples)
        
    # Compute running local variance (power profile) of the signal
    squared = np.square(noisy - np.mean(noisy))
    kernel = np.ones(win_len) / win_len
    local_variance = np.convolve(squared, kernel, mode='same')
    
    # Trim boundary effects where convolution padded with zeros
    trim = min(win_len // 2, (n_samples - 1) // 2)
    if trim > 0:
        valid_variance = local_variance[trim:-trim]
    else:
        valid_variance = local_variance
        
    # Noise floor is estimated as the minimum local variance over a sliding block window
    # representing quiet segments (e.g. diastole in ECG or calm periods in EEG)
    p_total = np.mean(squared)
    
    block_len = win_len * 2
    n_valid = len(valid_variance)
    n_blocks = n_valid // block_len
    if n_blocks > 1:
        min_vars = []
        for i in range(n_blocks):
            start = i * block_len
            end = min(n_valid, start + block_len)
            min_vars.append(np.min(valid_variance[start:end]))
        p_noise_est = np.median(min_vars)
    else:
        p_noise_est = np.min(valid_variance)
        
    # Bounding checks
    if p_noise_est <= 1e-15:
        p_noise_est = 1e-15
        
    p_sig_est = p_total - p_noise_est
    if p_sig_est <= 1e-15:
        return -20.0  # Assumes extremely noisy state
        
    return float(10.0 * np.log10(p_sig_est / p_noise_est))


def compute_snr_wavelet(
    clean: np.ndarray,
    noisy: np.ndarray,
    fs: float,
    level: int = 3
) -> Dict[str, float]:
    """
    Evaluate subband SNR at different discrete wavelet decomposition levels.
    
    Implements a pure-NumPy quadrature mirror filter bank (QMF) using Haar wavelets,
    avoiding extra external dependencies.
    
    Parameters
    ----------
    clean : np.ndarray
        Clean reference signal.
    noisy : np.ndarray
        Noisy signal.
    fs : float
        Sampling frequency in Hz.
    level : int
        Decomposition depth. Default is 3.
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping subbands (e.g. 'D1', 'D2', 'D3', 'A3') to their respective SNR in dB.
    """
    if clean.ndim == 2:
        # Multi-channel average
        n_ch = clean.shape[0]
        results = [compute_snr_wavelet(clean[c], noisy[c], fs, level) for c in range(n_ch)]
        keys = results[0].keys()
        avg_results = {}
        for k in keys:
            avg_results[k] = float(np.mean([res[k] for res in results]))
        return avg_results
        
    # 1. Custom 1-D Discrete Wavelet Transform (DWT) using Haar filters
    # Lowpass (h) and Highpass (g) decomposition filter coefficients
    h = np.array([1.0, 1.0]) / np.sqrt(2.0)
    g = np.array([1.0, -1.0]) / np.sqrt(2.0)
    
    def dwt_step(sig):
        # Convolve and downsample by 2
        approx = np.convolve(sig, h, mode='same')[::2]
        detail = np.convolve(sig, g, mode='same')[::2]
        return approx, detail
        
    # Decompose clean and noisy signals
    clean_approx = clean.copy()
    noisy_approx = noisy.copy()
    
    subbands_clean = {}
    subbands_noisy = {}
    
    for j in range(1, level + 1):
        clean_approx, clean_detail = dwt_step(clean_approx)
        noisy_approx, noisy_detail = dwt_step(noisy_approx)
        
        subbands_clean[f'D{j}'] = clean_detail
        subbands_noisy[f'D{j}'] = noisy_detail
        
    subbands_clean[f'A{level}'] = clean_approx
    subbands_noisy[f'A{level}'] = noisy_approx
    
    # 2. Compute SNR in each subband
    subband_snr = {}
    for key in subbands_clean:
        c_sub = subbands_clean[key]
        n_sub = subbands_noisy[key]
        
        # Trim arrays to match in case of minor offset mismatches
        min_len = min(len(c_sub), len(n_sub))
        if min_len > 1:
            subband_snr[key] = compute_snr_wideband(c_sub[:min_len], n_sub[:min_len], fs)
        else:
            subband_snr[key] = 0.0
            
    return subband_snr
