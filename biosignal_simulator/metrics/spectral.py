"""
Frequency-Domain and Spectral Characterization Metrics.

This module provides metrics to evaluate frequency distribution, complexity,
harmonic distortion, and spectral compression (e.g. fatigue monitoring).

Mathematical Formulations:
    1. Spectral Entropy:
       Measures the complexity/flatness of the signal spectrum. Given normalized PSD $p(f)$:
       $$p(f_i) = \\frac{P(f_i)}{\\sum_{j} P(f_j)}$$
       $$H_{\\text{spectral}} = -\\sum_{i} p(f_i) \\log_{2}(p(f_i))$$

    2. Mean Frequency ($f_{\\text{mean}}$):
       The spectral centroid frequency:
       $$f_{\\text{mean}} = \\frac{\\sum_{i} f_i P(f_i)}{\\sum_{i} P(f_i)}$$

    3. Median Frequency ($f_{\\text{median}}$):
       The frequency that divides the PSD into two equal halves:
       $$\\sum_{f_i < f_{\\text{median}}} P(f_i) = \\sum_{f_i \\ge f_{\\text{median}}} P(f_i) = 0.5 \\sum_{i} P(f_i)$$
       Widely used as a clinical metric for muscle fatigue detection in EMG.

    4. Spectral Edge Frequency (SEF_x):
       The frequency below which a specified percentile $x\\%$ (typically 95%) of the power resides:
       $$\\sum_{f_i < \\text{SEF}_x} P(f_i) = \\frac{x}{100} \\sum_{i} P(f_i)$$
       Commonly used in EEG sleep staging and anesthesia depth monitoring.
"""

from typing import Optional, Union, List, Tuple, Dict
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.core.math_utils import bandpower

def compute_psd(
    signal_arr: np.ndarray,
    fs: float,
    method: str = 'welch',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Power Spectral Density (PSD) using Welch's method or Periodogram.
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array (1-D).
    fs : float
        Sampling frequency in Hz.
    method : str
        Method: 'welch' or 'periodogram'. Default is 'welch'.
    **kwargs :
        Additional arguments passed to the scipy methods.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (frequencies, psd_values)
    """
    if signal_arr.ndim != 1:
        raise ValueError("compute_psd expects a 1-D signal array.")
        
    if method.lower() == 'welch':
        # Default nperseg to min of 256 and signal length
        nperseg = kwargs.pop('nperseg', min(256, len(signal_arr)))
        return sp_signal.welch(signal_arr, fs=fs, nperseg=nperseg, **kwargs)
    else:
        return sp_signal.periodogram(signal_arr, fs=fs, **kwargs)


def compute_band_power(signal_arr: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """
    Compute the total power of a signal in a specific frequency band.
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array (1-D or 2-D).
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Low cutoff frequency.
    fmax : float
        High cutoff frequency.
    """
    if signal_arr.ndim == 2:
        return float(np.mean([bandpower(signal_arr[c], fs, fmin, fmax) for c in range(signal_arr.shape[0])]))
    return bandpower(signal_arr, fs, fmin, fmax)


def compute_spectral_flatness(signal_arr: np.ndarray, fs: float) -> float:
    """
    Compute the spectral flatness of the signal (Wiener entropy).
    
    Ratio of geometric mean to arithmetic mean of the power spectrum.
    Ranges from 0 (tonal/pure tone) to 1 (white noise/flat spectrum).
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array (1-D or 2-D).
    fs : float
        Sampling frequency in Hz.
    """
    if signal_arr.ndim == 2:
        return float(np.mean([compute_spectral_flatness(signal_arr[c], fs) for c in range(signal_arr.shape[0])]))
        
    f, psd = compute_psd(signal_arr, fs, method='welch')
    valid_psd = psd[f > 0.1]  # Exclude DC offset
    
    if len(valid_psd) == 0:
        return 0.0
        
    valid_psd = np.clip(valid_psd, 1e-15, None)
    
    geom_mean = np.exp(np.mean(np.log(valid_psd)))
    arith_mean = np.mean(valid_psd)
    
    if arith_mean <= 1e-15:
        return 0.0
    return float(geom_mean / arith_mean)


def compute_thd(
    signal_arr: np.ndarray,
    fs: float,
    fundamental_hz: float = 50.0,
    n_harmonics: int = 3
) -> float:
    """
    Compute the Total Harmonic Distortion (THD) of a fundamental frequency component.
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array (1-D or 2-D).
    fs : float
        Sampling frequency in Hz.
    fundamental_hz : float
        Nominal fundamental frequency in Hz. Default is 50.0 Hz.
    n_harmonics : int
        Number of harmonics to integrate. Default is 3.
    """
    if signal_arr.ndim == 2:
        return float(np.mean([compute_thd(signal_arr[c], fs, fundamental_hz, n_harmonics) for c in range(signal_arr.shape[0])]))
        
    f0 = fundamental_hz
    df = 2.0  # Integration band half-width in Hz
    
    # Fundamental power
    p_fund = bandpower(signal_arr, fs, max(0.1, f0 - df), f0 + df)
    
    # Sum of harmonics power
    p_harm = 0.0
    for h in range(2, n_harmonics + 1):
        fh = h * f0
        if fh < 0.5 * fs:  # Nyquist boundary
            p_harm += bandpower(signal_arr, fs, max(0.1, fh - df), fh + df)
            
    if p_fund <= 1e-15:
        return 0.0
    return float(p_harm / p_fund)


def compute_spectral_entropy(signal_arr: np.ndarray, fs: float) -> float:
    """
    Compute the normalized Shannon Spectral Entropy.
    
    Measures the uncertainty/complexity of the frequency distribution.
    Ranges from 0 (single frequency component) to 1 (pure white noise).
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array. Supports 1-D and 2-D.
    fs : float
        Sampling frequency in Hz.
    """
    if signal_arr.ndim == 2:
        return float(np.mean([compute_spectral_entropy(signal_arr[c], fs) for c in range(signal_arr.shape[0])]))
        
    f, psd = compute_psd(signal_arr, fs, method='welch')
    valid_psd = psd[f > 0.1]
    
    if len(valid_psd) <= 1:
        return 0.0
        
    # Normalize PSD to create probability density distribution
    sum_psd = np.sum(valid_psd)
    if sum_psd <= 1e-15:
        return 0.0
        
    p = valid_psd / sum_psd
    p = np.clip(p, 1e-15, None)  # Prevent log(0)
    
    # Shannon entropy: H = -sum(p * log2(p))
    h = -np.sum(p * np.log2(p))
    # Normalize by maximum possible entropy (log2 of number of bins)
    h_max = np.log2(len(p))
    return float(h / h_max)


def compute_median_frequency(signal_arr: np.ndarray, fs: float) -> float:
    """
    Compute the Median Frequency (MDF) of the Power Spectral Density.
    
    Divides the power spectrum into two regions of equal power. Key metric
    for detecting muscle fatigue in EMG signals.
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array. Supports 1-D and 2-D.
    fs : float
        Sampling frequency in Hz.
        
    Returns
    -------
    float
        Median frequency in Hz.
    """
    if signal_arr.ndim == 2:
        return float(np.mean([compute_median_frequency(signal_arr[c], fs) for c in range(signal_arr.shape[0])]))
        
    f, psd = compute_psd(signal_arr, fs, method='welch')
    
    # Calculate cumulative power sum
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    
    if total_power <= 1e-15:
        return 0.0
        
    # Find index closest to 50% of total power
    target = 0.5 * total_power
    idx = np.searchsorted(cumulative_power, target)
    return float(f[idx])


def compute_mean_frequency(signal_arr: np.ndarray, fs: float) -> float:
    """
    Compute the Mean Frequency (MNF) of the Power Spectral Density.
    
    Centroid frequency of the PSD curve.
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array. Supports 1-D and 2-D.
    fs : float
        Sampling frequency in Hz.
        
    Returns
    -------
    float
        Mean frequency in Hz.
    """
    if signal_arr.ndim == 2:
        return float(np.mean([compute_mean_frequency(signal_arr[c], fs) for c in range(signal_arr.shape[0])]))
        
    f, psd = compute_psd(signal_arr, fs, method='welch')
    
    sum_psd = np.sum(psd)
    if sum_psd <= 1e-15:
        return 0.0
        
    # MNF = sum(f_i * P_i) / sum(P_i)
    return float(np.sum(f * psd) / sum_psd)


def compute_spectral_edge_frequency(
    signal_arr: np.ndarray,
    fs: float,
    percentile: float = 95.0
) -> float:
    """
    Compute the Spectral Edge Frequency (SEF) for a specified percentile.
    
    Identifies the frequency below which `percentile` percentage of total
    spectral power lies (typically SEF95).
    
    Parameters
    ----------
    signal_arr : np.ndarray
        Signal array. Supports 1-D and 2-D.
    fs : float
        Sampling frequency in Hz.
    percentile : float
        Percentage boundary [0, 100]. Default is 95.0.
        
    Returns
    -------
    float
        SEF in Hz.
    """
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("Percentile must be in range [0, 100].")
        
    if signal_arr.ndim == 2:
        return float(np.mean([compute_spectral_edge_frequency(signal_arr[c], fs, percentile) for c in range(signal_arr.shape[0])]))
        
    f, psd = compute_psd(signal_arr, fs, method='welch')
    
    cumulative_power = np.cumsum(psd)
    total_power = cumulative_power[-1]
    
    if total_power <= 1e-15:
        return 0.0
        
    target = (percentile / 100.0) * total_power
    idx = np.searchsorted(cumulative_power, target)
    return float(f[idx])
