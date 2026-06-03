"""
Signal Distortion, Similarity, and Reconstruction Fidelity Metrics.

This module provides metrics to evaluate reconstruction accuracy, shape matching,
and morphological distortion between a reference (clean) and a processed (noisy/filtered) signal.

Mathematical Formulations:
    1. Percent Residual Difference (PRD):
       Standard metric in biomedical signal compression (especially ECG) to evaluate
       distortion relative to overall signal energy:
       $$PRD = \\sqrt{\\frac{\\sum_{n=1}^{N} (x[n] - \\hat{x}[n])^2}{\\sum_{n=1}^{N} x^2[n]}} \\times 100\\%$$

    2. Normalized PRD (PRDN):
       Removes the influence of DC offset/mean from PRD, focusing purely on shape distortion:
       $$PRDN = \\sqrt{\\frac{\\sum_{n=1}^{N} (x[n] - \\hat{x}[n])^2}{\\sum_{n=1}^{N} (x[n] - \\mu_x)^2}} \\times 100\\%$$

    3. 1D Structural Similarity Index (SSIM):
       Adaptation of image SSIM for 1D physiological time-series. Compares local luminance (mean),
       contrast (variance), and structure (correlation) over sliding windows:
       $$SSIM(x, y) = \\frac{(2\\mu_x\\mu_y + C_1)(2\\sigma_{xy} + C_2)}{(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}$$
       where $\\mu_x, \\sigma_x^2$ are computed in local windows of size $W$, and $C_1, C_2$ are stabilizing constants.
"""

from typing import Optional, Union, List, Tuple, Dict
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.core.math_utils import compute_rms

def compute_mse(reference: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal (1-D or 2-D).
    filtered : np.ndarray
        Processed or corrupted signal matching the shape of `reference`.
    """
    if reference.ndim == 2:
        return float(np.mean([np.mean((reference[c] - filtered[c]) ** 2) for c in range(reference.shape[0])]))
    return float(np.mean((reference - filtered) ** 2))


def compute_rmse(reference: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE).
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal.
    filtered : np.ndarray
        Processed or corrupted signal.
    """
    return float(np.sqrt(compute_mse(reference, filtered)))


def compute_psnr(reference: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) in dB.
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal.
    filtered : np.ndarray
        Processed or corrupted signal.
    """
    if reference.ndim == 2:
        results = [compute_psnr(reference[c], filtered[c]) for c in range(reference.shape[0])]
        return float(np.mean(results))
        
    rmse = compute_rmse(reference, filtered)
    if rmse <= 1e-15:
        return 100.0
        
    max_ref = np.max(np.abs(reference))
    if max_ref <= 1e-15:
        return -100.0
        
    return float(20.0 * np.log10(max_ref / rmse))


def compute_correlation(reference: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient.
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal.
    filtered : np.ndarray
        Processed or corrupted signal.
    """
    if reference.ndim == 2:
        results = []
        for c in range(reference.shape[0]):
            corr = np.corrcoef(reference[c], filtered[c])[0, 1]
            results.append(0.0 if np.isnan(corr) else corr)
        return float(np.mean(results))
        
    corr = np.corrcoef(reference, filtered)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_ste(reference: np.ndarray, filtered: np.ndarray, fs: float, threshold_factor: float = 1.5) -> float:
    """
    Compute Signal-to-Error ratio (STE) in high-amplitude windows (e.g. QRS complex).
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal.
    filtered : np.ndarray
        Processed or corrupted signal.
    fs : float
        Sampling frequency in Hz.
    threshold_factor : float
        Multiplier of overall RMS to establish threshold for high-amplitude segment detection.
        Default is 1.5.
    """
    if reference.ndim == 2:
        results = [compute_ste(reference[c], filtered[c], fs, threshold_factor) for c in range(reference.shape[0])]
        return float(np.mean(results))
        
    rms = compute_rms(reference)
    mask = np.abs(reference) > threshold_factor * rms
    if not np.any(mask):
        return compute_psnr(reference, filtered)
        
    p_ref = np.mean(reference[mask] ** 2)
    p_err = np.mean((filtered[mask] - reference[mask]) ** 2)
    
    if p_err <= 1e-15:
        return 100.0
    if p_ref <= 1e-15:
        return -100.0
        
    return float(10.0 * np.log10(p_ref / p_err))


def compute_qrs_correlation(
    clean_ecg: np.ndarray,
    filtered_ecg: np.ndarray,
    fs: float,
    pan_tompkins_threshold: float = 0.2
) -> Dict[str, float]:
    """
    Detect QRS peaks in clean ECG and compute average shape correlation of QRS complexes in filtered ECG.
    
    Parameters
    ----------
    clean_ecg : np.ndarray
        Reference clean ECG channel.
    filtered_ecg : np.ndarray
        Processed or corrupted ECG channel.
    fs : float
        Sampling frequency.
    pan_tompkins_threshold : float
        Peak detection threshold fraction. Default is 0.2.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'mean' and 'std' of Pearson correlation.
    """
    if clean_ecg.ndim == 2:
        # If multi-channel, evaluate on the first channel (e.g. Lead II)
        clean_ecg = clean_ecg[0]
        filtered_ecg = filtered_ecg[0]
        
    # 1. Simplified Pan-Tompkins Peak Detector
    nyq = 0.5 * fs
    low = 5.0
    high = min(15.0, nyq - 0.1)
    b, a = sp_signal.butter(2, [low / nyq, high / nyq], btype='bandpass')
    filtered = sp_signal.filtfilt(b, a, clean_ecg)
    
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2
    
    win_len = int(np.round(0.12 * fs))
    if win_len <= 0:
        win_len = 1
    window = np.ones(win_len) / win_len
    integrated = np.convolve(squared, window, mode='same')
    
    dist = max(1, int(0.4 * fs))
    prom = np.max(integrated) * pan_tompkins_threshold
    
    peaks, _ = sp_signal.find_peaks(integrated, distance=dist, prominence=prom)
    
    if len(peaks) == 0:
        corr = compute_correlation(clean_ecg, filtered_ecg)
        return {'mean': corr, 'std': 0.0}
        
    # 2. Extract QRS segments [-0.08 s, +0.12 s] around peaks
    pre_samples = int(np.round(0.08 * fs))
    post_samples = int(np.round(0.12 * fs))
    
    corrs = []
    for pk in peaks:
        start = pk - pre_samples
        end = pk + post_samples
        if start >= 0 and end <= len(clean_ecg):
            seg_clean = clean_ecg[start:end]
            seg_filt = filtered_ecg[start:end]
            c = compute_correlation(seg_clean, seg_filt)
            corrs.append(c)
            
    if len(corrs) == 0:
        return {'mean': 0.0, 'std': 0.0}
        
    return {
        'mean': float(np.mean(corrs)),
        'std': float(np.std(corrs))
    }


def compute_prd(reference: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute the Percent Residual Difference (PRD).
    
    Common metric for biomedical signal compression quality assessment.
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal.
    filtered : np.ndarray
        Processed or corrupted signal.
    """
    if reference.ndim == 2:
        return float(np.mean([compute_prd(reference[c], filtered[c]) for c in range(reference.shape[0])]))
        
    p_err = np.sum((reference - filtered) ** 2)
    p_ref = np.sum(reference ** 2)
    
    if p_ref <= 1e-15:
        return 0.0
    return float(np.sqrt(p_err / p_ref) * 100.0)


def compute_prdn(reference: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute the Normalized Percent Residual Difference (PRDN).
    
    Removes DC bias influence to evaluate shape-only distortion.
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal.
    filtered : np.ndarray
        Processed or corrupted signal.
    """
    if reference.ndim == 2:
        return float(np.mean([compute_prdn(reference[c], filtered[c]) for c in range(reference.shape[0])]))
        
    p_err = np.sum((reference - filtered) ** 2)
    mean_ref = np.mean(reference)
    p_ref_centered = np.sum((reference - mean_ref) ** 2)
    
    if p_ref_centered <= 1e-15:
        return 0.0
    return float(np.sqrt(p_err / p_ref_centered) * 100.0)


def compute_max_absolute_error(reference: np.ndarray, filtered: np.ndarray) -> float:
    """
    Compute the Maximum Absolute Error (MAE) / Chebyshev distance.
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal.
    filtered : np.ndarray
        Processed or corrupted signal.
    """
    return float(np.max(np.abs(reference - filtered)))


def compute_ssim_1d(
    reference: np.ndarray,
    filtered: np.ndarray,
    window_size: int = 15,
    c1: float = 1e-4,
    c2: float = 9e-4
) -> float:
    """
    Compute 1D Structural Similarity Index (SSIM) over sliding windows.
    
    Compares local luminance (means), contrast (variances), and structural
    correlation, returning value in [-1, 1] (1 = perfect match).
    
    Parameters
    ----------
    reference : np.ndarray
        Clean original reference signal (1-D or 2-D).
    filtered : np.ndarray
        Processed or corrupted signal matching the shape of `reference`.
    window_size : int
        Size of local sliding window for statistics estimation. Default is 15.
    c1 : float
        Stabilizing constant for mean division. Default is 1e-4.
    c2 : float
        Stabilizing constant for variance division. Default is 9e-4.
        
    Returns
    -------
    float
        Average 1D SSIM score.
    """
    if reference.ndim == 2:
        return float(np.mean([compute_ssim_1d(reference[c], filtered[c], window_size, c1, c2) for c in range(reference.shape[0])]))
        
    n_samples = len(reference)
    if n_samples < window_size:
        return compute_correlation(reference, filtered)
        
    # Compute moving averages (means) using box convolution
    kernel = np.ones(window_size) / window_size
    mu_x = np.convolve(reference, kernel, mode='valid')
    mu_y = np.convolve(filtered, kernel, mode='valid')
    
    # Compute local variances and covariances:
    # Var(x) = E(x^2) - E(x)^2
    mu_x2 = np.convolve(reference ** 2, kernel, mode='valid')
    mu_y2 = np.convolve(filtered ** 2, kernel, mode='valid')
    mu_xy = np.convolve(reference * filtered, kernel, mode='valid')
    
    var_x = np.clip(mu_x2 - mu_x ** 2, 0, None)
    var_y = np.clip(mu_y2 - mu_y ** 2, 0, None)
    cov_xy = mu_xy - mu_x * mu_y
    
    # SSIM equation evaluated at each window index
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denom = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    
    ssim_map = num / denom
    return float(np.mean(ssim_map))
