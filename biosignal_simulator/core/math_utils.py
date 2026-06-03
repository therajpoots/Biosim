"""
Mathematical and Digital Signal Processing (DSP) utilities for biosignals.

Provides:
1. Basic metrics (RMS, db/linear scaling, SNR breakdowns).
2. Advanced spectral estimation (Welch bandpower, colored noise shaping).
3. Windowing and zero-phase Butterworth/Notch filter design.
4. Robust statistical estimators (Crest factor, robust standard deviation, autocorrelation).
5. Interpolation methods for high-fidelity resampling.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional, Union, List

# =====================================================================
# 1. Basic Amplitude and Power Conversions
# =====================================================================

def compute_rms(x: np.ndarray) -> float:
    """
    Compute the Root Mean Square (RMS) of an array.
    
    The RMS is calculated as:
        RMS = sqrt( (1 / N) * sum( x_i^2 ) )
        
    Handles multi-dimensional arrays by flattening them, empty arrays,
    and single scalar values safely.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal vector or matrix.
        
    Returns
    -------
    float
        The calculated RMS value. Returns 0.0 if empty or zero-filled.
        
    Examples
    --------
    >>> compute_rms(np.array([3.0, 4.0]))
    3.5355339059327378
    """
    if x is None or x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def normalize_to_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    """
    Scale a signal array so its Root Mean Square (RMS) equals the target value.
    
    If the input array's RMS is near-zero (i.e. <= 1e-12), it returns the original
    array unscaled to prevent division-by-zero numerical overflow.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array to scale.
    target_rms : float
        Target RMS value (must be >= 0.0).
        
    Returns
    -------
    np.ndarray
        The scaled signal array.
        
    Raises
    ------
    ValueError
        If target_rms is negative.
    """
    if target_rms < 0.0:
        raise ValueError(f"target_rms must be non-negative, got {target_rms}")
        
    rms = compute_rms(x)
    if rms <= 1e-12:
        return x
    return x * (target_rms / rms)


def db_to_linear(db: float) -> float:
    """
    Convert a decibel power ratio to a linear power ratio.
    
    Formula:
        Linear = 10^(dB / 10)
        
    Parameters
    ----------
    db : float
        Decibel value.
        
    Returns
    -------
    float
        Linear power scale factor.
    """
    return 10.0 ** (db / 10.0)


def linear_to_db(ratio: float) -> float:
    """
    Convert a linear power ratio to decibels.
    
    Formula:
        dB = 10 * log10(Ratio)
        
    Handles zero and negative ratios by clamping to -150.0 dB.
    
    Parameters
    ----------
    ratio : float
        Linear power ratio.
        
    Returns
    -------
    float
        The corresponding value in decibels.
    """
    if ratio <= 1e-15:
        return -150.0
    return 10.0 * np.log10(ratio)


# =====================================================================
# 2. Spectral Analysis Utilities
# =====================================================================

def bandpower(
    x: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    method: str = 'welch',
    nperseg: Optional[int] = None
) -> float:
    """
    Estimate the average power of a signal within a specific frequency band.
    
    Integrates the Power Spectral Density (PSD) over the range [fmin, fmax]
    using the trapezoidal rule:
        Power = integral_{fmin}^{fmax} PSD(f) df
        
    Supported PSD estimation methods:
    - 'welch': Welch's averaged periodogram (reduces variance, recommended).
    - 'periodogram': Standard periodogram (higher resolution, higher variance).
    
    Parameters
    ----------
    x : np.ndarray
        1-D signal array.
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Lower bound of the frequency band in Hz.
    fmax : float
        Upper bound of the frequency band in Hz.
    method : str
        Method for PSD estimation: 'welch' or 'periodogram' (default: 'welch').
    nperseg : Optional[int]
        Length of each segment for Welch's method (default: min(256, len(x))).
        
    Returns
    -------
    float
        Average band power.
        
    Raises
    ------
    ValueError
        If an invalid method is supplied or boundaries are out-of-bounds.
    """
    if len(x) == 0:
        return 0.0
    if fmin < 0.0 or fmax < 0.0:
        raise ValueError("Frequency limits must be non-negative")
    if fmin >= fmax:
        raise ValueError("fmin must be strictly less than fmax")
        
    method_clean = method.strip().lower()
    
    # Select segment size for Welch
    if nperseg is None:
        nperseg = min(256, len(x))
        
    if method_clean == 'welch' and len(x) >= 8:
        # Welch PSD
        f, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    elif method_clean == 'periodogram' or len(x) < 8:
        # Periodogram PSD
        f, psd = sp_signal.periodogram(x, fs=fs)
    else:
        raise ValueError(f"Unknown PSD method: {method}. Use 'welch' or 'periodogram'.")
        
    # Find frequency bin indices
    idx = (f >= fmin) & (f <= fmax)
    if not np.any(idx):
        return 0.0
        
    # Integrate PSD slice
    return float(trapezoid(psd[idx], f[idx]))


def spectral_shape(n: int, fs: float, exponent: float) -> np.ndarray:
    """
    Generate a frequency-domain scaling vector for colored noise shaping.
    
    The scaling factor at frequency f is given by:
        Scale(f) = f^(-exponent / 2)
        
    Sets the DC component (f = 0) to 0.0 to eliminate baseline offsets.
    
    Parameters
    ----------
    n : int
        Total samples in the time domain.
    fs : float
        Sampling frequency in Hz.
    exponent : float
        Spectral exponent (e.g. 0=white, 1=pink, 2=brown).
        
    Returns
    -------
    np.ndarray
        1-D array of shape (n // 2 + 1,) containing scale coefficients.
    """
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    shape = np.zeros_like(freqs)
    
    active_mask = freqs > 0.0
    # Apply magnitude scaling
    shape[active_mask] = freqs[active_mask] ** (-exponent / 2.0)
    shape[~active_mask] = 0.0
    
    return shape


def compute_snr_components(clean: np.ndarray, noise_components: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute separate Signal-to-Noise Ratio (SNR) contributions for each noise component.
    
    SNR is defined as:
        SNR_c = 10 * log10( P_signal / P_noise_c )
        
    Parameters
    ----------
    clean : np.ndarray
        Clean source signal.
    noise_components : Dict[str, np.ndarray]
        Dictionary of individual noise arrays.
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping component names to individual SNRs in dB.
    """
    p_signal = np.mean(np.square(clean))
    if p_signal <= 1e-15:
        p_signal = 1e-15
        
    snrs = {}
    for name, noise in noise_components.items():
        p_noise = np.mean(np.square(noise))
        if p_noise <= 1e-15:
            snrs[name] = 150.0  # Infinite SNR simulation
        else:
            snrs[name] = 10.0 * np.log10(p_signal / p_noise)
            
    return snrs


# =====================================================================
# 3. Digital Filter Design and Application
# =====================================================================

def butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase bidirectional lowpass Butterworth filter to a signal.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Lowpass cutoff frequency in Hz.
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return x  # Cutoff above Nyquist; return original signal
        
    b, a = sp_signal.butter(order, cutoff / nyq, btype='low')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def butter_highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase bidirectional highpass Butterworth filter.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Highpass cutoff frequency in Hz.
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs
    if cutoff <= 0.0 or cutoff >= nyq:
        return x - np.mean(x)  # Fall back to removing DC offset
        
    b, a = sp_signal.butter(order, cutoff / nyq, btype='high')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def butter_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase bidirectional bandpass Butterworth filter.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    low : float
        Lower cutoff frequency in Hz.
    high : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal array.
    """
    nyq = 0.5 * fs
    low_norm = max(0.01, low) / nyq
    high_norm = min(high, nyq - 0.01) / nyq
    
    if low_norm >= 1.0 or high_norm <= 0.0 or low_norm >= high_norm:
        return x
        
    b, a = sp_signal.butter(order, [low_norm, high_norm], btype='bandpass')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def butter_notch(x: np.ndarray, fs: float, f_notch: float, q: float = 30.0) -> np.ndarray:
    """
    Apply a zero-phase bidirectional IIR notch filter to remove line noise.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    f_notch : float
        Target notch frequency in Hz (e.g. 50 or 60 Hz).
    q : float
        Quality factor (Q-factor). Higher Q means narrower notch band (default: 30.0).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if f_notch <= 0.0 or f_notch >= nyq:
        return x
        
    w0 = f_notch / nyq
    b, a = sp_signal.iirnotch(w0, q)
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


# =====================================================================
# 4. Windowing and Temporal Tapering
# =====================================================================

def apply_window(x: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """
    Apply a temporal window (tapering) to smooth boundaries of a signal.
    
    Supported windows: 'hann', 'hamming', 'blackman', 'bartlett', 'cosine'.
    
    Parameters
    ----------
    x : np.ndarray
        1-D input array.
    window_type : str
        Window category (default: 'hann').
        
    Returns
    -------
    np.ndarray
        Window-scaled signal.
    """
    n = len(x)
    if n <= 1:
        return x
        
    w = sp_signal.get_window(window_type, n)
    return x * w


# =====================================================================
# 5. Robust Statistical Utility Estimators
# =====================================================================

def robust_std(x: np.ndarray) -> float:
    """
    Estimate standard deviation robustly using the Interquartile Range (IQR).
    
    Formula:
        Robust STD = IQR / 1.34897
        
    Reduces influence of outliers, baseline drift, and transient spikes.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal.
        
    Returns
    -------
    float
        Robust standard deviation.
    """
    if len(x) < 2:
        return 0.0
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    return float(iqr / 1.3489797)


def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    """
    Calculate the normalized autocorrelation sequence of a 1-D signal.
    
    Formula:
        R(k) = E[ (x_t - mu)*(x_{t+k} - mu) ] / var(x)
        
    Parameters
    ----------
    x : np.ndarray
        1-D input vector.
    max_lag : Optional[int]
        Maximum sample lag to compute (default: len(x) - 1).
        
    Returns
    -------
    np.ndarray
        Autocorrelation sequence indexed by lag [0, max_lag].
    """
    n = len(x)
    if n == 0:
        return np.array([])
        
    if max_lag is None:
        max_lag = n - 1
        
    max_lag = min(max_lag, n - 1)
    
    # Center and normalize
    mean_val = np.mean(x)
    std_val = np.std(x)
    if std_val < 1e-12:
        return np.zeros(max_lag + 1)
        
    x_centered = (x - mean_val) / std_val
    
    # Compute using FFT correlation
    corr = np.correlate(x_centered, x_centered, mode='full')
    # Pull lags starting from midpoint (lag 0)
    mid = len(corr) // 2
    return corr[mid : mid + max_lag + 1] / n


def crest_factor(x: np.ndarray) -> float:
    """
    Calculate the Crest Factor of a signal.
    
    Formula:
        Crest Factor = Peak Amplitude / RMS
        
    Parameters
    ----------
    x : np.ndarray
        Signal array.
        
    Returns
    -------
    float
        The crest factor ratio. Returns 0.0 if RMS is near zero.
    """
    rms = compute_rms(x)
    if rms <= 1e-12:
        return 0.0
    return float(np.max(np.abs(x)) / rms)


def peak_to_average_power_ratio(x: np.ndarray) -> float:
    """
    Calculate the Peak-to-Average Power Ratio (PAPR) in decibels.
    
    Formula:
        PAPR = 10 * log10( Peak^2 / RMS^2 )
        
    Parameters
    ----------
    x : np.ndarray
        Signal array.
        
    Returns
    -------
    float
        PAPR in dB.
    """
    cf = crest_factor(x)
    if cf <= 0.0:
        return -150.0
    return 20.0 * np.log10(cf)


# =====================================================================
# 6. Interpolation and Resampling Helpers
# =====================================================================

def interpolate_1d_grid(t_orig: np.ndarray, x: np.ndarray, t_new: np.ndarray, method: str = 'cubic') -> np.ndarray:
    """
    General interpolation function to map a signal between non-uniform time grids.
    
    Parameters
    ----------
    t_orig : np.ndarray
        Original time grid.
    x : np.ndarray
        Original signal values.
    t_new : np.ndarray
        New target time grid.
    method : str
        Interpolation method: 'linear', 'nearest', 'cubic', 'quadratic' (default: 'cubic').
        
    Returns
    -------
    np.ndarray
        Interpolated signal matching shape of t_new.
    """
    if len(t_orig) != len(x):
        raise ValueError("Lengths of original time grid and signal must match")
        
    if len(t_orig) < 4 and method in {'cubic', 'quadratic'}:
        method = 'linear'
        
    f = interp1d(t_orig, x, kind=method, fill_value="extrapolate")
    return f(t_new)


# =====================================================================
# 7. Advanced Statistical and Quality Indicators
# =====================================================================

def compute_skewness(x: np.ndarray, bias: bool = True) -> float:
    """
    Calculate the skewness (third standardized moment) of a signal.
    
    Skewness measures the asymmetry of the probability distribution of a
    real-valued random variable about its mean.
    
    Mathematical Formulation:
        Skewness = E[ ( (X - mu) / sigma )^3 ]
        
    For a sample of size N, the biased skewness is:
        g_1 = ( (1 / N) * sum_{i=1}^N (x_i - mu)^3 ) / ( (1 / N) * sum_{i=1}^N (x_i - mu)^2 )^(1.5)
        
    If bias is False, the unbiased skewness estimator is computed:
        G_1 = g_1 * sqrt(N * (N - 1)) / (N - 2)
        
    Parameters
    ----------
    x : np.ndarray
        1-D input array.
    bias : bool
        If True, returns the biased sample skewness. Otherwise, returns
        the unbiased estimator (default: True).
        
    Returns
    -------
    float
        The skewness value. Returns 0.0 if standard deviation is near zero
        or length is too small.
    """
    n = len(x)
    if n < 3:
        return 0.0
    
    mean_val = np.mean(x)
    std_val = np.std(x)
    if std_val < 1e-12:
        return 0.0
        
    diff = x - mean_val
    m3 = np.mean(diff ** 3)
    m2 = np.mean(diff ** 2)
    
    g1 = m3 / (m2 ** 1.5)
    
    if bias:
        return float(g1)
    else:
        # Unbiased estimator
        return float(g1 * np.sqrt(n * (n - 1)) / (n - 2))


def compute_kurtosis(x: np.ndarray, fisher: bool = True, bias: bool = True) -> float:
    """
    Calculate the kurtosis (fourth standardized moment) of a signal.
    
    Kurtosis is a measure of the "tailedness" of the probability distribution.
    A higher kurtosis indicates heavier tails (more extreme outliers).
    
    Mathematical Formulation:
        Kurtosis = E[ ( (X - mu) / sigma )^4 ]
        
    For a sample of size N, the biased kurtosis is:
        g_2 = ( (1 / N) * sum_{i=1}^N (x_i - mu)^4 ) / ( (1 / N) * sum_{i=1}^N (x_i - mu)^2 )^2
        
    If fisher is True, the kurtosis is relative to a normal distribution (excess kurtosis):
        g_2_excess = g_2 - 3
        
    If bias is False, the unbiased kurtosis/excess kurtosis estimators are:
        G_2 = ( (N - 1) / ((N - 2) * (N - 3)) ) * ( (N + 1) * g_2 - 3 * (N - 1) )
        G_2_excess = G_2 - ( 3 * (N - 1)^2 / ((N - 2) * (N - 3)) )
        
    Parameters
    ----------
    x : np.ndarray
        1-D input array.
    fisher : bool
        If True, Fisher's definition is used (excess kurtosis, normal = 0.0).
        If False, Pearson's definition is used (normal = 3.0) (default: True).
    bias : bool
        If True, returns the biased sample kurtosis. Otherwise, returns
        the unbiased estimator (default: True).
        
    Returns
    -------
    float
        The kurtosis value. Returns 0.0 (or 3.0 if fisher=False) if standard
        deviation is near zero or length is too small.
    """
    n = len(x)
    if n < 4:
        return 0.0 if fisher else 3.0
        
    mean_val = np.mean(x)
    std_val = np.std(x)
    if std_val < 1e-12:
        return 0.0 if fisher else 3.0
        
    diff = x - mean_val
    m4 = np.mean(diff ** 4)
    m2 = np.mean(diff ** 2)
    
    g2 = m4 / (m2 ** 2)
    
    if bias:
        if fisher:
            return float(g2 - 3.0)
        else:
            return float(g2)
    else:
        # Unbiased estimator
        val = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 - 3 * (n - 1))
        if fisher:
            return float(val - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        else:
            return float(val)


def compute_zcr(x: np.ndarray) -> float:
    """
    Calculate the Zero-Crossing Rate (ZCR) of a signal.
    
    The zero-crossing rate is the rate of sign-changes along a signal,
    i.e., the rate at which the signal changes from positive to negative
    or back.
    
    Mathematical Formulation:
        ZCR = (1 / (N - 1)) * sum_{t=1}^{N-1} I( x_t * x_{t-1} < 0 )
        
    Parameters
    ----------
    x : np.ndarray
        1-D input array.
        
    Returns
    -------
    float
        The zero-crossing rate (fraction between 0.0 and 1.0).
    """
    n = len(x)
    if n <= 1:
        return 0.0
        
    # Subtract mean to account for DC offset
    x_centered = x - np.mean(x)
    
    # Detect zero crossings
    crossings = np.diff(np.sign(x_centered)) != 0
    return float(np.sum(crossings) / (n - 1))


def compute_shannon_entropy(x: np.ndarray, bins: int = 50) -> float:
    """
    Calculate the Shannon Entropy of the signal's amplitude distribution.
    
    Entropy represents the uncertainty or information content in the signal.
    
    Mathematical Formulation:
        H(X) = - sum_{i=1}^M p(x_i) * log2( p(x_i) )
        
    where p(x_i) is the probability of the signal falling into bin i.
    
    Parameters
    ----------
    x : np.ndarray
        1-D input array.
    bins : int
        Number of bins for estimating the probability density function (default: 50).
        
    Returns
    -------
    float
        Shannon entropy in bits.
    """
    if len(x) == 0:
        return 0.0
        
    # Calculate histogram
    counts, _ = np.histogram(x, bins=bins)
    probs = counts / len(x)
    
    # Filter out zero probabilities to avoid log2(0)
    probs = probs[probs > 0]
    
    return float(-np.sum(probs * np.log2(probs)))


def compute_cross_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between two 1-D signals.
    
    Mathematical Formulation:
        r = cov(X, Y) / (sigma_X * sigma_Y)
        
    Parameters
    ----------
    x : np.ndarray
        First 1-D signal array.
    y : np.ndarray
        Second 1-D signal array.
        
    Returns
    -------
    float
        Pearson correlation coefficient in range [-1.0, 1.0].
        Returns 0.0 if either signal has zero variance.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
        
    std_x = np.std(x)
    std_y = np.std(y)
    if std_x < 1e-12 or std_y < 1e-12:
        return 0.0
        
    cov = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    return float(cov / (std_x * std_y))


# =====================================================================
# 8. Additional Classical IIR Filters and Windowing
# =====================================================================

def chebyshev1_lowpass(x: np.ndarray, fs: float, cutoff: float, rp: float = 1.0, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Chebyshev Type I lowpass filter.
    
    Chebyshev Type I filters have passband ripple but a sharper rolloff than Butterworth.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    rp : float
        Maximum ripple allowed in the passband in dB (default: 1.0).
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return x
    b, a = sp_signal.cheby1(order, rp, cutoff / nyq, btype='low')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def chebyshev2_lowpass(x: np.ndarray, fs: float, cutoff: float, rs: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Chebyshev Type II lowpass filter.
    
    Chebyshev Type II filters have stopband ripple but a monotonic passband.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    rs : float
        Minimum attenuation required in the stopband in dB (default: 40.0).
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return x
    b, a = sp_signal.cheby2(order, rs, cutoff / nyq, btype='low')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def ellip_lowpass(x: np.ndarray, fs: float, cutoff: float, rp: float = 1.0, rs: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Elliptic (Cauer) lowpass filter.
    
    Elliptic filters have ripple in both passband and stopband, but transition fastest.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    rp : float
        Maximum ripple in passband in dB (default: 1.0).
    rs : float
        Minimum stopband attenuation in dB (default: 40.0).
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return x
    b, a = sp_signal.ellip(order, rp, rs, cutoff / nyq, btype='low')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def bessel_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Bessel lowpass filter.
    
    Bessel filters have maximally flat group delay, preserving wave shapes.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return x
    b, a = sp_signal.bessel(order, cutoff / nyq, btype='low', norm='mag')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def detrend_polynomial(x: np.ndarray, deg: int = 1) -> np.ndarray:
    """
    Remove polynomial trend of degree `deg` from a signal.
    
    Mathematical Formulation:
        Trend(t) = sum_{j=0}^{deg} c_j * t^j
        Clean(t) = x(t) - Trend(t)
        
    Parameters
    ----------
    x : np.ndarray
        1-D input signal.
    deg : int
        Degree of the detrending polynomial (default: 1, linear detrend).
        
    Returns
    -------
    np.ndarray
        Detrended signal array.
    """
    n = len(x)
    if n <= deg:
        return x - np.mean(x)
        
    t = np.arange(n)
    coeffs = np.polyfit(t, x, deg)
    trend = np.polyval(coeffs, t)
    return x - trend


def moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a moving average smoothing filter to a 1-D signal.
    
    Mathematical Formulation:
        y[n] = (1 / W) * sum_{k=0}^{W-1} x[n - k]
        
    Parameters
    ----------
    x : np.ndarray
        1-D input array.
    window_size : int
        Length of the moving window. Must be >= 1.
        
    Returns
    -------
    np.ndarray
        Smoothed signal array matching input length (with boundary padding).
    """
    if window_size <= 1:
        return x
        
    window = np.ones(window_size) / window_size
    # Use mode='same' to preserve length and pad boundary using edge values
    padded = np.pad(x, window_size // 2, mode='edge')
    smoothed = np.convolve(padded, window, mode='valid')
    
    # Adjust length to match original precisely
    if len(smoothed) < len(x):
        smoothed = np.pad(smoothed, (0, len(x) - len(smoothed)), mode='edge')
    elif len(smoothed) > len(x):
        smoothed = smoothed[:len(x)]
        
    return smoothed


def compute_rmse(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two signals.
    
    Mathematical Formulation:
        RMSE = sqrt( (1 / N) * sum_{t=1}^N (x_t - y_t)^2 )
        
    Parameters
    ----------
    x : np.ndarray
        First signal array (typically clean reference).
    y : np.ndarray
        Second signal array (typically noisy estimate).
        
    Returns
    -------
    float
        RMSE value.
    """
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Signal lengths must be equal and non-zero")
    return float(np.sqrt(np.mean(np.square(x - y))))


def compute_ssim_1d(x: np.ndarray, y: np.ndarray, window_size: int = 101, k1: float = 0.01, k2: float = 0.03, dynamic_range: Optional[float] = None) -> float:
    """
    Calculate a 1-D approximation of the Structural Similarity (SSIM) Index.
    
    SSIM measures the similarity between two signals by comparing luminance
    (mean), contrast (variance), and structure (covariance) over a sliding window.
    
    Mathematical Formulation:
        SSIM(x, y) = ( (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2) ) / ( (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2) )
        
    where C1 = (k1 * L)^2 and C2 = (k2 * L)^2, where L is the dynamic range.
    
    Parameters
    ----------
    x : np.ndarray
        1-D reference signal.
    y : np.ndarray
        1-D target signal.
    window_size : int
        Size of the sliding window for local computations (default: 101).
        Must be odd and >= 3.
    k1 : float
        Luminance scaling constant (default: 0.01).
    k2 : float
        Contrast/structure scaling constant (default: 0.03).
    dynamic_range : Optional[float]
        Dynamic range L of the signal. If None, it is calculated as
        max(x) - min(x) (default: None).
        
    Returns
    -------
    float
        Mean 1-D SSIM index in range [-1.0, 1.0], where 1.0 indicates identity.
    """
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Signal lengths must be equal and non-zero")
        
    n = len(x)
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, n)
    if window_size < 3:
        window_size = 3
        
    if dynamic_range is None:
        dr = np.max(x) - np.min(x)
        if dr <= 1e-12:
            dr = 1.0
    else:
        dr = dynamic_range
        
    c1 = (k1 * dr) ** 2
    c2 = (k2 * dr) ** 2
    
    # B-11 FIX: Replaced O(N) pure-Python loop with vectorised cumsum-based
    # sliding window statistics — 100-1000x faster on real biosignal lengths.
    w = window_size
    
    # Use cumsum trick for O(1)-per-window sliding statistics
    def _sliding_sum(arr: np.ndarray) -> np.ndarray:
        cs = np.cumsum(arr, dtype=np.float64)
        out = cs[w - 1:]
        out = out.copy()
        out[1:] -= cs[:-(w)]
        return out
    
    # Sliding window sums
    sum_x   = _sliding_sum(x)
    sum_y   = _sliding_sum(y)
    sum_xx  = _sliding_sum(x * x)
    sum_yy  = _sliding_sum(y * y)
    sum_xy  = _sliding_sum(x * y)
    
    mu_x    = sum_x  / w
    mu_y    = sum_y  / w
    
    # Variance and covariance (biased estimator, consistent with original loop)
    var_x   = sum_xx / w - mu_x ** 2
    var_y   = sum_yy / w - mu_y ** 2
    cov_xy  = sum_xy / w - mu_x * mu_y
    
    # SSIM map
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    
    ssim_map = num / den
    
    return float(np.mean(ssim_map))


def chebyshev1_highpass(x: np.ndarray, fs: float, cutoff: float, rp: float = 1.0, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Chebyshev Type I highpass filter.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    rp : float
        Maximum ripple allowed in the passband in dB (default: 1.0).
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff <= 0.0 or cutoff >= nyq:
        return x - np.mean(x)
    b, a = sp_signal.cheby1(order, rp, cutoff / nyq, btype='high')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def chebyshev2_highpass(x: np.ndarray, fs: float, cutoff: float, rs: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Chebyshev Type II highpass filter.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    rs : float
        Minimum stopband attenuation in dB (default: 40.0).
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff <= 0.0 or cutoff >= nyq:
        return x - np.mean(x)
    b, a = sp_signal.cheby2(order, rs, cutoff / nyq, btype='high')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def ellip_highpass(x: np.ndarray, fs: float, cutoff: float, rp: float = 1.0, rs: float = 40.0, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Elliptic (Cauer) highpass filter.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    rp : float
        Maximum ripple in passband in dB (default: 1.0).
    rs : float
        Minimum stopband attenuation in dB (default: 40.0).
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff <= 0.0 or cutoff >= nyq:
        return x - np.mean(x)
    b, a = sp_signal.ellip(order, rp, rs, cutoff / nyq, btype='high')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)


def bessel_highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Bessel highpass filter.
    
    Parameters
    ----------
    x : np.ndarray
        Signal array.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Filter cutoff frequency in Hz.
    order : int
        Filter order (default: 4).
        
    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs
    if cutoff <= 0.0 or cutoff >= nyq:
        return x - np.mean(x)
    b, a = sp_signal.bessel(order, cutoff / nyq, btype='high', norm='mag')
    pad_len = min(150, len(x) // 3)
    return sp_signal.filtfilt(b, a, x, padlen=pad_len)
