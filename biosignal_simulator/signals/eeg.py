"""
High-fidelity Electroencephalogram (EEG) simulator.

This module provides a comprehensive, physiologically accurate simulation of brain
electrical activity as recorded from scalp electrodes (EEG). It implements:

1. **Multi-band rhythm synthesis**: Delta, Theta, Alpha, Beta, Gamma, and 1/f background
   noise, each filtered using Butterworth bandpass filters and combined with spatial
   cross-channel correlation.

2. **Sleep transients**: Sleep spindles (11-16 Hz wax-and-wane oscillations), K-complexes
   (high-amplitude biphasic slow waves), and Vertex sharp waves (negative deflections
   during NREM transitions).

3. **Sleep staging**: Four distinct EEG brain states:
   - Awake active (Beta/Gamma dominance, 10-30 µV)
   - Awake relaxed (Alpha dominance, 20-50 µV)
   - N2 sleep (Theta + spindles + K-complexes, 40-80 µV)
   - N3 deep sleep (Delta dominance, 100-200 µV)

4. **Epileptic seizure patterns**:
   - Tonic-Clonic: tonic (~10 Hz), clonic (3 Hz spike-wave), post-ictal suppression
   - Absence: continuous 3 Hz generalized spike-and-wave
   - Focal Onset: localized slow build-up spreading across channels
   - Interictal Epileptiform Discharges (IEDs): isolated sharp spikes + slow wave

5. **Event-Related Potentials (ERPs)**:
   - P300: positive deflection ~300 ms after auditory/visual oddball stimuli
   - N400: semantic mismatch potential
   - Alpha ERD/ERS: event-related desynchronization/synchronization

6. **Spatial modeling**: Realistic cross-channel correlation using Cholesky
   decomposition on a configurable correlation matrix.

References
----------
- Niedermeyer, E. & da Silva, F.L. (2004). Electroencephalography. Williams & Wilkins.
- Rechtschaffen, A. & Kales, A. (1968). A Manual of Standardized Terminology.
- Lopes da Silva, F. (2013). EEG and MEG: Relevance to Neuroscience. Neuron.
- Iber, C. et al. (2007). The AASM Manual for the Scoring of Sleep.
- Fisher, R.S. et al. (2017). Operational Classification of Seizure Types.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

from biosignal_simulator.core.base import BaseSignal
from biosignal_simulator.core.config import EEGConfig
from biosignal_simulator.core.math_utils import normalize_to_rms, spectral_shape


# ──────────────────────────────────────────────────────────────────────────────
# EEG Band Definitions
# ──────────────────────────────────────────────────────────────────────────────

class EEGBand(str, Enum):
    """Standard EEG frequency bands."""
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"
    HIGH_GAMMA = "high_gamma"
    INFRASLOW = "infraslow"
    PINK = "1f"


# Standard band frequency ranges (Hz)
BAND_RANGES: Dict[str, Tuple[float, float]] = {
    "infraslow": (0.01, 0.5),
    "delta":     (0.5, 4.0),
    "theta":     (4.0, 8.0),
    "alpha":     (8.0, 13.0),
    "beta":      (13.0, 30.0),
    "gamma":     (30.0, 80.0),
    "high_gamma":(80.0, 150.0),
}

# Standard EEG electrode positions (10-20 system)
ELECTRODE_10_20 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7",  "C3",  "Cz", "C4", "T8",
    "P7",  "P3",  "Pz", "P4", "P8",
    "O1",  "Oz",  "O2",
]


# ──────────────────────────────────────────────────────────────────────────────
# EEG State Definitions
# ──────────────────────────────────────────────────────────────────────────────

class EEGState(str, Enum):
    """Enumeration of supported EEG brain states."""
    ACTIVE          = "active"
    RELAXED         = "relaxed"
    N1_SLEEP        = "n1_sleep"
    N2_SLEEP        = "n2_sleep"
    N3_SLEEP        = "n3_sleep"
    REM_SLEEP       = "rem_sleep"
    TONIC_CLONIC    = "tonic_clonic"
    ABSENCE         = "absence"
    FOCAL_SEIZURE   = "focal_seizure"
    EPIFOARM_SPIKES = "epileptiform_spikes"
    ANESTHESIA      = "anesthesia"
    BURST_SUPPRESS  = "burst_suppression"


# Brain state power spectrum weights: {state: {band: relative_power, amplitude_uV, 1f_weight}}
BRAIN_STATE_PROFILES: Dict[str, Dict[str, Any]] = {
    "active": {
        "delta": 0.10, "theta": 0.10, "alpha": 0.20, "beta": 0.70,
        "gamma": 0.45, "high_gamma": 0.15, "infraslow": 0.05,
        "1f": 0.20, "amplitude_uv": 15.0,
    },
    "relaxed": {
        "delta": 0.10, "theta": 0.15, "alpha": 1.00, "beta": 0.30,
        "gamma": 0.10, "high_gamma": 0.05, "infraslow": 0.10,
        "1f": 0.30, "amplitude_uv": 35.0,
    },
    "n1_sleep": {
        "delta": 0.20, "theta": 0.80, "alpha": 0.35, "beta": 0.10,
        "gamma": 0.05, "high_gamma": 0.02, "infraslow": 0.15,
        "1f": 0.35, "amplitude_uv": 50.0,
    },
    "n2_sleep": {
        "delta": 0.40, "theta": 1.00, "alpha": 0.05, "beta": 0.10,
        "gamma": 0.02, "high_gamma": 0.01, "infraslow": 0.15,
        "1f": 0.40, "amplitude_uv": 65.0,
    },
    "n3_sleep": {
        "delta": 1.50, "theta": 0.20, "alpha": 0.02, "beta": 0.05,
        "gamma": 0.01, "high_gamma": 0.00, "infraslow": 0.20,
        "1f": 0.20, "amplitude_uv": 120.0,
    },
    "rem_sleep": {
        "delta": 0.10, "theta": 0.60, "alpha": 0.30, "beta": 0.50,
        "gamma": 0.25, "high_gamma": 0.10, "infraslow": 0.10,
        "1f": 0.25, "amplitude_uv": 20.0,
    },
    "tonic_clonic": {
        "delta": 0.30, "theta": 0.30, "alpha": 0.70, "beta": 0.40,
        "gamma": 0.15, "high_gamma": 0.10, "infraslow": 0.05,
        "1f": 0.30, "amplitude_uv": 45.0,
    },
    "absence": {
        "delta": 0.20, "theta": 0.25, "alpha": 0.50, "beta": 0.20,
        "gamma": 0.08, "high_gamma": 0.05, "infraslow": 0.05,
        "1f": 0.25, "amplitude_uv": 40.0,
    },
    "focal_seizure": {
        "delta": 0.25, "theta": 0.35, "alpha": 0.45, "beta": 0.30,
        "gamma": 0.20, "high_gamma": 0.15, "infraslow": 0.10,
        "1f": 0.25, "amplitude_uv": 40.0,
    },
    "epileptiform_spikes": {
        "delta": 0.25, "theta": 0.30, "alpha": 0.70, "beta": 0.35,
        "gamma": 0.15, "high_gamma": 0.08, "infraslow": 0.05,
        "1f": 0.25, "amplitude_uv": 40.0,
    },
    "anesthesia": {
        "delta": 1.20, "theta": 0.40, "alpha": 0.80, "beta": 0.10,
        "gamma": 0.05, "high_gamma": 0.01, "infraslow": 0.30,
        "1f": 0.30, "amplitude_uv": 80.0,
    },
    "burst_suppression": {
        "delta": 0.50, "theta": 0.20, "alpha": 0.10, "beta": 0.05,
        "gamma": 0.02, "high_gamma": 0.01, "infraslow": 0.20,
        "1f": 0.20, "amplitude_uv": 60.0,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper: build spatial correlation matrix
# ──────────────────────────────────────────────────────────────────────────────

def build_spatial_correlation_matrix(
    n_channels: int,
    base_correlation: float = 0.60,
    decay_rate: float = 0.25,
) -> np.ndarray:
    """
    Construct a positive-definite spatial correlation matrix for EEG channels.

    Uses an exponential decay model: R[i, j] = base_corr * exp(-decay * |i-j|).
    This approximates the spatial smoothing effect of the skull and scalp.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    base_correlation : float
        Correlation coefficient between adjacent channels. Default: 0.60.
    decay_rate : float
        Exponential decay rate with channel distance. Default: 0.25.

    Returns
    -------
    np.ndarray
        Positive-definite correlation matrix of shape (n_channels, n_channels).

    Examples
    --------
    >>> R = build_spatial_correlation_matrix(4, base_correlation=0.5, decay_rate=0.3)
    >>> R.shape
    (4, 4)
    >>> np.allclose(np.diag(R), 1.0)
    True
    """
    R = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            dist = abs(i - j)
            if i == j:
                R[i, j] = 1.0
            else:
                R[i, j] = base_correlation * np.exp(-decay_rate * dist)

    # Ensure positive definiteness via Higham's method (simplified)
    eigvals = np.linalg.eigvalsh(R)
    if np.any(eigvals <= 0):
        # Add small regularization
        R += np.eye(n_channels) * (abs(np.min(eigvals)) + 1e-6)
        # Renormalize diagonal
        diag_sqrt = np.sqrt(np.diag(R))
        R /= np.outer(diag_sqrt, diag_sqrt)

    return R


def compute_cholesky_mixing(
    corr_matrix: Optional[np.ndarray], n_channels: int, base_corr: float = 0.6
) -> np.ndarray:
    """
    Compute the Cholesky lower-triangular mixing matrix L such that L @ L^T = R.

    Parameters
    ----------
    corr_matrix : np.ndarray, optional
        User-specified correlation matrix. If None, a default exponential
        decay matrix is constructed.
    n_channels : int
        Number of EEG channels.
    base_corr : float
        Base correlation for default matrix construction.

    Returns
    -------
    np.ndarray
        Cholesky factor L of shape (n_channels, n_channels).
    """
    if corr_matrix is not None:
        R = np.array(corr_matrix, dtype=np.float64)
    else:
        R = build_spatial_correlation_matrix(n_channels, base_correlation=base_corr)

    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # Fallback: identity matrix if Cholesky fails
        warnings.warn("Correlation matrix is not positive definite. Using identity mixing.")
        L = np.eye(n_channels)

    return L


# ──────────────────────────────────────────────────────────────────────────────
# EEG Band Synthesis Helper
# ──────────────────────────────────────────────────────────────────────────────

def synthesize_band(
    rng: np.random.Generator,
    band: str,
    n_channels: int,
    n_samples: int,
    fs: float,
    alpha_peak_hz: float = 10.0,
    high_gamma_cutoff: float = 120.0,
) -> np.ndarray:
    """
    Synthesize filtered band noise for a single EEG frequency band.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    band : str
        Band name: 'delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma',
        'infraslow', or '1f'.
    n_channels : int
        Number of channels.
    n_samples : int
        Number of samples.
    fs : float
        Sampling frequency in Hz.
    alpha_peak_hz : float
        Alpha band center frequency (Hz). Default: 10.0.
    high_gamma_cutoff : float
        Upper cutoff for high gamma band (Hz). Default: 120.0.

    Returns
    -------
    np.ndarray
        Band-filtered noise of shape (n_channels, n_samples).
    """
    raw_noise = rng.normal(0.0, 1.0, size=(n_channels, n_samples))
    filtered = np.zeros_like(raw_noise)

    if band == "1f":
        # Pink noise via spectral shaping
        shaper = spectral_shape(n_samples, fs, exponent=1.0)
        for c in range(n_channels):
            w_dft = np.fft.rfft(raw_noise[c])
            x_dft = w_dft * shaper
            filtered[c] = np.fft.irfft(x_dft, n=n_samples)

    elif band == "infraslow":
        nyq = 0.5 * fs
        low, high = 0.01, 0.5
        if high >= nyq:
            high = nyq * 0.9
        if low < high:
            b, a = sp_signal.butter(2, [low / nyq, high / nyq], btype="bandpass")
            for c in range(n_channels):
                filtered[c] = sp_signal.filtfilt(b, a, raw_noise[c])

    elif band == "high_gamma":
        nyq = 0.5 * fs
        low = 80.0
        high = min(high_gamma_cutoff, nyq * 0.95)
        if low < nyq and low < high:
            b, a = sp_signal.butter(4, [low / nyq, high / nyq], btype="bandpass")
            for c in range(n_channels):
                filtered[c] = sp_signal.filtfilt(b, a, raw_noise[c])

    else:
        nyq = 0.5 * fs
        brange = BAND_RANGES.get(band, (0.5, 4.0))
        low, high = brange

        # Alpha band uses user-specified peak ± bandwidth
        if band == "alpha":
            bw = 2.0  # Half-bandwidth in Hz
            low = max(0.5, alpha_peak_hz - bw)
            high = min(nyq * 0.95, alpha_peak_hz + bw)
            order = 6
        else:
            order = 4

        high = min(high, nyq * 0.95)
        if low >= nyq or low >= high:
            return filtered

        b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
        for c in range(n_channels):
            filtered[c] = sp_signal.filtfilt(b, a, raw_noise[c])

    return filtered


def compute_band_power_spectrum(
    eeg: np.ndarray,
    fs: float,
    window_s: float = 4.0,
    overlap: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Estimate power spectral density for each EEG band using Welch's method.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal. Shape (n_channels, n_samples) or (n_samples,).
    fs : float
        Sampling frequency in Hz.
    window_s : float
        Welch window length in seconds.
    overlap : float
        Fractional window overlap (0-1).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping band name to average power per channel:
        - 'freqs': np.ndarray — frequency array (Hz)
        - 'psd': np.ndarray — power spectral density (V²/Hz)
        - 'delta', 'theta', 'alpha', 'beta', 'gamma': float — band power values
    """
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]

    n_channels, n_samples = eeg.shape
    nperseg = int(window_s * fs)
    noverlap = int(nperseg * overlap)

    all_freqs = None
    all_psd = []

    for c in range(n_channels):
        freqs, psd = sp_signal.welch(eeg[c], fs=fs, nperseg=nperseg, noverlap=noverlap)
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    all_psd = np.array(all_psd)
    mean_psd = np.mean(all_psd, axis=0)

    result: Dict[str, Any] = {"freqs": all_freqs, "psd": mean_psd}

    # Compute absolute band powers (area under PSD in each band)
    for band_name, (low, high) in BAND_RANGES.items():
        mask = (all_freqs >= low) & (all_freqs < high)
        if np.any(mask):
            df = all_freqs[1] - all_freqs[0]
            result[band_name] = float(np.sum(mean_psd[mask]) * df)
        else:
            result[band_name] = 0.0

    return result


def measure_alpha_peak(eeg: np.ndarray, fs: float) -> float:
    """
    Detect the dominant alpha peak frequency from the EEG PSD.

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal (n_channels, n_samples) or (n_samples,).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Alpha peak frequency in Hz. Returns 10.0 if no peak detected.
    """
    band_data = compute_band_power_spectrum(eeg, fs)
    freqs = band_data["freqs"]
    psd = band_data["psd"]

    alpha_mask = (freqs >= 7.5) & (freqs <= 13.5)
    if not np.any(alpha_mask):
        return 10.0

    alpha_psd = psd[alpha_mask]
    alpha_freqs = freqs[alpha_mask]

    peak_idx = np.argmax(alpha_psd)
    return float(alpha_freqs[peak_idx])


# ──────────────────────────────────────────────────────────────────────────────
# Sleep Transient Generators
# ──────────────────────────────────────────────────────────────────────────────

class SleepTransientGenerator:
    """
    Generator for EEG sleep transients: spindles, K-complexes, and vertex waves.

    This class encapsulates the generation of NREM sleep-specific EEG transients
    that are superimposed on the background rhythm.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    fs : float
        Sampling frequency in Hz.
    """

    def __init__(self, rng: np.random.Generator, fs: float) -> None:
        self.rng = rng
        self.fs = fs

    def generate_sleep_spindle(
        self,
        center_time: float,
        frequency_hz: float,
        duration_s: float,
        amplitude_uv: float,
        phase_shift: float = 0.0,
        t: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Synthesize a single sleep spindle.

        A sleep spindle is a waxing-and-waning sinusoidal burst in the sigma band
        (11-16 Hz), typically lasting 0.5-2 seconds, generated by thalamo-cortical
        oscillatory loops.

        Parameters
        ----------
        center_time : float
            Center time of the spindle in seconds.
        frequency_hz : float
            Carrier frequency (11-16 Hz).
        duration_s : float
            Duration of the spindle envelope (seconds).
        amplitude_uv : float
            Peak amplitude in microvolts.
        phase_shift : float
            Phase shift of the carrier sinusoid (radians).
        t : np.ndarray, optional
            Pre-computed time array. If None, must provide n_samples.
        n_samples : int, optional
            Number of samples (only used if t is None).

        Returns
        -------
        np.ndarray
            Sleep spindle waveform.

        Notes
        -----
        The spindle is modeled as:
            spindle(t) = A * sin(2π * f * t + φ) * Gaussian_envelope(t, center, σ)
        where σ = duration_s / 4.
        """
        if t is None:
            raise ValueError("Must provide time array t.")

        sigma = duration_s / 4.0
        envelope = np.exp(-0.5 * ((t - center_time) / sigma) ** 2)
        carrier = np.sin(2.0 * np.pi * frequency_hz * (t - center_time) + phase_shift)
        return amplitude_uv * carrier * envelope

    def generate_k_complex(
        self,
        center_time: float,
        amplitude_uv: float,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize a single K-complex waveform.

        A K-complex consists of:
        1. A sharp negative deflection (negative peak, ~120 ms width, large amplitude)
        2. Followed by a slow positive wave (broad, ~300 ms width, ~55% amplitude)

        K-complexes are the largest EEG events of the normal EEG and are a defining
        feature of N2 sleep. They are believed to play a role in memory consolidation
        and cortical arousal suppression.

        Parameters
        ----------
        center_time : float
            Timestamp of the negative peak (center of K-complex) in seconds.
        amplitude_uv : float
            Peak negative amplitude in microvolts (will be negated).
        t : np.ndarray
            Time array in seconds.

        Returns
        -------
        np.ndarray
            K-complex waveform.
        """
        # Sharp negative deflection
        sigma_neg = 0.10  # ~200 ms total width
        neg_wave = -amplitude_uv * np.exp(-0.5 * ((t - center_time) / sigma_neg) ** 2)

        # Slow positive wave (delayed, broader)
        dt_pos = 0.28  # Offset of positive peak from negative
        sigma_pos = 0.25  # Broader positive wave
        pos_amp = amplitude_uv * 0.55
        pos_wave = pos_amp * np.exp(-0.5 * ((t - (center_time + dt_pos)) / sigma_pos) ** 2)

        return neg_wave + pos_wave

    def generate_vertex_wave(
        self,
        center_time: float,
        amplitude_uv: float,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize a Vertex Sharp Wave (VSW).

        Vertex waves are monophasic negative sharp transients occurring at the vertex
        (Cz electrode), most prominent during N1 sleep and early N2 transitions.
        They are time-locked to arousal stimuli and spontaneous transitions.

        Parameters
        ----------
        center_time : float
            Peak time of the vertex wave.
        amplitude_uv : float
            Peak negative amplitude in microvolts.
        t : np.ndarray
            Time array in seconds.

        Returns
        -------
        np.ndarray
            Vertex wave signal.
        """
        sigma = 0.15  # ~300 ms duration
        return -amplitude_uv * np.exp(-0.5 * ((t - center_time) / sigma) ** 2)

    def generate_pos(
        self,
        center_time: float,
        amplitude_uv: float,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize a Positive Occipital Sharp Transient of Sleep (POSTS).

        POSTS are small positive deflections seen in occipital channels during
        N1 and N2 sleep. They appear as saw-tooth or V-shaped waves.

        Parameters
        ----------
        center_time : float
            Peak time.
        amplitude_uv : float
            Peak amplitude.
        t : np.ndarray
            Time array.

        Returns
        -------
        np.ndarray
            POSTS waveform.
        """
        # Sharper rising slope, gentle falling slope
        result = np.zeros_like(t)
        rise_sigma = 0.04
        fall_sigma = 0.10
        pre_mask = t < center_time
        post_mask = t >= center_time
        result[pre_mask] = amplitude_uv * np.exp(
            -0.5 * ((t[pre_mask] - center_time) / rise_sigma) ** 2
        )
        result[post_mask] = amplitude_uv * np.exp(
            -0.5 * ((t[post_mask] - center_time) / fall_sigma) ** 2
        )
        return result

    def generate_saw_tooth_rem(
        self,
        center_time: float,
        amplitude_uv: float,
        duration_s: float,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize REM sleep sawtooth waves.

        Sawtooth waves are characteristic of REM sleep, appearing as bursts of
        notched, 2-6 Hz theta activity, particularly prominent in frontal and
        central derivations.

        Parameters
        ----------
        center_time : float
            Center time of the sawtooth burst.
        amplitude_uv : float
            Amplitude of the sawtooth waves.
        duration_s : float
            Duration of the burst in seconds.
        t : np.ndarray
            Time array.

        Returns
        -------
        np.ndarray
            Sawtooth wave burst.
        """
        # Sawtooth at 4-5 Hz within a Gaussian envelope
        f_saw = self.rng.uniform(3.0, 5.0)
        sigma = duration_s / 4.0
        envelope = amplitude_uv * np.exp(-0.5 * ((t - center_time) / sigma) ** 2)
        sawtooth = sp_signal.sawtooth(2.0 * np.pi * f_saw * (t - center_time), width=0.7)
        return envelope * sawtooth


# ──────────────────────────────────────────────────────────────────────────────
# Seizure Pattern Generator
# ──────────────────────────────────────────────────────────────────────────────

class SeizurePatternGenerator:
    """
    Generator for EEG epileptic seizure patterns.

    Implements tonic-clonic, absence, focal onset, and interictal
    epileptiform discharge (IED) patterns with realistic morphology.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    fs : float
        Sampling frequency in Hz.
    n_channels : int
        Number of EEG channels.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        fs: float,
        n_channels: int,
    ) -> None:
        self.rng = rng
        self.fs = fs
        self.n_channels = n_channels

    def generate_tonic_clonic(
        self,
        t: np.ndarray,
        duration_s: float,
        channels: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize a generalized Tonic-Clonic Seizure pattern.

        The seizure consists of three sequential phases:
        1. **Tonic phase** (first 35%): sustained high-frequency muscle activation
           at ~10 Hz. EEG shows continuous spiking at 8-14 Hz, amplitude 150-250 µV.
        2. **Clonic phase** (middle 45%): rhythmic muscle jerks. EEG shows 3 Hz
           spike-and-wave discharges with declining frequency (8→3 Hz).
        3. **Post-ictal phase** (last 20%): EEG suppression to <10 µV, diffuse
           slowing, exhaustion of GABAergic inhibitory mechanisms.

        Parameters
        ----------
        t : np.ndarray
            Time axis.
        duration_s : float
            Total duration in seconds.
        channels : np.ndarray
            Background EEG channels of shape (n_ch, n_samples) to modify.

        Returns
        -------
        np.ndarray
            Modified EEG channels with seizure activity.
        """
        n_ch = self.n_channels
        n_samples = len(t)

        # Phase boundaries
        t_tonic_end = 0.35 * duration_s
        t_clonic_end = 0.80 * duration_s

        seizure_wave = np.zeros(n_samples)

        # ── Tonic Phase ───────────────────────────────────────────────────────
        tonic_mask = t <= t_tonic_end
        if np.any(tonic_mask):
            t_tonic = t[tonic_mask]
            f_tonic = 10.0  # 10 Hz spiking

            # Amplitude envelope: ramp from 0 to 180 µV in first 2 seconds
            amp_env = np.full(np.sum(tonic_mask), 180.0)
            ramp_dur = min(2.0, t_tonic_end * 0.5)
            ramp_mask_local = t_tonic < ramp_dur
            if np.any(ramp_mask_local):
                amp_env[ramp_mask_local] = 180.0 * (t_tonic[ramp_mask_local] / ramp_dur) ** 2

            # High-frequency spike train (abs(sin) mimics sharp spikes)
            spikes = amp_env * (np.abs(np.sin(np.pi * f_tonic * t_tonic)) ** 0.7 - 0.3)
            seizure_wave[tonic_mask] = spikes

        # ── Clonic Phase ──────────────────────────────────────────────────────
        clonic_mask = (t > t_tonic_end) & (t <= t_clonic_end)
        if np.any(clonic_mask):
            t_clonic = t[clonic_mask]
            # Frequency decreases from 8 Hz to 3 Hz over clonic phase
            clonic_duration = t_clonic_end - t_tonic_end
            clonic_progress = (t_clonic - t_tonic_end) / clonic_duration
            f_clonic_inst = 8.0 - 5.0 * clonic_progress  # 8 → 3 Hz

            # Phase via integration of instantaneous frequency
            dt_clonic = 1.0 / self.fs
            phase_clonic = 2.0 * np.pi * np.cumsum(f_clonic_inst) * dt_clonic

            # Sharp spike component (negative)
            spike = -220.0 * np.exp(-0.5 * ((np.sin(phase_clonic / 2.0)) / 0.08) ** 2)
            # Broad positive slow wave
            wave = 155.0 * np.exp(-0.5 * ((np.sin(phase_clonic / 2.0 - 0.7)) / 0.26) ** 2)

            # Gradual amplitude decline over clonic phase
            fade = 1.0 - 0.4 * clonic_progress
            seizure_wave[clonic_mask] = (spike + wave) * fade

        # ── Apply to all channels ─────────────────────────────────────────────
        for c in range(n_ch):
            suppression = np.ones(n_samples)
            # Suppress background during tonic and clonic
            suppression[t <= t_clonic_end] = 0.12
            # Post-ictal: deep suppression
            suppression[t > t_clonic_end] = 0.06

            # Small channel-wise amplitude variation
            scale = self.rng.uniform(0.82, 1.18)
            channels[c] = channels[c] * suppression + seizure_wave * scale

        return channels

    def generate_absence_seizure(
        self,
        t: np.ndarray,
        channels: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize a generalized Absence Seizure pattern.

        Characterized by abrupt onset and offset of continuous 2.5-3.5 Hz
        generalized spike-and-wave discharges across all electrodes, highly
        synchronous. Patient appears "absent" (blank stare) during the event.

        Clinical features:
        - Duration: 4-30 seconds
        - Frequency: exactly 3 Hz (sometimes 2.5-3.5 Hz range)
        - Morphology: sharp spike followed by slow wave in each cycle
        - High synchrony across all channels (correlation > 0.95)

        Parameters
        ----------
        t : np.ndarray
            Time axis.
        channels : np.ndarray
            Background EEG channels to modify.

        Returns
        -------
        np.ndarray
            Modified EEG channels with absence seizure activity.
        """
        n_ch = self.n_channels

        # Precise 3 Hz spike-and-wave
        f_abs = self.rng.uniform(2.8, 3.2)  # Slight variation around 3 Hz
        phase = 2.0 * np.pi * f_abs * t

        # Spike component: very sharp negative deflection
        spike = -260.0 * np.exp(-0.5 * ((np.sin(phase / 2.0)) / 0.065) ** 2)
        # Slow wave: broader positive component
        wave = 185.0 * np.exp(-0.5 * ((np.sin(phase / 2.0 - 0.85)) / 0.28) ** 2)
        absence_wave = spike + wave

        # Bilateral synchrony: very high correlation across all channels
        for c in range(n_ch):
            # Tiny amplitude variation per channel (highly synchronous)
            noise = self.rng.uniform(0.92, 1.08)
            # Normal background is almost completely replaced
            channels[c] = channels[c] * 0.18 + absence_wave * noise

        return channels

    def generate_focal_seizure(
        self,
        t: np.ndarray,
        channels: np.ndarray,
        onset_channel: int = 0,
    ) -> np.ndarray:
        """
        Synthesize a Focal (Partial) Seizure with secondary generalization.

        Focal seizures begin in a localized brain region (one or two channels)
        and may spread to adjacent channels over 5-15 seconds.

        Phases:
        1. **Focal onset**: rapid, low-amplitude, high-frequency (15-30 Hz) activity
           in the onset channels.
        2. **Evolution**: slowing of frequency, increasing amplitude, and spatial spread.
        3. **Terminal**: slow irregular activity and brief suppression.

        Parameters
        ----------
        t : np.ndarray
            Time axis.
        channels : np.ndarray
            Background EEG channels.
        onset_channel : int
            Index of the seizure onset channel.

        Returns
        -------
        np.ndarray
            Modified EEG channels with focal seizure.
        """
        n_ch = self.n_channels
        duration = t[-1]

        # Seizure evolves over the full duration
        # Phase 1 (onset): first 20%, high freq low amp
        # Phase 2 (build): 20-70%, slowing, spreading
        # Phase 3 (clonic-like): 70-90%, rhythmic slow
        # Phase 4 (postictal): 90-100%, suppression

        t1 = 0.20 * duration
        t2 = 0.70 * duration
        t3 = 0.90 * duration

        for c in range(n_ch):
            # Spatial distance from onset channel
            dist = abs(c - onset_channel)
            # Channels further from onset have delayed onset and reduced amplitude
            delay = dist * 0.3  # 300 ms delay per channel distance
            spread_amp = np.exp(-0.25 * dist)  # Exponential amplitude decay with distance

            seizure = np.zeros(len(t))

            t_adj = t - delay  # Adjusted time with propagation delay

            # Phase 1: low-amplitude high-frequency
            mask1 = (t_adj > 0) & (t_adj <= t1)
            if np.any(mask1):
                f1 = 20.0 - 8.0 * t_adj[mask1] / t1  # 20 → 12 Hz
                phase1 = 2.0 * np.pi * np.cumsum(f1) / self.fs
                env1 = 15.0 * spread_amp * (t_adj[mask1] / t1) ** 2
                seizure[mask1] = env1 * np.sin(phase1)

            # Phase 2: building, slowing spread
            mask2 = (t_adj > t1) & (t_adj <= t2)
            if np.any(mask2):
                progress2 = (t_adj[mask2] - t1) / (t2 - t1)
                f2 = 12.0 - 9.0 * progress2  # 12 → 3 Hz
                phase2 = 2.0 * np.pi * np.cumsum(f2) / self.fs
                env2 = spread_amp * (30.0 + 120.0 * progress2)
                seizure[mask2] = env2 * np.sin(phase2)

            # Phase 3: clonic-like
            mask3 = (t_adj > t2) & (t_adj <= t3)
            if np.any(mask3):
                progress3 = (t_adj[mask3] - t2) / (t3 - t2)
                f3 = 3.0 - 1.0 * progress3
                phase3 = 2.0 * np.pi * np.cumsum(f3) / self.fs
                env3 = spread_amp * 150.0 * (1.0 - progress3)
                spike3 = -env3 * np.exp(-0.5 * ((np.sin(phase3)) / 0.12) ** 2)
                seizure[mask3] = spike3

            # Phase 4: postictal suppression
            mask4 = t_adj > t3
            if np.any(mask4):
                channels[c, mask4] *= 0.08

            # Apply clamp
            seizure = np.clip(seizure, -400.0, 400.0)
            channels[c] += seizure

        return channels

    def inject_interictal_spikes(
        self,
        t: np.ndarray,
        channels: np.ndarray,
        spike_rate_hz: float = 0.25,
        focal: bool = False,
        focal_channels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Inject interictal epileptiform discharges (IEDs) onto background EEG.

        IEDs are stereotyped sharp waves or spikes occurring between seizures.
        They are a hallmark of epileptic brain disorders and indicate cortical
        hyperexcitability.

        Morphology:
        - Sharp spike: negative peak, 70-200 ms duration, 100-300 µV
        - Slow wave: positive afterpotential, 200-500 ms duration, 30-80 µV

        Parameters
        ----------
        t : np.ndarray
            Time axis.
        channels : np.ndarray
            EEG channels to inject spikes into.
        spike_rate_hz : float
            Average rate of IEDs per second (Poisson process).
        focal : bool
            If True, IEDs are restricted to focal_channels. Otherwise generalized.
        focal_channels : List[int], optional
            Channel indices for focal IEDs.

        Returns
        -------
        np.ndarray
            Modified EEG channels.
        """
        n_ch = self.n_channels
        duration = t[-1]
        n_samples = len(t)

        # Number of IEDs
        n_spikes = self.rng.poisson(spike_rate_hz * duration)
        if n_spikes == 0 and duration >= 4.0:
            n_spikes = 1

        if n_spikes == 0:
            return channels

        # Spike timing (Poisson process)
        spike_times = self.rng.uniform(0.3, max(0.31, duration - 0.3), size=n_spikes)

        for t_c in spike_times:
            # Spike and slow-wave morphology
            sig_spike = self.rng.uniform(0.010, 0.018)  # 20-36 ms half-width
            sig_slow = self.rng.uniform(0.05, 0.08)
            dt_slow = self.rng.uniform(0.07, 0.10)
            amp = self.rng.uniform(140.0, 250.0)

            # Spike template
            spike_wave = -amp * np.exp(-0.5 * ((t - t_c) / sig_spike) ** 2)
            # Slow afterpotential (positive)
            slow_wave = (amp * self.rng.uniform(0.25, 0.40)) * np.exp(
                -0.5 * ((t - (t_c + dt_slow)) / sig_slow) ** 2
            )
            template = spike_wave + slow_wave

            # Distribute across channels
            if focal and focal_channels:
                target_channels = focal_channels
            else:
                target_channels = list(range(n_ch))

            for c in target_channels:
                coupling = self.rng.uniform(0.5, 1.3)
                channels[c] += template * coupling

        return channels


# ──────────────────────────────────────────────────────────────────────────────
# Event-Related Potential Generator
# ──────────────────────────────────────────────────────────────────────────────

class ERPGenerator:
    """
    Generator for Event-Related Potentials (ERPs) in EEG.

    ERPs are time-locked neural responses to sensory, cognitive, or motor events.
    This class generates standard ERP components including P300, N400, N200,
    and mismatch negativity (MMN).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    fs : float
        Sampling frequency in Hz.
    """

    def __init__(self, rng: np.random.Generator, fs: float) -> None:
        self.rng = rng
        self.fs = fs

    def generate_p300(
        self,
        event_time: float,
        t: np.ndarray,
        amplitude_uv: float = 8.0,
        latency_ms: float = 300.0,
        width_ms: float = 100.0,
    ) -> np.ndarray:
        """
        Synthesize a P300 event-related potential.

        The P300 is a positive deflection occurring approximately 300 ms after
        an infrequent target stimulus (oddball paradigm). It reflects cognitive
        processes of attention and working memory updating.

        Clinical relevance:
        - Reduced amplitude in ADHD, schizophrenia, depression
        - Prolonged latency in dementia, mild cognitive impairment
        - Used in brain-computer interfaces (BCIs)

        Parameters
        ----------
        event_time : float
            Onset time of the eliciting stimulus in seconds.
        t : np.ndarray
            Time axis.
        amplitude_uv : float
            Peak amplitude in microvolts. Normal P300: 5-20 µV.
        latency_ms : float
            Latency of P300 peak after stimulus onset (ms). Normal: 250-400 ms.
        width_ms : float
            Half-width of the P300 component (ms).

        Returns
        -------
        np.ndarray
            P300 waveform.
        """
        peak_time = event_time + latency_ms / 1000.0
        sigma = width_ms / 1000.0 / 2.0  # Convert ms to seconds
        return amplitude_uv * np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)

    def generate_n400(
        self,
        event_time: float,
        t: np.ndarray,
        amplitude_uv: float = -5.0,
        latency_ms: float = 400.0,
        width_ms: float = 120.0,
    ) -> np.ndarray:
        """
        Synthesize an N400 event-related potential.

        The N400 is a negative deflection peaking around 400 ms after a
        semantically unexpected word or image. It reflects the effort of semantic
        integration in language processing.

        Parameters
        ----------
        event_time : float
            Stimulus onset time (seconds).
        t : np.ndarray
            Time axis.
        amplitude_uv : float
            Peak amplitude (negative by convention). Default: -5 µV.
        latency_ms : float
            Latency of N400 peak (ms). Default: 400 ms.
        width_ms : float
            Half-width (ms). Default: 120 ms.

        Returns
        -------
        np.ndarray
            N400 waveform.
        """
        peak_time = event_time + latency_ms / 1000.0
        sigma = width_ms / 1000.0 / 2.0
        return amplitude_uv * np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)

    def generate_mismatch_negativity(
        self,
        event_time: float,
        t: np.ndarray,
        amplitude_uv: float = -3.5,
        latency_ms: float = 150.0,
        width_ms: float = 60.0,
    ) -> np.ndarray:
        """
        Synthesize a Mismatch Negativity (MMN) ERP component.

        The MMN is generated pre-attentively when an auditory deviant stimulus
        occurs in a sequence of standard stimuli. It reflects automatic change
        detection in the auditory cortex.

        Parameters
        ----------
        event_time : float
            Stimulus onset time (seconds).
        t : np.ndarray
            Time axis.
        amplitude_uv : float
            Peak amplitude (negative). Default: -3.5 µV.
        latency_ms : float
            MMN peak latency (ms). Default: 150 ms.
        width_ms : float
            MMN width (ms). Default: 60 ms.

        Returns
        -------
        np.ndarray
            MMN waveform.
        """
        peak_time = event_time + latency_ms / 1000.0
        sigma = width_ms / 1000.0 / 2.0
        return amplitude_uv * np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)

    def generate_n200(
        self,
        event_time: float,
        t: np.ndarray,
        amplitude_uv: float = -4.0,
        latency_ms: float = 200.0,
        width_ms: float = 80.0,
    ) -> np.ndarray:
        """
        Synthesize an N200 (N2) ERP component.

        The N200 is a negative deflection peaking at ~200 ms, related to
        stimulus discrimination and conflict monitoring (NoGo N2 in Go/NoGo tasks).

        Parameters
        ----------
        event_time : float
            Stimulus onset (seconds).
        t : np.ndarray
            Time axis.
        amplitude_uv : float
            Peak amplitude (negative). Default: -4.0 µV.
        latency_ms : float
            N200 latency (ms). Default: 200 ms.
        width_ms : float
            Width (ms). Default: 80 ms.

        Returns
        -------
        np.ndarray
            N200 waveform.
        """
        peak_time = event_time + latency_ms / 1000.0
        sigma = width_ms / 1000.0 / 2.0
        return amplitude_uv * np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Burst Suppression Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_burst_suppression(
    t: np.ndarray,
    channels: np.ndarray,
    rng: np.random.Generator,
    fs: float,
    burst_probability: float = 0.35,
    burst_duration_s: float = 1.0,
    suppression_amplitude_scale: float = 0.05,
) -> np.ndarray:
    """
    Synthesize burst-suppression EEG pattern.

    Burst-suppression is a pathological EEG pattern seen during deep anesthesia,
    severe anoxic brain injury, and certain metabolic encephalopathies.
    It alternates between:
    - **Bursts**: high-amplitude, polymorphic 0.5-3 s episodes
    - **Suppression**: near-flat EEG (< 10 µV) lasting 1-5 s

    Parameters
    ----------
    t : np.ndarray
        Time axis.
    channels : np.ndarray
        Background EEG channels of shape (n_ch, n_samples).
    rng : np.random.Generator
        Random number generator.
    fs : float
        Sampling frequency.
    burst_probability : float
        Fraction of time in burst (vs. suppression). Default: 0.35.
    burst_duration_s : float
        Mean burst duration in seconds. Default: 1.0 s.
    suppression_amplitude_scale : float
        Amplitude scaling during suppression (fraction of burst amplitude). Default: 0.05.

    Returns
    -------
    np.ndarray
        Modified EEG channels with burst-suppression pattern.
    """
    n_ch, n_samples = channels.shape
    duration = t[-1]

    # Build alternating burst/suppression timeline
    burst_mask = np.zeros(n_samples, dtype=bool)
    curr_t = 0.0

    while curr_t < duration:
        if rng.random() < burst_probability:
            # Burst period
            b_dur = rng.exponential(scale=burst_duration_s)
            b_end = min(curr_t + b_dur, duration)
            idx_start = int(curr_t * fs)
            idx_end = int(b_end * fs)
            burst_mask[idx_start:idx_end] = True
            curr_t = b_end
        else:
            # Suppression period
            s_dur = rng.exponential(scale=burst_duration_s * 1.5)
            curr_t += s_dur

    suppression_mask = ~burst_mask

    # Apply to channels
    for c in range(n_ch):
        channels[c][suppression_mask] *= suppression_amplitude_scale

    return channels


# ──────────────────────────────────────────────────────────────────────────────
# Main EEG Generator Class
# ──────────────────────────────────────────────────────────────────────────────

class EEGGenerator(BaseSignal):
    """
    High-fidelity multi-channel EEG simulator.

    Generates realistic EEG across brain states, sleep stages, seizure types,
    and event-related potentials. Supports multi-channel spatial correlation
    modeling via Cholesky decomposition.

    Parameters
    ----------
    config : EEGConfig
        EEG simulation configuration.

    Attributes
    ----------
    config : EEGConfig
        Configuration object.
    sleep_gen : SleepTransientGenerator
        Generator for sleep transients.
    seizure_gen : SeizurePatternGenerator
        Generator for seizure patterns.
    erp_gen : ERPGenerator
        Generator for event-related potentials.

    Examples
    --------
    >>> from biosignal_simulator.core.config import EEGConfig
    >>> config = EEGConfig(fs=256.0, duration_s=30.0, state='n2_sleep', n_channels=8)
    >>> gen = EEGGenerator(config)
    >>> eeg = gen.generate()
    >>> eeg.shape
    (8, 7680)

    >>> # Generate a single-channel awake-active EEG
    >>> config_awake = EEGConfig(fs=256.0, duration_s=10.0, state='active', n_channels=1)
    >>> gen_awake = EEGGenerator(config_awake)
    >>> eeg_awake = gen_awake.generate()
    >>> eeg_awake.shape
    (10,)
    """

    def __init__(self, config: EEGConfig) -> None:
        config.__post_init__()
        n_ch = config.n_channels
        super().__init__(
            fs=config.fs,
            duration_s=config.duration_s,
            seed=config.seed,
            multichannel=True if n_ch > 1 else False,
            n_channels=n_ch,
        )
        self.config = config
        self.sleep_gen = SleepTransientGenerator(rng=self.rng, fs=self.fs)
        self.seizure_gen = SeizurePatternGenerator(
            rng=self.rng, fs=self.fs, n_channels=n_ch
        )
        self.erp_gen = ERPGenerator(rng=self.rng, fs=self.fs)

    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate EEG configuration parameters.

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
        Generate multi-channel EEG signal.

        Returns
        -------
        np.ndarray
            - Shape (n_channels, n_samples) if n_channels > 1
            - Shape (n_samples,) if n_channels == 1

        Notes
        -----
        The generation pipeline:
        1. Determine state-specific band power weights and amplitude
        2. Compute spatial Cholesky mixing matrix
        3. Synthesize background band noise (all bands in parallel)
        4. Apply spatial mixing
        5. Inject state-specific transients (spindles, K-complexes, seizures)
        6. Zero-mean and return
        """
        n_ch = self.config.n_channels
        n_samples = self.n_samples
        t = self.t
        state = self.config.state.lower()

        # 1. Get state-specific power weights
        profile = self._get_state_profile(state)
        amplitude = profile.get("amplitude_uv", self.config.amplitude_uv)
        w_1f = profile.get("1f", 0.3)

        # 2. Spatial mixing matrix (Cholesky)
        L = compute_cholesky_mixing(
            corr_matrix=self.config.corr_matrix,
            n_channels=n_ch,
            base_corr=0.60,
        )

        # 3. Synthesize background rhythms
        background = self._synthesize_background(profile, w_1f, amplitude, L)

        # 4. Inject state-specific transients
        output = self._inject_state_transients(background.copy(), t, state)

        # 5. Zero-mean centering
        for c in range(n_ch):
            output[c] -= np.mean(output[c])

        # 6. Return
        if n_ch == 1:
            return output[0]
        return output

    def _get_state_profile(self, state: str) -> Dict[str, Any]:
        """
        Retrieve the brain state power profile.

        If state matches a known profile, return it. Otherwise fall back to
        the user-specified config band powers.

        Parameters
        ----------
        state : str
            Brain state name.

        Returns
        -------
        Dict[str, Any]
            Power profile dictionary.
        """
        if state in BRAIN_STATE_PROFILES:
            return dict(BRAIN_STATE_PROFILES[state])
        else:
            # Custom state: use user config
            profile = dict(self.config.band_powers)
            profile["1f"] = self.config.background_1f_power
            profile["amplitude_uv"] = self.config.amplitude_uv
            return profile

    def _synthesize_background(
        self,
        profile: Dict[str, Any],
        w_1f: float,
        amplitude: float,
        L: np.ndarray,
    ) -> np.ndarray:
        """
        Synthesize the spatially correlated multi-band background EEG.

        Each band is synthesized independently, normalized, Cholesky-mixed
        for spatial correlation, then weighted and summed.

        Parameters
        ----------
        profile : Dict[str, Any]
            Band power profile.
        w_1f : float
            Weight of 1/f (pink noise) background.
        amplitude : float
            Target RMS amplitude in µV.
        L : np.ndarray
            Cholesky mixing matrix.

        Returns
        -------
        np.ndarray
            Background EEG channels of shape (n_ch, n_samples).
        """
        n_ch = self.config.n_channels
        n_samples = self.n_samples
        fs = self.fs

        # Extract band weights (exclude non-band keys)
        non_band_keys = {"amplitude_uv", "1f"}
        band_weights: Dict[str, float] = {
            k: v for k, v in profile.items() if k not in non_band_keys
        }
        total_weight = sum(band_weights.values())
        if total_weight <= 0:
            total_weight = 1.0

        # Normalize to 1-w_1f
        normalized = {b: (1.0 - w_1f) * (w / total_weight) for b, w in band_weights.items()}
        normalized["1f"] = w_1f

        background = np.zeros((n_ch, n_samples))

        for band, weight in normalized.items():
            if weight <= 0.0:
                continue

            band_noise = synthesize_band(
                rng=self.rng,
                band=band,
                n_channels=n_ch,
                n_samples=n_samples,
                fs=fs,
                alpha_peak_hz=self.config.alpha_peak_hz,
                high_gamma_cutoff=min(120.0, fs * 0.45),
            )

            # Normalize each channel to unit RMS
            for c in range(n_ch):
                band_noise[c] = normalize_to_rms(band_noise[c], 1.0)

            # Apply spatial mixing
            mixed = L @ band_noise

            # Add weighted contribution
            background += mixed * np.sqrt(weight)

        # Scale to target amplitude
        for c in range(n_ch):
            background[c] = normalize_to_rms(background[c], amplitude)

        return background

    def _inject_state_transients(
        self,
        channels: np.ndarray,
        t: np.ndarray,
        state: str,
    ) -> np.ndarray:
        """
        Inject state-specific EEG transients onto the background signal.

        Parameters
        ----------
        channels : np.ndarray
            Background EEG of shape (n_ch, n_samples).
        t : np.ndarray
            Time axis.
        state : str
            Brain state name.

        Returns
        -------
        np.ndarray
            Modified EEG channels.
        """
        duration = self.duration_s

        if state == "n1_sleep":
            channels = self._inject_n1_transients(channels, t)
        elif state == "n2_sleep":
            channels = self._inject_n2_transients(channels, t)
        elif state == "n3_sleep":
            channels = self._inject_n3_transients(channels, t)
        elif state == "rem_sleep":
            channels = self._inject_rem_transients(channels, t)
        elif state == "tonic_clonic":
            channels = self.seizure_gen.generate_tonic_clonic(t, duration, channels)
        elif state == "absence":
            channels = self.seizure_gen.generate_absence_seizure(t, channels)
        elif state == "focal_seizure":
            onset_ch = self.rng.integers(0, max(1, self.config.n_channels))
            channels = self.seizure_gen.generate_focal_seizure(t, channels, onset_ch)
        elif state == "epileptiform_spikes":
            channels = self.seizure_gen.inject_interictal_spikes(
                t, channels, spike_rate_hz=self.rng.uniform(0.2, 0.4)
            )
        elif state == "burst_suppression":
            channels = generate_burst_suppression(t, channels, self.rng, self.fs)
        elif state == "anesthesia":
            channels = self._inject_anesthesia_features(channels, t)

        return channels

    def _inject_n1_transients(self, channels: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Inject N1 sleep transients: vertex waves and POSTS.

        N1 is characterized by:
        - Disappearance of alpha
        - Appearance of theta
        - Vertex sharp waves at the transition
        - Slow rolling eye movements (not simulated here)

        Parameters
        ----------
        channels : np.ndarray
            EEG channels.
        t : np.ndarray
            Time axis.

        Returns
        -------
        np.ndarray
            Modified channels.
        """
        n_ch = self.config.n_channels
        duration = self.duration_s

        # Vertex waves (appear every 30-120 s in real sleep; scale for duration)
        v_rate = 0.08  # Events per second
        n_v = max(1, self.rng.poisson(v_rate * duration))

        v_centers = self.rng.uniform(0.5, max(0.6, duration - 0.5), size=n_v)
        for t_c in v_centers:
            amp_v = self.rng.uniform(45.0, 80.0)
            v_wave = self.sleep_gen.generate_vertex_wave(t_c, amp_v, t)
            for c in range(n_ch):
                # Vertex waves strongest at midline channels
                midline_weight = 1.0 - 0.4 * abs(c - n_ch // 2) / max(1, n_ch // 2)
                channels[c] += v_wave * midline_weight * self.rng.uniform(0.85, 1.15)

        return channels

    def _inject_n2_transients(self, channels: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Inject N2 sleep transients: sleep spindles and K-complexes.

        N2 sleep features:
        - Spindles: 11-16 Hz wax-and-wane bursts, 0.5-2 s duration
        - K-complexes: high-amplitude biphasic waves (±80-200 µV)
        - Theta background with minimal alpha

        Parameters
        ----------
        channels : np.ndarray
            EEG channels.
        t : np.ndarray
            Time axis.

        Returns
        -------
        np.ndarray
            Modified channels.
        """
        n_ch = self.config.n_channels
        duration = self.duration_s

        # ── Sleep Spindles ────────────────────────────────────────────────────
        # Average spindle rate: 2-3 per minute → ~0.04 Hz
        spin_rate = 0.04  # Events/second
        n_spindles = self.rng.poisson(spin_rate * duration)
        n_spindles = max(1, n_spindles) if duration >= 2.0 else n_spindles

        spin_centers = self.rng.uniform(1.0, max(1.1, duration - 1.0), size=n_spindles)
        for t_c in spin_centers:
            f_spin = self.rng.uniform(11.5, 15.5)
            spin_dur = self.rng.uniform(0.5, 1.8)  # 0.5-1.8 s duration
            amp_spin = self.rng.uniform(25.0, 50.0)

            for c in range(n_ch):
                phase_shift = self.rng.uniform(-0.2, 0.2)
                spindle = self.sleep_gen.generate_sleep_spindle(
                    center_time=t_c,
                    frequency_hz=f_spin,
                    duration_s=spin_dur,
                    amplitude_uv=amp_spin,
                    phase_shift=phase_shift,
                    t=t,
                )
                # Spindles predominant in centroparietal channels
                frontal_weight = 1.0 + 0.3 * (c / max(1, n_ch - 1))
                channels[c] += spindle * frontal_weight

        # ── K-Complexes ───────────────────────────────────────────────────────
        k_rate = 0.018  # Events/second (about 1 per minute)
        n_k = self.rng.poisson(k_rate * duration)
        n_k = max(1, n_k) if duration >= 5.0 else n_k

        k_centers = self.rng.uniform(1.5, max(1.6, duration - 1.5), size=n_k)
        for t_c in k_centers:
            amp_k = self.rng.uniform(100.0, 170.0)
            k_wave = self.sleep_gen.generate_k_complex(t_c, amp_k, t)
            for c in range(n_ch):
                channels[c] += k_wave * self.rng.uniform(0.82, 1.18)

        # ── Vertex Waves ──────────────────────────────────────────────────────
        n_v = max(0, self.rng.poisson(0.03 * duration))
        if n_v > 0:
            v_centers = self.rng.uniform(0.5, max(0.6, duration - 0.5), size=n_v)
            for t_c in v_centers:
                amp_v = self.rng.uniform(40.0, 65.0)
                v_wave = self.sleep_gen.generate_vertex_wave(t_c, amp_v, t)
                for c in range(n_ch):
                    channels[c] += v_wave * self.rng.uniform(0.88, 1.12)

        return channels

    def _inject_n3_transients(self, channels: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Inject N3 deep sleep transients: high-amplitude vertex waves and slow wave bursts.

        N3 (slow-wave sleep) features:
        - High-amplitude delta oscillations (75-200 µV, 0.5-2 Hz)
        - Occasional vertex sharp waves
        - Minimal spindles (early N3 only)

        Parameters
        ----------
        channels : np.ndarray
            EEG channels.
        t : np.ndarray
            Time axis.

        Returns
        -------
        np.ndarray
            Modified channels.
        """
        n_ch = self.config.n_channels
        duration = self.duration_s

        # High-amplitude vertex-like transients in N3
        v_rate = 0.12
        n_v = max(1, self.rng.poisson(v_rate * duration)) if duration >= 2.0 else 0

        v_centers = self.rng.uniform(1.0, max(1.1, duration - 1.0), size=n_v)
        for t_c in v_centers:
            amp_v = self.rng.uniform(60.0, 110.0)
            v_wave = self.sleep_gen.generate_vertex_wave(t_c, amp_v, t)
            for c in range(n_ch):
                channels[c] += v_wave * self.rng.uniform(0.88, 1.12)

        # Occasional early-N3 spindles (sigma band remnants)
        if self.rng.random() < 0.3:
            n_spin = max(1, self.rng.poisson(0.01 * duration))
            if duration >= 3.0:
                spin_centers = self.rng.uniform(1.0, max(1.1, duration - 1.0), size=n_spin)
                for t_c in spin_centers:
                    f_spin = self.rng.uniform(12.0, 14.0)
                    amp_spin = self.rng.uniform(15.0, 30.0)  # Smaller in N3
                    for c in range(n_ch):
                        spindle = self.sleep_gen.generate_sleep_spindle(
                            center_time=t_c,
                            frequency_hz=f_spin,
                            duration_s=0.7,
                            amplitude_uv=amp_spin,
                            t=t,
                        )
                        channels[c] += spindle

        return channels

    def _inject_rem_transients(self, channels: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Inject REM sleep transients: sawtooth waves and rapid eye movement artifacts.

        REM sleep features:
        - Sawtooth waves: 2-6 Hz notched theta, frontal-central
        - Low-amplitude mixed-frequency background
        - Tonic EMG suppression (not modeled here)
        - Phasic bursts of rapid eye movements (EOG artifact injection)

        Parameters
        ----------
        channels : np.ndarray
            EEG channels.
        t : np.ndarray
            Time axis.

        Returns
        -------
        np.ndarray
            Modified channels with REM transients.
        """
        n_ch = self.config.n_channels
        duration = self.duration_s

        # Sawtooth waves (REM-specific theta bursts)
        saw_rate = 0.06  # Bursts per second
        n_saw = max(1, self.rng.poisson(saw_rate * duration)) if duration >= 2.0 else 0

        if n_saw > 0:
            saw_centers = self.rng.uniform(1.0, max(1.1, duration - 1.0), size=n_saw)
            for t_c in saw_centers:
                amp_saw = self.rng.uniform(15.0, 35.0)
                dur_saw = self.rng.uniform(0.8, 2.0)
                saw = self.sleep_gen.generate_saw_tooth_rem(t_c, amp_saw, dur_saw, t)
                for c in range(n_ch):
                    # Sawtooth predominant in frontal channels
                    frontal_w = 1.0 - 0.5 * (c / max(1, n_ch - 1))
                    channels[c] += saw * frontal_w * self.rng.uniform(0.8, 1.2)

        return channels

    def _inject_anesthesia_features(
        self, channels: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """
        Inject anesthesia-specific EEG features.

        Deep propofol anesthesia shows:
        - Spindle-like oscillations in the alpha/sigma range (8-14 Hz)
        - Slow (<1 Hz) oscillations with high amplitude
        - Burst-suppression at deeper levels

        Parameters
        ----------
        channels : np.ndarray
            EEG channels.
        t : np.ndarray
            Time axis.

        Returns
        -------
        np.ndarray
            Modified channels.
        """
        n_ch = self.config.n_channels
        duration = self.duration_s

        # Propofol-induced alpha oscillations (distinct from sleep spindles)
        alpha_rate = 0.05
        n_alpha_bursts = max(1, self.rng.poisson(alpha_rate * duration))

        alpha_centers = self.rng.uniform(1.0, max(1.1, duration - 1.0), size=n_alpha_bursts)
        for t_c in alpha_centers:
            f_alpha = self.rng.uniform(9.0, 12.0)  # Propofol alpha: 9-12 Hz
            dur_alpha = self.rng.uniform(1.0, 3.0)
            amp_alpha = self.rng.uniform(35.0, 75.0)
            for c in range(n_ch):
                spindle = self.sleep_gen.generate_sleep_spindle(
                    center_time=t_c,
                    frequency_hz=f_alpha,
                    duration_s=dur_alpha,
                    amplitude_uv=amp_alpha,
                    t=t,
                )
                channels[c] += spindle

        return channels

    def inject_erp(
        self,
        channels: np.ndarray,
        t: np.ndarray,
        erp_type: str = "p300",
        event_times: Optional[List[float]] = None,
        erp_channels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Inject event-related potentials onto generated EEG channels.

        Parameters
        ----------
        channels : np.ndarray
            EEG channels of shape (n_ch, n_samples).
        t : np.ndarray
            Time axis.
        erp_type : str
            ERP type: 'p300', 'n400', 'mmn', 'n200'.
        event_times : List[float], optional
            List of event onset times (seconds). If None, uniformly distributed.
        erp_channels : List[int], optional
            Channel indices to add ERP to. If None, all channels.

        Returns
        -------
        np.ndarray
            Modified EEG channels with ERPs.
        """
        n_ch = self.config.n_channels
        duration = self.duration_s

        if event_times is None:
            n_events = max(1, int(duration / 2.0))
            event_times = list(np.linspace(0.5, duration - 1.0, n_events))

        if erp_channels is None:
            erp_channels = list(range(n_ch))

        for t_ev in event_times:
            if erp_type == "p300":
                erp = self.erp_gen.generate_p300(
                    t_ev, t,
                    amplitude_uv=self.rng.uniform(5.0, 12.0),
                    latency_ms=self.rng.uniform(250.0, 380.0),
                )
            elif erp_type == "n400":
                erp = self.erp_gen.generate_n400(
                    t_ev, t,
                    amplitude_uv=self.rng.uniform(-8.0, -3.0),
                    latency_ms=self.rng.uniform(350.0, 500.0),
                )
            elif erp_type == "mmn":
                erp = self.erp_gen.generate_mismatch_negativity(t_ev, t)
            elif erp_type == "n200":
                erp = self.erp_gen.generate_n200(t_ev, t)
            else:
                warnings.warn(f"Unknown ERP type '{erp_type}'. Skipping.")
                continue

            for c in erp_channels:
                if c < n_ch:
                    channels[c] += erp * self.rng.uniform(0.8, 1.2)

        return channels

    def generate_with_erp(
        self,
        erp_type: str = "p300",
        event_times: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Generate EEG with embedded event-related potentials.

        Parameters
        ----------
        erp_type : str
            Type of ERP to inject: 'p300', 'n400', 'mmn', 'n200'.
        event_times : List[float], optional
            Event onset times. If None, uses uniform spacing.

        Returns
        -------
        Tuple[np.ndarray, List[float]]
            - EEG signal (n_ch, n_samples) or (n_samples,)
            - List of event onset times used
        """
        n_ch = self.config.n_channels
        n_samples = self.n_samples
        t = self.t
        state = self.config.state.lower()

        profile = self._get_state_profile(state)
        amplitude = profile.get("amplitude_uv", self.config.amplitude_uv)
        w_1f = profile.get("1f", 0.3)

        L = compute_cholesky_mixing(
            corr_matrix=self.config.corr_matrix,
            n_channels=n_ch,
        )

        background = self._synthesize_background(profile, w_1f, amplitude, L)
        output = self._inject_state_transients(background.copy(), t, state)

        # Set up event times
        if event_times is None:
            n_events = max(1, int(self.duration_s / 2.0))
            event_times = list(np.linspace(0.5, self.duration_s - 1.0, n_events))

        output = self.inject_erp(output, t, erp_type=erp_type, event_times=event_times)

        # Zero-mean
        for c in range(n_ch):
            output[c] -= np.mean(output[c])

        if n_ch == 1:
            return output[0], event_times
        return output, event_times

    def get_band_powers(self) -> Dict[str, float]:
        """
        Generate EEG and compute frequency band power estimates.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping band name to absolute power (µV²).
        """
        eeg = self.generate()
        band_data = compute_band_power_spectrum(eeg, self.fs)
        return {k: v for k, v in band_data.items() if k not in ("freqs", "psd")}

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of the EEG generator configuration.

        Returns
        -------
        Dict[str, Any]
            Configuration summary dictionary.
        """
        return {
            "state": self.config.state,
            "n_channels": self.config.n_channels,
            "fs_hz": self.fs,
            "duration_s": self.duration_s,
            "amplitude_uv": self.config.amplitude_uv,
            "alpha_peak_hz": self.config.alpha_peak_hz,
            "band_powers": dict(self.config.band_powers),
            "background_1f_power": self.config.background_1f_power,
            "n_samples": self.n_samples,
            "seed": self.config.seed,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Factory Functions
# ──────────────────────────────────────────────────────────────────────────────

def make_eeg_awake_active(
    duration_s: float = 30.0,
    n_channels: int = 8,
    fs: float = 256.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an awake, active EEG (Beta/Gamma dominant).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    n_channels : int
        Number of EEG channels.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        EEG signal.
    """
    config = EEGConfig(
        fs=fs, duration_s=duration_s, state="active",
        n_channels=n_channels, seed=seed,
    )
    return EEGGenerator(config).generate()


def make_eeg_relaxed(
    duration_s: float = 30.0,
    n_channels: int = 8,
    fs: float = 256.0,
    alpha_peak_hz: float = 10.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a relaxed, eyes-closed EEG (Alpha dominant).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    n_channels : int
        Number of EEG channels.
    fs : float
        Sampling frequency in Hz.
    alpha_peak_hz : float
        Individual alpha peak frequency. Default: 10.0 Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        EEG signal.
    """
    config = EEGConfig(
        fs=fs, duration_s=duration_s, state="relaxed",
        n_channels=n_channels, alpha_peak_hz=alpha_peak_hz, seed=seed,
    )
    return EEGGenerator(config).generate()


def make_eeg_n2_sleep(
    duration_s: float = 60.0,
    n_channels: int = 8,
    fs: float = 256.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate N2 sleep EEG with spindles and K-complexes.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    n_channels : int
        Number of channels.
    fs : float
        Sampling frequency.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        N2 sleep EEG signal.
    """
    config = EEGConfig(
        fs=fs, duration_s=duration_s, state="n2_sleep",
        n_channels=n_channels, seed=seed,
    )
    return EEGGenerator(config).generate()


def make_eeg_n3_sleep(
    duration_s: float = 60.0,
    n_channels: int = 8,
    fs: float = 256.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate N3 deep sleep EEG (Delta dominant).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    n_channels : int
        Number of channels.
    fs : float
        Sampling frequency.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        N3 sleep EEG signal.
    """
    config = EEGConfig(
        fs=fs, duration_s=duration_s, state="n3_sleep",
        n_channels=n_channels, seed=seed,
    )
    return EEGGenerator(config).generate()


def make_eeg_tonic_clonic(
    duration_s: float = 30.0,
    n_channels: int = 8,
    fs: float = 256.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Tonic-Clonic seizure EEG.

    Parameters
    ----------
    duration_s : float
        Total duration (seizure spans full duration).
    n_channels : int
        Number of channels.
    fs : float
        Sampling frequency.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Tonic-clonic seizure EEG signal.
    """
    config = EEGConfig(
        fs=fs, duration_s=duration_s, state="tonic_clonic",
        n_channels=n_channels, seed=seed,
    )
    return EEGGenerator(config).generate()


def make_eeg_absence(
    duration_s: float = 15.0,
    n_channels: int = 8,
    fs: float = 256.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an Absence seizure EEG (3 Hz spike-wave).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    n_channels : int
        Number of channels.
    fs : float
        Sampling frequency.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Absence seizure EEG signal.
    """
    config = EEGConfig(
        fs=fs, duration_s=duration_s, state="absence",
        n_channels=n_channels, seed=seed,
    )
    return EEGGenerator(config).generate()


def make_eeg_rem_sleep(
    duration_s: float = 60.0,
    n_channels: int = 8,
    fs: float = 256.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate REM sleep EEG with sawtooth waves.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    n_channels : int
        Number of channels.
    fs : float
        Sampling frequency.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        REM sleep EEG signal.
    """
    config = EEGConfig(
        fs=fs, duration_s=duration_s, state="rem_sleep",
        n_channels=n_channels, seed=seed,
    )
    return EEGGenerator(config).generate()


def batch_generate_eeg_states(
    states: List[str],
    duration_s: float = 30.0,
    n_channels: int = 8,
    fs: float = 256.0,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate EEG signals for multiple brain states.

    Parameters
    ----------
    states : List[str]
        List of brain state names.
    duration_s : float
        Duration for each state.
    n_channels : int
        Number of channels.
    fs : float
        Sampling frequency.
    seed : int, optional
        Base random seed (offset per state for independence).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping state name to EEG signal array.

    Examples
    --------
    >>> signals = batch_generate_eeg_states(['active', 'n2_sleep', 'absence'])
    >>> list(signals.keys())
    ['active', 'n2_sleep', 'absence']
    """
    results: Dict[str, np.ndarray] = {}
    for i, state in enumerate(states):
        rng_seed = (seed + i) if seed is not None else None
        config = EEGConfig(
            fs=fs,
            duration_s=duration_s,
            state=state,
            n_channels=n_channels,
            seed=rng_seed,
        )
        try:
            results[state] = EEGGenerator(config).generate()
        except Exception as exc:
            warnings.warn(f"Failed to generate EEG state '{state}': {exc}")
    return results


def compute_sleep_staging_features(
    eeg: np.ndarray,
    fs: float,
    epoch_s: float = 30.0,
) -> List[Dict[str, float]]:
    """
    Extract standard sleep staging features from EEG in 30-second epochs.

    Computes band power ratios commonly used by automated sleep staging
    algorithms (e.g., for AASM epoch-based staging).

    Parameters
    ----------
    eeg : np.ndarray
        EEG signal: (n_channels, n_samples) or (n_samples,).
    fs : float
        Sampling frequency.
    epoch_s : float
        Epoch length in seconds (AASM standard: 30 s).

    Returns
    -------
    List[Dict[str, float]]
        List of feature dictionaries, one per epoch. Each contains:
        - 'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'
        - 'delta_ratio', 'theta_ratio', 'alpha_ratio' (relative to total)
        - 'alpha_delta_ratio', 'theta_alpha_ratio'
        - 'spectral_edge_95_hz': frequency below which 95% of power lies
    """
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]

    n_samples = eeg.shape[1]
    epoch_samples = int(epoch_s * fs)
    n_epochs = n_samples // epoch_samples

    features = []
    for ep in range(n_epochs):
        start = ep * epoch_samples
        end = start + epoch_samples
        epoch_data = eeg[:, start:end]

        band_data = compute_band_power_spectrum(epoch_data, fs, window_s=min(4.0, epoch_s))

        total = sum(band_data.get(b, 0.0) for b in BAND_RANGES.keys())
        if total <= 0:
            total = 1.0

        delta = band_data.get("delta", 0.0)
        theta = band_data.get("theta", 0.0)
        alpha = band_data.get("alpha", 0.0)
        beta = band_data.get("beta", 0.0)
        gamma = band_data.get("gamma", 0.0)

        # Spectral edge frequency (95th percentile)
        freqs = band_data.get("freqs", np.array([0.0]))
        psd = band_data.get("psd", np.array([1.0]))
        cum_power = np.cumsum(psd)
        if cum_power[-1] > 0:
            sef95_idx = np.searchsorted(cum_power, 0.95 * cum_power[-1])
            sef95 = float(freqs[min(sef95_idx, len(freqs) - 1)])
        else:
            sef95 = 0.0

        features.append({
            "epoch": ep,
            "delta_power": delta,
            "theta_power": theta,
            "alpha_power": alpha,
            "beta_power": beta,
            "gamma_power": gamma,
            "delta_ratio": delta / total,
            "theta_ratio": theta / total,
            "alpha_ratio": alpha / total,
            "beta_ratio": beta / total,
            "alpha_delta_ratio": alpha / max(delta, 1e-12),
            "theta_alpha_ratio": theta / max(alpha, 1e-12),
            "spectral_edge_95_hz": sef95,
        })

    return features
