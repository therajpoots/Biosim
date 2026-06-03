"""
High-fidelity Electrodermal Activity (EDA) / Galvanic Skin Response (GSR) simulator.

This module provides a physiologically accurate simulation of the skin conductance
signal, comprising two components:

1. **Tonic Component (Skin Conductance Level, SCL)**: The slowly varying baseline of
   skin conductance, reflecting the overall level of sympathetic arousal. It changes
   over minutes in response to sustained stress, temperature, and autonomic states.

2. **Phasic Component (Skin Conductance Response, SCR)**: Rapid, event-related bursts
   superimposed on the tonic level. Each SCR follows a characteristic rise (0.5-5 s)
   and slow exponential decay (5-15 s), driven by sweat gland activity under sympathetic
   nervous system control.

Physiological Basis
-------------------
EDA is controlled exclusively by the sympathetic branch of the autonomic nervous system.
Eccrine sweat glands in the palms, fingers, and soles of the feet are densely innervated
and respond to emotional arousal, cognitive load, and physical activity.

Signal Components
-----------------
- **SCL**: Baseline conductance, typically 2-20 µS at rest, up to 50 µS under stress.
- **SCR**: Phasic responses of 0.1-1.0 µS amplitude (specific SCR) or 1-3 µS (non-specific).
- **SCR Rise Time**: 1-3 seconds (onset to peak)
- **SCR Decay Time**: 5-20 seconds (half-decay)

Pathological and Physiological States
--------------------------------------
- **High arousal** (acute stress, anxiety): high SCR amplitude, fast rate
- **Low arousal** (relaxation, sleep): minimal SCRs, low, stable SCL
- **Autonomic dysfunction**: reduced or absent SCRs (e.g., diabetic neuropathy)
- **Hyperhidrosis**: continuously elevated SCL with frequent SCRs
- **Dehydration**: reduced overall conductance

Signal Variants
---------------
1. **Conductance** (standard EDA, µS): direct skin conductance measurement
2. **Resistance** (skin resistance, kΩ): inverse of conductance, older measurement
3. **Admittance** (AC impedance analysis)

References
----------
- Boucsein, W. (2012). Electrodermal Activity, 2nd Ed. Springer.
- Benedek, M. & Kaernbach, C. (2010). A continuous measure of phasic
  electrodermal activity. J. Neurosci. Methods.
- Lim, C.L. et al. (1997). A decomposition method for dermal and epidermal
  EDA. Psychophysiology.
- Electrodermal Activity Processing Toolkit (neurokit2): MIT License.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

from biosignal_simulator.core.base import BaseSignal
from biosignal_simulator.core.config import EDAConfig
from biosignal_simulator.core.math_utils import normalize_to_rms
from biosignal_simulator.utils.validation import validate_config


# ──────────────────────────────────────────────────────────────────────────────
# SCR Impulse Response Functions
# ──────────────────────────────────────────────────────────────────────────────

def scr_biexponential(
    t_relative: np.ndarray,
    amplitude: float,
    tau_rise: float,
    tau_decay: float,
) -> np.ndarray:
    """
    Compute the bi-exponential SCR (Skin Conductance Response) impulse response.

    This is the standard model for a single phasic EDA response to a discrete
    event (stimulus). The SCR rises with time constant tau_rise and decays
    with time constant tau_decay.

    Mathematical form:
        SCR(t) = A * (1 - exp(-t/τ_rise)) * exp(-t/τ_decay)  for t ≥ 0

    The amplitude is normalized so that the peak value equals `amplitude`.

    Parameters
    ----------
    t_relative : np.ndarray
        Time relative to event onset (seconds). Negative values → 0.
    amplitude : float
        Peak amplitude of the SCR in microsiemens (µS).
    tau_rise : float
        Rise time constant in seconds. Typical: 0.5-2.0 s.
    tau_decay : float
        Decay time constant in seconds. Typical: 5-15 s.

    Returns
    -------
    np.ndarray
        SCR waveform values (same shape as t_relative).

    Notes
    -----
    Peak occurs at:
        t_peak = τ_rise * τ_decay / (τ_decay - τ_rise) * ln(τ_decay / τ_rise)
    """
    result = np.zeros_like(t_relative, dtype=float)
    pos_mask = t_relative >= 0.0

    if not np.any(pos_mask):
        return result

    t_pos = t_relative[pos_mask]
    raw_scr = (1.0 - np.exp(-t_pos / max(tau_rise, 1e-9))) * np.exp(-t_pos / max(tau_decay, 1e-9))

    # Compute peak value for normalization
    if tau_decay > tau_rise and tau_rise > 0:
        t_peak = (tau_rise * tau_decay / (tau_decay - tau_rise)) * np.log(tau_decay / tau_rise)
        peak_val = (1.0 - np.exp(-t_peak / tau_rise)) * np.exp(-t_peak / tau_decay)
    else:
        # Fallback for edge cases
        peak_val = float(np.max(raw_scr)) if len(raw_scr) > 0 else 0.25

    if peak_val > 0:
        result[pos_mask] = (amplitude / peak_val) * raw_scr
    else:
        result[pos_mask] = raw_scr

    return result


def scr_gaussian_derivative(
    t_relative: np.ndarray,
    amplitude: float,
    peak_time: float,
    sigma: float,
) -> np.ndarray:
    """
    Compute an SCR using a Gaussian-derivative model.

    An alternative to the bi-exponential model, this uses a Gaussian-smoothed
    step function. Useful for faster computation when many SCRs are needed.

    Parameters
    ----------
    t_relative : np.ndarray
        Time relative to event onset.
    amplitude : float
        Peak SCR amplitude.
    peak_time : float
        Time of SCR peak (seconds after event).
    sigma : float
        Width of the SCR in seconds.

    Returns
    -------
    np.ndarray
        SCR waveform.
    """
    result = np.zeros_like(t_relative, dtype=float)
    pos_mask = t_relative >= 0.0
    if not np.any(pos_mask):
        return result

    t_pos = t_relative[pos_mask]
    scr = amplitude * np.exp(-0.5 * ((t_pos - peak_time) / max(sigma, 1e-9)) ** 2)
    result[pos_mask] = scr
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Tonic Component Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_tonic_scl(
    t: np.ndarray,
    fs: float,
    scl_amplitude: float,
    drift_rate: float,
    rng: np.random.Generator,
    drift_type: str = "random_walk",
) -> np.ndarray:
    """
    Generate the Skin Conductance Level (SCL) tonic component.

    The SCL represents the slowly varying background conductance level.
    It is modeled as a combination of:
    - A random walk process (low-frequency noise)
    - A linear drift (reflecting sustained autonomic changes)
    - A low-frequency sinusoidal oscillation (ultradian rhythms)

    Parameters
    ----------
    t : np.ndarray
        Time axis in seconds.
    fs : float
        Sampling frequency.
    scl_amplitude : float
        Target baseline SCL in microsiemens (µS).
    drift_rate : float
        Linear drift rate in µS/second.
    rng : np.random.Generator
        Random number generator.
    drift_type : str
        Type of SCL drift: 'random_walk', 'linear', 'sinusoidal', 'constant'.

    Returns
    -------
    np.ndarray
        SCL tonic component.
    """
    n_samples = len(t)
    duration = t[-1] if len(t) > 0 else 1.0

    # ── Base SCL ──────────────────────────────────────────────────────────────
    raw_noise = rng.normal(0.0, 1.0, size=n_samples)

    # Lowpass filter to create smooth random walk
    nyq = 0.5 * fs
    fc = 0.05  # 0.05 Hz — very slow variations
    fc_norm = min(fc / nyq, 0.9)
    b, a = sp_signal.butter(2, fc_norm, btype="lowpass")
    filtered_noise = sp_signal.filtfilt(b, a, raw_noise)

    if drift_type == "random_walk":
        # Cumulative sum to create a random walk
        random_walk = np.cumsum(filtered_noise)
        rw_std = np.std(random_walk)
        if rw_std > 1e-12:
            random_walk = (random_walk - np.mean(random_walk)) / rw_std * (0.15 * scl_amplitude)
        else:
            random_walk = np.zeros(n_samples)

        drift_component = drift_rate * t
        scl = scl_amplitude + random_walk + drift_component

    elif drift_type == "linear":
        scl = scl_amplitude + drift_rate * t

    elif drift_type == "sinusoidal":
        # Slow sinusoidal oscillation (e.g., ultradian rhythm)
        f_ult = 1.0 / max(duration * 0.5, 1.0)  # Half-period spans recording
        scl = scl_amplitude + 0.2 * scl_amplitude * np.sin(2.0 * np.pi * f_ult * t)
        scl += drift_rate * t

    elif drift_type == "constant":
        scl = np.full(n_samples, scl_amplitude)

    else:
        scl = scl_amplitude + drift_rate * t

    # Ensure physiological positivity (SCL must be > 0)
    scl = np.clip(scl, 0.05, None)

    return scl


# ──────────────────────────────────────────────────────────────────────────────
# Phasic Component Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_phasic_scr(
    t: np.ndarray,
    event_times: np.ndarray,
    amplitudes: np.ndarray,
    tau_rise: float,
    tau_decay: float,
    rng: np.random.Generator,
    use_biexponential: bool = True,
) -> np.ndarray:
    """
    Generate the Skin Conductance Response (SCR) phasic component.

    Superimposes individual SCR impulse responses at each event time.

    Parameters
    ----------
    t : np.ndarray
        Time axis.
    event_times : np.ndarray
        Array of SCR onset times (seconds).
    amplitudes : np.ndarray
        Array of SCR amplitudes (µS).
    tau_rise : float
        SCR rise time constant (seconds).
    tau_decay : float
        SCR decay time constant (seconds).
    rng : np.random.Generator
        Random generator (for amplitude jitter).
    use_biexponential : bool
        If True, use bi-exponential model. Otherwise use Gaussian derivative.

    Returns
    -------
    np.ndarray
        SCR phasic component.
    """
    scr = np.zeros_like(t)

    for t_ev, amp in zip(event_times, amplitudes):
        t_rel = t - t_ev

        if use_biexponential:
            # Slight per-event jitter on time constants
            tr = tau_rise * rng.uniform(0.8, 1.2)
            td = tau_decay * rng.uniform(0.8, 1.2)
            scr += scr_biexponential(t_rel, amp, tr, td)
        else:
            peak_t = tau_rise * 2.0
            sigma = tau_rise
            scr += scr_gaussian_derivative(t_rel, amp, peak_t, sigma)

    return scr


# ──────────────────────────────────────────────────────────────────────────────
# EDA Feature Analysis
# ──────────────────────────────────────────────────────────────────────────────

def decompose_eda(
    eda: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose EDA signal into tonic (SCL) and phasic (SCR) components.

    Uses a simple lowpass filter approach:
    - Tonic: lowpass filtered EDA (cutoff < 0.05 Hz)
    - Phasic: residual (EDA - tonic)

    Parameters
    ----------
    eda : np.ndarray
        Raw EDA signal.
    fs : float
        Sampling frequency.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (tonic_scl, phasic_scr) components.
    """
    nyq = 0.5 * fs
    fc = 0.05  # Hz
    fc_norm = min(fc / nyq, 0.9)
    b, a = sp_signal.butter(2, fc_norm, btype="lowpass")
    tonic = sp_signal.filtfilt(b, a, eda)
    phasic = eda - tonic
    return tonic, phasic


def detect_scr_events(
    phasic_scr: np.ndarray,
    fs: float,
    min_amplitude: float = 0.02,
    min_rise_time_s: float = 0.3,
    min_distance_s: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Detect individual SCR events in the phasic EDA component.

    Parameters
    ----------
    phasic_scr : np.ndarray
        Phasic EDA component.
    fs : float
        Sampling frequency.
    min_amplitude : float
        Minimum SCR amplitude (µS). Default: 0.02 µS.
    min_rise_time_s : float
        Minimum rise time for an SCR (seconds). Default: 0.3 s.
    min_distance_s : float
        Minimum time between SCRs (seconds). Default: 1.0 s.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with:
        - 'onset_indices': SCR onset indices
        - 'peak_indices': SCR peak indices
        - 'amplitudes': SCR amplitudes (µS)
        - 'rise_times_s': Rise times (seconds)
    """
    min_dist = int(min_distance_s * fs)

    # Find positive peaks
    peaks, props = sp_signal.find_peaks(
        phasic_scr,
        distance=min_dist,
        prominence=min_amplitude,
        height=min_amplitude,
    )

    if len(peaks) == 0:
        return {
            "onset_indices": np.array([]),
            "peak_indices": np.array([]),
            "amplitudes": np.array([]),
            "rise_times_s": np.array([]),
        }

    amplitudes = phasic_scr[peaks]
    onset_indices = np.zeros(len(peaks), dtype=int)
    rise_times = np.zeros(len(peaks))

    min_rise_samples = int(min_rise_time_s * fs)

    for i, peak_idx in enumerate(peaks):
        # Find onset: search backwards for minimum before this peak
        search_start = max(0, peak_idx - int(5.0 * fs))
        segment = phasic_scr[search_start:peak_idx]
        if len(segment) > 0:
            local_min_idx = np.argmin(segment) + search_start
            onset_indices[i] = local_min_idx
            rise_times[i] = (peak_idx - local_min_idx) / fs
        else:
            onset_indices[i] = max(0, peak_idx - min_rise_samples)
            rise_times[i] = min_rise_time_s

    return {
        "onset_indices": onset_indices,
        "peak_indices": peaks,
        "amplitudes": amplitudes,
        "rise_times_s": rise_times,
    }


def compute_eda_features(eda: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Compute standard EDA signal features.

    Parameters
    ----------
    eda : np.ndarray
        EDA signal in µS.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict[str, float]
        Feature dictionary containing:
        - 'mean_scl_us': Mean skin conductance level (µS)
        - 'std_scl_us': Standard deviation of SCL
        - 'n_scr_events': Number of detected SCR events
        - 'mean_scr_amplitude_us': Mean SCR amplitude (µS)
        - 'scr_rate_per_min': SCR event rate per minute
        - 'total_power': Total signal power
        - 'phasic_power_fraction': Fraction of power in phasic component
    """
    tonic, phasic = decompose_eda(eda, fs)
    scr_events = detect_scr_events(phasic, fs)

    n_scr = len(scr_events["peak_indices"])
    duration_min = len(eda) / fs / 60.0

    mean_scr_amp = float(np.mean(scr_events["amplitudes"])) if n_scr > 0 else 0.0

    total_power = float(np.mean(eda ** 2))
    phasic_power = float(np.mean(phasic ** 2))
    phasic_fraction = phasic_power / max(total_power, 1e-12)

    return {
        "mean_scl_us": float(np.mean(tonic)),
        "std_scl_us": float(np.std(tonic)),
        "n_scr_events": n_scr,
        "mean_scr_amplitude_us": mean_scr_amp,
        "scr_rate_per_min": n_scr / max(duration_min, 1e-6),
        "total_power": total_power,
        "phasic_power_fraction": phasic_fraction,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main EDA Generator
# ──────────────────────────────────────────────────────────────────────────────

class EDAGenerator(BaseSignal):
    """
    High-fidelity Electrodermal Activity (EDA) / Galvanic Skin Response (GSR) generator.

    Generates physiologically realistic EDA signals comprising:
    - Slow-varying tonic component (Skin Conductance Level, SCL)
    - Event-related phasic components (Skin Conductance Responses, SCR)
    - Thermal regulation drift
    - Individual response variability

    Parameters
    ----------
    config : EDAConfig
        EDA simulation configuration.

    Examples
    --------
    >>> from biosignal_simulator.core.config import EDAConfig
    >>> config = EDAConfig(fs=32.0, duration_s=120.0, scl_amplitude_us=5.0,
    ...                     event_rate_hz=0.05)
    >>> gen = EDAGenerator(config)
    >>> eda = gen.generate()
    >>> eda.shape
    (3840,)
    """

    def __init__(self, config: EDAConfig) -> None:
        validate_config(config)
        super().__init__(fs=config.fs, duration_s=config.duration_s, seed=config.seed)
        self.config = config

    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate EDA configuration.

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
        Generate the full EDA signal (SCL + SCR).

        Returns
        -------
        np.ndarray
            EDA signal in microsiemens (µS), shape (n_samples,).

        Notes
        -----
        Pipeline:
        1. Generate tonic SCL component (random walk + drift)
        2. Generate Poisson-distributed SCR event times
        3. Assign log-normal amplitudes to each SCR
        4. Superimpose bi-exponential SCR impulse responses
        5. Return combined SCL + SCR
        """
        t = self.t
        fs = self.fs
        n_samples = self.n_samples
        config = self.config

        # ── 1. Tonic Component (SCL) ─────────────────────────────────────────
        drift_type = getattr(config, "drift_type", "random_walk")
        scl = generate_tonic_scl(
            t=t,
            fs=fs,
            scl_amplitude=config.scl_amplitude_us,
            drift_rate=config.scl_drift_rate,
            rng=self.rng,
            drift_type=drift_type,
        )

        # ── 2. Phasic Component (SCR) ────────────────────────────────────────
        scr = np.zeros(n_samples)

        if config.event_rate_hz > 0.0:
            # Poisson process for event timing
            lam = config.event_rate_hz
            est_events = max(5, int(lam * self.duration_s * 3))
            intervals = self.rng.exponential(scale=1.0 / lam, size=est_events)
            event_times = np.cumsum(intervals)
            # Keep only events within signal duration (with 5s tail)
            event_times = event_times[event_times < self.duration_s]

            if len(event_times) > 0:
                # Log-normal amplitude distribution (realistic SCR variability)
                mean_amp = 0.5  # µS mean amplitude
                sigma_amp = 0.6  # Log-space standard deviation
                amplitudes = self.rng.lognormal(
                    mean=np.log(mean_amp), sigma=sigma_amp, size=len(event_times)
                )

                tau_rise = config.scr_rise_s
                tau_decay = config.scr_decay_s

                scr = generate_phasic_scr(
                    t=t,
                    event_times=event_times,
                    amplitudes=amplitudes,
                    tau_rise=tau_rise,
                    tau_decay=tau_decay,
                    rng=self.rng,
                    use_biexponential=True,
                )

        # ── 3. Combined EDA ───────────────────────────────────────────────────
        eda = scl + scr

        # Ensure physiological positivity
        eda = np.clip(eda, 0.01, None)

        return eda

    def generate_high_arousal(
        self, event_rate_multiplier: float = 3.0
    ) -> np.ndarray:
        """
        Generate EDA for a high-arousal state (stress, anxiety).

        High arousal features:
        - Elevated SCL (high baseline conductance)
        - Frequent, large-amplitude SCRs
        - Faster decay

        Parameters
        ----------
        event_rate_multiplier : float
            Multiplier for SCR rate relative to config baseline.

        Returns
        -------
        np.ndarray
            High-arousal EDA signal.
        """
        import copy
        high_config = copy.copy(self.config)
        high_config.scl_amplitude_us = self.config.scl_amplitude_us * 2.0
        high_config.event_rate_hz = self.config.event_rate_hz * event_rate_multiplier
        high_config.scr_rise_s = self.config.scr_rise_s * 0.7  # Faster rise
        high_config.scr_decay_s = self.config.scr_decay_s * 0.8  # Faster decay

        gen = EDAGenerator(high_config)
        gen.rng = self.rng  # Use same RNG
        return gen.generate()

    def generate_low_arousal(self) -> np.ndarray:
        """
        Generate EDA for a low-arousal state (relaxation, meditation, sleep).

        Low arousal features:
        - Low SCL
        - Very few SCRs

        Returns
        -------
        np.ndarray
            Low-arousal EDA signal.
        """
        import copy
        low_config = copy.copy(self.config)
        low_config.scl_amplitude_us = self.config.scl_amplitude_us * 0.4
        low_config.event_rate_hz = self.config.event_rate_hz * 0.15

        gen = EDAGenerator(low_config)
        gen.rng = self.rng
        return gen.generate()

    def decompose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate EDA and decompose into tonic and phasic components.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (tonic_scl, phasic_scr) components.
        """
        eda = self.generate()
        return decompose_eda(eda, self.fs)

    def compute_features(self) -> Dict[str, float]:
        """
        Generate EDA and compute feature metrics.

        Returns
        -------
        Dict[str, float]
            EDA feature dictionary.
        """
        return compute_eda_features(self.generate(), self.fs)

    def summary(self) -> Dict[str, Any]:
        """
        Return configuration summary.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary.
        """
        return {
            "scl_amplitude_us": self.config.scl_amplitude_us,
            "scl_drift_rate": self.config.scl_drift_rate,
            "event_rate_hz": self.config.event_rate_hz,
            "scr_rise_s": self.config.scr_rise_s,
            "scr_decay_s": self.config.scr_decay_s,
            "fs_hz": self.fs,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Factory Functions
# ──────────────────────────────────────────────────────────────────────────────

def make_eda_resting(
    duration_s: float = 120.0,
    scl_amplitude_us: float = 5.0,
    event_rate_hz: float = 0.03,
    fs: float = 32.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a resting-state EDA signal.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    scl_amplitude_us : float
        Baseline SCL in µS.
    event_rate_hz : float
        Non-specific SCR rate (events/second).
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Resting EDA signal.
    """
    config = EDAConfig(
        fs=fs, duration_s=duration_s,
        scl_amplitude_us=scl_amplitude_us,
        event_rate_hz=event_rate_hz,
        seed=seed,
    )
    return EDAGenerator(config).generate()


def make_eda_stress(
    duration_s: float = 60.0,
    scl_amplitude_us: float = 12.0,
    event_rate_hz: float = 0.15,
    fs: float = 32.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a stress-response EDA signal.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    scl_amplitude_us : float
        Elevated SCL baseline.
    event_rate_hz : float
        High SCR rate.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Stress-response EDA signal.
    """
    config = EDAConfig(
        fs=fs, duration_s=duration_s,
        scl_amplitude_us=scl_amplitude_us,
        event_rate_hz=event_rate_hz,
        scr_rise_s=0.8,
        scr_decay_s=6.0,
        seed=seed,
    )
    return EDAGenerator(config).generate()


def make_eda_sleep(
    duration_s: float = 300.0,
    fs: float = 32.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a sleep-state EDA signal (low arousal, minimal SCRs).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Sleep EDA signal.
    """
    config = EDAConfig(
        fs=fs, duration_s=duration_s,
        scl_amplitude_us=2.0,
        event_rate_hz=0.005,
        scr_rise_s=2.5,
        scr_decay_s=20.0,
        seed=seed,
    )
    return EDAGenerator(config).generate()
