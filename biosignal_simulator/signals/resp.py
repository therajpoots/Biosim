"""
High-fidelity Respiratory (Breathing) signal simulator.

This module provides a comprehensive, physiologically accurate simulation of
respiratory airflow, thoracic movement, or chest impedance signals as recorded
by spirometers, chest belts (RIP bands), or bioimpedance systems.

Physiological Basis
-------------------
Normal breathing (eupnea) is controlled by the brainstem respiratory centers
(pre-Bötzinger complex). It consists of alternating inspiration and expiration
phases, with the inspiration:expiration (I:E) ratio typically 1:2 to 1:3.

Respiratory Waveform Components
---------------------------------
The respiratory waveform is characterized by:

1. **Inspiration**: Diaphragm contracts → lung volume increases → airflow in
   - Rise time: 1-2 seconds
   - Typically faster than expiration

2. **Expiration**: Passive recoil → lung volume decreases → airflow out
   - Slower than inspiration (passive elastic recoil)
   - I:E ratio: normally ~1:2 to 1:3

3. **Pause**: Brief end-expiratory pause before next inspiration
   - Duration: 0.2-1 s in eupnea

Asymmetric Waveform Modeling
------------------------------
The non-sinusoidal nature of breathing is captured using:
- Fourier harmonic synthesis with phase-shifted second harmonic
- Sigmoid-based inspiration/expiration templates
- Physiological flow rate curves (peak flow → deceleration)

Respiratory Patterns Supported
--------------------------------
- **Eupnea**: Normal resting breathing (12-20 breaths/min)
- **Tachypnea**: Rapid shallow breathing (> 20 breaths/min)
- **Bradypnea**: Slow breathing (< 12 breaths/min)
- **Hyperventilation**: Rapid deep breathing (anxiety, metabolic acidosis)
- **Hypoventilation**: Slow shallow breathing
- **Cheyne-Stokes**: Crescendo-decrescendo pattern with apneic pauses (CHF, CNS)
- **Biot's Breathing**: Irregular clusters of breaths followed by apneas
- **Kussmaul**: Deep, labored, rapid breathing (diabetic ketoacidosis)
- **Apnea**: Complete cessation of breathing (central or obstructive)
- **Sighing**: Periodic deep breaths (1-3× normal tidal volume)

References
----------
- Tobin, M.J. (2006). Principles and Practice of Mechanical Ventilation. McGraw-Hill.
- Douglas, C.G. & Haldane, J.S. (1909). The regulation of normal breathing. J. Physiol.
- Ranft, J. et al. (2021). Respiratory Monitoring in Clinical Applications. Respir. Physiol. Neurobiol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

from biosignal_simulator.core.base import BaseSignal
from biosignal_simulator.core.config import RespConfig
from biosignal_simulator.core.math_utils import normalize_to_rms
from biosignal_simulator.utils.validation import validate_config


# ──────────────────────────────────────────────────────────────────────────────
# Respiratory Pattern Definitions
# ──────────────────────────────────────────────────────────────────────────────

class BreathingPattern(str):
    """Supported respiratory pattern names."""
    EUPNEA = "eupnea"
    NORMAL = "normal"
    TACHYPNEA = "tachypnea"
    BRADYPNEA = "bradypnea"
    HYPERVENTILATION = "hyperventilation"
    HYPOVENTILATION = "hypoventilation"
    CHEYNE_STOKES = "cheyne_stokes"
    BIOT = "biot"
    KUSSMAUL = "kussmaul"
    APNEA = "apnea"


@dataclass
class RespBreath:
    """
    Parameters for a single breath event.

    Attributes
    ----------
    onset_s : float
        Breath onset time in seconds.
    cycle_duration_s : float
        Total breath cycle duration (seconds).
    ie_ratio : float
        Inspiration:Expiration ratio. Default 1:2 → ie_ratio=0.33.
    amplitude : float
        Tidal volume scale factor (1.0 = normal).
    has_pause : bool
        Whether to include an end-expiratory pause.
    pause_duration_s : float
        Duration of end-expiratory pause (seconds).
    """
    onset_s: float = 0.0
    cycle_duration_s: float = 4.0
    ie_ratio: float = 0.33
    amplitude: float = 1.0
    has_pause: bool = True
    pause_duration_s: float = 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Single Breath Waveform Templates
# ──────────────────────────────────────────────────────────────────────────────

def generate_single_breath_template(
    t_local: np.ndarray,
    cycle_duration_s: float,
    ie_ratio: float,
    amplitude: float,
    pause_fraction: float = 0.08,
    waveform_type: str = "harmonic",
) -> np.ndarray:
    """
    Compute a single breath waveform using the specified template.

    Parameters
    ----------
    t_local : np.ndarray
        Time within the breath cycle (0 to cycle_duration_s).
    cycle_duration_s : float
        Total breath cycle duration.
    ie_ratio : float
        Fraction of cycle occupied by inspiration (0.25-0.45).
    amplitude : float
        Tidal volume amplitude (normalized).
    pause_fraction : float
        Fraction of cycle occupied by end-expiratory pause.
    waveform_type : str
        Template type: 'harmonic', 'sigmoid', 'triangular', 'sinusoidal'.

    Returns
    -------
    np.ndarray
        Breath waveform values (positive = inspiration, negative = expiration).
    """
    T = cycle_duration_s
    if T <= 0:
        return np.zeros_like(t_local)

    # Phase fractions
    insp_frac = ie_ratio
    pause_frac = pause_fraction
    exp_frac = 1.0 - insp_frac - pause_frac
    if exp_frac <= 0:
        exp_frac = 0.1
        pause_frac = max(0.0, 1.0 - insp_frac - exp_frac)

    # Time boundaries
    t_insp_end = insp_frac * T
    t_exp_start = t_insp_end
    t_exp_end = (insp_frac + exp_frac) * T

    result = np.zeros_like(t_local)

    if waveform_type == "harmonic":
        # Harmonic synthesis: primary sine + second harmonic (asymmetric)
        # Phase offset places peak at ~30% of cycle (faster rise)
        phase = 2.0 * np.pi * (t_local / T)
        phi_shift = -np.pi * 0.5  # Phase offset to align peak
        k = 0.35  # Harmonic coefficient (controls asymmetry)
        raw = np.sin(phase + phi_shift) + k * np.sin(2.0 * (phase + phi_shift) - np.pi * 0.6)
        # Normalize to [-1, 1]
        raw_range = np.max(raw) - np.min(raw)
        if raw_range > 0:
            raw = 2.0 * (raw - np.min(raw)) / raw_range - 1.0
        result = amplitude * raw

    elif waveform_type == "sigmoid":
        # Sigmoid-based: sharp inspiration, slower expiration
        for i, t_i in enumerate(t_local):
            if t_i < t_insp_end:
                # Inspiration: sigmoid ramp from 0 to 1
                progress = t_i / max(t_insp_end, 1e-9)
                result[i] = amplitude * (1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5))))
            elif t_i < t_exp_end:
                # Expiration: reverse sigmoid from 1 to 0
                progress = (t_i - t_exp_start) / max(t_exp_end - t_exp_start, 1e-9)
                result[i] = amplitude * (1.0 - 1.0 / (1.0 + np.exp(-10.0 * (progress - 0.5))))
            else:
                # Pause at baseline
                result[i] = 0.0
        # Scale to [0, 1] convention
        result -= np.min(result)
        if np.max(result) > 0:
            result = result / np.max(result) * amplitude
        result = result * 2.0 - amplitude  # Center around 0

    elif waveform_type == "triangular":
        # Triangular: linear ramp up, linear ramp down
        for i, t_i in enumerate(t_local):
            if t_i <= t_insp_end:
                result[i] = amplitude * (t_i / max(t_insp_end, 1e-9))
            elif t_i <= t_exp_end:
                progress = (t_i - t_insp_end) / max(t_exp_end - t_insp_end, 1e-9)
                result[i] = amplitude * (1.0 - progress)
            else:
                result[i] = 0.0

    elif waveform_type == "sinusoidal":
        # Pure sinusoidal (simplest model)
        result = amplitude * np.sin(2.0 * np.pi * t_local / T)

    else:
        raise ValueError(f"Unknown waveform_type '{waveform_type}'.")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Respiratory Cycle Generator
# ──────────────────────────────────────────────────────────────────────────────

class BreathCycleGenerator:
    """
    Generator for respiratory breath cycle sequences.

    Produces sequences of breath events (onset, duration, amplitude) for
    different respiratory patterns. Handles inter-breath variability,
    sighs, and pathological patterns.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    resp_rate_hz : float
        Mean respiratory rate in Hz (breaths per second).
    duration_s : float
        Total signal duration in seconds.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        resp_rate_hz: float,
        duration_s: float,
    ) -> None:
        self.rng = rng
        self.resp_rate_hz = resp_rate_hz
        self.mean_cycle = 1.0 / max(resp_rate_hz, 0.05)
        self.duration_s = duration_s

    def _sample_cycle_duration(
        self, variability_frac: float = 0.10
    ) -> float:
        """Sample a single cycle duration with lognormal variability."""
        sigma = np.sqrt(np.log(1.0 + variability_frac ** 2))
        mu = np.log(self.mean_cycle) - 0.5 * sigma ** 2
        return max(0.5, float(self.rng.lognormal(mean=mu, sigma=sigma)))

    def generate_eupnea(
        self, ie_ratio: float = 0.33, amplitude_cv: float = 0.10
    ) -> List[RespBreath]:
        """
        Generate normal eupneic breathing cycle sequence.

        Parameters
        ----------
        ie_ratio : float
            Inspiration fraction (0.25-0.45). Default: 0.33 (I:E = 1:2).
        amplitude_cv : float
            Coefficient of variation for tidal volume. Default: 0.10.

        Returns
        -------
        List[RespBreath]
            Ordered list of breath events.
        """
        breaths: List[RespBreath] = []
        curr_t = 0.0

        while curr_t < self.duration_s + 2.0:
            cycle_dur = self._sample_cycle_duration(variability_frac=0.10)
            amp = max(0.1, float(self.rng.normal(1.0, amplitude_cv)))

            # Occasional deep sighs (1-3% of breaths)
            is_sigh = self.rng.random() < 0.02
            if is_sigh:
                amp = self.rng.uniform(1.5, 2.5)
                cycle_dur *= self.rng.uniform(1.3, 1.8)

            breaths.append(RespBreath(
                onset_s=curr_t,
                cycle_duration_s=cycle_dur,
                ie_ratio=self.rng.uniform(ie_ratio * 0.85, ie_ratio * 1.15),
                amplitude=amp,
                has_pause=True,
                pause_duration_s=cycle_dur * self.rng.uniform(0.05, 0.12),
            ))
            curr_t += cycle_dur

        return breaths

    def generate_tachypnea(self) -> List[RespBreath]:
        """
        Generate tachypneic breathing (> 20 breaths/min, shallow).

        Returns
        -------
        List[RespBreath]
            Tachypneic breath events.
        """
        # Tachypnea: short cycles, reduced tidal volume
        breaths: List[RespBreath] = []
        curr_t = 0.0

        while curr_t < self.duration_s + 1.0:
            cycle_dur = self._sample_cycle_duration(0.12)
            amp = self.rng.uniform(0.3, 0.6)  # Shallow breaths
            breaths.append(RespBreath(
                onset_s=curr_t,
                cycle_duration_s=cycle_dur,
                ie_ratio=self.rng.uniform(0.30, 0.40),
                amplitude=amp,
                has_pause=False,
                pause_duration_s=0.0,
            ))
            curr_t += cycle_dur

        return breaths

    def generate_cheyne_stokes(self) -> List[RespBreath]:
        """
        Generate Cheyne-Stokes respiration.

        A crescendo-decrescendo pattern where tidal volume gradually increases
        then decreases, separated by periods of apnea (10-30 s).

        Seen in: Congestive heart failure, CNS lesions, uremia.

        Returns
        -------
        List[RespBreath]
            Cheyne-Stokes breath events.
        """
        breaths: List[RespBreath] = []
        curr_t = 0.0

        while curr_t < self.duration_s + 2.0:
            # Apneic pause: 10-30 s
            apnea_dur = self.rng.uniform(10.0, 25.0)
            curr_t += apnea_dur

            if curr_t >= self.duration_s:
                break

            # Crescendo phase: 5-12 breaths with increasing amplitude
            n_cs_breaths = self.rng.integers(5, 13)
            amps = np.concatenate([
                np.linspace(0.2, 1.5, n_cs_breaths // 2 + 1),
                np.linspace(1.5, 0.2, n_cs_breaths - n_cs_breaths // 2)
            ])

            for i, amp in enumerate(amps):
                cycle_dur = self._sample_cycle_duration(0.08)
                breaths.append(RespBreath(
                    onset_s=curr_t,
                    cycle_duration_s=cycle_dur,
                    ie_ratio=0.33,
                    amplitude=float(amp),
                    has_pause=False,
                    pause_duration_s=0.0,
                ))
                curr_t += cycle_dur

                if curr_t >= self.duration_s:
                    break

        return breaths

    def generate_kussmaul(self) -> List[RespBreath]:
        """
        Generate Kussmaul breathing pattern.

        Deep, regular, labored respirations at 20-30 breaths/min.
        Seen in metabolic acidosis (e.g., diabetic ketoacidosis).

        Returns
        -------
        List[RespBreath]
            Kussmaul breath events.
        """
        breaths: List[RespBreath] = []
        curr_t = 0.0

        while curr_t < self.duration_s + 1.0:
            cycle_dur = self._sample_cycle_duration(0.07)
            amp = self.rng.uniform(1.8, 2.5)  # Very deep breaths
            breaths.append(RespBreath(
                onset_s=curr_t,
                cycle_duration_s=cycle_dur,
                ie_ratio=0.45,  # Slightly prolonged inspiration
                amplitude=amp,
                has_pause=False,
                pause_duration_s=0.0,
            ))
            curr_t += cycle_dur

        return breaths

    def generate_biot(self) -> List[RespBreath]:
        """
        Generate Biot's breathing pattern.

        Irregular clusters of 2-6 breaths, abruptly followed by apneic pauses.
        Seen in CNS dysfunction (meningitis, increased ICP).

        Returns
        -------
        List[RespBreath]
            Biot's breath events.
        """
        breaths: List[RespBreath] = []
        curr_t = 0.0

        while curr_t < self.duration_s + 1.0:
            # Cluster of 2-6 irregular breaths
            n_cluster = self.rng.integers(2, 7)
            for _ in range(n_cluster):
                cycle_dur = self._sample_cycle_duration(0.25)  # High variability
                amp = self.rng.uniform(0.6, 1.4)
                breaths.append(RespBreath(
                    onset_s=curr_t,
                    cycle_duration_s=cycle_dur,
                    ie_ratio=self.rng.uniform(0.25, 0.50),
                    amplitude=amp,
                    has_pause=False,
                    pause_duration_s=0.0,
                ))
                curr_t += cycle_dur
                if curr_t >= self.duration_s:
                    break

            if curr_t >= self.duration_s:
                break

            # Abrupt apneic pause: 8-20 s
            apnea_dur = self.rng.uniform(8.0, 20.0)
            curr_t += apnea_dur

        return breaths

    def generate_hyperventilation(self) -> List[RespBreath]:
        """
        Generate hyperventilation pattern (fast + deep breathing).

        Seen in: Panic attacks, anxiety disorders, pulmonary embolism.

        Returns
        -------
        List[RespBreath]
            Hyperventilation breath events.
        """
        breaths: List[RespBreath] = []
        curr_t = 0.0

        while curr_t < self.duration_s + 1.0:
            cycle_dur = self._sample_cycle_duration(0.08)
            amp = self.rng.uniform(1.4, 2.2)  # Deep
            breaths.append(RespBreath(
                onset_s=curr_t,
                cycle_duration_s=cycle_dur,
                ie_ratio=self.rng.uniform(0.38, 0.50),
                amplitude=amp,
                has_pause=False,
                pause_duration_s=0.0,
            ))
            curr_t += cycle_dur

        return breaths

    def generate_apnea_events(
        self, breaths: List[RespBreath], apnea_rate: float = 0.05
    ) -> List[RespBreath]:
        """
        Insert apnea events (zero-amplitude breaths) into a breath sequence.

        Models obstructive sleep apnea (OSA) or central apnea events.

        Parameters
        ----------
        breaths : List[RespBreath]
            Existing breath event list.
        apnea_rate : float
            Fraction of breath cycles replaced by apneas. Default: 0.05.

        Returns
        -------
        List[RespBreath]
            Modified breath events with apneas inserted.
        """
        for breath in breaths:
            if self.rng.random() < apnea_rate:
                # Apnea: no airflow for a prolonged period
                breath.amplitude = 0.0
                breath.cycle_duration_s = self.rng.uniform(10.0, 30.0)

        return breaths


# ──────────────────────────────────────────────────────────────────────────────
# Respiratory Signal Analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_respiratory_features(resp: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Compute respiratory signal features.

    Parameters
    ----------
    resp : np.ndarray
        Respiratory signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict[str, float]
        Feature dictionary containing:
        - 'dominant_freq_hz': Dominant respiratory frequency from PSD
        - 'resp_rate_bpm': Estimated respiratory rate in breaths/min
        - 'mean_amplitude': Mean peak-to-trough amplitude
        - 'amplitude_cv': Coefficient of variation of breath amplitude
        - 'spectral_entropy': Entropy of PSD
        - 'ie_ratio_estimate': Estimated I:E ratio
    """
    n = len(resp)
    nperseg = min(512, n // 4)

    freqs, psd = sp_signal.welch(resp, fs=fs, nperseg=nperseg)

    # Find dominant respiratory frequency (0.05-1.0 Hz)
    resp_mask = (freqs >= 0.05) & (freqs <= 1.0)
    if not np.any(resp_mask):
        dom_freq = 0.25
    else:
        dom_idx = np.argmax(psd[resp_mask]) + np.where(resp_mask)[0][0]
        dom_freq = float(freqs[dom_idx])

    # Peak detection
    min_dist = max(1, int(0.5 * fs))
    peaks, _ = sp_signal.find_peaks(resp, distance=min_dist, prominence=0.05 * np.std(resp))
    troughs, _ = sp_signal.find_peaks(-resp, distance=min_dist, prominence=0.05 * np.std(resp))

    if len(peaks) > 1 and len(troughs) > 0:
        peak_amps = resp[peaks]
        trough_amps = resp[troughs]
        mean_amp = float(np.mean(peak_amps) - np.mean(trough_amps))
        amp_cv = float(np.std(peak_amps) / max(np.mean(peak_amps), 1e-12))
    else:
        mean_amp = float(np.max(resp) - np.min(resp))
        amp_cv = 0.0

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_pos = psd_norm[psd_norm > 0]
    spectral_entropy = float(-np.sum(psd_pos * np.log2(psd_pos)))

    # I:E ratio estimation (rough: positive vs negative half-cycle lengths)
    positive_samples = np.sum(resp > 0)
    negative_samples = np.sum(resp < 0)
    ie_ratio_est = positive_samples / max(negative_samples, 1)

    return {
        "dominant_freq_hz": dom_freq,
        "resp_rate_bpm": dom_freq * 60.0,
        "mean_amplitude": mean_amp,
        "amplitude_cv": amp_cv,
        "spectral_entropy": spectral_entropy,
        "ie_ratio_estimate": ie_ratio_est,
    }


def detect_apnea_events(
    resp: np.ndarray,
    fs: float,
    min_apnea_duration_s: float = 10.0,
    amplitude_threshold: float = 0.1,
) -> List[Dict[str, float]]:
    """
    Detect apnea events in a respiratory signal.

    Parameters
    ----------
    resp : np.ndarray
        Respiratory signal.
    fs : float
        Sampling frequency.
    min_apnea_duration_s : float
        Minimum apnea duration to detect (seconds). Default: 10 s.
    amplitude_threshold : float
        Fraction of max amplitude below which is considered apnea.

    Returns
    -------
    List[Dict[str, float]]
        List of detected apnea events:
        - 'onset_s': Start time (seconds)
        - 'offset_s': End time (seconds)
        - 'duration_s': Duration (seconds)
    """
    n = len(resp)
    envelope = np.abs(resp)

    # Smooth envelope
    win = max(1, int(1.0 * fs))
    kernel = np.ones(win) / win
    smooth_env = np.convolve(envelope, kernel, mode="same")

    threshold = amplitude_threshold * np.max(smooth_env)
    below_thresh = smooth_env < threshold

    min_samples = int(min_apnea_duration_s * fs)
    apneas = []
    i = 0

    while i < n:
        if below_thresh[i]:
            j = i
            while j < n and below_thresh[j]:
                j += 1
            if (j - i) >= min_samples:
                apneas.append({
                    "onset_s": i / fs,
                    "offset_s": j / fs,
                    "duration_s": (j - i) / fs,
                })
            i = j
        else:
            i += 1

    return apneas


# ──────────────────────────────────────────────────────────────────────────────
# Main Respiration Generator
# ──────────────────────────────────────────────────────────────────────────────

class RespGenerator(BaseSignal):
    """
    High-fidelity respiratory signal generator.

    Generates realistic breathing waveforms for multiple respiratory patterns
    including normal eupnea, pathological rhythms (Cheyne-Stokes, Biot),
    and exercise/hyperventilation states.

    Parameters
    ----------
    config : RespConfig
        Respiratory signal configuration.

    Examples
    --------
    >>> from biosignal_simulator.core.config import RespConfig
    >>> config = RespConfig(fs=50.0, duration_s=60.0, resp_rate_hz=0.25)
    >>> gen = RespGenerator(config)
    >>> resp = gen.generate()
    >>> resp.shape
    (3000,)

    >>> # Generate Cheyne-Stokes pattern
    >>> config_cs = RespConfig(fs=50.0, duration_s=120.0, resp_rate_hz=0.25,
    ...                         pattern='cheyne_stokes')
    >>> gen_cs = RespGenerator(config_cs)
    >>> resp_cs = gen_cs.generate()
    """

    def __init__(self, config: RespConfig) -> None:
        validate_config(config)
        super().__init__(
            fs=config.fs,
            duration_s=config.duration_s,
            seed=config.seed,
        )
        self.config = config

    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate respiratory configuration.

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
        Generate the respiratory signal.

        Returns
        -------
        np.ndarray
            Respiratory signal of shape (n_samples,).

        Notes
        -----
        Uses harmonic synthesis with phase noise for the default case.
        For named patterns, uses the BreathCycleGenerator with beat-by-beat rendering.
        """
        config = self.config
        pattern = getattr(config, "pattern", "normal").lower()

        if pattern in {"normal", "eupnea", "default"}:
            return self._generate_harmonic()
        else:
            return self._generate_pattern(pattern)

    def _generate_harmonic(self) -> np.ndarray:
        """
        Generate respiratory signal using harmonic Fourier synthesis.

        This is the fast default method using the asymmetric sinusoidal model:
            resp(t) = A * [sin(2π*f*t + θ) + k * sin(4π*f*t + θ - φ)]

        Returns
        -------
        np.ndarray
            Respiratory signal.
        """
        fs = self.fs
        n_samples = self.n_samples
        t = self.t
        config = self.config

        # Phase noise (frequency jitter)
        raw_phase_noise = self.rng.normal(0.0, 1.0, size=n_samples)
        nyq = 0.5 * fs
        fc = 0.02  # 0.02 Hz — very slow phase drift
        fc_norm = min(fc / nyq, 0.9)
        b, a = sp_signal.butter(2, fc_norm, btype="lowpass")
        phase_noise = sp_signal.filtfilt(b, a, raw_phase_noise)

        # Scale to desired std
        std_val = np.std(phase_noise)
        if std_val > 1e-12:
            phase_noise = (phase_noise / std_val) * config.phase_noise_std
        else:
            phase_noise = np.zeros(n_samples)

        # Compute instantaneous phase
        f = config.resp_rate_hz
        theta = 2.0 * np.pi * f * t + phase_noise

        # Asymmetric waveform: fundamental + second harmonic
        phi = -np.pi / 2.0  # Phase offset for asymmetry
        k = config.harmonic_k
        A = config.amplitude

        resp = A * (np.sin(theta) + k * np.sin(2.0 * theta + phi))

        return normalize_to_rms(resp, A)

    def _generate_pattern(self, pattern: str) -> np.ndarray:
        """
        Generate a named respiratory pattern using beat-by-beat synthesis.

        Parameters
        ----------
        pattern : str
            Pattern name.

        Returns
        -------
        np.ndarray
            Respiratory signal.
        """
        config = self.config
        n_samples = self.n_samples
        fs = self.fs
        t = self.t

        resp_rate_hz = config.resp_rate_hz
        cycle_gen = BreathCycleGenerator(
            rng=self.rng,
            resp_rate_hz=resp_rate_hz,
            duration_s=self.duration_s,
        )

        # Generate breath cycle sequence
        if pattern in {"eupnea", "normal"}:
            breaths = cycle_gen.generate_eupnea()
        elif pattern == "tachypnea":
            breaths = cycle_gen.generate_tachypnea()
        elif pattern == "cheyne_stokes":
            breaths = cycle_gen.generate_cheyne_stokes()
        elif pattern == "kussmaul":
            breaths = cycle_gen.generate_kussmaul()
        elif pattern == "biot":
            breaths = cycle_gen.generate_biot()
        elif pattern == "hyperventilation":
            breaths = cycle_gen.generate_hyperventilation()
        elif pattern == "apnea":
            breaths = cycle_gen.generate_eupnea()
            breaths = cycle_gen.generate_apnea_events(breaths, apnea_rate=0.30)
        else:
            warnings.warn(f"Unknown pattern '{pattern}'. Using harmonic default.")
            return self._generate_harmonic()

        # Render breath sequence to signal
        resp = self._render_breaths(breaths, t)

        # Add phase noise for realism
        phase_noise_arr = self.rng.normal(0.0, config.phase_noise_std * 0.3, size=n_samples)
        resp += phase_noise_arr

        return normalize_to_rms(resp, config.amplitude)

    def _render_breaths(
        self, breaths: List[RespBreath], t: np.ndarray
    ) -> np.ndarray:
        """
        Render a sequence of breath events to a signal array.

        Parameters
        ----------
        breaths : List[RespBreath]
            Breath event sequence.
        t : np.ndarray
            Time axis.

        Returns
        -------
        np.ndarray
            Rendered respiratory signal.
        """
        n_samples = len(t)
        fs = self.fs
        resp = np.zeros(n_samples)

        for breath in breaths:
            onset = breath.onset_s
            cycle_dur = breath.cycle_duration_s
            amp = breath.amplitude

            if onset > self.duration_s:
                break
            if amp <= 0.0:
                continue  # Apnea: zero amplitude → no contribution

            # Sample indices for this breath
            start_idx = max(0, int(onset * fs))
            end_idx = min(n_samples, int((onset + cycle_dur) * fs) + 1)

            if start_idx >= end_idx:
                continue

            t_local = t[start_idx:end_idx] - onset

            wave = generate_single_breath_template(
                t_local=t_local,
                cycle_duration_s=cycle_dur,
                ie_ratio=breath.ie_ratio,
                amplitude=amp,
                pause_fraction=breath.pause_duration_s / max(cycle_dur, 1e-6),
                waveform_type="harmonic",
            )

            resp[start_idx:end_idx] += wave

        return resp

    def generate_with_heartbeat_coupling(
        self, hr_bpm: float = 72.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate respiratory signal coupled with cardiac RSA (Respiratory Sinus Arrhythmia).

        RSA is the normal increase in heart rate during inspiration and decrease
        during expiration. This coupling is mediated by the vagus nerve.

        Parameters
        ----------
        hr_bpm : float
            Mean heart rate in bpm.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (resp_signal, rr_intervals_s) — respiratory signal and RSA-modulated RR intervals.
        """
        resp = self._generate_harmonic()
        t = self.t

        # Mean RR interval
        mean_rr = 60.0 / hr_bpm

        # RSA amplitude: ~0.06 s (60 ms) peak-to-peak variation
        rsa_amplitude = 0.03  # seconds

        # RSA coupling: RR interval decreases during inspiration (positive resp)
        # Normalize resp to [-1, 1] for modulation
        resp_norm = resp / (np.max(np.abs(resp)) + 1e-12)
        rr_modulation = -rsa_amplitude * resp_norm  # Negative: HR increases on inspiration

        # Smooth modulation (RSA is low-frequency)
        nyq = 0.5 / (t[1] - t[0]) if len(t) > 1 else 1.0
        fc = min(self.config.resp_rate_hz * 2.0 / nyq, 0.9)
        b, a = sp_signal.butter(2, fc, btype="lowpass")
        rr_modulation = sp_signal.filtfilt(b, a, rr_modulation)

        # Generate RR intervals with RSA
        rr_at_time = mean_rr + rr_modulation

        # Sample RR intervals for each beat
        beat_times = []
        curr_t = 0.0
        while curr_t < self.duration_s:
            idx = min(int(curr_t * self.fs), len(rr_at_time) - 1)
            rr = rr_at_time[idx] + self.rng.normal(0.0, 0.01)
            rr = max(0.3, rr)
            beat_times.append(curr_t)
            curr_t += rr

        rr_intervals = np.diff(beat_times) if len(beat_times) > 1 else np.array([mean_rr])

        return resp, rr_intervals

    def compute_features(self) -> Dict[str, float]:
        """
        Generate and compute respiratory features.

        Returns
        -------
        Dict[str, float]
            Respiratory feature dictionary.
        """
        return compute_respiratory_features(self.generate(), self.fs)

    def detect_apneas(
        self, min_duration_s: float = 10.0
    ) -> List[Dict[str, float]]:
        """
        Generate signal and detect apnea events.

        Parameters
        ----------
        min_duration_s : float
            Minimum apnea duration in seconds.

        Returns
        -------
        List[Dict[str, float]]
            List of detected apnea events.
        """
        resp = self.generate()
        return detect_apnea_events(resp, self.fs, min_apnea_duration_s=min_duration_s)

    def summary(self) -> Dict[str, Any]:
        """
        Return configuration summary.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary.
        """
        return {
            "resp_rate_hz": self.config.resp_rate_hz,
            "resp_rate_bpm": self.config.resp_rate_hz * 60.0,
            "amplitude": self.config.amplitude,
            "harmonic_k": self.config.harmonic_k,
            "phase_noise_std": self.config.phase_noise_std,
            "pattern": getattr(self.config, "pattern", "normal"),
            "fs_hz": self.fs,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Factory Functions
# ──────────────────────────────────────────────────────────────────────────────

def make_resp_normal(
    duration_s: float = 60.0,
    resp_rate_hz: float = 0.25,
    fs: float = 50.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a normal respiratory signal.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    resp_rate_hz : float
        Respiratory rate in Hz (default: 0.25 Hz = 15 breaths/min).
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Normal respiratory signal.
    """
    config = RespConfig(
        fs=fs, duration_s=duration_s, resp_rate_hz=resp_rate_hz, seed=seed,
    )
    return RespGenerator(config).generate()


def make_resp_cheyne_stokes(
    duration_s: float = 120.0,
    resp_rate_hz: float = 0.25,
    fs: float = 50.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Cheyne-Stokes respiratory pattern.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    resp_rate_hz : float
        Mean respiratory rate in Hz.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Cheyne-Stokes respiratory signal.
    """
    config = RespConfig(
        fs=fs, duration_s=duration_s, resp_rate_hz=resp_rate_hz, seed=seed,
    )
    # Inject pattern attribute
    config.pattern = "cheyne_stokes"
    return RespGenerator(config).generate()


def make_resp_hyperventilation(
    duration_s: float = 60.0,
    resp_rate_hz: float = 0.50,
    fs: float = 50.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a hyperventilation respiratory signal.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    resp_rate_hz : float
        Respiratory rate (≥ 0.40 Hz for hyperventilation = ≥ 24 breaths/min).
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Hyperventilation respiratory signal.
    """
    config = RespConfig(
        fs=fs, duration_s=duration_s, resp_rate_hz=resp_rate_hz, seed=seed,
    )
    config.pattern = "hyperventilation"
    return RespGenerator(config).generate()


def batch_generate_resp_patterns(
    patterns: List[str],
    duration_s: float = 60.0,
    resp_rate_hz: float = 0.25,
    fs: float = 50.0,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate respiratory signals for multiple patterns.

    Parameters
    ----------
    patterns : List[str]
        Pattern names.
    duration_s : float
        Duration.
    resp_rate_hz : float
        Mean respiratory rate.
    fs : float
        Sampling frequency.
    seed : int, optional
        Base random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping pattern name to signal.

    Examples
    --------
    >>> signals = batch_generate_resp_patterns(['eupnea', 'cheyne_stokes', 'kussmaul'])
    >>> list(signals.keys())
    ['eupnea', 'cheyne_stokes', 'kussmaul']
    """
    results: Dict[str, np.ndarray] = {}
    for i, pattern in enumerate(patterns):
        rng_seed = (seed + i) if seed is not None else None
        config = RespConfig(
            fs=fs, duration_s=duration_s, resp_rate_hz=resp_rate_hz, seed=rng_seed,
        )
        config.pattern = pattern
        try:
            results[pattern] = RespGenerator(config).generate()
        except Exception as exc:
            warnings.warn(f"Failed to generate pattern '{pattern}': {exc}")
    return results
