"""
High-fidelity Electromyogram (EMG) simulator.

This module provides a comprehensive, physiologically accurate simulation of muscle
electrical activity as recorded from surface or intramuscular electrodes (sEMG/iEMG).

Physiological Basis
-------------------
Motor neuron activity drives muscle contractions. Each motor neuron innervates a group
of muscle fibers (a motor unit). When a motor neuron fires, all its fibers contract
synchronously, producing a Motor Unit Action Potential (MUAP).

Surface EMG records the superposition of MUAPs from many motor units, creating an
interference pattern whose amplitude scales with contraction force and motor unit
recruitment threshold. Intramuscular EMG can record individual MUAP waveforms.

MUAP Morphology
---------------
Each MUAP has a characteristic shape determined by:
- **Duration**: 5-15 ms for normal MUAPs; prolonged in neuropathy (20-30 ms)
- **Amplitude**: 100-1000 µV for surface; 200-5000 µV for intramuscular
- **Phases**: Biphasic, triphasic, or polyphasic (>4 phases in myopathy)
- **Firing Rate**: 5-30 Hz depending on contraction force and pathology

EMG Types Implemented
---------------------
1. **Surface EMG (sEMG)**:
   - Interference pattern from multiple recruited motor units
   - Bandpass-filtered Gaussian process (20-500 Hz)
   - Amplitude modulated by voluntary contraction envelopes (constant, ramp, burst)
   - Multi-channel with lead-specific motor unit recruitment zones

2. **Intramuscular EMG (iEMG)**:
   - Discrete MUAP spikes with specified morphology (biphasic/triphasic/polyphasic)
   - Poisson-process firing with inter-spike interval jitter
   - Multiple concurrent motor unit action potential trains (MUAPTs)

Pathologies Implemented
------------------------
- **Normal**: Standard triphasic MUAPs, 5-20 Hz firing rate
- **Neuropathic**: Giant MUAPs (>5x amplitude), prolonged duration, slow firing
- **Myopathic**: Small, short, polyphasic MUAPs, high firing rate
- **ALS**: Giant polyphasic MUAPs + spontaneous fasciculation potentials
- **Myasthenia Gravis**: Normal MUAPs with progressive decrement (amplitude fatigue)
- **Parkinson's Tremor**: 4-6 Hz bursting synchronization of motor units
- **Peripheral Neuropathy**: Reduced recruitment, denervated fibers, sharp waves
- **Muscular Dystrophy**: Myopathic with early fatigue and irregular polyphasic bursts
- **Lambert-Eaton Syndrome**: Initial increment then fatigue in repetitive stimulation

References
----------
- Merletti, R. & Farina, D. (2016). Surface Electromyography. Wiley-IEEE Press.
- Dumitru, D. et al. (2002). Electrodiagnostic Medicine. Hanley & Belfus.
- Farina, D. et al. (2014). The Extraction of Neural Information from Surface EMG.
  J Neural Eng.
- Pino, L.J. et al. (2008). Surface EMG Decomposition Using High-Density Arrays.
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
from biosignal_simulator.core.config import EMGConfig
from biosignal_simulator.core.math_utils import normalize_to_rms


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class EMGType(str, Enum):
    """EMG electrode type."""
    SURFACE = "surface"
    INTRAMUSCULAR = "intramuscular"


class MUAPShape(str, Enum):
    """MUAP morphology shape types."""
    BIPHASIC = "biphasic"
    TRIPHASIC = "triphasic"
    POLYPHASIC = "polyphasic"
    COMPLEX_POLYPHASIC = "complex_polyphasic"  # >5 phases (myopathic)
    GIANT = "giant"                             # Neuropathic giant MUAP
    FIBRILLATION = "fibrillation"              # Spontaneous fibrillation potential


class EMGPathology(str, Enum):
    """Supported EMG pathology types."""
    NORMAL = "normal"
    NEUROPATHIC = "neuropathic"
    MYOPATHIC = "myopathic"
    ALS = "als"
    MYASTHENIA_GRAVIS = "myasthenia_gravis"
    PARKINSONS_TREMOR = "parkinsons_tremor"
    PERIPHERAL_NEUROPATHY = "peripheral_neuropathy"
    MUSCULAR_DYSTROPHY = "muscular_dystrophy"
    LAMBERT_EATON = "lambert_eaton"
    FATIGUE = "fatigue"


# ──────────────────────────────────────────────────────────────────────────────
# Motor Unit Action Potential (MUAP) Parameters by Pathology
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MUAPParameters:
    """
    Parameters defining the MUAP morphology for a given pathology.

    Attributes
    ----------
    firing_rate_hz : float
        Mean MUAP firing rate in Hz.
    amplitude_scale : float
        Amplitude scaling factor relative to baseline.
    duration_ms : float
        Total MUAP duration in milliseconds.
    shape : str
        MUAP shape type (see MUAPShape enum).
    jitter_fraction : float
        Coefficient of variation for inter-spike interval (ISI jitter).
    n_motor_units : int
        Number of concurrent motor units (for sEMG).
    recruitment_threshold : float
        Contraction level threshold (0-1) for motor unit activation.
    fatigue_tau_s : float
        Time constant for amplitude fatigue (seconds). 0 = no fatigue.
    has_fibrillations : bool
        Whether to generate spontaneous fibrillation potentials.
    has_fasciculations : bool
        Whether to generate spontaneous fasciculation potentials.
    """
    firing_rate_hz: float = 15.0
    amplitude_scale: float = 1.0
    duration_ms: float = 8.0
    shape: str = "triphasic"
    jitter_fraction: float = 0.12
    n_motor_units: int = 8
    recruitment_threshold: float = 0.0
    fatigue_tau_s: float = 0.0
    has_fibrillations: bool = False
    has_fasciculations: bool = False


# Pathology parameter table
PATHOLOGY_PARAMS: Dict[str, MUAPParameters] = {
    "normal": MUAPParameters(
        firing_rate_hz=15.0, amplitude_scale=1.0, duration_ms=8.0,
        shape="triphasic", jitter_fraction=0.12, n_motor_units=8,
    ),
    "neuropathic": MUAPParameters(
        firing_rate_hz=6.0, amplitude_scale=5.0, duration_ms=20.0,
        shape="triphasic", jitter_fraction=0.20, n_motor_units=4,
        has_fasciculations=True,
    ),
    "myopathic": MUAPParameters(
        firing_rate_hz=25.0, amplitude_scale=0.3, duration_ms=4.0,
        shape="polyphasic", jitter_fraction=0.15, n_motor_units=12,
    ),
    "als": MUAPParameters(
        firing_rate_hz=5.0, amplitude_scale=4.5, duration_ms=22.0,
        shape="polyphasic", jitter_fraction=0.30, n_motor_units=3,
        has_fibrillations=True, has_fasciculations=True,
    ),
    "myasthenia_gravis": MUAPParameters(
        firing_rate_hz=18.0, amplitude_scale=1.0, duration_ms=7.5,
        shape="triphasic", jitter_fraction=0.15, n_motor_units=7,
        fatigue_tau_s=3.5,
    ),
    "parkinsons_tremor": MUAPParameters(
        firing_rate_hz=14.0, amplitude_scale=1.0, duration_ms=8.0,
        shape="triphasic", jitter_fraction=0.25, n_motor_units=6,
    ),
    "peripheral_neuropathy": MUAPParameters(
        firing_rate_hz=8.0, amplitude_scale=3.5, duration_ms=16.0,
        shape="triphasic", jitter_fraction=0.25, n_motor_units=4,
        has_fibrillations=True,
    ),
    "muscular_dystrophy": MUAPParameters(
        firing_rate_hz=22.0, amplitude_scale=0.4, duration_ms=5.0,
        shape="complex_polyphasic", jitter_fraction=0.20, n_motor_units=10,
        fatigue_tau_s=2.0, has_fibrillations=True,
    ),
    "lambert_eaton": MUAPParameters(
        firing_rate_hz=20.0, amplitude_scale=0.5, duration_ms=7.0,
        shape="triphasic", jitter_fraction=0.18, n_motor_units=6,
        fatigue_tau_s=8.0,  # Initial increment, then fatigue
    ),
    "fatigue": MUAPParameters(
        firing_rate_hz=12.0, amplitude_scale=0.8, duration_ms=9.0,
        shape="triphasic", jitter_fraction=0.18, n_motor_units=8,
        fatigue_tau_s=5.0,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# MUAP Waveform Templates
# ──────────────────────────────────────────────────────────────────────────────

def generate_muap_biphasic(
    t_local: np.ndarray,
    amplitude: float,
    duration_ms: float,
) -> np.ndarray:
    """
    Generate a biphasic MUAP waveform.

    A biphasic MUAP consists of a negative-positive or positive-negative deflection.
    This morphology is typically seen in surface recordings from muscles recorded at
    moderate distances from the active fibers.

    The waveform is modeled as the first derivative of a Gaussian (a biphasic pulse):
        MUAP(t) = -A * (t/σ) * exp(-0.5 * (t/σ)^2)

    Parameters
    ----------
    t_local : np.ndarray
        Time relative to MUAP onset (seconds). Center at t=0.
    amplitude : float
        Peak amplitude in microvolts.
    duration_ms : float
        Total MUAP duration in milliseconds.

    Returns
    -------
    np.ndarray
        Biphasic MUAP waveform.

    Examples
    --------
    >>> t = np.linspace(-0.02, 0.02, 100)
    >>> muap = generate_muap_biphasic(t, amplitude=500.0, duration_ms=8.0)
    >>> float(np.max(np.abs(muap)))  # Peak amplitude
    500.0
    """
    sigma = (duration_ms / 1000.0) / 4.0
    if sigma <= 0.0:
        sigma = 1e-6
    tau = t_local / sigma
    val = -amplitude * tau * np.exp(-0.5 * tau ** 2)
    # Normalize to exact peak amplitude
    peak = np.max(np.abs(val))
    if peak > 0:
        val = val * (amplitude / peak)
    return val


def generate_muap_triphasic(
    t_local: np.ndarray,
    amplitude: float,
    duration_ms: float,
) -> np.ndarray:
    """
    Generate a triphasic MUAP waveform.

    Triphasic MUAPs are the most common morphology in normal muscle.
    They consist of a small initial positive deflection, a large negative
    spike, and a final positive deflection.

    Modeled as the second derivative of a Gaussian (Mexican hat wavelet):
        MUAP(t) = A * (1 - (t/σ)^2) * exp(-0.5 * (t/σ)^2)

    Parameters
    ----------
    t_local : np.ndarray
        Time relative to MUAP onset.
    amplitude : float
        Peak amplitude in microvolts.
    duration_ms : float
        MUAP duration in milliseconds.

    Returns
    -------
    np.ndarray
        Triphasic MUAP waveform.
    """
    sigma = (duration_ms / 1000.0) / 4.0
    if sigma <= 0.0:
        sigma = 1e-6
    tau = t_local / sigma
    val = amplitude * (1.0 - tau ** 2) * np.exp(-0.5 * tau ** 2)
    peak = np.max(np.abs(val))
    if peak > 0:
        val = val * (amplitude / peak)
    return val


def generate_muap_polyphasic(
    t_local: np.ndarray,
    amplitude: float,
    duration_ms: float,
    n_phases: int = 5,
) -> np.ndarray:
    """
    Generate a polyphasic MUAP waveform.

    Polyphasic MUAPs (>4 zero-crossings) are characteristic of:
    - Myopathic conditions: many small, briefly active fibers
    - Re-innervation in neuropathy (early stages)
    - Normal variation in some muscles

    Modeled as a damped sinusoidal oscillation within a Gaussian envelope:
        MUAP(t) = A * sin(n * π * t/σ) * exp(-0.5 * (t/σ)^2)

    Parameters
    ----------
    t_local : np.ndarray
        Time relative to MUAP center.
    amplitude : float
        Peak amplitude.
    duration_ms : float
        MUAP duration in milliseconds.
    n_phases : int
        Approximate number of phases (determines oscillation frequency).

    Returns
    -------
    np.ndarray
        Polyphasic MUAP waveform.
    """
    sigma = (duration_ms / 1000.0) / 4.0
    if sigma <= 0.0:
        sigma = 1e-6
    tau = t_local / sigma
    # Sinusoidal oscillation frequency scales with n_phases
    omega = n_phases * 0.8
    val = amplitude * np.sin(omega * np.pi * tau) * np.exp(-0.5 * tau ** 2)
    peak = np.max(np.abs(val))
    if peak > 0:
        val = val * (amplitude / peak)
    return val


def generate_muap_complex_polyphasic(
    t_local: np.ndarray,
    amplitude: float,
    duration_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a complex polyphasic MUAP (>5 phases, irregular shape).

    Highly complex MUAPs occur in:
    - Advanced myopathic conditions (muscular dystrophy, polymyositis)
    - Long-standing re-innervation
    - Satellite potentials (late components)

    Modeled as a sum of multiple Gaussian pulses with random polarities and timings,
    all within a broader Gaussian envelope.

    Parameters
    ----------
    t_local : np.ndarray
        Time relative to MUAP center.
    amplitude : float
        Peak amplitude.
    duration_ms : float
        MUAP duration in milliseconds.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Complex polyphasic MUAP waveform.
    """
    sigma_env = (duration_ms / 1000.0) / 3.0
    if sigma_env <= 0:
        sigma_env = 1e-6

    # Build 5-8 sub-component Gaussians within the envelope
    n_components = rng.integers(5, 9)
    val = np.zeros_like(t_local)

    t_max = sigma_env * 3.0  # ±3σ range
    component_centers = rng.uniform(-t_max * 0.9, t_max * 0.9, size=n_components)
    component_amps = rng.choice([-1.0, 1.0], size=n_components) * rng.uniform(0.2, 1.0, size=n_components)
    component_widths = rng.uniform(sigma_env * 0.1, sigma_env * 0.35, size=n_components)

    for ct, ca, cw in zip(component_centers, component_amps, component_widths):
        val += ca * np.exp(-0.5 * ((t_local - ct) / max(cw, 1e-9)) ** 2)

    # Apply outer Gaussian envelope
    outer_env = np.exp(-0.5 * (t_local / sigma_env) ** 2)
    val *= outer_env

    # Normalize
    peak = np.max(np.abs(val))
    if peak > 0:
        val = val * (amplitude / peak)
    return val


def generate_muap_fibrillation(
    t_local: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """
    Generate a fibrillation potential waveform.

    Fibrillation potentials are spontaneous, involuntary discharges of denervated
    single muscle fibers. They appear as brief (1-2 ms), diphasic or triphasic
    spikes with regular firing at 2-20 Hz.

    Clinical significance:
    - Indicate recent or active denervation (neuropathy, radiculopathy, ALS)
    - Appear 2-3 weeks after nerve injury
    - Distinguish acute from chronic neuropathy

    Parameters
    ----------
    t_local : np.ndarray
        Time relative to spike center (seconds).
    amplitude : float
        Peak amplitude in microvolts. Typical range: 100-1000 µV.

    Returns
    -------
    np.ndarray
        Fibrillation potential waveform.
    """
    # Very brief triphasic spike (1-2 ms duration)
    sigma = 0.0005  # 0.5 ms width
    tau = t_local / sigma
    val = amplitude * (1.0 - tau ** 2) * np.exp(-0.5 * tau ** 2)
    peak = np.max(np.abs(val))
    if peak > 0:
        val = val * (amplitude / peak)
    return val


def generate_muap(
    t_local: np.ndarray,
    amplitude: float,
    duration_ms: float,
    shape: str,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Dispatch to the correct MUAP shape generator.

    Parameters
    ----------
    t_local : np.ndarray
        Local time array centered at MUAP peak.
    amplitude : float
        Peak amplitude in microvolts.
    duration_ms : float
        MUAP duration in milliseconds.
    shape : str
        MUAP shape: 'biphasic', 'triphasic', 'polyphasic', 'complex_polyphasic',
        'fibrillation', or 'giant'.
    rng : np.random.Generator, optional
        Required for 'complex_polyphasic' and 'giant'.

    Returns
    -------
    np.ndarray
        MUAP waveform values.
    """
    shape = shape.lower()
    if shape == "biphasic":
        return generate_muap_biphasic(t_local, amplitude, duration_ms)
    elif shape == "triphasic":
        return generate_muap_triphasic(t_local, amplitude, duration_ms)
    elif shape == "polyphasic":
        return generate_muap_polyphasic(t_local, amplitude, duration_ms, n_phases=5)
    elif shape == "complex_polyphasic":
        if rng is None:
            rng = np.random.default_rng()
        return generate_muap_complex_polyphasic(t_local, amplitude, duration_ms, rng)
    elif shape == "giant":
        # Giant MUAP: oversized triphasic with satellite potential
        main = generate_muap_triphasic(t_local, amplitude, duration_ms)
        # Add late satellite potential (mini-MUAP, 10-30% of main amplitude)
        if rng is not None:
            satellite_delay = (duration_ms / 1000.0) * rng.uniform(0.5, 1.5)
            sat_amp = amplitude * rng.uniform(0.10, 0.30)
            sat_sigma = 0.0008
            satellite = sat_amp * np.exp(-0.5 * ((t_local - satellite_delay) / sat_sigma) ** 2)
            return main + satellite
        return main
    elif shape == "fibrillation":
        return generate_muap_fibrillation(t_local, amplitude)
    else:
        return generate_muap_triphasic(t_local, amplitude, duration_ms)


# ──────────────────────────────────────────────────────────────────────────────
# Motor Unit Spike Train Generator
# ──────────────────────────────────────────────────────────────────────────────

class MotorUnitSpikeTrainGenerator:
    """
    Generator for individual motor unit spike trains (MUAPTs).

    A motor unit fires asynchronously in response to neural drive. The inter-spike
    interval (ISI) follows a lognormal distribution with pathology-specific
    variability.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    pathology : str
        Pathology name for parameter lookup.
    base_amplitude_uv : float
        Baseline amplitude in microvolts.
    duration_s : float
        Total signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        pathology: str,
        base_amplitude_uv: float,
        duration_s: float,
        fs: float,
    ) -> None:
        self.rng = rng
        self.pathology = pathology.lower()
        self.base_amplitude_uv = base_amplitude_uv
        self.duration_s = duration_s
        self.fs = fs
        self.params = PATHOLOGY_PARAMS.get(self.pathology, PATHOLOGY_PARAMS["normal"])

    def generate_spike_times(
        self,
        firing_rate_hz: float,
        jitter_fraction: float,
        tremor_modulate: bool = False,
    ) -> np.ndarray:
        """
        Generate a sequence of motor unit firing timestamps.

        Parameters
        ----------
        firing_rate_hz : float
            Mean firing rate in Hz.
        jitter_fraction : float
            Coefficient of variation for ISI (higher = more irregular).
        tremor_modulate : bool
            If True, modulate firing probability with a 5 Hz tremor oscillation.

        Returns
        -------
        np.ndarray
            Array of firing timestamps in seconds.
        """
        mean_isi = 1.0 / max(firing_rate_hz, 0.1)
        times: List[float] = []

        if tremor_modulate:
            # Parkinson's tremor: 4-6 Hz rhythmic burst modulation
            tremor_freq = self.rng.uniform(4.0, 6.0)
            curr_t = mean_isi / 2.0
            while curr_t < self.duration_s:
                # Probability of firing modulated by tremor phase
                phase = 2.0 * np.pi * tremor_freq * curr_t
                mod = 0.5 * (1.0 + np.sin(phase))
                fire_prob = 0.2 + 0.8 * mod
                if self.rng.random() < fire_prob:
                    times.append(curr_t)
                curr_t += mean_isi * self.rng.uniform(0.5, 1.5)
        else:
            # Lognormal ISI distribution
            sigma_isi = np.sqrt(np.log(1.0 + jitter_fraction ** 2))
            mu_isi = np.log(mean_isi) - 0.5 * sigma_isi ** 2

            curr_t = mean_isi / 2.0
            while curr_t < self.duration_s:
                times.append(curr_t)
                isi = self.rng.lognormal(mean=mu_isi, sigma=sigma_isi)
                curr_t += max(0.005, isi)  # Minimum 5 ms refractory period

        return np.array(times)

    def generate_spike_train(
        self,
        params: MUAPParameters,
        firing_rate_override: Optional[float] = None,
        amplitude_override: Optional[float] = None,
    ) -> np.ndarray:
        """
        Generate a complete motor unit spike train signal.

        Parameters
        ----------
        params : MUAPParameters
            MUAP parameters for this motor unit.
        firing_rate_override : float, optional
            Override the firing rate.
        amplitude_override : float, optional
            Override the amplitude.

        Returns
        -------
        np.ndarray
            Spike train signal array of shape (n_samples,).
        """
        fs = self.fs
        n_samples = int(self.duration_s * fs)
        t = np.arange(n_samples) / fs

        firing_rate = firing_rate_override or params.firing_rate_hz
        amplitude = (amplitude_override or 1.0) * self.base_amplitude_uv * params.amplitude_scale

        tremor_mod = self.pathology == "parkinsons_tremor"
        spike_times = self.generate_spike_times(
            firing_rate_hz=firing_rate,
            jitter_fraction=params.jitter_fraction,
            tremor_modulate=tremor_mod,
        )

        signal = np.zeros(n_samples)
        half_width = (params.duration_ms / 1000.0) * 2.5  # Half-width in seconds

        for k, t_fire in enumerate(spike_times):
            if t_fire >= self.duration_s:
                break

            # Compute amplitude with fatigue modulation
            amp_k = amplitude
            if params.fatigue_tau_s > 0:
                if self.pathology == "lambert_eaton":
                    # Lambert-Eaton: initial increment then fatigue
                    # First 2 seconds: increments to 3x; then decays
                    if t_fire < 2.0:
                        amp_k = amplitude * (1.0 + 2.0 * (t_fire / 2.0))
                    else:
                        amp_k = amplitude * 3.0 * np.exp(-(t_fire - 2.0) / params.fatigue_tau_s)
                else:
                    # Exponential fatigue decay
                    amp_k = amplitude * np.exp(-t_fire / params.fatigue_tau_s)

            # Find sample window for this MUAP
            start_idx = max(0, int((t_fire - half_width) * fs))
            end_idx = min(n_samples, int((t_fire + half_width) * fs))

            if start_idx >= end_idx:
                continue

            t_local = t[start_idx:end_idx] - t_fire
            muap_val = generate_muap(
                t_local=t_local,
                amplitude=amp_k,
                duration_ms=params.duration_ms,
                shape=params.shape,
                rng=self.rng,
            )
            signal[start_idx:end_idx] += muap_val

        return signal


# ──────────────────────────────────────────────────────────────────────────────
# Surface EMG Envelope Types
# ──────────────────────────────────────────────────────────────────────────────

def build_emg_envelope(
    t: np.ndarray,
    env_type: str,
    contraction_level: float,
    ramp_duration_s: float = 1.0,
    burst_rate_hz: float = 2.0,
    burst_duration_s: float = 0.2,
    burst_amplitude: float = 1.0,
) -> np.ndarray:
    """
    Build a voluntary contraction envelope for surface EMG.

    The envelope shapes the amplitude of the EMG signal over time,
    representing the voluntary activation pattern.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds.
    env_type : str
        Envelope type: 'constant', 'ramp', 'burst', 'trapezoid', 'random',
        'sinusoidal', 'staircase'.
    contraction_level : float
        Peak contraction level (0-1).
    ramp_duration_s : float
        Duration of rise/fall ramp (seconds). Used for 'ramp' and 'trapezoid'.
    burst_rate_hz : float
        Burst repetition rate (Hz). Used for 'burst'.
    burst_duration_s : float
        Duration of each burst (seconds). Used for 'burst'.
    burst_amplitude : float
        Peak amplitude of bursts. Used for 'burst'.

    Returns
    -------
    np.ndarray
        Envelope array, same shape as t.

    Raises
    ------
    ValueError
        If env_type is not recognized.

    Examples
    --------
    >>> t = np.linspace(0, 10, 1000)
    >>> env = build_emg_envelope(t, 'ramp', contraction_level=0.8)
    >>> float(np.max(env))
    0.8
    """
    dur = t[-1] if len(t) > 0 else 1.0
    envelope = np.zeros_like(t)

    if env_type == "constant":
        envelope[:] = contraction_level

    elif env_type == "ramp":
        rd = ramp_duration_s
        cl = contraction_level
        if dur >= 2.0 * rd:
            rise_m = t < rd
            hold_m = (t >= rd) & (t < dur - rd)
            fall_m = t >= dur - rd
            envelope[rise_m] = cl * (t[rise_m] / rd)
            envelope[hold_m] = cl
            envelope[fall_m] = cl * ((dur - t[fall_m]) / rd)
        else:
            half = dur / 2.0
            rise_m = t < half
            fall_m = t >= half
            envelope[rise_m] = cl * (t[rise_m] / half)
            envelope[fall_m] = cl * ((dur - t[fall_m]) / half)

    elif env_type == "trapezoid":
        # Trapezoid: ramp up → sustained plateau → ramp down
        rise_end = ramp_duration_s
        fall_start = max(rise_end + 0.01, dur - ramp_duration_s)
        rise_m = t < rise_end
        hold_m = (t >= rise_end) & (t < fall_start)
        fall_m = t >= fall_start
        envelope[rise_m] = contraction_level * (t[rise_m] / rise_end)
        envelope[hold_m] = contraction_level
        dur_fall = dur - fall_start
        if dur_fall > 0:
            envelope[fall_m] = contraction_level * ((dur - t[fall_m]) / dur_fall)

    elif env_type == "burst":
        if burst_rate_hz > 0.0:
            interval = 1.0 / burst_rate_hz
            centers = np.arange(interval / 2.0, dur, interval)
            sigma_burst = burst_duration_s / 4.0
            if sigma_burst <= 0:
                sigma_burst = 1e-6
            for center in centers:
                envelope += burst_amplitude * np.exp(-0.5 * ((t - center) / sigma_burst) ** 2)

    elif env_type == "sinusoidal":
        # Sinusoidal oscillation of contraction level
        f_env = burst_rate_hz  # Reuse burst_rate as oscillation frequency
        envelope = contraction_level * (0.5 + 0.5 * np.sin(2.0 * np.pi * f_env * t))

    elif env_type == "staircase":
        # Staircase: increasing steps of contraction
        n_steps = max(2, int(burst_rate_hz))  # Reuse as step count
        step_dur = dur / n_steps
        for step in range(n_steps):
            step_level = contraction_level * (step + 1) / n_steps
            step_m = (t >= step * step_dur) & (t < (step + 1) * step_dur)
            envelope[step_m] = step_level

    elif env_type == "random":
        # Random voluntary contraction fluctuations (smoothed noise)
        raw_noise = np.random.default_rng().normal(0.0, 0.25, size=len(t))
        nyq = 0.5 / (t[1] - t[0]) if len(t) > 1 else 1.0
        fc = min(2.0 / nyq, 0.9)
        b, a = sp_signal.butter(2, fc, btype="lowpass")
        smooth = sp_signal.filtfilt(b, a, raw_noise)
        envelope = contraction_level * np.clip(0.5 + smooth, 0.0, 1.0)

    else:
        raise ValueError(
            f"Unknown envelope type '{env_type}'. "
            f"Valid: 'constant', 'ramp', 'burst', 'trapezoid', 'random', 'sinusoidal', 'staircase'."
        )

    return np.clip(envelope, 0.0, None)  # No negative contraction


# ──────────────────────────────────────────────────────────────────────────────
# Surface EMG Frequency Shaping
# ──────────────────────────────────────────────────────────────────────────────

def get_emg_bandpass_params(
    pathology: str,
    fmin_hz: float,
    fmax_hz: float,
    fs: float,
) -> Tuple[float, float]:
    """
    Compute pathology-adjusted EMG bandpass filter cutoff frequencies.

    Different pathologies shift the dominant spectral content of the EMG:
    - Neuropathic/ALS: lower spectral content (giant, slow MUAPs)
    - Myopathic/Muscular Dystrophy: higher spectral content (brief MUAPs)
    - Fatigue: spectral compression toward low frequencies (median frequency decreases)

    Parameters
    ----------
    pathology : str
        Pathology name.
    fmin_hz : float
        Configured minimum frequency in Hz.
    fmax_hz : float
        Configured maximum frequency in Hz.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Tuple[float, float]
        Adjusted (low, high) cutoff frequencies.
    """
    nyq = 0.5 * fs

    if pathology in {"neuropathic", "als", "peripheral_neuropathy"}:
        # Spectral shift down (giant MUAPs with long duration)
        low = max(10.0, fmin_hz * 0.65)
        high = min(fmax_hz * 0.55, nyq * 0.45)
    elif pathology in {"myopathic", "muscular_dystrophy"}:
        # Spectral shift up (brief, high-frequency MUAPs)
        low = max(40.0, fmin_hz * 1.5)
        high = min(fmax_hz * 1.35, nyq * 0.48)
    elif pathology == "fatigue":
        # Fatigue compresses spectrum toward low frequencies
        low = max(15.0, fmin_hz * 0.8)
        high = min(fmax_hz * 0.65, nyq * 0.45)
    else:
        low = fmin_hz
        high = min(fmax_hz, nyq * 0.48)

    return max(1.0, low), max(low + 1.0, high)


# ──────────────────────────────────────────────────────────────────────────────
# Main EMG Generator Class
# ──────────────────────────────────────────────────────────────────────────────

class EMGGenerator(BaseSignal):
    """
    High-fidelity EMG signal generator.

    Supports surface EMG (sEMG) and intramuscular EMG (iEMG) configurations
    with a comprehensive suite of normal and pathological patterns.

    Parameters
    ----------
    config : EMGConfig
        EMG simulation configuration.

    Attributes
    ----------
    config : EMGConfig
        Configuration object.
    params : MUAPParameters
        Pathology-specific MUAP parameters.

    Examples
    --------
    >>> from biosignal_simulator.core.config import EMGConfig
    >>> config = EMGConfig(fs=2000.0, duration_s=5.0, emg_type='surface',
    ...                     pathology='normal', amplitude_uv=500.0)
    >>> gen = EMGGenerator(config)
    >>> emg = gen.generate()
    >>> emg.shape
    (10000,)

    >>> # Intramuscular EMG with ALS pathology
    >>> config_als = EMGConfig(fs=10000.0, duration_s=3.0, emg_type='intramuscular',
    ...                         pathology='als', amplitude_uv=1000.0)
    >>> gen_als = EMGGenerator(config_als)
    >>> emg_als = gen_als.generate()
    """

    def __init__(self, config: EMGConfig) -> None:
        config.__post_init__()
        super().__init__(
            fs=config.fs,
            duration_s=config.duration_s,
            seed=config.seed,
        )
        self.config = config
        self.params = PATHOLOGY_PARAMS.get(
            config.pathology.lower(),
            PATHOLOGY_PARAMS["normal"],
        )

    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate EMG configuration parameters.

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message).
        """
        try:
            self.config.__post_init__()
            if self.config.pathology.lower() not in PATHOLOGY_PARAMS:
                return False, f"Unknown pathology: {self.config.pathology}"
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def generate(self) -> np.ndarray:
        """
        Generate the EMG signal.

        Returns
        -------
        np.ndarray
            EMG signal array of shape (n_samples,).
        """
        emg_type = self.config.emg_type.lower()
        if emg_type == "intramuscular":
            return self._generate_iemg()
        elif emg_type == "surface":
            return self._generate_semg()
        else:
            raise ValueError(f"Unknown EMG type '{emg_type}'. Use 'surface' or 'intramuscular'.")

    def _generate_iemg(self) -> np.ndarray:
        """
        Generate intramuscular EMG by superposing discrete MUAP spike trains.

        Creates n_motor_units concurrent MUAPTs with slight amplitude and
        timing variation between units, then superimposes them.

        Returns
        -------
        np.ndarray
            iEMG signal of shape (n_samples,).
        """
        n_samples = self.n_samples
        pathology = self.config.pathology.lower()
        params = self.params
        base_amp = self.config.amplitude_uv

        # Build inter-unit spike train generator
        spike_gen = MotorUnitSpikeTrainGenerator(
            rng=self.rng,
            pathology=pathology,
            base_amplitude_uv=base_amp,
            duration_s=self.duration_s,
            fs=self.fs,
        )

        # 1. Generate multiple motor unit spike trains
        n_units = params.n_motor_units
        signal = np.zeros(n_samples)

        for unit_idx in range(n_units):
            # Vary firing rate and amplitude across motor units
            fr_variation = self.rng.uniform(0.70, 1.30)
            amp_variation = self.rng.uniform(0.60, 1.50)

            unit_params = MUAPParameters(
                firing_rate_hz=params.firing_rate_hz * fr_variation,
                amplitude_scale=params.amplitude_scale * amp_variation,
                duration_ms=params.duration_ms * self.rng.uniform(0.85, 1.20),
                shape=params.shape,
                jitter_fraction=params.jitter_fraction,
                fatigue_tau_s=params.fatigue_tau_s,
            )

            unit_signal = spike_gen.generate_spike_train(unit_params)
            signal += unit_signal

        # 2. Inject spontaneous fibrillation potentials (denervation)
        if params.has_fibrillations:
            signal = self._inject_fibrillations(signal)

        # 3. Inject fasciculation potentials (spontaneous motor unit discharges)
        if params.has_fasciculations:
            signal = self._inject_fasciculations(signal)

        # 4. Add background electrical noise (physiological + electrode)
        noise_level = 0.03 * base_amp
        signal += self.rng.normal(0.0, noise_level, size=n_samples)

        return signal - np.mean(signal)

    def _generate_semg(self) -> np.ndarray:
        """
        Generate surface EMG as amplitude-modulated bandpass-filtered noise.

        The surface EMG interference pattern is well-approximated as filtered
        Gaussian noise modulated by the voluntary contraction envelope.

        The spectral content depends on the pathology (shifted low for neuropathy,
        high for myopathy).

        Returns
        -------
        np.ndarray
            sEMG signal of shape (n_samples,).
        """
        fs = self.fs
        n_samples = self.n_samples
        t = self.t
        pathology = self.config.pathology.lower()

        # 1. Build voluntary activation envelope
        envelope = build_emg_envelope(
            t=t,
            env_type=self.config.envelope_type.lower(),
            contraction_level=self.config.contraction_level,
            ramp_duration_s=self.config.ramp_duration_s,
            burst_rate_hz=self.config.burst_rate_hz,
            burst_duration_s=self.config.burst_duration_s,
            burst_amplitude=self.config.burst_amplitude,
        )

        # 2. Apply pathology-specific envelope modulations
        envelope = self._apply_envelope_pathology_modulation(envelope, t, pathology)

        # 3. Compute bandpass filter cutoffs (pathology-dependent)
        low, high = get_emg_bandpass_params(
            pathology=pathology,
            fmin_hz=self.config.fmin_hz,
            fmax_hz=self.config.fmax_hz,
            fs=fs,
        )

        # 4. Generate and filter Gaussian noise
        raw_noise = self.rng.normal(0.0, 1.0, size=n_samples)
        nyq = 0.5 * fs
        low_n = max(0.01, low) / nyq
        high_n = min(high, nyq - 0.01) / nyq

        if low_n < high_n:
            b, a = sp_signal.butter(4, [low_n, high_n], btype="bandpass")
            filtered_noise = sp_signal.filtfilt(b, a, raw_noise)
        else:
            filtered_noise = raw_noise

        # 5. Normalize filtered noise to unit RMS
        filtered_noise = normalize_to_rms(filtered_noise, 1.0)

        # 6. Compute target amplitude (pathology-dependent)
        target_amp = self._get_target_amplitude(pathology)

        # 7. Apply envelope × filtered noise × target amplitude
        emg = filtered_noise * envelope * target_amp

        # 8. Inject spontaneous potentials (fibrillations, fasciculations)
        if self.params.has_fibrillations:
            emg = self._inject_fibrillations(emg)
        if self.params.has_fasciculations:
            emg = self._inject_fasciculations(emg)

        # 9. Inject muscle fatigue spectral shift (if fatigue pathology)
        if pathology == "fatigue":
            emg = self._apply_fatigue_spectral_shift(emg, fs)

        return emg - np.mean(emg)

    def _apply_envelope_pathology_modulation(
        self, envelope: np.ndarray, t: np.ndarray, pathology: str
    ) -> np.ndarray:
        """
        Apply pathology-specific modulations to the contraction envelope.

        Parameters
        ----------
        envelope : np.ndarray
            Raw voluntary contraction envelope.
        t : np.ndarray
            Time axis.
        pathology : str
            Pathology type.

        Returns
        -------
        np.ndarray
            Modified envelope.
        """
        if pathology == "parkinsons_tremor":
            # Modulate envelope with 4-6 Hz tremor oscillation (resting tremor)
            tremor_freq = self.rng.uniform(4.0, 6.0)
            tremor_mod = 0.5 * (1.0 + np.sin(2.0 * np.pi * tremor_freq * t))
            envelope = envelope * (0.25 + 0.75 * tremor_mod)

        elif pathology == "myasthenia_gravis":
            # Decrement: amplitude decays exponentially with time
            fatigue_decay = np.exp(-t / 4.0)
            envelope = envelope * fatigue_decay

        elif pathology == "lambert_eaton":
            # Initial amplitude increment during first 2 s, then fatigue
            increment = np.where(t < 2.0, 1.0 + 2.0 * (t / 2.0), 3.0)
            fatigue = np.where(t >= 2.0, np.exp(-(t - 2.0) / 8.0), 1.0)
            envelope = envelope * increment * fatigue

        elif pathology == "muscular_dystrophy":
            # Rapid early fatigue + irregular amplitude fluctuations
            fatigue_decay = np.exp(-t / 2.5)
            noise_mod = 1.0 + 0.15 * self.rng.normal(0.0, 1.0, size=len(t))
            envelope = envelope * fatigue_decay * np.abs(noise_mod)

        elif pathology == "fatigue":
            # Gradual fatigue over the recording
            fatigue_decay = np.exp(-t / 5.0)
            envelope = envelope * (0.4 + 0.6 * fatigue_decay)

        return envelope

    def _get_target_amplitude(self, pathology: str) -> float:
        """
        Get the target RMS amplitude for the sEMG based on pathology.

        Parameters
        ----------
        pathology : str
            Pathology type.

        Returns
        -------
        float
            Target amplitude in microvolts.
        """
        base = self.config.amplitude_uv
        params = self.params

        if pathology in {"neuropathic", "als"}:
            return base * 2.5  # Giant MUAPs → high amplitude
        elif pathology == "peripheral_neuropathy":
            return base * 2.0
        elif pathology in {"myopathic", "muscular_dystrophy"}:
            return base * 0.35  # Small MUAPs → low amplitude
        elif pathology == "myasthenia_gravis":
            return base * 0.85  # Slight initial reduction
        else:
            return base * params.amplitude_scale

    def _inject_fibrillations(self, signal: np.ndarray) -> np.ndarray:
        """
        Inject spontaneous fibrillation potentials into the EMG signal.

        Parameters
        ----------
        signal : np.ndarray
            EMG signal to modify in-place.

        Returns
        -------
        np.ndarray
            Modified signal with fibrillations.
        """
        fs = self.fs
        n_samples = len(signal)
        t = np.arange(n_samples) / fs

        # Fibrillations at 2-15 Hz (very regular)
        fib_rate = self.rng.uniform(4.0, 12.0)
        mean_isi = 1.0 / fib_rate
        sigma_isi = 0.05 * mean_isi  # Very regular

        curr_t = mean_isi / 2.0
        fib_amp = self.config.amplitude_uv * self.rng.uniform(0.08, 0.20)

        while curr_t < self.duration_s:
            fib_amp_k = fib_amp * self.rng.uniform(0.85, 1.15)
            half_w = 0.002  # ±2 ms
            start_idx = max(0, int((curr_t - half_w) * fs))
            end_idx = min(n_samples, int((curr_t + half_w) * fs))
            t_local = t[start_idx:end_idx] - curr_t
            fib = generate_muap_fibrillation(t_local, fib_amp_k)
            signal[start_idx:end_idx] += fib
            curr_t += self.rng.normal(mean_isi, sigma_isi)

        return signal

    def _inject_fasciculations(self, signal: np.ndarray) -> np.ndarray:
        """
        Inject spontaneous fasciculation potentials into the EMG signal.

        Fasciculations are brief, involuntary discharges of a whole motor unit
        (not a single fiber). They appear as large, brief MUAPs at slow,
        irregular rates (0.5-5 Hz).

        Parameters
        ----------
        signal : np.ndarray
            EMG signal to modify.

        Returns
        -------
        np.ndarray
            Modified signal with fasciculations.
        """
        fs = self.fs
        n_samples = len(signal)
        t = np.arange(n_samples) / fs

        # Fasciculation rate: 0.5-3 Hz, very irregular (Poisson process)
        fasc_rate = self.rng.uniform(0.5, 2.0)
        n_fasc = self.rng.poisson(fasc_rate * self.duration_s)
        if n_fasc == 0 and self.duration_s > 1.0:
            n_fasc = 1

        fasc_times = self.rng.uniform(0.1, max(0.2, self.duration_s - 0.1), size=n_fasc)

        for t_f in fasc_times:
            # Fasciculation: large (1.5-4x normal) triphasic MUAP
            fasc_amp = self.config.amplitude_uv * self.rng.uniform(1.2, 3.0)
            fasc_dur = self.rng.uniform(8.0, 16.0)  # ms
            shape = self.rng.choice(["triphasic", "biphasic"])

            half_w = (fasc_dur / 1000.0) * 2.5
            start_idx = max(0, int((t_f - half_w) * fs))
            end_idx = min(n_samples, int((t_f + half_w) * fs))
            t_local = t[start_idx:end_idx] - t_f

            fasc_val = generate_muap(t_local, fasc_amp, fasc_dur, shape, self.rng)
            signal[start_idx:end_idx] += fasc_val

        return signal

    def _apply_fatigue_spectral_shift(
        self, emg: np.ndarray, fs: float
    ) -> np.ndarray:
        """
        Apply progressive spectral compression to simulate muscle fatigue.

        As fatigue develops, the median frequency of the EMG power spectrum
        decreases due to slowed muscle fiber conduction velocity and changes
        in motor unit recruitment. We model this by applying a time-varying
        lowpass filter with decreasing cutoff over time.

        Parameters
        ----------
        emg : np.ndarray
            EMG signal.
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Fatigue-modulated EMG signal.
        """
        n_samples = len(emg)
        t = np.arange(n_samples) / fs
        result = np.zeros(n_samples)

        # Process in overlapping windows with linearly decreasing cutoff
        win_size = int(0.5 * fs)  # 500 ms windows
        hop = win_size // 2
        nyq = 0.5 * fs

        for i in range(0, n_samples - win_size, hop):
            t_center = t[i + win_size // 2]
            progress = t_center / max(self.duration_s, 1.0)

            # Median frequency decreases from ~100 Hz to ~60 Hz
            fcut = max(40.0, 100.0 - 40.0 * progress)
            wn = min(fcut / nyq, 0.95)
            b, a = sp_signal.butter(2, wn, btype="lowpass")

            win = emg[i:i + win_size]
            filtered_win = sp_signal.lfilter(b, a, win)

            # Window blending
            window_fn = np.hanning(win_size)
            result[i:i + win_size] += filtered_win * window_fn

        return result

    def generate_high_density_semg(
        self, n_channels: int = 64, inter_electrode_distance_mm: float = 8.0
    ) -> np.ndarray:
        """
        Generate a high-density sEMG array (HD-EMG).

        High-density EMG uses a 2D grid of electrodes (e.g., 8×8 = 64 electrodes)
        placed over the muscle. Signals are spatially correlated based on inter-
        electrode distance.

        Parameters
        ----------
        n_channels : int
            Number of electrodes. Default: 64.
        inter_electrode_distance_mm : float
            Distance between adjacent electrodes (mm). Default: 8 mm.

        Returns
        -------
        np.ndarray
            HD-EMG array of shape (n_channels, n_samples).
        """
        # Generate base single-channel EMG
        base_emg = self.generate()

        # Build spatially correlated version using exponential distance decay
        channels = np.zeros((n_channels, len(base_emg)))

        # Correlation decays exponentially with distance
        # λ = decay constant (spatial) — typically ~20-40 mm for muscle EMG
        lambda_mm = 25.0
        for c in range(n_channels):
            dist_mm = c * inter_electrode_distance_mm
            correlation = np.exp(-dist_mm / lambda_mm)

            # Individual noise per channel (uncorrelated component)
            noise = self.rng.normal(0.0, self.config.amplitude_uv * 0.15, size=len(base_emg))

            channels[c] = correlation * base_emg + np.sqrt(1.0 - correlation ** 2) * noise

        return channels

    def compute_emg_features(self) -> Dict[str, float]:
        """
        Compute standard EMG feature metrics on the generated signal.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'rms_uv': RMS amplitude
            - 'mean_abs_value': Mean absolute value (MAV)
            - 'zero_crossing_rate': ZCR per second
            - 'waveform_length': Total waveform length (sum of abs differences)
            - 'median_freq_hz': Median frequency from PSD
            - 'mean_freq_hz': Mean (centroid) frequency from PSD
            - 'slope_sign_changes': SSC per second
        """
        emg = self.generate()
        fs = self.fs
        n = len(emg)

        rms = float(np.sqrt(np.mean(emg ** 2)))
        mav = float(np.mean(np.abs(emg)))
        wl = float(np.sum(np.abs(np.diff(emg))))

        # Zero crossing rate
        zc = np.sum(np.diff(np.sign(emg)) != 0)
        zcr = float(zc / self.duration_s)

        # Slope sign changes
        diff_emg = np.diff(emg)
        ssc = np.sum(np.diff(np.sign(diff_emg)) != 0)
        ssc_rate = float(ssc / self.duration_s)

        # PSD (Welch)
        nperseg = min(256, n // 4)
        freqs, psd = sp_signal.welch(emg, fs=fs, nperseg=nperseg)
        total_power = np.sum(psd)

        if total_power > 0:
            # Mean frequency (centroid)
            mean_freq = float(np.sum(freqs * psd) / total_power)
            # Median frequency (50th percentile of cumulative power)
            cum_power = np.cumsum(psd)
            median_idx = np.searchsorted(cum_power, 0.5 * cum_power[-1])
            median_freq = float(freqs[min(median_idx, len(freqs) - 1)])
        else:
            mean_freq = 0.0
            median_freq = 0.0

        return {
            "rms_uv": rms,
            "mean_abs_value_uv": mav,
            "waveform_length": wl,
            "zero_crossing_rate_hz": zcr,
            "slope_sign_changes_hz": ssc_rate,
            "median_freq_hz": median_freq,
            "mean_freq_hz": mean_freq,
            "pathology": self.config.pathology,
        }

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary of the EMG generator configuration.

        Returns
        -------
        Dict[str, Any]
            Configuration summary.
        """
        return {
            "emg_type": self.config.emg_type,
            "pathology": self.config.pathology,
            "amplitude_uv": self.config.amplitude_uv,
            "fmin_hz": self.config.fmin_hz,
            "fmax_hz": self.config.fmax_hz,
            "envelope_type": self.config.envelope_type,
            "contraction_level": self.config.contraction_level,
            "fs_hz": self.fs,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
            "seed": self.config.seed,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Factory Functions
# ──────────────────────────────────────────────────────────────────────────────

def make_normal_semg(
    duration_s: float = 5.0,
    amplitude_uv: float = 500.0,
    fs: float = 2000.0,
    contraction_level: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a normal surface EMG signal at constant contraction.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    amplitude_uv : float
        Target RMS amplitude in microvolts.
    fs : float
        Sampling frequency in Hz.
    contraction_level : float
        Voluntary contraction level (0-1).
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Normal sEMG signal.
    """
    config = EMGConfig(
        fs=fs, duration_s=duration_s, emg_type="surface",
        pathology="normal", amplitude_uv=amplitude_uv,
        contraction_level=contraction_level, seed=seed,
    )
    return EMGGenerator(config).generate()


def make_neuropathic_iemg(
    duration_s: float = 3.0,
    amplitude_uv: float = 1000.0,
    fs: float = 10000.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a neuropathic intramuscular EMG (giant MUAPs).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    amplitude_uv : float
        Baseline amplitude.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Neuropathic iEMG signal.
    """
    config = EMGConfig(
        fs=fs, duration_s=duration_s, emg_type="intramuscular",
        pathology="neuropathic", amplitude_uv=amplitude_uv, seed=seed,
    )
    return EMGGenerator(config).generate()


def make_myopathic_iemg(
    duration_s: float = 3.0,
    amplitude_uv: float = 500.0,
    fs: float = 10000.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a myopathic intramuscular EMG (small, polyphasic MUAPs).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    amplitude_uv : float
        Baseline amplitude.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Myopathic iEMG signal.
    """
    config = EMGConfig(
        fs=fs, duration_s=duration_s, emg_type="intramuscular",
        pathology="myopathic", amplitude_uv=amplitude_uv, seed=seed,
    )
    return EMGGenerator(config).generate()


def make_als_iemg(
    duration_s: float = 3.0,
    amplitude_uv: float = 1000.0,
    fs: float = 10000.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an ALS-like intramuscular EMG with fasciculations.

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    amplitude_uv : float
        Baseline amplitude.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        ALS-like iEMG signal.
    """
    config = EMGConfig(
        fs=fs, duration_s=duration_s, emg_type="intramuscular",
        pathology="als", amplitude_uv=amplitude_uv, seed=seed,
    )
    return EMGGenerator(config).generate()


def make_parkinsons_tremor_semg(
    duration_s: float = 5.0,
    amplitude_uv: float = 500.0,
    fs: float = 2000.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Parkinson's resting tremor surface EMG (4-6 Hz bursting).

    Parameters
    ----------
    duration_s : float
        Duration in seconds.
    amplitude_uv : float
        Baseline amplitude.
    fs : float
        Sampling frequency in Hz.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Tremor sEMG signal.
    """
    config = EMGConfig(
        fs=fs, duration_s=duration_s, emg_type="surface",
        pathology="parkinsons_tremor", amplitude_uv=amplitude_uv,
        envelope_type="constant", contraction_level=0.3, seed=seed,
    )
    return EMGGenerator(config).generate()


def make_ramp_emg(
    duration_s: float = 10.0,
    amplitude_uv: float = 500.0,
    fs: float = 2000.0,
    pathology: str = "normal",
    ramp_duration_s: float = 2.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a ramp EMG with gradual contraction buildup.

    Parameters
    ----------
    duration_s : float
        Duration.
    amplitude_uv : float
        Amplitude.
    fs : float
        Sampling frequency.
    pathology : str
        Pathology type.
    ramp_duration_s : float
        Duration of rise/fall ramp.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Ramp-envelope EMG signal.
    """
    config = EMGConfig(
        fs=fs, duration_s=duration_s, emg_type="surface",
        pathology=pathology, amplitude_uv=amplitude_uv,
        envelope_type="ramp", contraction_level=1.0,
        ramp_duration_s=ramp_duration_s, seed=seed,
    )
    return EMGGenerator(config).generate()


def batch_generate_emg_pathologies(
    pathologies: List[str],
    emg_type: str = "intramuscular",
    duration_s: float = 3.0,
    fs: float = 10000.0,
    amplitude_uv: float = 1000.0,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate EMG signals for multiple pathologies.

    Parameters
    ----------
    pathologies : List[str]
        Pathology names (see EMGPathology enum values).
    emg_type : str
        'surface' or 'intramuscular'.
    duration_s : float
        Signal duration.
    fs : float
        Sampling frequency.
    amplitude_uv : float
        Baseline amplitude.
    seed : int, optional
        Base random seed.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping pathology name to signal.

    Examples
    --------
    >>> signals = batch_generate_emg_pathologies(['normal', 'neuropathic', 'myopathic'])
    >>> list(signals.keys())
    ['normal', 'neuropathic', 'myopathic']
    """
    results: Dict[str, np.ndarray] = {}
    for i, path in enumerate(pathologies):
        rng_seed = (seed + i) if seed is not None else None
        config = EMGConfig(
            fs=fs, duration_s=duration_s, emg_type=emg_type,
            pathology=path, amplitude_uv=amplitude_uv, seed=rng_seed,
        )
        try:
            results[path] = EMGGenerator(config).generate()
        except Exception as exc:
            warnings.warn(f"Failed to generate EMG pathology '{path}': {exc}")
    return results
