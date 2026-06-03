"""
Configuration schemas and serialization methods for signals, noise, and wearable conditions.

This module defines:
1. Signal configuration classes for ECG, EEG, EMG, PPG, EDA, and Respiration.
2. Noise configuration classes including Gaussian, Colored, Baseline Wander, Powerline,
   Motion, Electrode, EMG Artifact, Impulse, and Quantization.
3. Crosstalk and wearable condition configurations (Sensor Detachment, Electrode Displacement,
   Light Leakage, and Packet Loss).
4. ConfigSerializer for JSON, YAML, and dict conversions.
5. BenchmarkSuite for automated parameter grid sweep filter evaluation.
"""

import json
import itertools
import warnings
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Callable, Type, TypeVar, Dict, List, Optional, Union, Tuple
import numpy as np
from biosignal_simulator.core.base import ParameterValidationError

T = TypeVar('T')

# =====================================================================
# 1. Signal Configuration Dataclasses
# =====================================================================

@dataclass
class ECGConfig:
    """
    Configuration parameters for standard VCG projected ECG signals.
    
    Attributes
    ----------
    fs : float
        Sampling frequency in Hz (typically [100, 2000]).
    duration_s : float
        Signal length in seconds.
    heart_rate : float
        Target heart rate in beats per minute.
    hr_variability_std : float
        R-to-R interval standard deviation (fraction of RR interval mean).
    p_amplitude : float
        Atrial depolarization peak amplitude in mV.
    qrs_amplitude : float
        Ventricular depolarization peak amplitude in mV.
    t_amplitude : float
        Ventricular repolarization peak amplitude in mV.
    qrs_width : float
        Duration of the QRS complex in seconds.
    pr_interval : float
        PR interval duration in seconds.
    st_elevation : float
        Ventricular baseline shift in mV (ischemia simulation).
    lead_type : str
        Leads configuration: 'single', '12lead', or 'vcg'.
    lead_name : str
        Target lead for 'single' mode (e.g. 'I', 'II', 'V5').
    rhythm_type : str
        Arrhythmia category: 'normal', 'afib', 'pvc', 'vtach', 'bradycardia', 'tachycardia', 'av_block'.
    seed : Optional[int]
        Seed for pseudo-random reproducibility.
    """
    fs: float = 500.0
    duration_s: float = 10.0
    heart_rate: float = 75.0
    hr_variability_std: float = 0.05
    p_amplitude: float = 0.15
    qrs_amplitude: float = 1.0
    t_amplitude: float = 0.35
    qrs_width: float = 0.08
    pr_interval: float = 0.16
    st_elevation: float = 0.0
    lead_type: str = 'single'
    lead_name: str = 'II'
    rhythm_type: str = 'normal'
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate parameter boundaries."""
        errors = []
        if self.fs < 50.0 or self.fs > 5000.0:
            errors.append(f"fs must be between 50.0 and 5000.0 Hz, got {self.fs}")
        if self.duration_s <= 0.0:
            errors.append(f"duration_s must be positive, got {self.duration_s}")
        if self.heart_rate < 40.0 or self.heart_rate > 200.0:
            errors.append(f"heart_rate must be between 40.0 and 200.0 bpm, got {self.heart_rate}")
        if self.hr_variability_std < 0.0 or self.hr_variability_std > 0.5:
            errors.append(f"hr_variability_std must be between 0.0 and 0.5, got {self.hr_variability_std}")
        if self.p_amplitude < 0.0 or self.p_amplitude > 2.0:
            errors.append(f"p_amplitude must be between 0.0 and 2.0 mV, got {self.p_amplitude}")
        if self.qrs_amplitude < 0.3 or self.qrs_amplitude > 3.0:
            errors.append(f"qrs_amplitude must be between 0.3 and 3.0 mV, got {self.qrs_amplitude}")
        if self.t_amplitude < 0.0 or self.t_amplitude > 2.0:
            errors.append(f"t_amplitude must be between 0.0 and 2.0 mV, got {self.t_amplitude}")
        if self.qrs_width < 0.03 or self.qrs_width > 0.25:
            errors.append(f"qrs_width must be between 0.03 and 0.25 s, got {self.qrs_width}")
        if self.pr_interval < 0.08 or self.pr_interval > 0.4:
            errors.append(f"pr_interval must be between 0.08 and 0.4 s, got {self.pr_interval}")
        if abs(self.st_elevation) > 2.0:
            errors.append(f"ST elevation amplitude must be within [-2.0, 2.0] mV, got {self.st_elevation}")
            
        allowed_leads = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
        if self.lead_name not in allowed_leads:
            errors.append(f"lead_name must be one of {allowed_leads}, got {self.lead_name}")
            
        allowed_lead_types = {'single', '12lead', 'vcg'}
        if self.lead_type.lower() not in allowed_lead_types:
            errors.append(f"lead_type must be one of {allowed_lead_types}, got {self.lead_type}")
            
        allowed_rhythms = {
            'normal', 'bradycardia', 'tachycardia', 'afib', 'aflutter',
            'pvc', 'pac', 'vtach', 'vfib', 'av_block', 'wenckebach',
            'complete_av_block', 'rbbb', 'lbbb', 'wpw', 'long_qt', 'stemi', 'ischemia'
        }
        if self.rhythm_type.lower() not in allowed_rhythms:
            errors.append(f"rhythm_type must be one of {allowed_rhythms}, got {self.rhythm_type}")

        if errors:
            raise ValueError(f"ECGConfig Validation Errors:\n" + "\n".join(errors))


@dataclass
class EEGConfig:
    """
    Configuration parameters for multi-channel brainwave simulations.
    
    Attributes
    ----------
    fs : float
        Sampling frequency in Hz (typically [128, 1024]).
    duration_s : float
        Signal length in seconds.
    band_powers : Dict[str, float]
        Dictionary mapping brain rhythm bands to relative power densities.
    background_1f_power : float
        Power weight of the 1/f background noise distribution.
    alpha_peak_hz : float
        Peak frequency of the alpha rhythm in Hz (usually ~10 Hz).
    n_channels : int
        Number of active EEG channels.
    corr_matrix : Optional[List[List[float]]]
        Cross-channel correlation matrix for spatial mixing.
    amplitude_uv : float
        Target standard deviation of signal in microvolts.
    state : str
        Brain state representation: 'active', 'relaxed', 'n2_sleep', 'n3_sleep',
        'tonic_clonic', 'absence', 'epileptiform_spikes'.
    seed : Optional[int]
        Seed for reproducibility.
    """
    fs: float = 256.0
    duration_s: float = 10.0
    band_powers: Dict[str, float] = field(default_factory=lambda: {
        'delta': 0.2, 'theta': 0.3, 'alpha': 1.0, 'beta': 0.5, 'gamma': 0.1
    })
    background_1f_power: float = 0.3
    alpha_peak_hz: float = 10.0
    n_channels: int = 1
    corr_matrix: Optional[List[List[float]]] = None
    amplitude_uv: float = 50.0
    state: str = 'relaxed'
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate EEG parameters."""
        errors = []
        if self.fs < 32.0 or self.fs > 4000.0:
            errors.append(f"fs must be between 32.0 and 4000.0 Hz, got {self.fs}")
        if self.duration_s <= 0.0:
            errors.append(f"duration_s must be positive, got {self.duration_s}")
        if self.background_1f_power < 0.0 or self.background_1f_power > 1.0:
            errors.append(f"background_1f_power must be between 0.0 and 1.0, got {self.background_1f_power}")
        if self.alpha_peak_hz < 6.0 or self.alpha_peak_hz > 14.0:
            errors.append(f"alpha_peak_hz must be in range [6.0, 14.0] Hz, got {self.alpha_peak_hz}")
        if self.n_channels < 1:
            errors.append(f"n_channels must be >= 1, got {self.n_channels}")
        if self.amplitude_uv <= 0.0:
            errors.append(f"amplitude_uv must be positive, got {self.amplitude_uv}")
            
        allowed_states = {'active', 'relaxed', 'n2_sleep', 'n3_sleep', 'tonic_clonic', 'absence', 'epileptiform_spikes'}
        if self.state.lower() not in allowed_states:
            errors.append(f"state must be one of {allowed_states}, got {self.state}")
            
        if self.corr_matrix is not None:
            mat = np.array(self.corr_matrix)
            if mat.shape != (self.n_channels, self.n_channels):
                errors.append(f"corr_matrix shape must match {self.n_channels}x{self.n_channels}, got {mat.shape}")
            if not np.allclose(mat, mat.T):
                errors.append("corr_matrix must be symmetric")
            try:
                np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
                errors.append("corr_matrix must be positive-definite")

        if errors:
            raise ValueError(f"EEGConfig Validation Errors:\n" + "\n".join(errors))


@dataclass
class EMGConfig:
    """
    Configuration parameters for simulated electromyographic activation.
    
    Attributes
    ----------
    fs : float
        Sampling frequency in Hz (typically [500, 4000]).
    duration_s : float
        Signal length in seconds.
    fmin_hz : float
        Lower cutoff frequency for EMG bandpass.
    fmax_hz : float
        Upper cutoff frequency for EMG bandpass.
    envelope_type : str
        Envelope profile: 'constant', 'ramp', or 'burst'.
    contraction_level : float
        Maximum level of activation scale factor [0.0, 1.0].
    amplitude_uv : float
        Target RMS value under maximum activation in microvolts.
    burst_rate_hz : float
        Frequency of contraction cycles in 'burst' mode.
    burst_duration_s : float
        Duration of active envelope bursts.
    burst_amplitude : float
        Envelope scale factor of bursts.
    ramp_duration_s : float
        Contraction build-up/fade duration in 'ramp' mode.
    emg_type : str
        EMG category: 'surface' or 'intramuscular'.
    pathology : str
        Muscle phenotype: 'normal', 'neuropathic', 'myopathic', 'als', 'myasthenia_gravis', 'parkinsons_tremor'.
    seed : Optional[int]
        Reproducibility seed.
    """
    fs: float = 2000.0
    duration_s: float = 10.0
    fmin_hz: float = 20.0
    fmax_hz: float = 500.0
    envelope_type: str = 'constant'
    contraction_level: float = 1.0
    amplitude_uv: float = 500.0
    burst_rate_hz: float = 1.0
    burst_duration_s: float = 0.2
    burst_amplitude: float = 1.0
    ramp_duration_s: float = 2.0
    emg_type: str = 'surface'
    pathology: str = 'normal'
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate EMG parameters."""
        # Auto-adjust defaults for low sampling rates
        nyq = self.fs / 2.0
        if self.fmax_hz >= nyq:
            if self.fmax_hz == 500.0:
                self.fmax_hz = nyq - 0.1
            else:
                raise ValueError(f"fmax_hz must be below Nyquist ({nyq} Hz), got {self.fmax_hz}")
        if self.fmin_hz >= self.fmax_hz:
            if self.fmin_hz == 20.0:
                self.fmin_hz = max(1.0, self.fmax_hz / 10.0)
            else:
                raise ValueError(f"fmin_hz must be less than fmax_hz, got {self.fmin_hz}")

        errors = []
        if self.fs < 100.0 or self.fs > 10000.0:
            errors.append(f"fs must be between 100.0 and 10000.0 Hz, got {self.fs}")
        if self.duration_s <= 0.0:
            errors.append(f"duration_s must be positive, got {self.duration_s}")
        if self.fmin_hz < 1.0 or self.fmin_hz >= self.fmax_hz:
            errors.append(f"fmin_hz must be >= 1.0 and less than fmax_hz, got {self.fmin_hz}")
        if self.fmax_hz >= nyq:
            errors.append(f"fmax_hz must be below Nyquist ({nyq} Hz), got {self.fmax_hz}")
        if self.contraction_level < 0.0 or self.contraction_level > 1.0:
            errors.append(f"contraction_level must be within [0.0, 1.0], got {self.contraction_level}")
        if self.amplitude_uv <= 0.0:
            errors.append(f"amplitude_uv must be positive, got {self.amplitude_uv}")
            
        allowed_envs = {'constant', 'ramp', 'burst'}
        if self.envelope_type.lower() not in allowed_envs:
            errors.append(f"envelope_type must be one of {allowed_envs}, got {self.envelope_type}")
            
        allowed_types = {'surface', 'intramuscular'}
        if self.emg_type.lower() not in allowed_types:
            errors.append(f"emg_type must be one of {allowed_types}, got {self.emg_type}")
            
        allowed_paths = {'normal', 'neuropathic', 'myopathic', 'als', 'myasthenia_gravis', 'parkinsons_tremor'}
        if self.pathology.lower() not in allowed_paths:
            errors.append(f"pathology must be one of {allowed_paths}, got {self.pathology}")

        if errors:
            raise ValueError(f"EMGConfig Validation Errors:\n" + "\n".join(errors))


@dataclass
class PPGConfig:
    """
    Configuration parameters for optical photoplethysmography simulations.
    
    Attributes
    ----------
    fs : float
        Sampling frequency in Hz (typically [50, 500]).
    duration_s : float
        Signal length in seconds.
    heart_rate : float
        Base pulse rate in bpm.
    systolic_fraction : float
        Width of the systolic peak as a fraction of cardiac cycle.
    dicrotic_fraction : float
        Relative amplitude/width scale factor of the dicrotic wave.
    resp_modulation : float
        Respiratory amplitude modulation depth index [0.0, 0.5].
    resp_rate : float
        Respiratory frequency in Hz.
    seed : Optional[int]
        Seed.
    """
    fs: float = 100.0
    duration_s: float = 10.0
    heart_rate: float = 75.0
    systolic_fraction: float = 0.25
    dicrotic_fraction: float = 0.45
    resp_modulation: float = 0.15
    resp_rate: float = 0.25
    derivative: str = 'none'
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate PPG parameters."""
        errors = []
        if self.fs < 10.0 or self.fs > 2000.0:
            errors.append(f"fs must be between 10.0 and 2000.0 Hz, got {self.fs}")
        if self.duration_s <= 0.0:
            errors.append(f"duration_s must be positive, got {self.duration_s}")
        if self.heart_rate < 30.0 or self.heart_rate > 220.0:
            errors.append(f"heart_rate must be between 30.0 and 220.0 bpm, got {self.heart_rate}")
        if self.systolic_fraction <= 0.0 or self.systolic_fraction >= 0.6:
            errors.append(f"systolic_fraction must be in range (0, 0.6), got {self.systolic_fraction}")
        if self.dicrotic_fraction < 0.0 or self.dicrotic_fraction > 1.0:
            errors.append(f"dicrotic_fraction must be in range [0, 1], got {self.dicrotic_fraction}")
        if self.resp_modulation < 0.0 or self.resp_modulation > 0.8:
            errors.append(f"resp_modulation must be in range [0.0, 0.8], got {self.resp_modulation}")
        if self.resp_rate <= 0.0 or self.resp_rate > 2.0:
            errors.append(f"resp_rate must be in range (0.0, 2.0] Hz, got {self.resp_rate}")
        
        allowed_derivs = {'none', 'first', 'second', 'vpg', 'apg'}
        if self.derivative is not None and str(self.derivative).lower().strip() not in allowed_derivs:
            errors.append(f"derivative must be one of {allowed_derivs}, got {self.derivative}")

        if errors:
            raise ValueError(f"PPGConfig Validation Errors:\n" + "\n".join(errors))


@dataclass
class EDAConfig:
    """
    Configuration parameters for Electrodermal Activity (Skin Conductance).
    
    Attributes
    ----------
    fs : float
        Sampling frequency in Hz (typically [4, 64]).
    duration_s : float
        Signal length in seconds.
    scl_amplitude_us : float
        Tonic baseline skin conductance level in microSiemens.
    scl_drift_rate : float
        Tonic random walk drift slope.
    event_rate_hz : float
        Average rate of skin conductance response peaks (Poisson events).
    scr_rise_s : float
        Rise time of phasic responses in seconds.
    scr_decay_s : float
        Decay constant of phasic responses in seconds.
    seed : Optional[int]
        Seed.
    """
    fs: float = 32.0
    duration_s: float = 60.0
    scl_amplitude_us: float = 10.0
    scl_drift_rate: float = 0.01
    event_rate_hz: float = 0.2
    scr_rise_s: float = 1.0
    scr_decay_s: float = 4.0
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate EDA parameters."""
        errors = []
        if self.fs < 1.0 or self.fs > 500.0:
            errors.append(f"fs must be between 1.0 and 500.0 Hz, got {self.fs}")
        if self.duration_s <= 0.0:
            errors.append(f"duration_s must be positive, got {self.duration_s}")
        if self.scl_amplitude_us <= 0.0:
            errors.append(f"scl_amplitude_us must be positive, got {self.scl_amplitude_us}")
        if self.event_rate_hz < 0.0 or self.event_rate_hz > 5.0:
            errors.append(f"event_rate_hz must be in range [0, 5.0] Hz, got {self.event_rate_hz}")
        if self.scr_rise_s <= 0.0 or self.scr_rise_s > 10.0:
            errors.append(f"scr_rise_s must be in range (0, 10.0], got {self.scr_rise_s}")
        if self.scr_decay_s <= self.scr_rise_s or self.scr_decay_s > 30.0:
            errors.append(f"scr_decay_s must be greater than rise time and <= 30.0 s, got {self.scr_decay_s}")

        if errors:
            raise ValueError(f"EDAConfig Validation Errors:\n" + "\n".join(errors))


@dataclass
class RespConfig:
    """
    Configuration parameters for breathing patterns.
    
    Attributes
    ----------
    fs : float
        Sampling frequency in Hz.
    duration_s : float
        Signal length in seconds.
    resp_rate_hz : float
        Base respiration rate in Hz (typically [0.1, 0.6]).
    amplitude : float
        Wave amplitude.
    harmonic_k : float
        Asymmetry ratio between inspiration and expiration.
    phase_noise_std : float
        Standard deviation of breath cycle duration variability.
    seed : Optional[int]
        Seed.
    """
    fs: float = 32.0
    duration_s: float = 60.0
    resp_rate_hz: float = 0.25
    amplitude: float = 1.0
    harmonic_k: float = 0.3
    phase_noise_std: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate Respiration parameters."""
        errors = []
        if self.fs < 2.0 or self.fs > 1000.0:
            errors.append(f"fs must be between 2.0 and 1000.0 Hz, got {self.fs}")
        if self.duration_s <= 0.0:
            errors.append(f"duration_s must be positive, got {self.duration_s}")
        if self.resp_rate_hz < 0.05 or self.resp_rate_hz > 2.0:
            errors.append(f"resp_rate_hz must be in range [0.05, 2.0] Hz, got {self.resp_rate_hz}")
        if self.amplitude <= 0.0:
            errors.append(f"amplitude must be positive, got {self.amplitude}")
        if self.harmonic_k < 0.0 or self.harmonic_k > 0.9:
            errors.append(f"harmonic_k must be within [0, 0.9], got {self.harmonic_k}")
        if self.phase_noise_std < 0.0 or self.phase_noise_std > 0.5:
            errors.append(f"phase_noise_std must be in range [0, 0.5], got {self.phase_noise_std}")

        if errors:
            raise ValueError(f"RespConfig Validation Errors:\n" + "\n".join(errors))


# =====================================================================
# 2. Additive and Environmental Noise Configuration Dataclasses
# =====================================================================

@dataclass
class GaussianNoiseConfig:
    """AWGN parameters."""
    std: float = 1.0
    mean: float = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.std < 0.0:
            raise ValueError(f"std must be non-negative, got {self.std}")


@dataclass
class ColoredNoiseConfig:
    """1/f^exponent spectral noise parameters."""
    exponent: float = 1.0
    std: float = 1.0
    method: str = 'fft'
    seed: Optional[int] = None

    def __post_init__(self):
        if self.std < 0.0:
            raise ValueError("std must be non-negative")
        if self.method not in {'fft', 'voss', 'iir'}:
            raise ValueError("method must be 'fft', 'voss', or 'iir'")


@dataclass
class BaselineWanderConfig:
    """Baseline wander parameters."""
    amplitude: float = 0.1
    f_resp_hz: float = 0.25
    resp_fraction: float = 0.6
    drift_fraction: float = 0.3
    trend_fraction: float = 0.1
    trend_degree: int = 1
    f_resp_variation: float = 0.02
    seed: Optional[int] = None

    def __post_init__(self):
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")
        if self.f_resp_hz <= 0.0:
            raise ValueError("f_resp_hz must be positive")
        total = self.resp_fraction + self.drift_fraction + self.trend_fraction
        if not np.isclose(total, 1.0):
            # B-18 FIX: warn the caller instead of silently mutating their values
            warnings.warn(
                f"BaselineWanderConfig fractions sum to {total:.6f} (expected 1.0). "
                f"Automatically normalising: resp={self.resp_fraction/total:.4f}, "
                f"drift={self.drift_fraction/total:.4f}, "
                f"trend={self.trend_fraction/total:.4f}. "
                "Pass fractions that sum to 1.0 to avoid this normalisation.",
                UserWarning,
                stacklevel=2,
            )
            self.resp_fraction /= total
            self.drift_fraction /= total
            self.trend_fraction /= total


@dataclass
class PowerlineNoiseConfig:
    """Line noise interference (50/60 Hz)."""
    f_line_hz: float = 50.0
    n_harmonics: int = 3
    amplitude: float = 0.05
    harmonic_decay: float = 1.0
    freq_std_hz: float = 0.1
    amplitude_mod_depth: float = 0.1
    phase_drift_std: float = 0.02
    seed: Optional[int] = None

    def __post_init__(self):
        if self.f_line_hz not in {50.0, 60.0}:
            # Keep as warning or validation boundary
            pass
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")
        if self.n_harmonics < 1:
            raise ValueError("n_harmonics must be at least 1")


@dataclass
class MotionArtifactConfig:
    """Motion artifacts configuration."""
    lf_amplitude: float = 0.2
    lf_fmin_hz: float = 0.1
    lf_fmax_hz: float = 10.0
    impact_rate_hz: float = 0.1
    impact_amplitude: float = 1.0
    impact_decay_s: float = 0.2
    impact_freq_hz: float = 20.0
    cable_amplitude: float = 0.1
    enable_lf: bool = True
    enable_impacts: bool = False
    enable_cable: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        if self.lf_amplitude < 0.0 or self.impact_amplitude < 0.0 or self.cable_amplitude < 0.0:
            raise ValueError("Amplitudes must be non-negative")


@dataclass
class ElectrodeNoiseConfig:
    """Loose contact or thermal electrode contact noise."""
    enable_popcorn: bool = True
    popcorn_amplitude: float = 0.05
    popcorn_rate_hz: float = 5.0
    enable_impedance_noise: bool = True
    impedance_ohms: float = 5000.0
    temperature_k: float = 310.0
    bandwidth_hz: Optional[float] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.popcorn_amplitude < 0.0 or self.impedance_ohms < 0.0:
            raise ValueError("Impedance and amplitudes must be positive")


@dataclass
class EMGArtifactConfig:
    """EMG pollution on other records."""
    fmin_hz: float = 20.0
    fmax_hz: float = 500.0
    burst_rate_hz: float = 2.0
    burst_duration_s: float = 0.3
    amplitude_fraction: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.amplitude_fraction < 0.0 or self.amplitude_fraction > 1.0:
            raise ValueError("amplitude_fraction must be in range [0, 1]")


@dataclass
class ImpulseNoiseConfig:
    """Impulsive popcorn spike noise."""
    rate_hz: float = 1.0
    amplitude_scale: float = 2.0
    amplitude_shape: float = 0.5
    pulse_width_s: float = 0.0
    polarity: str = 'bipolar'
    seed: Optional[int] = None

    def __post_init__(self):
        if self.rate_hz < 0.0:
            raise ValueError("rate_hz must be non-negative")
        if self.polarity not in {'bipolar', 'positive', 'negative'}:
            raise ValueError("polarity must be 'bipolar', 'positive', or 'negative'")


@dataclass
class QuantizationNoiseConfig:
    """ADC converter discretization parameters."""
    n_bits: int = 12
    v_range: float = 5.0
    dither: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        if self.n_bits < 4 or self.n_bits > 32:
            raise ValueError("n_bits must be between 4 and 32")
        if self.v_range <= 0.0:
            raise ValueError("v_range must be positive")


# =====================================================================
# 3. Crosstalk and Wearable Condition Configuration Dataclasses
# =====================================================================

@dataclass
class CrosstalkNoiseConfig:
    """
    Configuration parameters for Physiological Crosstalk/leakage simulation.
    
    Attributes
    ----------
    coupling_factor : float
        Ratio of amplitude bleeding in.
    source_type : str
        Leakage signal source: 'ecg', 'resp', 'emg', 'ppg', 'eda', 'eeg'.
    source_config : Optional[Any]
        Custom configuration object for the source signal generator.
    seed : Optional[int]
        Seed.
    """
    coupling_factor: float = 0.1
    source_type: str = "ecg"
    source_config: Optional[Any] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.coupling_factor < 0.0:
            raise ValueError("coupling_factor must be non-negative")
        allowed_sources = {"ecg", "resp", "emg", "ppg", "eda", "eeg"}
        if self.source_type.lower() not in allowed_sources:
            raise ValueError(f"source_type must be one of {allowed_sources}, got {self.source_type}")


@dataclass
class SensorDetachmentConfig:
    """
    Configuration parameters for sudden sensor detachment.
    
    Attributes
    ----------
    detachment_time_s : float
        Time in seconds when sensor detachment event occurs.
    transient_duration_s : float
        Duration of the initial high-voltage spike transient.
    transient_amplitude : float
        Amplitude of the offset bounce spike.
    noise_level_uv : float
        Amplifier/thermal flatline background noise std in microvolts.
    seed : Optional[int]
        Seed.
    """
    detachment_time_s: float = 5.0
    transient_duration_s: float = 0.2
    transient_amplitude: float = 5.0
    noise_level_uv: float = 10.0
    seed: Optional[int] = None

    def __post_init__(self):
        if self.detachment_time_s < 0.0:
            raise ValueError("detachment_time_s must be non-negative")
        if self.transient_duration_s < 0.0:
            raise ValueError("transient_duration_s must be non-negative")
        if self.noise_level_uv < 0.0:
            raise ValueError("noise_level_uv must be non-negative")


@dataclass
class ElectrodeDisplacementConfig:
    """
    Configuration parameters for loose contacts causing displacement jumps.
    
    Attributes
    ----------
    displacement_times : List[float]
        Time indices of sudden sensor shifts.
    shift_amplitudes : List[float]
        Offset shifts in physical units added to the signal at each event.
    noise_increments : List[float]
        Relative noise standard deviation multiplier increments following events.
    seed : Optional[int]
        Seed.
    """
    displacement_times: List[float] = field(default_factory=lambda: [3.0, 7.0])
    shift_amplitudes: List[float] = field(default_factory=lambda: [0.5, -0.3])
    noise_increments: List[float] = field(default_factory=lambda: [2.0, 1.5])
    seed: Optional[int] = None

    def __post_init__(self):
        if len(self.displacement_times) != len(self.shift_amplitudes) or len(self.displacement_times) != len(self.noise_increments):
            raise ValueError("displacement_times, shift_amplitudes, and noise_increments lists must be of equal size")
        for t in self.displacement_times:
            if t < 0.0:
                raise ValueError("displacement times must be non-negative")


@dataclass
class LightLeakageConfig:
    """
    PPG Ambient light leakage parameters.
    
    Attributes
    ----------
    leakage_amplitude : float
        Base magnitude of light leak interference.
    modulation_frequency_hz : float
        Frequency of motion/coupling modulation (e.g. respiration/movement).
    f_line_hz : float
        Line frequency of indoor light flicker (50 Hz or 60 Hz).
    harmonic_leakage : float
        Relative amplitude of light harmonics.
    seed : Optional[int]
        Seed.
    """
    leakage_amplitude: float = 0.2
    modulation_frequency_hz: float = 0.25
    f_line_hz: float = 50.0
    harmonic_leakage: float = 0.05
    seed: Optional[int] = None

    def __post_init__(self):
        if self.leakage_amplitude < 0.0:
            raise ValueError("leakage_amplitude must be non-negative")
        if self.modulation_frequency_hz < 0.0:
            raise ValueError("modulation_frequency_hz must be non-negative")


@dataclass
class PacketLossConfig:
    """
    Simulated wireless transmission dropout parameters.
    
    Attributes
    ----------
    loss_rate : float
        Probability of dropping a data segment [0.0, 1.0].
    burst_length_samples : int
        Average length of consecutive sample losses.
    interpolation_mode : str
        Method to fill dropped frames: 'zero', 'hold', or 'linear'.
    seed : Optional[int]
        Seed.
    """
    loss_rate: float = 0.05
    burst_length_samples: int = 5
    interpolation_mode: str = 'zero'
    seed: Optional[int] = None

    def __post_init__(self):
        if self.loss_rate < 0.0 or self.loss_rate > 1.0:
            raise ValueError("loss_rate must be between 0.0 and 1.0")
        if self.burst_length_samples < 1:
            raise ValueError("burst_length_samples must be at least 1")
        allowed_modes = {'zero', 'hold', 'linear', 'cubic', 'spline'}
        if self.interpolation_mode.lower() not in allowed_modes:
            raise ValueError(f"interpolation_mode must be one of {allowed_modes}, got {self.interpolation_mode}")


# =====================================================================
# 3.5. Clinical Presets and Sweep Utility Functions
# =====================================================================

def sweep_config(config: T, param_name: str, values: List[Any]) -> List[T]:
    """
    Generate a list of configuration objects with a single parameter swept across values.
    
    Parameters
    ----------
    config : T
        The base configuration dataclass instance.
    param_name : str
        The name of the parameter to sweep.
    values : List[Any]
        The values to assign to the parameter.
        
    Returns
    -------
    List[T]
        A list of configuration copies with the swept parameter applied.
    """
    import copy
    if not hasattr(config, param_name):
        raise ValueError(f"Configuration {config.__class__.__name__} has no attribute '{param_name}'")
    swept = []
    errors: List[Tuple[Any, Exception]] = []
    import copy
    for val in values:
        cfg_copy = copy.deepcopy(config)
        setattr(cfg_copy, param_name, val)
        # Re-trigger validation, but don't abort the whole sweep on one bad value
        if hasattr(cfg_copy, '__post_init__'):
            try:
                cfg_copy.__post_init__()
            except Exception as exc:
                # B-12 FIX: collect error and skip this value rather than raising
                warnings.warn(
                    f"sweep_config: value {val!r} for '{param_name}' failed validation "
                    f"and was skipped: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                errors.append((val, exc))
                continue
        swept.append(cfg_copy)
    if not swept and errors:
        raise ValueError(
            f"sweep_config: all {len(errors)} value(s) failed validation for '{param_name}'. "
            f"First error: {errors[0][1]}"
        )
    return swept


class ClinicalPresets:
    """Clinical configurations presets for standard biosignal simulations."""
    
    @staticmethod
    def get_normal_ecg(fs: float = 500.0, duration_s: float = 10.0, heart_rate: float = 72.0) -> ECGConfig:
        """Get ECGConfig with standard sinus rhythm parameters."""
        return ECGConfig(
            fs=fs,
            duration_s=duration_s,
            heart_rate=heart_rate,
            hr_variability_std=0.04,
            lead_type='12lead',
            rhythm_type='normal'
        )

    @staticmethod
    def get_afib_ecg(fs: float = 500.0, duration_s: float = 10.0) -> ECGConfig:
        """
        Get ECGConfig for Atrial Fibrillation.
        
        AFib is characterized by:
        - Irregularly irregular ventricular rate (high RR variability).
        - Absence of distinct P-waves (replaced by high-frequency f-waves).
        - Tachycardia (elevated heart rate).
        """
        return ECGConfig(
            fs=fs,
            duration_s=duration_s,
            heart_rate=115.0,
            hr_variability_std=0.22,
            p_amplitude=0.0,  # No P-wave
            lead_type='12lead',
            rhythm_type='afib'
        )

    @staticmethod
    def get_pvc_ecg(fs: float = 500.0, duration_s: float = 10.0) -> ECGConfig:
        """
        Get ECGConfig with Premature Ventricular Contractions (PVC).
        
        PVCs are characterized by:
        - Ectopic beats originating in the ventricles.
        - Wide, bizarre QRS complexes.
        - Compensatory pauses following the ectopic beat.
        """
        return ECGConfig(
            fs=fs,
            duration_s=duration_s,
            heart_rate=80.0,
            hr_variability_std=0.05,
            lead_type='12lead',
            rhythm_type='pvc'
        )

    @staticmethod
    def get_vtach_ecg(fs: float = 500.0, duration_s: float = 10.0) -> ECGConfig:
        """
        Get ECGConfig for Ventricular Tachycardia (VTach).
        
        VTach is a life-threatening arrhythmia characterized by:
        - Rapid heart rate (>150 bpm).
        - Broad QRS complexes.
        - Complete absence of P-waves.
        """
        return ECGConfig(
            fs=fs,
            duration_s=duration_s,
            heart_rate=170.0,
            hr_variability_std=0.01,  # Very regular rapid pacing
            p_amplitude=0.0,
            lead_type='12lead',
            rhythm_type='vtach'
        )

    @staticmethod
    def get_sleep_eeg(stage: str, fs: float = 256.0, duration_s: float = 30.0) -> EEGConfig:
        """
        Get EEGConfig for specific sleep stages.
        
        Supported stages:
        - 'awake_active': Beta/Gamma rhythm dominance.
        - 'awake_relaxed': Alpha rhythm dominance.
        - 'n2_sleep': Theta rhythm dominance with sleep spindles and K-complexes.
        - 'n3_deep_sleep': Delta rhythm dominance.
        """
        stage_clean = stage.strip().lower()
        if stage_clean == 'awake_active':
            return EEGConfig(
                fs=fs,
                duration_s=duration_s,
                band_powers={'delta': 0.1, 'theta': 0.1, 'alpha': 0.2, 'beta': 1.0, 'gamma': 0.4},
                background_1f_power=0.5,
                state='active'
            )
        elif stage_clean == 'awake_relaxed':
            return EEGConfig(
                fs=fs,
                duration_s=duration_s,
                band_powers={'delta': 0.1, 'theta': 0.2, 'alpha': 1.2, 'beta': 0.3, 'gamma': 0.1},
                background_1f_power=0.4,
                state='relaxed'
            )
        elif stage_clean == 'n2_sleep':
            return EEGConfig(
                fs=fs,
                duration_s=duration_s,
                band_powers={'delta': 0.3, 'theta': 1.0, 'alpha': 0.2, 'beta': 0.1, 'gamma': 0.05},
                background_1f_power=0.6,
                state='n2_sleep'
            )
        elif stage_clean in ('n3_deep_sleep', 'n3_sleep', 'n3'):
            return EEGConfig(
                fs=fs,
                duration_s=duration_s,
                band_powers={'delta': 1.5, 'theta': 0.4, 'alpha': 0.1, 'beta': 0.05, 'gamma': 0.01},
                background_1f_power=0.8,
                state='n3_sleep'
            )
        else:
            raise ValueError(f"Unknown sleep stage: {stage}. Use 'awake_active', 'awake_relaxed', 'n2_sleep', or 'n3_deep_sleep'")

    @staticmethod
    def get_seizure_eeg(fs: float = 256.0, duration_s: float = 20.0) -> EEGConfig:
        """
        Get EEGConfig for an epileptic seizure.
        
        Characterized by high-amplitude tonic-clonic discharges.
        """
        return EEGConfig(
            fs=fs,
            duration_s=duration_s,
            band_powers={'delta': 0.5, 'theta': 0.5, 'alpha': 0.5, 'beta': 1.5, 'gamma': 1.0},
            background_1f_power=1.0,
            state='tonic_clonic'
        )

    @staticmethod
    def get_emg_als(fs: float = 2000.0, duration_s: float = 10.0) -> EMGConfig:
        """
        Get EMGConfig representing Amyotrophic Lateral Sclerosis (ALS) pathology.
        
        Characterized by spontaneous fasciculation potentials and giant motor units.
        """
        return EMGConfig(
            fs=fs,
            duration_s=duration_s,
            pathology='als',
            contraction_level=1.0,
            amplitude_uv=800.0
        )

    @staticmethod
    def get_emg_myopathy(fs: float = 2000.0, duration_s: float = 10.0) -> EMGConfig:
        """
        Get EMGConfig representing myopathic disorders.
        
        Characterized by small, short-duration, polyphasic motor unit potentials.
        """
        return EMGConfig(
            fs=fs,
            duration_s=duration_s,
            pathology='myopathic',
            contraction_level=1.0,
            amplitude_uv=200.0
        )

    @staticmethod
    def get_resting_resp(fs: float = 50.0, duration_s: float = 60.0) -> RespConfig:
        """Get RespConfig with resting breathing rate (12 breaths per minute, i.e. 0.2 Hz)."""
        return RespConfig(
            fs=fs,
            duration_s=duration_s,
            resp_rate_hz=0.2,
            harmonic_k=0.3
        )

    @staticmethod
    def get_tachypnea_resp(fs: float = 50.0, duration_s: float = 60.0) -> RespConfig:
        """Get RespConfig with rapid breathing rate (28 breaths per minute, i.e. 0.467 Hz)."""
        return RespConfig(
            fs=fs,
            duration_s=duration_s,
            resp_rate_hz=28.0 / 60.0,
            harmonic_k=0.3
        )


# =====================================================================
# 4. Configuration Serializer Class
# =====================================================================

class ConfigSerializer:
    """Serializes and deserializes configurations to Dict, JSON, or YAML."""
    
    @staticmethod
    def to_dict(config: Any) -> Dict[str, Any]:
        """Convert configuration dataclass into dictionary."""
        return asdict(config)

    @staticmethod
    def from_dict(d: Dict[str, Any], config_class: Type[T]) -> T:
        """Construct configuration instance from dictionary."""
        valid_fields = {f.name for f in fields(config_class)}
        filtered_d = {k: v for k, v in d.items() if k in valid_fields}
        return config_class(**filtered_d)

    @staticmethod
    def to_json(config: Any, path: str) -> None:
        """Serialize configuration to a JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ConfigSerializer.to_dict(config), f, indent=4)

    @staticmethod
    def from_json(path: str, config_class: Type[T]) -> T:
        """Load configuration from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return ConfigSerializer.from_dict(d, config_class)

    @staticmethod
    def to_yaml(config: Any, path: str) -> None:
        """Serialize configuration to a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML operations. Run `pip install pyyaml`.")
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(ConfigSerializer.to_dict(config), f, default_flow_style=False)

    @staticmethod
    def from_yaml(path: str, config_class: Type[T]) -> T:
        """Load configuration from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML operations. Run `pip install pyyaml`.")
        with open(path, 'r', encoding='utf-8') as f:
            d = yaml.safe_load(f)
        return ConfigSerializer.from_dict(d, config_class)


# =====================================================================
# 5. Parameter Sweeping Benchmark Suite Class
# =====================================================================

class BenchmarkSuite:
    """Executes systematic grid sweeps of signal and noise configurations."""
    
    def __init__(
        self,
        signal_class: Any,
        noise_class: Any,
        signal_param_grid: Dict[str, List[Any]],
        noise_param_grid: Dict[str, List[Any]],
        metrics: List[Union[str, Callable[..., float]]],
        filter_fn: Callable[[np.ndarray, float], np.ndarray]
    ):
        """
        Initialize the Benchmark Suite.
        
        Parameters
        ----------
        signal_class : Any
            Generator class (e.g. ECGGenerator).
        noise_class : Any
            Noise model class (e.g. PowerlineNoise).
        signal_param_grid : Dict[str, List[Any]]
            Parameter grid ranges for the signal.
        noise_param_grid : Dict[str, List[Any]]
            Parameter grid ranges for the noise.
        metrics : List[Union[str, Callable[..., float]]]
            Metric names or functions for post-filtering validation.
        filter_fn : Callable[[np.ndarray, float], np.ndarray]
            Function matching `f(signal_array, fs) -> filtered_array`.
        """
        self.signal_class = signal_class
        self.noise_class = noise_class
        self.signal_param_grid = signal_param_grid
        self.noise_param_grid = noise_param_grid
        self.metrics = metrics
        self.filter_fn = filter_fn

    def run(self) -> Any:
        """
        Execute parameter grid sweeps.
        
        Returns
        -------
        pd.DataFrame
            A pandas DataFrame mapping grid combinations to computed metrics.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for BenchmarkSuite. Run `pip install pandas`.")

        from biosignal_simulator.composer.mixer import SignalMixer
        from biosignal_simulator.metrics.snr import compute_snr_wideband
        from biosignal_simulator.metrics.distortion import compute_rmse, compute_correlation

        # Resolve metrics
        metric_funcs = []
        for m in self.metrics:
            if isinstance(m, str):
                name = m.lower()
                if name == 'snr_wideband':
                    metric_funcs.append(('snr_wideband', compute_snr_wideband))
                elif name == 'rmse':
                    metric_funcs.append(('rmse', compute_rmse))
                elif name == 'correlation':
                    metric_funcs.append(('correlation', compute_correlation))
                else:
                    raise ValueError(f"Unknown metric string: '{m}'. Available options: 'snr_wideband', 'rmse', 'correlation'")
            else:
                metric_funcs.append((m.__name__, m))

        # Generate config permutations
        sig_keys, sig_vals = (zip(*self.signal_param_grid.items()) if self.signal_param_grid else ([], []))
        noise_keys, noise_vals = (zip(*self.noise_param_grid.items()) if self.noise_param_grid else ([], []))

        sig_combos = [dict(zip(sig_keys, v)) for v in itertools.product(*sig_vals)] if sig_vals else [{}]
        noise_combos = [dict(zip(noise_keys, v)) for v in itertools.product(*noise_vals)] if noise_vals else [{}]

        results = []
        
        # B-07 FIX: use an explicit mapping for aliases (PinkNoise, BrownNoise, etc.)
        # because __name__ + 'Config' doesn't work for subclasses of ColoredNoise.
        _NOISE_CLASS_TO_CONFIG: Dict[str, str] = {
            'ColoredNoise': 'ColoredNoiseConfig',
            'PinkNoise': 'ColoredNoiseConfig',
            'BrownNoise': 'ColoredNoiseConfig',
            'BlueNoise': 'ColoredNoiseConfig',
            'VioletNoise': 'ColoredNoiseConfig',
        }
        import biosignal_simulator.core.config as config_module
        sig_config_cls_name = self.signal_class.__name__.replace('Generator', 'Config')
        # Try the explicit alias map first, then fall back to __name__ + 'Config'
        noise_cls_name = self.noise_class.__name__
        noise_config_cls_name = _NOISE_CLASS_TO_CONFIG.get(
            noise_cls_name, noise_cls_name + 'Config'
        )
        
        sig_config_cls = getattr(config_module, sig_config_cls_name, None)
        noise_config_cls = getattr(config_module, noise_config_cls_name, None)

        for sig_params in sig_combos:
            for noise_params in noise_combos:
                # Reconstruct configs
                if sig_config_cls:
                    sig_cfg = sig_config_cls(**sig_params)
                else:
                    sig_cfg = sig_params
                    
                if noise_config_cls:
                    noise_cfg = noise_config_cls(**noise_params)
                else:
                    noise_cfg = noise_params

                # Create generator instances
                gen = self.signal_class(sig_cfg)
                noise_model = self.noise_class(noise_cfg)

                # Compose and mix
                mixer = SignalMixer(signal_generator=gen, noise_models=[noise_model])
                record = mixer.mix()

                # Filter noisy output
                if record.clean.ndim == 2:
                    # Multichannel filtering
                    filtered = np.zeros_like(record.noisy)
                    for c in range(record.clean.shape[0]):
                        filtered[c] = self.filter_fn(record.noisy[c], record.fs)
                else:
                    filtered = self.filter_fn(record.noisy, record.fs)

                # Compute performance metrics
                row = {}
                for k, v in sig_params.items():
                    row[f"sig_{k}"] = v
                for k, v in noise_params.items():
                    row[f"noise_{k}"] = v
                    
                row['fs'] = record.fs
                
                # Check dimensional power SNR inputs
                if record.clean.ndim == 2:
                    snr_in_list = [compute_snr_wideband(record.clean[c], record.noisy[c], record.fs) for c in range(record.clean.shape[0])]
                    row['snr_in'] = float(np.mean(snr_in_list))
                else:
                    row['snr_in'] = compute_snr_wideband(record.clean, record.noisy, record.fs)

                # Compute post-filtering metrics
                for name, func in metric_funcs:
                    if record.clean.ndim == 2:
                        scores = []
                        for c in range(record.clean.shape[0]):
                            try:
                                scores.append(func(record.clean[c], filtered[c], record.fs))
                            except TypeError:
                                scores.append(func(record.clean[c], filtered[c]))
                        row[name] = float(np.mean(scores))
                    else:
                        try:
                            row[name] = func(record.clean, filtered, record.fs)
                        except TypeError:
                            row[name] = func(record.clean, filtered)

                results.append(row)

        return pd.DataFrame(results)
