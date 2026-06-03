"""
Advanced Powerline Interference (PLI) and electromagnetic coupling simulator.

This module provides the `PowerlineNoise` class, which generates high-fidelity line noise
interference (commonly 50 Hz or 60 Hz) including harmonics, phase drift, and amplitude modulation.

Physiological and Electrical Context:
    Powerline noise is one of the most ubiquitous noise sources in biopotential recordings.
    It occurs due to:
    1. Electromagnetic coupling: Ambient capacitive and inductive coupling between the AC
       power grid wires in walls/appliances and the patient lead wire loops.
    2. Common-mode voltage leakage: Current leaking through the skin-electrode contact impedance
       to ground, converting common-mode line noise into differential biopotential signal.
    3. Harmonic leakage: Non-linearities in instrumentation amplifiers or nearby switching power
       supplies cause higher-frequency integer multiples of the fundamental frequency (50/60 Hz)
       to couple into the signal.

Mathematical Formulation:
    The powerline interference signal $PL(t)$ is modeled as a sum of fundamental and harmonic
    components, subject to frequency fluctuations, amplitude envelopes, and phase drifts:
    $$PL(t) = \\sum_{h=1}^{H} A_h(t) \\cdot \\sin\\left( 2\\pi h \\cdot \\Phi(t) + \\theta_h + \\theta_{\\text{ch}} + \\phi_{h,\\text{drift}}(t) \\right)$$

    where:
    1. Instantaneous Integrated Phase $\\Phi(t)$:
       $$\\Phi(t) = f_{\\text{line}} \\cdot t + \\int_{0}^{t} \\delta_f(\\tau) d\\tau$$
       where $f_{\\text{line}}$ is the nominal power grid frequency, and $\\delta_f(t)$ is a slow
       frequency drift modeled by lowpass-filtered noise (representing grid frequency changes).

    2. Time-Varying Harmonic Amplitudes $A_h(t)$:
       $$A_h(t) = A_{\\text{base}} \\cdot h^{-\\alpha} \\cdot E_{\\text{am}}(t)$$
       where $\\alpha$ is the harmonic decay rate, and $E_{\\text{am}}(t)$ is the amplitude envelope
       representing load fluctuations (flicker).

    3. Phase Parameters:
       - $\\theta_h$: Random starting phase for harmonic $h$ in $[0, 2\\pi]$.
       - $\\theta_{\\text{ch}}$: Phase offset specific to the channel (modeling spatial orientation).
       - $\\phi_{h,\\text{drift}}(t)$: Phase drift over time modeled by a lowpass process.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import PowerlineNoiseConfig
from biosignal_simulator.core.math_utils import normalize_to_rms
from biosignal_simulator.utils.validation import validate_config

class PowerlineNoise(BaseNoiseModel):
    """
    High-Fidelity Powerline Interference Simulator.
    
    Synthesizes fundamental 50/60 Hz electrical line noise with decaying harmonics,
    power grid frequency fluctuations, amplitude modulation, and channel phase offsets.
    """
    
    def __init__(
        self,
        config: Optional[PowerlineNoiseConfig] = None,
        channel_phases: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs
    ):
        """
        Initialize the Powerline Noise model.
        
        Parameters
        ----------
        config : Optional[PowerlineNoiseConfig]
            Base configuration containing nominal frequency, harmonics, and decay rates.
        channel_phases : Optional[Union[List[float], np.ndarray]]
            Pre-defined phase offsets (in radians) for multi-channel noise.
            Shape must be (n_channels,).
        **kwargs :
            Alternative parameters passed to PowerlineNoiseConfig if config is None.
        """
        if config is None:
            config = PowerlineNoiseConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.channel_phases = np.array(channel_phases) if channel_phases is not None else None

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D powerline noise.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            1-D noise array of shape (n_samples,).
        """
        if n_samples <= 0:
            return np.empty(0, dtype=np.float64)
            
        t = np.arange(n_samples) / fs
        nyq = 0.5 * fs
        
        # --- 1. Grid Frequency Fluctuation delta_f(t) ---
        # Frequency fluctuations modeled as slow noise process (bandwidth ~0.01 Hz)
        raw_noise_f = self.rng.normal(0.0, 1.0, size=n_samples)
        bf, af = sp_signal.butter(2, min(0.01, nyq - 0.005) / nyq, btype='lowpass')
        delta_f = sp_signal.filtfilt(bf, af, raw_noise_f)
        std_f = np.std(delta_f)
        if std_f > 1e-12:
            delta_f = (delta_f / std_f) * self.config.freq_std_hz
        else:
            delta_f = np.zeros_like(delta_f)
            
        # Integrate frequency to get Phase Accumulator
        f_nominal = self.config.f_line_hz
        f_integral = f_nominal * t + np.cumsum(delta_f) / fs

        # --- 2. Amplitude Modulation (Flicker/Load Changes) ---
        # Modeling slow AC load fluctuation (envelope bandwidth ~0.1 Hz)
        raw_noise_am = self.rng.normal(0.0, 1.0, size=n_samples)
        bam, aam = sp_signal.butter(2, min(0.1, nyq - 0.01) / nyq, btype='lowpass')
        lowpass_am = sp_signal.filtfilt(bam, aam, raw_noise_am)
        std_am = np.std(lowpass_am)
        if std_am > 1e-12:
            lowpass_am = lowpass_am / std_am
        else:
            lowpass_am = np.zeros_like(lowpass_am)
            
        # Amplitude envelope swings based on modulation depth
        env = 1.0 + self.config.amplitude_mod_depth * lowpass_am

        # --- 3. Synthesize Fundamental and Harmonics ---
        pl = np.zeros_like(t)
        H = self.config.n_harmonics
        decay = self.config.harmonic_decay
        drift_std = self.config.phase_drift_std
        
        # Phase drift bandwidth ~0.05 Hz
        bp, ap = sp_signal.butter(2, min(0.05, nyq - 0.01) / nyq, btype='lowpass')

        for h in range(1, H + 1):
            if h * f_nominal >= nyq:
                # Aliasing prevention: skip harmonics above Nyquist
                break
                
            # Amplitude decays with harmonic order
            A_h_t = (1.0 / (h ** decay)) * env
            
            # Stochastic start phase
            phi_h = self.rng.uniform(0.0, 2.0 * np.pi)
            
            # Stochastic phase drift over time
            raw_noise_p = self.rng.normal(0.0, 1.0, size=n_samples)
            phase_drift = sp_signal.filtfilt(bp, ap, raw_noise_p)
            std_p = np.std(phase_drift)
            if std_p > 1e-12:
                phase_drift = (phase_drift / std_p) * drift_std
            else:
                phase_drift = np.zeros_like(phase_drift)
                
            # Accumulate this harmonic component
            pl += A_h_t * np.sin(2.0 * np.pi * h * f_integral + phi_h + phase_drift)

        # Scale output to match fundamental amplitude RMS config
        return normalize_to_rms(pl, self.config.amplitude)

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel line noise with spatial phase distribution.
        
        Different spatial orientations of electrode leads lead to distinct phase
        offsets relative to the power grid emission field.
        
        Parameters
        ----------
        n_channels : int
            Number of channels.
        n_samples : int
            Number of samples.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            2-D noise array of shape (n_channels, n_samples).
        """
        if n_channels <= 0 or n_samples <= 0:
            return np.empty((n_channels, 0), dtype=np.float64)
            
        # Determine channel phase shifts
        phases = self.channel_phases
        if phases is None:
            # Assign random phase offsets across channels
            phases = self.rng.uniform(-np.pi, np.pi, size=n_channels)
        elif len(phases) < n_channels:
            # Pad phases if shorter than requested channels
            padded = np.zeros(n_channels)
            padded[:len(phases)] = phases
            padded[len(phases):] = self.rng.uniform(-np.pi, np.pi, size=n_channels - len(phases))
            phases = padded
            
        t = np.arange(n_samples) / fs
        nyq = 0.5 * fs
        
        # Share the frequency fluctuation grid across all channels (common electrical source)
        raw_noise_f = self.rng.normal(0.0, 1.0, size=n_samples)
        bf, af = sp_signal.butter(2, min(0.01, nyq - 0.005) / nyq, btype='lowpass')
        delta_f = sp_signal.filtfilt(bf, af, raw_noise_f)
        std_f = np.std(delta_f)
        if std_f > 1e-12:
            delta_f = (delta_f / std_f) * self.config.freq_std_hz
        else:
            delta_f = np.zeros_like(delta_f)
        f_nominal = self.config.f_line_hz
        f_integral = f_nominal * t + np.cumsum(delta_f) / fs
        
        # Share amplitude modulation (grid load changes)
        raw_noise_am = self.rng.normal(0.0, 1.0, size=n_samples)
        bam, aam = sp_signal.butter(2, min(0.1, nyq - 0.01) / nyq, btype='lowpass')
        lowpass_am = sp_signal.filtfilt(bam, aam, raw_noise_am)
        std_am = np.std(lowpass_am)
        if std_am > 1e-12:
            lowpass_am = lowpass_am / std_am
        else:
            lowpass_am = np.zeros_like(lowpass_am)
        env = 1.0 + self.config.amplitude_mod_depth * lowpass_am
        
        noise_matrix = np.zeros((n_channels, n_samples))
        H = self.config.n_harmonics
        decay = self.config.harmonic_decay
        drift_std = self.config.phase_drift_std
        bp, ap = sp_signal.butter(2, min(0.05, nyq - 0.01) / nyq, btype='lowpass')
        
        for c in range(n_channels):
            pl_channel = np.zeros(n_samples)
            ch_phase = phases[c]
            
            for h in range(1, H + 1):
                if h * f_nominal >= nyq:
                    break
                A_h_t = (1.0 / (h ** decay)) * env
                
                # Starting phase for harmonic + channel specific phase offset
                phi_h = self.rng.uniform(0.0, 2.0 * np.pi) + h * ch_phase
                
                # Independent phase drift per channel
                raw_noise_p = self.rng.normal(0.0, 1.0, size=n_samples)
                phase_drift = sp_signal.filtfilt(bp, ap, raw_noise_p)
                std_p = np.std(phase_drift)
                if std_p > 1e-12:
                    phase_drift = (phase_drift / std_p) * drift_std
                else:
                    phase_drift = np.zeros_like(phase_drift)
                    
                pl_channel += A_h_t * np.sin(2.0 * np.pi * h * f_integral + phi_h + phase_drift)
                
            # Scale each channel to the fundamental amplitude
            noise_matrix[c] = normalize_to_rms(pl_channel, self.config.amplitude)
            
        return noise_matrix
