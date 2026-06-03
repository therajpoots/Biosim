"""
Physiologically and physically motivated Motion Artifact (MA) simulator.

This module provides the `MotionArtifact` class, which models complex signal deformations
caused by subject movement, sensor acceleration, and cable movement.

Physical and Physiological Context:
    Motion artifacts are major confounders in wearable health monitoring (e.g. smartwatches,
    patches). They manifest through multiple physical pathways:
    1. Skin-Electrode Deformation (Low Frequency): Movement deforms the skin under the
       electrode, altering the skin-battery contact potential and causing baseline shifts
       ranging from 0.1 to 10 Hz.
    2. Cable Artifacts (Triboelectric Effect): Movement of electrode cables causes friction
       and electrostatic charge accumulation, releasing high-frequency burst charges (50-200 Hz).
    3. Sensor Deceleration / Impacts (Transients): Sudden impacts (e.g., heel strikes during
       running or knocks to the sensor) generate Poisson-distributed, decaying oscillatory spikes.
    4. Postural Swings: Steady cyclical shifts in baseline corresponding to walking, running,
       or breathing motion harmonics.

Mathematical Formulation:
    The motion artifact signal $MA(t)$ is a mixture of low-frequency displacement $M_{\\text{lf}}(t)$,
    impact transients $M_{\\text{imp}}(t)$, and high-frequency triboelectric bursts $M_{\\text{cable}}(t)$:
    $$MA(t) = M_{\\text{lf}}(t) + M_{\\text{imp}}(t) + M_{\\text{cable}}(t)$$

    where:
    1. Low-Frequency Displacement $M_{\\text{lf}}(t)$:
       Bandpass filtered Gaussian noise ($f \\in [f_{\\text{min}}, f_{\\text{max}}]$) multiplied by
       a slow low-frequency envelope $E(t)$ representing active movement periods:
       $$M_{\\text{lf}}(t) = \\text{Bandpass}\\left( w(t) \\right) \\cdot E_{\\text{lowpass}}(t)$$

    2. Decaying Impact Transients $M_{\\text{imp}}(t)$:
       A Poisson process with event times $\\{\\tau_i\\}$ triggering decaying sinusoidal oscillations:
       $$M_{\\text{imp}}(t) = \\sum_{i} A_i \\cdot e^{-(t - \\tau_i)/\\tau_d} \\cdot \\sin\\left(2\\pi f_{\\text{imp}} (t - \\tau_i)\\right) \\cdot \\mathcal{H}(t - \\tau_i)$$
       where $\\tau_d$ is the decay constant, $f_{\\text{imp}}$ is the mechanical resonance frequency,
       and $\\mathcal{H}$ is the Heaviside step function.

    3. Cable Triboelectric Bursts $M_{\\text{cable}}(t)$:
       A series of high-frequency white noise bursts filtered to $[50, 200]$ Hz and windowed
       using localized Gaussian envelopes centered around Poisson event times:
       $$M_{\\text{cable}}(t) = w_{\\text{bp}}(t) \\cdot \\sum_{k} e^{-\\frac{(t - T_k)^2}{2\\sigma^2}}$$
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import MotionArtifactConfig
from biosignal_simulator.core.math_utils import normalize_to_rms
from biosignal_simulator.utils.validation import validate_config

class MotionArtifact(BaseNoiseModel):
    """
    Comprehensive Motion Artifact Simulator.
    
    Synthesizes multi-component movement artifacts: low-frequency skin deformation shifts,
    Poisson-triggered deceleration impact transients, and triboelectric high-frequency cable bursts.
    """
    
    def __init__(
        self,
        config: Optional[MotionArtifactConfig] = None,
        motion_direction_vector: Optional[Union[List[float], np.ndarray]] = None,
        **kwargs
    ):
        """
        Initialize the Motion Artifact generator.
        
        Parameters
        ----------
        config : Optional[MotionArtifactConfig]
            Base configuration containing amplitudes, frequencies, decay rates.
        motion_direction_vector : Optional[Union[List[float], np.ndarray]]
            Spatial projection vector of shape (n_channels,) used to scale motion
            across multi-channel electrodes to model directional movement.
        **kwargs :
            Alternative parameters passed to MotionArtifactConfig if config is None.
        """
        if config is None:
            config = MotionArtifactConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.direction_vector = np.array(motion_direction_vector) if motion_direction_vector is not None else None

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D motion artifact signal.
        
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
        dur = n_samples / fs
        nyq = 0.5 * fs
        
        combined_noise = np.zeros(n_samples)

        # --- 1. Low-Frequency Displacement ---
        if self.config.enable_lf:
            raw_noise = self.rng.normal(0.0, 1.0, size=n_samples)
            low = max(0.01, self.config.lf_fmin_hz)
            high = min(self.config.lf_fmax_hz, nyq - 0.1)
            
            if low < high:
                b, a = sp_signal.butter(2, [low / nyq, high / nyq], btype='bandpass')
                lf_signal = sp_signal.filtfilt(b, a, raw_noise)
            else:
                lf_signal = raw_noise
                
            # Compute low-frequency activity envelope (representing motion intensity bouts)
            raw_noise_env = self.rng.normal(0.0, 1.0, size=n_samples)
            b_env, a_env = sp_signal.butter(2, min(0.1, nyq - 0.01) / nyq, btype='lowpass')
            env = np.abs(sp_signal.filtfilt(b_env, a_env, raw_noise_env))
            env_max = np.max(env)
            env = env / env_max if env_max > 1e-12 else env
            
            lf_noise = lf_signal * env
            combined_noise += normalize_to_rms(lf_noise, self.config.lf_amplitude)

        # --- 2. Impact Transients (Poisson Decay Sinusoids) ---
        if self.config.enable_impacts and self.config.impact_rate_hz > 0:
            impacts = np.zeros(n_samples)
            lam = self.config.impact_rate_hz
            
            # Draw event count using Poisson distribution
            num_events = self.rng.poisson(lam=lam * dur)
            if num_events > 0:
                # Distribute events uniformly in time
                event_times = self.rng.uniform(0.0, dur, size=num_events)
                
                tau = self.config.impact_decay_s
                f_imp = self.config.impact_freq_hz
                amp = self.config.impact_amplitude
                
                for t_ev in event_times:
                    t_rel = t - t_ev
                    mask = t_rel >= 0
                    if np.any(mask):
                        t_val = t_rel[mask]
                        # Decaying harmonic oscillation
                        decaying_wave = amp * np.exp(-t_val / tau) * np.sin(2.0 * np.pi * f_imp * t_val)
                        impacts[mask] += decaying_wave
                        
            combined_noise += impacts

        # --- 3. Cable Artifact (Triboelectric Burst Noise) ---
        if self.config.enable_cable:
            cable = np.zeros(n_samples)
            # Cable bursts occur at a slower Poisson rate (e.g. 0.2 Hz)
            lam = 0.2
            num_events = self.rng.poisson(lam=lam * dur)
            
            if num_events > 0:
                event_times = self.rng.uniform(0.0, dur, size=num_events)
                
                # Filter white noise to triboelectric bands (50-200 Hz)
                raw_hfn = self.rng.normal(0.0, 1.0, size=n_samples)
                low_cb = 50.0
                high_cb = min(200.0, nyq - 1.0)
                
                if low_cb < high_cb:
                    b_cb, a_cb = sp_signal.butter(4, [low_cb / nyq, high_cb / nyq], btype='bandpass')
                    filtered_hfn = sp_signal.filtfilt(b_cb, a_cb, raw_hfn)
                else:
                    filtered_hfn = raw_hfn
                    
                filtered_hfn = normalize_to_rms(filtered_hfn, 1.0)
                
                # Localized Gaussian windows around event times
                win_dur = 0.6
                sigma = win_dur / 4.0
                envelope = np.zeros(n_samples)
                
                for t_ev in event_times:
                    envelope += np.exp(-0.5 * ((t - t_ev) / sigma) ** 2)
                    
                cable_noise = filtered_hfn * envelope
                combined_noise += normalize_to_rms(cable_noise, self.config.cable_amplitude)

        return combined_noise

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel motion artifacts projected along spatial direction vector.
        
        Movement direction affects electrodes differently depending on geometry.
        
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
            
        # Determine directional projection vector
        direction = self.direction_vector
        if direction is None:
            direction = self.rng.uniform(0.2, 1.0, size=n_channels)
        elif len(direction) < n_channels:
            padded = np.ones(n_channels)
            padded[:len(direction)] = direction
            direction = padded
            
        # Generate shared core source of motion
        # This models the physical movement source (e.g. wrist acceleration)
        shared_motion = self.generate(n_samples, fs)
        
        noise_matrix = np.zeros((n_channels, n_samples))
        for c in range(n_channels):
            # Each channel receives the motion projected along its axis
            # Plus small local uncorrelated sensor displacement
            local_noise = np.zeros(n_samples)
            if self.config.enable_lf:
                # Add tiny uncorrelated local displacement
                raw_local = self.rng.normal(0.0, 1.0, size=n_samples)
                b, a = sp_signal.butter(2, [0.5 / (0.5 * fs), 5.0 / (0.5 * fs)], btype='bandpass')
                local_noise = sp_signal.filtfilt(b, a, raw_local)
                local_noise = normalize_to_rms(local_noise, 0.05 * self.config.lf_amplitude)
                
            noise_matrix[c] = shared_motion * direction[c] + local_noise
            
        return noise_matrix
