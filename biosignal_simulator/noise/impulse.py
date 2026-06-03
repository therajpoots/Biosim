"""
Advanced Impulsive and Transient Spike Noise Simulator.

This module provides the `ImpulseNoise` class, which generates short, high-amplitude,
isolated biopotential spikes (transients) modeled after electrostatic discharges,
amplifier clipping recovery, or motion-induced voltage pops.

Physical and Physiological Context:
    Impulsive noise consists of sudden, high-voltage spikes of very short duration.
    1. Electrostatic spark discharges: Fast charging/discharging between dry electrode surfaces
       and synthetic clothing fibers (triboelectric charge dumps).
    2. Swallowing/Blinking artifacts: High-velocity physiological events creating brief spikes in
       sensitive channels (e.g. eye-blink artifacts in EEG).
    3. Wireless communication dropouts: Glitches or sync clicks in analog-to-digital converters.

Mathematical Formulation:
    1. Arrival Times:
       Modeled as a Poisson process where the probability of $k$ spikes occurring in duration $T$ is:
       $$P(k) = \\frac{(\\lambda T)^k e^{-\\lambda T}}{k!}$$
       where $\\lambda$ is the arrival rate in spikes per second (Hz). Spike times $\\{\\tau_i\\}$ are
       drawn uniformly within the simulation duration.

    2. Amplitude Distributions:
       - Generalized Pareto Distribution (GPD): Heavy-tailed model for extreme transients:
         $$P(A; c, \\sigma) = \\frac{1}{\\sigma} \\left(1 + c \\frac{A}{\\sigma}\\right)^{-\\left(\\frac{1}{c} + 1\\right)}$$
       - Cauchy Distribution: Modeling extremely heavy-tailed, infinite-variance impulses:
         $$P(A; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\left(1 + \\left(\\frac{A - x_0}{\\gamma}\\right)^2\\right)}$$

    3. Pulse Shapes:
       - Dirac Delta (Single Sample): $h_i[n] = A_i \\cdot \\delta[n - n_i]$.
       - Gaussian Pulse: $h_i(t) = A_i \\cdot \\exp\\left( -\\frac{(t - \\tau_i)^2}{2\\sigma_p^2} \\right)$.
       - Double-Exponential Transient: Models physical charging/discharging with finite rise time $\\tau_r$ and decay time $\\tau_d$:
         $$h_i(t) = A_i \\cdot K_0 \\cdot \\left( e^{-\\frac{t - \\tau_i}{\\tau_d}} - e^{-\\frac{t - \\tau_i}{\\tau_r}} \\right) \\cdot \\mathcal{H}(t - \\tau_i)$$
         where $K_0$ is a normalization constant to ensure peak amplitude reaches $A_i$.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy.stats import genpareto, cauchy
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import ImpulseNoiseConfig
from biosignal_simulator.utils.validation import validate_config

class ImpulseNoise(BaseNoiseModel):
    """
    Impulsive and Transient Spike Noise Generator.
    
    Synthesizes Poisson-distributed pulse spikes with variable amplitudes (Pareto, Cauchy),
    polarities (positive, negative, bipolar), pulse shapes (Dirac, Gaussian, Double-Exponential),
    and multi-channel spatial transmission.
    """
    
    def __init__(
        self,
        config: Optional[ImpulseNoiseConfig] = None,
        pulse_shape: str = "gaussian",
        rise_time_s: float = 0.005,
        spatial_leakage_factor: float = 0.3,
        **kwargs
    ):
        """
        Initialize the Impulse Noise generator.
        
        Parameters
        ----------
        config : Optional[ImpulseNoiseConfig]
            Base configuration containing spike rate, amplitude shape/scale, and width.
        pulse_shape : str
            Spike profile shape: 'dirac', 'gaussian', 'double_exponential', 'rectangular', 'triangular'.
            Default is 'gaussian'.
        rise_time_s : float
            Rise time constant in seconds (only used for 'double_exponential' pulses).
            Default is 0.005 s.
        spatial_leakage_factor : float
            Coupling coefficient [0, 1] scaling how impulses bleed into adjacent channels.
            Default is 0.3.
        **kwargs :
            Alternative parameters passed to ImpulseNoiseConfig if config is None.
        """
        if config is None:
            config = ImpulseNoiseConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.pulse_shape = pulse_shape.lower().strip()
        self.rise_time_s = rise_time_s
        self.spatial_leakage_factor = spatial_leakage_factor
        
        self._validate_impulse_parameters()

    def _validate_impulse_parameters(self) -> None:
        """Validate input parameter bounds."""
        allowed_shapes = {"dirac", "gaussian", "double_exponential", "rectangular", "triangular"}
        if self.pulse_shape not in allowed_shapes:
            raise ValueError(f"Unsupported pulse shape '{self.pulse_shape}'. Allowed: {allowed_shapes}")
            
        if self.rise_time_s <= 0.0:
            raise ValueError("Rise time constant 'rise_time_s' must be positive.")
            
        if self.config.pulse_width_s < 0.0:
            raise ValueError("Pulse width 'pulse_width_s' must be non-negative.")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D impulse noise.
        
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
            
        noise = np.zeros(n_samples)
        dur = n_samples / fs
        t = np.arange(n_samples) / fs

        # --- 1. Determine Spike Count and Arrival Times ---
        rate = self.config.rate_hz
        if rate <= 0.0:
            return noise
            
        # Draw total arrivals using Poisson distribution
        num_spikes = self.rng.poisson(lam=rate * dur)
        if num_spikes == 0:
            return noise
            
        # Distribute spikes uniformly across duration
        spike_times = self.rng.uniform(0.0, dur, size=num_spikes)

        # --- 2. Draw Impulse Amplitudes ---
        c = self.config.amplitude_shape
        scale = self.config.amplitude_scale
        
        if c <= 0.0:
            # Fall back to exponential distribution if shape is zero or negative
            amplitudes = self.rng.exponential(scale=scale, size=num_spikes)
        else:
            # Draw from Generalized Pareto (GPD)
            amplitudes = genpareto.rvs(c, scale=scale, size=num_spikes, random_state=self.rng)

        # --- 3. Apply Polarity Settings ---
        polarity = self.config.polarity.lower().strip()
        if polarity == 'positive':
            polarities = np.ones(num_spikes)
        elif polarity == 'negative':
            polarities = -np.ones(num_spikes)
        else:  # bipolar
            polarities = self.rng.choice([1.0, -1.0], size=num_spikes)
            
        signed_amplitudes = amplitudes * polarities

        # --- 4. Synthesize Pulse Waveforms ---
        p_width = self.config.pulse_width_s
        
        if self.pulse_shape == "dirac" or p_width <= 0.0:
            # Ideal single-sample delta impulses
            for t_ev, amp in zip(spike_times, signed_amplitudes):
                idx = int(np.round(t_ev * fs))
                if 0 <= idx < n_samples:
                    noise[idx] += amp
        else:
            # Continuous-time finite width pulse shapes
            sigma = p_width / 4.0
            
            for t_ev, amp in zip(spike_times, signed_amplitudes):
                # Optimize computation by only updating samples inside a localized window
                half_width_samples = int(4.5 * sigma * fs) + 1
                center_idx = int(np.round(t_ev * fs))
                start_idx = max(0, center_idx - half_width_samples)
                end_idx = min(n_samples, center_idx + half_width_samples + 1)
                
                if start_idx >= end_idx:
                    continue
                    
                t_sub = t[start_idx:end_idx]
                t_rel = t_sub - t_ev
                
                if self.pulse_shape == "gaussian":
                    pulse = amp * np.exp(-0.5 * (t_rel / sigma) ** 2)
                    
                elif self.pulse_shape == "double_exponential":
                    # Rise time constant (self.rise_time_s) and decay time constant (sigma)
                    tau_r = self.rise_time_s
                    tau_d = max(tau_r + 1e-4, sigma)  # decay must be slower than rise
                    
                    # Peak time: t_peak = (tau_r * tau_d) / (tau_d - tau_r) * ln(tau_d / tau_r)
                    t_peak = (tau_r * tau_d / (tau_d - tau_r)) * np.log(tau_d / tau_r)
                    # Normalization factor K0
                    k0 = 1.0 / (np.exp(-t_peak / tau_d) - np.exp(-t_peak / tau_r))
                    
                    pulse = np.zeros_like(t_rel)
                    mask = t_rel >= 0.0
                    t_pos = t_rel[mask]
                    pulse[mask] = amp * k0 * (np.exp(-t_pos / tau_d) - np.exp(-t_pos / tau_r))
                    
                elif self.pulse_shape == "rectangular":
                    pulse = np.zeros_like(t_rel)
                    mask = np.abs(t_rel) <= (p_width / 2.0)
                    pulse[mask] = amp
                    
                elif self.pulse_shape == "triangular":
                    pulse = np.zeros_like(t_rel)
                    mask = np.abs(t_rel) <= (p_width / 2.0)
                    pulse[mask] = amp * (1.0 - (2.0 * np.abs(t_rel[mask]) / p_width))
                    
                else:
                    pulse = amp * np.exp(-0.5 * (t_rel / sigma) ** 2)
                    
                noise[start_idx:end_idx] += pulse

        return noise

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel impulse noise with electromagnetic spatial propagation.
        
        Models how a transient spark is captured by a primary channel and leaks into
        neighboring channels with spatial attenuation (leakage) and small phase delays.
        
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
            
        noise_matrix = np.zeros((n_channels, n_samples))
        
        # 1. Synthesize primary independent impulses on each channel
        for c in range(n_channels):
            noise_matrix[c] = self.generate(n_samples, fs)
            
        # 2. Add cross-channel coupling (leakage)
        # Represents electrostatic charge coupling to adjacent lines
        if n_channels > 1 and self.spatial_leakage_factor > 0.0:
            leaked_matrix = np.zeros((n_channels, n_samples))
            for c in range(n_channels):
                # Mix adjacent channel leakage
                adj_channels = [c - 1, c + 1]
                leak = np.zeros(n_samples)
                n_adj = 0
                for adj in adj_channels:
                    if 0 <= adj < n_channels:
                        # Introduce a tiny delay (1-2 samples) during leakage propagation
                        delay_samples = self.rng.choice([1, 2])
                        leak += np.roll(noise_matrix[adj], delay_samples) * self.spatial_leakage_factor
                        n_adj += 1
                leaked_matrix[c] = noise_matrix[c] + (leak / n_adj if n_adj > 0 else 0.0)
            return leaked_matrix
            
        return noise_matrix
