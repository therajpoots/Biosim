"""
Physiologically accurate Baseline Wander (BW) and low-frequency drift simulator.

This module provides the `BaselineWander` class, which simulates slow, low-frequency,
non-stationary drifts that corrupt physiological recordings (such as ECG, PPG, EEG).

Physiological Reference:
    Baseline wander is primarily caused by:
    1. Respiration: Movement of the chest cavity changes the physical position of
       electrodes relative to the heart/muscles and modulates contact impedance,
       causing a cyclical baseline swing at the breathing frequency (typically 0.1 - 0.4 Hz).
    2. Body Movement & Postural Shifts: Slow changes in contact potential, sweating (which
       alters skin conductance and electrode potential), or gradual sensor migration.
    3. Instrumentation Drift: Temperature fluctuations and slow capacitor charge/discharge
       cycles within the amplifier circuitry.

Mathematical Formulation:
    The baseline wander signal $BW(t)$ is modeled as a weighted linear combination of three
    physiologically and physically motivated components:
    $$BW(t) = \\gamma \\cdot \\left[ w_{\\text{resp}} \\cdot B_{\\text{resp}}(t) + w_{\\text{drift}} \\cdot B_{\\text{drift}}(t) + w_{\\text{trend}} \\cdot B_{\\text{trend}}(t) \\right]$$

    where:
    1. Respiratory Component $B_{\\text{resp}}(t)$:
       A sinusoidal or asymmetric waveform at frequency $f_{\\text{resp}}$, with amplitude
       modulation $A(t)$ and frequency drift (phase modulation) $\\phi(t)$:
       $$B_{\\text{resp}}(t) = A(t) \\cdot \\sin\\left( 2\\pi f_{\\text{resp}} t + \\int_{0}^{t} \\delta_f(\\tau) d\\tau \\right)$$
       where $\\delta_f(t)$ is a lowpass-filtered noise process modeling respiration frequency variation.

    2. Low-Frequency Stochastic Drift $B_{\\text{drift}}(t)$:
       A Brownian motion (random walk) process filtered using a bandpass Butterworth filter
       (typically $0.005$ to $0.05$ Hz) to isolate ultra-low frequency wander:
       $$B_{\\text{drift}}(t) = \\text{Bandpass}\\left( \\int w(\\tau) d\\tau \\right)$$

    3. Deterministic Trend $B_{\\text{trend}}(t)$:
       A polynomial of degree $D$ parameterized by normalized time $t_n \\in [-1, 1]$:
       $$B_{\\text{trend}}(t) = \\sum_{d=0}^{D} c_d t_n^d \\quad \\text{where } c_d \\sim \\mathcal{N}(0, 1)$$
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import BaselineWanderConfig
from biosignal_simulator.core.math_utils import normalize_to_rms
from biosignal_simulator.utils.validation import validate_config

class BaselineWander(BaseNoiseModel):
    """
    Physiological Baseline Wander Simulator.
    
    Synthesizes low-frequency drifts combining respiratory variations, stochastic walks,
    and polynomial trends. Supports multi-channel coupled or independent drifts.
    """
    
    def __init__(
        self,
        config: Optional[BaselineWanderConfig] = None,
        spatial_correlation: Optional[np.ndarray] = None,
        asymmetric_breathing: bool = True,
        **kwargs
    ):
        """
        Initialize the Baseline Wander model.
        
        Parameters
        ----------
        config : Optional[BaselineWanderConfig]
            The base configuration specifying amplitudes, fractions, and trends.
        spatial_correlation : Optional[np.ndarray]
            Covariance matrix of shape (n_channels, n_channels) for multi-channel systems.
        asymmetric_breathing : bool
            If True, simulates asymmetrical inspiration/expiration baseline swings.
            Default is True.
        **kwargs :
            Alternative parameters passed to BaselineWanderConfig if config is None.
        """
        if config is None:
            config = BaselineWanderConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.asymmetric_breathing = asymmetric_breathing
        
        # Spatial correlation setup
        self.spatial_correlation = spatial_correlation
        self.cholesky_matrix: Optional[np.ndarray] = None
        if self.spatial_correlation is not None:
            self._precompute_spatial_mixing()

    def _precompute_spatial_mixing(self) -> None:
        """Compute the Cholesky factor for multi-channel correlation mixing."""
        cov = np.atleast_2d(self.spatial_correlation)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"Spatial correlation matrix must be square, got shape {cov.shape}")
        
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("Spatial correlation matrix must be symmetric.")
            
        try:
            reg_cov = cov + np.eye(cov.shape[0]) * 1e-12
            self.cholesky_matrix = np.linalg.cholesky(reg_cov)
        except np.linalg.LinAlgError:
            raise ValueError("Spatial correlation matrix is not positive-definite.")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D baseline wander noise.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
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
        
        # --- 1. Respiratory Component ---
        # A_resp(t) represents amplitude envelope fluctuations (lowpass filtered at 0.05 Hz)
        raw_noise_1 = self.rng.normal(0.0, 1.0, size=n_samples)
        b1, a1 = sp_signal.butter(2, min(0.05, nyq - 0.01) / nyq, btype='lowpass')
        A_resp = 1.0 + 0.25 * sp_signal.filtfilt(b1, a1, raw_noise_1)
        
        # Frequency drift integrated to produce phase drift
        raw_noise_2 = self.rng.normal(0.0, 1.0, size=n_samples)
        b2, a2 = sp_signal.butter(2, min(0.02, nyq - 0.005) / nyq, btype='lowpass')
        freq_drift = self.config.f_resp_variation * sp_signal.filtfilt(b2, a2, raw_noise_2)
        phase_drift = 2.0 * np.pi * np.cumsum(freq_drift) / fs
        
        f_resp = self.config.f_resp_hz
        phase = 2.0 * np.pi * f_resp * t + phase_drift
        
        if self.asymmetric_breathing:
            # Model asymmetric breathing: inspiration is faster than expiration
            # We shape the phase using a nonlinear wrapping function
            wrapped_phase = phase % (2.0 * np.pi)
            shaped_phase = np.zeros_like(wrapped_phase)
            
            # Inspiration (first 40% of cycle) vs Expiration (last 60% of cycle)
            insp_ratio = 0.4
            c1 = 1.0 / insp_ratio
            c2 = 1.0 / (1.0 - insp_ratio)
            
            mask_insp = wrapped_phase < (2.0 * np.pi * insp_ratio)
            # Map [0, 2pi*insp_ratio] -> [0, pi]
            shaped_phase[mask_insp] = wrapped_phase[mask_insp] * c1 * 0.5
            # Map [2pi*insp_ratio, 2pi] -> [pi, 2pi]
            shaped_phase[~mask_insp] = (wrapped_phase[~mask_insp] - 2.0 * np.pi * insp_ratio) * c2 * 0.5 + np.pi
            
            bw_resp = A_resp * np.sin(shaped_phase)
        else:
            bw_resp = A_resp * np.sin(phase)
            
        bw_resp = normalize_to_rms(bw_resp, 1.0)

        # --- 2. Low-Frequency Stochastic Drift ---
        # Generate Brownian motion (random walk)
        raw_walk = np.cumsum(self.rng.normal(0.0, 1.0, size=n_samples))
        
        # Apply ultra-low-frequency bandpass filter [0.005, 0.05] Hz
        low_cut = 0.005
        high_cut = 0.05
        low_cut = max(1e-4, min(low_cut, nyq - 0.02))
        high_cut = min(high_cut, nyq - 0.01)
        
        if low_cut < high_cut:
            b_bp, a_bp = sp_signal.butter(2, [low_cut / nyq, high_cut / nyq], btype='bandpass')
            lf_drift = sp_signal.filtfilt(b_bp, a_bp, raw_walk)
        else:
            # Fallback if sampling frequency is extremely small
            lf_drift = raw_walk - np.mean(raw_walk)
            
        lf_drift = normalize_to_rms(lf_drift, 1.0)

        # --- 3. Slow Trend ---
        degree = self.config.trend_degree
        if degree >= 0:
            # Normalize time to [-1, 1] to ensure numerical stability during polynomial evaluation
            t_norm = (t - dur / 2.0) / (dur / 2.0) if dur > 0 else t
            coeffs = self.rng.normal(0.0, 1.0, size=degree + 1)
            slow_trend = np.zeros_like(t)
            for d in range(degree + 1):
                slow_trend += coeffs[d] * (t_norm ** d)
            slow_trend = normalize_to_rms(slow_trend, 1.0)
        else:
            slow_trend = np.zeros_like(t)

        # --- Combine Components ---
        w_resp = self.config.resp_fraction
        w_drift = self.config.drift_fraction
        w_trend = self.config.trend_fraction
        
        total_w = w_resp + w_drift + w_trend
        if total_w <= 0:
            total_w = 1.0
            
        combined = (
            (w_resp / total_w) * bw_resp +
            (w_drift / total_w) * lf_drift +
            (w_trend / total_w) * slow_trend
        )
        
        # Center the output and scale to target amplitude
        combined_centered = combined - np.mean(combined)
        return normalize_to_rms(combined_centered, self.config.amplitude)

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel baseline wander with optional spatial coupling.
        
        Parameters
        ----------
        n_channels : int
            Number of channels to generate.
        n_samples : int
            Number of samples per channel.
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
        for c in range(n_channels):
            noise_matrix[c] = self.generate(n_samples, fs)
            
        # Apply spatial mixing if configured (coupled chest movements)
        if self.cholesky_matrix is not None:
            if n_channels != self.cholesky_matrix.shape[0]:
                raise ValueError(
                    f"Requested {n_channels} channels, but spatial correlation matrix "
                    f"is configured for {self.cholesky_matrix.shape[0]} channels."
                )
            noise_matrix = np.dot(self.cholesky_matrix, noise_matrix)
            
        return noise_matrix
