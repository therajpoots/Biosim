"""
High-fidelity Colored Noise generators for physiological and environmental simulations.

This module implements models to generate colored noise sequences where the power
spectral density (PSD) follows a power-law relationship:
$$S(f) \\propto \\frac{1}{f^\\alpha}$$
where $\\alpha$ is the spectral exponent.

Supported Noise Colors:
    1. White Noise ($\\alpha = 0.0$): Equal power per Hz.
    2. Pink Noise ($\\alpha = 1.0$): Equal power per octave (1/f). Common in biological systems,
       brain dynamics (EEG background), and cardiac dynamics (heart rate variability).
    3. Brown/Red Noise ($\\alpha = 2.0$): Brownian motion / random walk (1/f^2). Models physical
       drift and diffusion processes (e.g., electrode baseline wander).
    4. Blue Noise ($\\alpha = -1.0$): Power proportional to frequency (f). Models high-frequency
       sensor noise or noise-shaping quantization errors.
    5. Violet Noise ($\\alpha = -2.0$): Power proportional to f^2. Models acoustic or thermal
       hydrodynamic noise.

Mathematical Algorithms:
    1. FFT Spectral Shaping (Frequency Domain):
       - Generate standard Gaussian white noise $w(t)$.
       - Compute its Fourier Transform $W(f) = \\mathcal{F}\\{w(t)\\}$.
       - Multiply by the spectral shaping vector:
         $$H(f) = f^{-\\alpha/2} \\quad \\text{for } f > 0, \\quad H(0) = 0$$
       - Compute the Inverse Fourier Transform to obtain the colored noise in the time domain:
         $$x(t) = \\mathcal{F}^{-1}\\{W(f) H(f)\\}$$
       - Advantages: Works for any real-valued $\\alpha$, highly accurate.
       - Disadvantages: Requires the entire sequence length beforehand, periodic boundary effects.

    2. Voss-McCartney Algorithm (Time Domain):
       - A cascaded structure of $K$ independent white noise generators (random walks) running
         at power-of-two sub-harmonics of the sampling frequency.
       - The generators are updated at progressive octave rates (e.g. generator $k$ is updated
         every $2^k$ samples).
       - The output is the sum of all generators:
         $$y[n] = \\sum_{k=0}^{K-1} u_k[n]$$
       - Advantages: Real-time, streaming-friendly.
       - Disadvantages: Only approximates pink noise ($\\alpha \\approx 1.0$).

    3. Rational IIR Filter Approximation (Poles/Zeros):
       - Applies a digital IIR filter to standard white noise.
       - The filter coefficients are optimized to approximate the $3$ dB/octave attenuation of pink noise:
         $$H(z) = \\frac{b_0 + b_1 z^{-1} + b_2 z^{-2} + b_3 z^{-3}}{1 + a_1 z^{-1} + a_2 z^{-2} + a_3 z^{-3}}$$
       - Advantages: Extremely fast, real-time, streaming.
       - Disadvantages: Harder to generalize to arbitrary exponents.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import ColoredNoiseConfig
from biosignal_simulator.core.math_utils import normalize_to_rms, spectral_shape
from biosignal_simulator.utils.validation import validate_config

class ColoredNoise(BaseNoiseModel):
    """
    Advanced Colored Noise Generator supporting multiple colors and generation methods.
    
    Generates 1/f^alpha noise using FFT spectral shaping, Voss-McCartney octave cascades,
    or rational IIR filters, with optional multi-channel spatial correlation.
    """
    
    def __init__(
        self,
        config: Optional[ColoredNoiseConfig] = None,
        spatial_correlation: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initialize the Colored Noise generator.
        
        Parameters
        ----------
        config : Optional[ColoredNoiseConfig]
            The base configuration specifying exponent, std, method, and seed.
        spatial_correlation : Optional[np.ndarray]
            Covariance or correlation matrix of shape (n_channels, n_channels)
            for generating multi-channel spatially correlated colored noise.
        **kwargs :
            Alternative parameters passed to ColoredNoiseConfig if config is None.
        """
        if config is None:
            config = ColoredNoiseConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        
        # Spatial correlation mixing precomputation
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
            raise ValueError("Spatial correlation matrix is not positive-definite and cannot be decomposed.")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D colored noise.
        
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
            
        exponent = self.config.exponent
        method = self.config.method.lower().strip()
        
        # 1. Select and run the core generation algorithm
        if method == "iir" and np.isclose(exponent, 1.0):
            # Optimized 3-pole/3-zero pink noise filter (Voss-McCartney alternative)
            # Fits -3dB/octave slope between 0.001 * fs/2 and fs/2
            white = self.rng.normal(0.0, 1.0, size=n_samples)
            b = [0.049922035, -0.095993537, 0.050293873, -0.005082711]
            a = [1.0, -2.494956002, 2.017265875, -0.522189400]
            colored = sp_signal.lfilter(b, a, white)
            
        elif method == "voss" and np.isclose(exponent, 1.0):
            # Voss-McCartney octave random walk cascade for pink noise
            colored = self._generate_voss_pink(n_samples)
            
        elif method == "iir" and np.isclose(exponent, 2.0):
            # Brownian noise using simple leaky integration (AR(1) process)
            white = self.rng.normal(0.0, 1.0, size=n_samples)
            # a=1.0 is pure integration (unstable/non-stationary), leaky integration a=0.999 is stable
            b = [1.0]
            a = [1.0, -0.999]
            colored = sp_signal.lfilter(b, a, white)
            
        else:
            # Default FFT spectral shaping (works for any arbitrary real-valued exponent)
            white = self.rng.normal(0.0, 1.0, size=n_samples)
            w_fft = np.fft.rfft(white)
            shaper = spectral_shape(n_samples, fs, exponent)
            shaped_fft = w_fft * shaper
            colored = np.fft.irfft(shaped_fft, n=n_samples)
            
        # 2. Post-processing: Remove DC offset and normalize to target RMS
        colored_centered = colored - np.mean(colored)
        return normalize_to_rms(colored_centered, self.config.std)

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel colored noise with optional spatial correlation.
        
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
            
        # Generate independent colored noise per channel
        noise_matrix = np.zeros((n_channels, n_samples))
        for c in range(n_channels):
            noise_matrix[c] = self.generate(n_samples, fs)
            
        # Apply spatial correlation mixing matrix if configured
        if self.cholesky_matrix is not None:
            if n_channels != self.cholesky_matrix.shape[0]:
                raise ValueError(
                    f"Requested {n_channels} channels, but spatial correlation matrix "
                    f"is configured for {self.cholesky_matrix.shape[0]} channels."
                )
            noise_matrix = np.dot(self.cholesky_matrix, noise_matrix)
            
        return noise_matrix

    def _generate_voss_pink(self, n_samples: int) -> np.ndarray:
        """
        Generate pink noise using the Voss-McCartney algorithm.
        
        Creates 1/f noise by summing random numbers updated at different frequencies.
        
        B-17 FIX: The original used a 2-column ping-pong buffer which caused
        each octave value to be overwritten every other step regardless of the
        intended update frequency, degrading spectral accuracy for large n_samples.
        The correct implementation keeps one value per octave, updated only at
        its power-of-two step interval.
        """
        # Determine number of octaves based on sample count (max 16)
        num_octaves = min(16, int(np.ceil(np.log2(max(n_samples, 2)))))
        if num_octaves < 1:
            num_octaves = 1
            
        # One state value per octave (updated at 2^k step intervals)
        octave_values = self.rng.normal(0.0, 1.0, size=num_octaves)
        current_sum = float(np.sum(octave_values))
        
        output = np.zeros(n_samples)
        
        for idx in range(n_samples):
            output[idx] = current_sum
            
            if idx + 1 < n_samples:
                # Find the lowest-order bit that changes at step idx+1
                # (trailing zeros of idx+1 tell us which octave to refresh)
                step = idx + 1
                # Number of trailing zeros == octave to update
                octave_to_update = (step & -step).bit_length() - 1
                if octave_to_update < num_octaves:
                    old_val = octave_values[octave_to_update]
                    new_val = self.rng.normal(0.0, 1.0)
                    octave_values[octave_to_update] = new_val
                    current_sum += (new_val - old_val)
                    
        return output


class PinkNoise(ColoredNoise):
    """
    Pink Noise (1/f) Generator.
    
    Power decreases by 3 dB per octave. Very common in biological and brain systems.
    """
    
    def __init__(self, std: float = 1.0, method: str = "fft", seed: Optional[int] = None, **kwargs):
        cfg = ColoredNoiseConfig(exponent=1.0, std=std, method=method, seed=seed)
        super().__init__(config=cfg, **kwargs)


class BrownNoise(ColoredNoise):
    """
    Brownian Noise (1/f^2) Generator (Red Noise).
    
    Power decreases by 6 dB per octave. Models physical diffusion and baseline drift.
    """
    
    def __init__(self, std: float = 1.0, method: str = "fft", seed: Optional[int] = None, **kwargs):
        cfg = ColoredNoiseConfig(exponent=2.0, std=std, method=method, seed=seed)
        super().__init__(config=cfg, **kwargs)


class BlueNoise(ColoredNoise):
    """
    Blue Noise (f) Generator.
    
    Power increases by 3 dB per octave. Models high-frequency quantization error/dither.
    """
    
    def __init__(self, std: float = 1.0, method: str = "fft", seed: Optional[int] = None, **kwargs):
        cfg = ColoredNoiseConfig(exponent=-1.0, std=std, method=method, seed=seed)
        super().__init__(config=cfg, **kwargs)


class VioletNoise(ColoredNoise):
    """
    Violet Noise (f^2) Generator.
    
    Power increases by 6 dB per octave. Models high-frequency sensor thermal dynamics.
    """
    
    def __init__(self, std: float = 1.0, method: str = "fft", seed: Optional[int] = None, **kwargs):
        cfg = ColoredNoiseConfig(exponent=-2.0, std=std, method=method, seed=seed)
        super().__init__(config=cfg, **kwargs)
