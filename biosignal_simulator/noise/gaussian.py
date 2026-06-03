"""
Comprehensive, high-fidelity Gaussian and general white noise generator for physiological signals.

This module provides the `GaussianNoise` class, which extends the `BaseNoiseModel`
to simulate various white noise distributions. While referred to as "Gaussian Noise" due
to its historical role in Additive White Gaussian Noise (AWGN) models, this class is
extended to support multiple probability distributions, multi-channel spatial correlation,
and non-stationary time-varying variance envelopes.

Physiological and Physical Context:
    White noise is commonly used to model wideband instrumentation thermal noise (Johnson-Nyquist),
    background amplifier noise, and unstructured environmental interference. In multi-channel
    systems (e.g., EEG or multi-lead ECG), sensor noise can be spatially correlated due to
    shared reference electrodes, electrical coupling, or volume conduction.

Mathematical Formulations:
    1. Probability Density Functions (PDFs):
       - Gaussian:
         $$P(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} \\exp\\left( -\\frac{(x - \\mu)^2}{2\\sigma^2} \\right)$$
       - Uniform:
         $$P(x) = \\frac{1}{b - a} \\quad \\text{for } a \\le x \\le b$$
       - Laplacian (Double Exponential):
         $$P(x) = \\frac{1}{2b} \\exp\\left( -\\frac{|x - \\mu|}{b} \\right) \\quad \\text{where } b = \\frac{\\sigma}{\\sqrt{2}}$$
       - Student-t:
         $$P(x) = \\frac{\\Gamma(\\frac{\\nu+1}{2})}{\\sqrt{\\nu\\pi}\\Gamma(\\frac{\\nu}{2})} \\left(1 + \\frac{x^2}{\\nu}\\right)^{-\\frac{\\nu+1}{2}}$$
       - Cauchy (Heavy-Tailed):
         $$P(x) = \\frac{1}{\\pi \\gamma \\left(1 + \\left(\\frac{x - x_0}{\\gamma}\\right)^2\\right)}$$

    2. Spatial Correlation:
       Given a target correlation matrix $R$, we compute its Cholesky decomposition:
       $$R = L L^T$$
       Where $L$ is a lower triangular matrix. Uncorrelated noise channels $W$ are mixed to obtain
       spatially correlated noise $X$:
       $$X = L W$$

    3. Time-Varying Envelope (Non-Stationarity):
       The noise variance can be modulated in time by an envelope $E(t)$:
       $$y(t) = \\mu + E(t) \\cdot \\sigma \\cdot w(t)$$
       Where $w(t)$ is a zero-mean unit-variance white noise process.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy.stats import t as student_t
from scipy.stats import cauchy
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import GaussianNoiseConfig
from biosignal_simulator.utils.validation import validate_config

class GaussianNoise(BaseNoiseModel):
    """
    Advanced White Noise Generator supporting multiple statistical distributions.
    
    Generates uncorrelated or spatially correlated white noise with stationary or
    time-varying amplitude envelopes.
    """
    
    def __init__(
        self,
        config: Optional[GaussianNoiseConfig] = None,
        distribution: str = "gaussian",
        distribution_params: Optional[dict] = None,
        envelope_type: Optional[str] = None,
        envelope_params: Optional[dict] = None,
        spatial_correlation: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Initialize the Gaussian/White Noise generator.
        
        Parameters
        ----------
        config : Optional[GaussianNoiseConfig]
            The base configuration containing mean, std, and seed.
        distribution : str
            The target distribution: 'gaussian', 'uniform', 'laplacian', 'student_t', 'cauchy'.
            Default is 'gaussian'.
        distribution_params : Optional[dict]
            Additional parameters for the chosen distribution:
              - 'student_t': {'df': degrees_of_freedom (float)}
              - 'cauchy': {'scale': scale_parameter (float)}
        envelope_type : Optional[str]
            Non-stationary amplitude envelope: None, 'sinusoidal', 'exponential', 'random_walk'.
        envelope_params : Optional[dict]
            Parameters matching the envelope type:
              - 'sinusoidal': {'freq_hz': frequency (float), 'depth': modulation_depth (float)}
              - 'exponential': {'decay_rate': decay_rate (float)}
              - 'random_walk': {'step_std': step_std (float)}
        spatial_correlation : Optional[np.ndarray]
            Covariance or correlation matrix of shape (n_channels, n_channels) for generating
            multi-channel correlated white noise.
        **kwargs :
            Alternative parameters passed to GaussianNoiseConfig if config is None.
        """
        if config is None:
            try:
                config = GaussianNoiseConfig(**kwargs)
            except TypeError as e:
                valid_keys = {'std', 'mean', 'seed'}
                invalid_keys = set(kwargs.keys()) - valid_keys
                if invalid_keys:
                    raise TypeError(
                        f"GaussianNoise got unexpected keyword argument(s): {invalid_keys}. "
                        f"Accepted config arguments are: {valid_keys}. "
                        "Note: SNR scaling is set at the SignalMixer level via 'target_snr_db', "
                        "not at the noise model level."
                    ) from e
                raise e
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.distribution = distribution.lower().strip()
        self.distribution_params = distribution_params or {}
        self.envelope_type = envelope_type.lower().strip() if envelope_type else None
        self.envelope_params = envelope_params or {}
        
        # Spatial correlation matrix handling
        self.spatial_correlation = spatial_correlation
        self.cholesky_matrix: Optional[np.ndarray] = None
        if self.spatial_correlation is not None:
            self._precompute_spatial_mixing()
            
        self._validate_noise_parameters()

    def _validate_noise_parameters(self) -> None:
        """Validate distribution-specific parameters."""
        allowed_dists = {"gaussian", "uniform", "laplacian", "student_t", "cauchy"}
        if self.distribution not in allowed_dists:
            raise ValueError(f"Unsupported distribution '{self.distribution}'. Allowed: {allowed_dists}")
            
        if self.distribution == "student_t":
            df = self.distribution_params.get("df", 5.0)
            if df <= 0:
                raise ValueError("Degrees of freedom 'df' for Student-t must be positive.")
                
        if self.distribution == "cauchy":
            scale = self.distribution_params.get("scale", 1.0)
            if scale <= 0:
                raise ValueError("Scale parameter for Cauchy distribution must be positive.")
                
        if self.envelope_type:
            allowed_envs = {"sinusoidal", "exponential", "random_walk"}
            if self.envelope_type not in allowed_envs:
                raise ValueError(f"Unsupported envelope type '{self.envelope_type}'. Allowed: {allowed_envs}")

    def _precompute_spatial_mixing(self) -> None:
        """Compute the Cholesky factor for multi-channel correlation mixing."""
        cov = np.atleast_2d(self.spatial_correlation)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"Spatial correlation matrix must be square, got shape {cov.shape}")
        
        # Check symmetry
        if not np.allclose(cov, cov.T, atol=1e-8):
            raise ValueError("Spatial correlation matrix must be symmetric.")
            
        # Compute Cholesky decomposition L L^T
        try:
            # Add small diagonal regularization (jitter) to ensure positive-definiteness
            reg_cov = cov + np.eye(cov.shape[0]) * 1e-12
            self.cholesky_matrix = np.linalg.cholesky(reg_cov)
        except np.linalg.LinAlgError:
            raise ValueError("Spatial correlation matrix is not positive-definite and cannot be decomposed.")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D white noise values.
        
        Note: If `spatial_correlation` was set, this method generates 1-D uncorrelated
        noise. To generate multi-channel correlated noise, call `generate_multichannel()`.
        
        Parameters
        ----------
        n_samples : int
            Number of sample points.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            1-D noise array of shape (n_samples,).
        """
        if n_samples <= 0:
            return np.empty(0, dtype=np.float64)
            
        # 1. Generate core raw noise sequence
        raw_noise = self._draw_samples(n_samples)
        
        # 2. Apply non-stationary envelope if configured
        if self.envelope_type:
            envelope = self._compute_envelope(n_samples, fs)
            raw_noise = raw_noise * envelope
            
        # 3. Scale and shift based on config
        scaled_noise = self.config.mean + (self.config.std * raw_noise)
        return scaled_noise

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel noise with optional spatial correlation.
        
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
            
        # Generate independent noise for each channel
        noise_matrix = np.zeros((n_channels, n_samples))
        for c in range(n_channels):
            # Temporarily disable spatial mixing during individual channel generation
            noise_matrix[c] = self._draw_samples(n_samples)
            
        # Apply time-varying envelope if configured (shared across channels or independent)
        if self.envelope_type:
            for c in range(n_channels):
                envelope = self._compute_envelope(n_samples, fs)
                noise_matrix[c] = noise_matrix[c] * envelope
                
        # Scale to mean/std
        noise_matrix = self.config.mean + (self.config.std * noise_matrix)
        
        # Apply spatial correlation matrix if configured
        if self.cholesky_matrix is not None:
            if n_channels != self.cholesky_matrix.shape[0]:
                raise ValueError(
                    f"Requested {n_channels} channels, but spatial correlation matrix "
                    f"is configured for {self.cholesky_matrix.shape[0]} channels."
                )
            # Mix channels using Cholesky factor: X = L * W
            # W is shape (n_channels, n_samples)
            noise_matrix = np.dot(self.cholesky_matrix, noise_matrix)
            
        return noise_matrix

    def _draw_samples(self, n_samples: int) -> np.ndarray:
        """Draw raw unit-variance zero-mean samples based on configured distribution."""
        dist = self.distribution
        
        if dist == "gaussian":
            return self.rng.normal(0.0, 1.0, size=n_samples)
            
        elif dist == "uniform":
            # Standard uniform distribution in [-sqrt(3), sqrt(3)] has variance 1.0
            bound = np.sqrt(3.0)
            return self.rng.uniform(-bound, bound, size=n_samples)
            
        elif dist == "laplacian":
            # Laplace distribution scale parameter b = 1/sqrt(2) gives variance 1.0
            scale = 1.0 / np.sqrt(2.0)
            return self.rng.laplace(0.0, scale, size=n_samples)
            
        elif dist == "student_t":
            df = self.distribution_params.get("df", 5.0)
            raw = student_t.rvs(df, size=n_samples, random_state=self.rng)
            # Student-t variance is df / (df - 2) for df > 2. Normalize to unit variance.
            if df > 2.0:
                std_dev = np.sqrt(df / (df - 2.0))
                return raw / std_dev
            return raw
            
        elif dist == "cauchy":
            scale = self.distribution_params.get("scale", 1.0)
            # Cauchy has undefined variance/mean. We scale standard Cauchy directly.
            return cauchy.rvs(loc=0.0, scale=scale, size=n_samples, random_state=self.rng)
            
        else:
            return self.rng.normal(0.0, 1.0, size=n_samples)

    def _compute_envelope(self, n_samples: int, fs: float) -> np.ndarray:
        """Compute the time-varying amplitude envelope array."""
        t = np.arange(n_samples) / fs
        
        if self.envelope_type == "sinusoidal":
            freq = self.envelope_params.get("freq_hz", 0.5)
            depth = self.envelope_params.get("depth", 0.5)  # modulation index in [0, 1]
            # Envelope swings between (1 - depth) and (1 + depth)
            return 1.0 + depth * np.sin(2.0 * np.pi * freq * t)
            
        elif self.envelope_type == "exponential":
            decay = self.envelope_params.get("decay_rate", 0.5)
            return np.exp(-decay * t)
            
        elif self.envelope_type == "random_walk":
            step = self.envelope_params.get("step_std", 0.01)
            # Cumulative sum of white noise steps
            walk = np.cumsum(self.rng.normal(0.0, step, size=n_samples))
            # Keep envelope strictly positive
            return np.abs(1.0 + walk)
            
        return np.ones(n_samples)
