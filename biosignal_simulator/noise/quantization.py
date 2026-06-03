"""
Advanced Analog-to-Digital Converter (ADC) Quantization and Dither Simulator.

This module provides the `QuantizationNoise` class, which models the error introduced
by converting continuous physiological voltages into discrete, finite-resolution digital values.

Physical and Metrological Context:
    1. Uniform Quantization:
       Continuous input signal values are mapped to the nearest discrete quantization level.
       The step size (Least Significant Bit, LSB) $\\Delta$ is:
       $$\\Delta = \\frac{V_{\\text{range}}}{2^{B}}$$
       where $V_{\\text{range}}$ is the input voltage range and $B$ is the ADC bit depth (resolution).
       The quantization error is bounded by $[-\\Delta/2, \\Delta/2]$.

    2. Dithering:
       In low-resolution systems (e.g. 8-bit or 10-bit ADCs), quantization error is highly
       correlated with the signal, introducing harmonics and distortion. Dithering breaks up
       this correlation by adding a small amount of random noise before quantization:
       - Rectangular Probability Density Function (RPDF): Uniform noise of $\\pm 0.5 \\Delta$.
       - Triangular Probability Density Function (TPDF): Sum of two independent RPDF processes,
         covering $\\pm 1.0 \\Delta$. TPDF completely eliminates noise-modulation and harmonic distortion.

    3. Non-Uniform Companding (A-law & $\\mu$-law):
       Logarithmic compression used to optimize dynamic range for low-amplitude signals:
       - $\\mu$-law compression:
         $$F(x) = \\text{sgn}(x) \\cdot \\frac{\\ln(1 + \\mu |x|)}{\\ln(1 + \\mu)} \\quad \\text{for } |x| \\le 1$$
       - A-law compression:
         $$F(x) = \\begin{cases}
         \\text{sgn}(x) \\cdot \\frac{A |x|}{1 + \\ln(A)} & \\text{for } |x| < \\frac{1}{A} \\\\
         \\text{sgn}(x) \\cdot \\frac{1 + \\ln(A |x|)}{1 + \\ln(A)} & \\text{for } \\frac{1}{A} \\le |x| \\le 1
         \\end{cases}$$

    4. Noise Shaping (Error Feedback):
       Quantization errors are filtered and subtracted from subsequent input samples. This
       shapes the noise spectrum, shifting power from low-frequency bands (where physiological
       signals reside) to high-frequency bands (near Nyquist).
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import QuantizationNoiseConfig
from biosignal_simulator.utils.validation import validate_config

class QuantizationNoise(BaseNoiseModel):
    """
    ADC Quantization, Companding, and Dither Simulator.
    
    Models uniform quantization, non-uniform logarithmic companding (A-law/mu-law),
    multiple dither profiles (RPDF/TPDF/Gaussian), and high-order noise-shaping filters.
    """
    
    def __init__(
        self,
        config: Optional[QuantizationNoiseConfig] = None,
        companding: Optional[str] = None,
        companding_factor: float = 255.0,
        noise_shaping_order: int = 0,
        **kwargs
    ):
        """
        Initialize the Quantization Noise model.
        
        Parameters
        ----------
        config : Optional[QuantizationNoiseConfig]
            Base configuration containing bit depth, voltage range, dither enablement, and seed.
        companding : Optional[str]
            Logarithmic compression scheme: None, 'mu_law', or 'a_law'. Default is None.
        companding_factor : float
            Compression scale factor (commonly mu=255 or A=87.6). Default is 255.0.
        noise_shaping_order : int
            Noise shaping feedback filter order: 0 (disabled), 1 (1st order), or 2 (2nd order).
            Default is 0.
        **kwargs :
            Alternative parameters passed to QuantizationNoiseConfig if config is None.
        """
        if config is None:
            config = QuantizationNoiseConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.companding = companding.lower().strip() if companding else None
        self.companding_factor = companding_factor
        self.noise_shaping_order = noise_shaping_order
        
        self._validate_quantization_parameters()

    def _validate_quantization_parameters(self) -> None:
        """Validate input parameters."""
        if self.companding:
            allowed_schemes = {"mu_law", "a_law"}
            if self.companding not in allowed_schemes:
                raise ValueError(f"Unsupported companding scheme '{self.companding}'. Allowed: {allowed_schemes}")
            if self.companding_factor <= 0:
                raise ValueError("Companding compression factor must be positive.")
                
        if self.noise_shaping_order not in {0, 1, 2}:
            raise ValueError("Noise shaping order must be 0 (disabled), 1, or 2.")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize a standard uniform quantization error sequence.
        
        Note: To apply quantization directly to a physiological waveform,
        call `apply()`.
        
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
            
        delta = self.config.v_range / (2.0 ** self.config.n_bits)
        
        if self.config.dither:
            # Triangular dither (TPDF)
            u1 = self.rng.uniform(-0.5 * delta, 0.5 * delta, size=n_samples)
            u2 = self.rng.uniform(-0.5 * delta, 0.5 * delta, size=n_samples)
            return u1 + u2
            
        # Uniform rounding error in [-0.5 LSB, 0.5 LSB]
        return self.rng.uniform(-0.5 * delta, 0.5 * delta, size=n_samples)

    def apply(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantize an input signal, returning the discrete result and the error.
        
        Parameters
        ----------
        signal : np.ndarray
            Continuous-value input signal array. Can be 1-D or 2-D.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (quantized_signal, quantization_error)
        """
        if signal.ndim == 2:
            n_ch, n_samples = signal.shape
            quantized = np.zeros_like(signal)
            error = np.zeros_like(signal)
            for c in range(n_ch):
                quantized[c], error[c] = self._apply_1d(signal[c])
            return quantized, error
        else:
            return self._apply_1d(signal)

    def _apply_1d(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply quantization to a 1D array."""
        n_samples = len(x)
        if n_samples == 0:
            return np.empty(0), np.empty(0)
            
        delta = self.config.v_range / (2.0 ** self.config.n_bits)
        
        # --- Step 1. Non-linear Companding (Compression) ---
        x_scaled = x.copy()
        sig_max = np.max(np.abs(x))
        if sig_max > 1e-12:
            # Normalize to [-1, 1] for companding formulas
            x_scaled = x / sig_max
            
        if self.companding == "mu_law":
            mu = self.companding_factor
            x_scaled = np.sign(x_scaled) * np.log(1.0 + mu * np.abs(x_scaled)) / np.log(1.0 + mu)
        elif self.companding == "a_law":
            a_val = self.companding_factor
            inv_a = 1.0 / a_val
            denom = 1.0 + np.log(a_val)
            
            mask_low = np.abs(x_scaled) < inv_a
            x_scaled[mask_low] = np.sign(x_scaled[mask_low]) * (a_val * np.abs(x_scaled[mask_low])) / denom
            x_scaled[~mask_low] = np.sign(x_scaled[~mask_low]) * (1.0 + np.log(a_val * np.abs(x_scaled[~mask_low]))) / denom
            
        if sig_max > 1e-12:
            # Rescale back to original physical amplitude limits
            x_scaled = x_scaled * sig_max

        # --- Step 2. Generate Dither Noise ---
        dither_val = np.zeros(n_samples)
        if self.config.dither:
            # Triangular dither (TPDF)
            u1 = self.rng.uniform(-0.5 * delta, 0.5 * delta, size=n_samples)
            u2 = self.rng.uniform(-0.5 * delta, 0.5 * delta, size=n_samples)
            dither_val = u1 + u2

        # --- Step 3. Quantize with optional Noise-Shaping (Error Feedback) ---
        quantized = np.zeros(n_samples)
        
        if self.noise_shaping_order > 0:
            # Allocate state arrays for error feedback
            error_feedback = np.zeros(self.noise_shaping_order)
            
            for idx in range(n_samples):
                # Subtract accumulated feedback error from current sample
                if self.noise_shaping_order == 1:
                    feedback = error_feedback[0]
                else:  # 2nd order feedback: 2*e[n-1] - e[n-2]
                    feedback = 2.0 * error_feedback[0] - error_feedback[1]
                    
                val_dithered = x_scaled[idx] + dither_val[idx] - feedback
                
                # Perform rounding quantization step
                q_val = np.round(val_dithered / delta) * delta
                quantized[idx] = q_val
                
                # Compute actual quantization error at this step
                err_val = q_val - val_dithered
                
                # Shift state arrays
                if self.noise_shaping_order == 1:
                    error_feedback[0] = err_val
                else:
                    error_feedback[1] = error_feedback[0]
                    error_feedback[0] = err_val
        else:
            # Standard rounding quantization
            quantized = np.round((x_scaled + dither_val) / delta) * delta

        # --- Step 4. Non-linear Companding (Expansion/Decompression) ---
        if sig_max > 1e-12:
            quantized = quantized / sig_max
            
        if self.companding == "mu_law":
            mu = self.companding_factor
            quantized = np.sign(quantized) * ((1.0 + mu) ** np.abs(quantized) - 1.0) / mu
        elif self.companding == "a_law":
            a_val = self.companding_factor
            denom = 1.0 + np.log(a_val)
            inv_denom = 1.0 / denom
            
            mask_low = np.abs(quantized) < inv_denom
            quantized[mask_low] = np.sign(quantized[mask_low]) * (np.abs(quantized[mask_low]) * denom) / a_val
            quantized[~mask_low] = np.sign(quantized[~mask_low]) * np.exp(np.abs(quantized[~mask_low]) * denom - 1.0) / a_val
            
        if sig_max > 1e-12:
            quantized = quantized * sig_max
            
        # Error is the difference between quantized output and original continuous input
        noise = quantized - x
        return quantized, noise
