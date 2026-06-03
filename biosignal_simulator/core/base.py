"""
Base abstractions, decorators, registry patterns, and core exceptions for physiological simulations.

This module provides the core framework of the BioSignal Simulator library:
1. A comprehensive, hierarchical exception tree for signal processing, modeling, and serialization.
2. The SignalRegistry metaclass for automatic registration and lookup of signal generators.
3. Base classes for signals (BaseSignal) and noise sources (BaseNoiseModel) with type enforcement.
4. Parameter validation decorators to verify inputs against physiological boundaries.
5. Advanced DSP baseline correction algorithms (double median, Savitzky-Golay, splines, highpass IIR).
6. Resampling and interpolation methods (Linear, Cubic, Quadratic Spline, Lanczos windowed sinc).

Physiological Reference:
    McSharry, P. E., et al. (2003). "A dynamical model for generating synthetic electrocardiogram signals."
    IEEE Transactions on Biomedical Engineering, 50(3), 289-294.
"""

import functools
from abc import ABC, abstractmethod, ABCMeta
from typing import Optional, Tuple, Dict, Any, Callable, TypeVar, Type, Union, List, ClassVar
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

# =====================================================================
# 1. Custom Exception Hierarchy
# =====================================================================

class BiosignalError(Exception):
    """Base exception class for all errors in the biosignal simulator library."""
    pass


class ParameterValidationError(BiosignalError):
    """
    Raised when configuration parameters fall outside physiological or physical limits.
    
    Examples: sampling rate <= 0, duration <= 0, heart rate outside [40, 200] bpm.
    """
    pass


class SignalGenerationError(BiosignalError):
    """
    Raised when an error occurs during signal synthesis.
    
    Examples: numerical solver instability, invalid state transitions, or convergence failures.
    """
    pass


class NoiseGenerationError(BiosignalError):
    """
    Raised when noise synthesis fails.
    
    Examples: unstable colored noise filter coefficients, invalid PSD grids.
    """
    pass


class SignalProcessingError(BiosignalError):
    """
    Raised during mathematical processing steps.
    
    Examples: resampling failures, out-of-bounds interpolation, or baseline removal instability.
    """
    pass


class RegistryError(BiosignalError):
    """
    Raised when looking up an unregistered generator or duplicate registration conflicts.
    """
    pass

# =====================================================================
# 2. Metaclasses and Registry Pattern
# =====================================================================

class SignalRegistry(ABCMeta):
    """
    Metaclass that automatically registers all subclasses of BaseSignal.
    
    Enables dynamic lookup of signal generators by their string identifier,
    facilitating dynamic configuration-driven factory patterns (e.g. from JSON/YAML).
    """
    _registry: ClassVar[Dict[str, Type['BaseSignal']]] = {}

    def __new__(mcs, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]) -> type:
        cls = super().__new__(mcs, name, bases, attrs)
        # Register classes that inherit from BaseSignal and are not abstract (have a generate method)
        if name != 'BaseSignal' and not attrs.get('__abstractmethods__', None):
            reg_name = name.replace('Generator', '').lower()
            if reg_name in mcs._registry:
                raise RegistryError(f"Duplicate registration conflict: '{reg_name}' is already registered.")
            mcs._registry[reg_name] = cls  # type: ignore
        return cls

    @classmethod
    def get_generator(mcs, name: str) -> Type['BaseSignal']:
        """
        Look up a registered signal generator class by its name.
        
        Parameters
        ----------
        name : str
            String identifier of the signal (e.g. 'ecg', 'eeg', 'emg').
            
        Returns
        -------
        Type[BaseSignal]
            The matching generator class.
            
        Raises
        ------
        RegistryError
            If no matching generator class is registered.
        """
        clean_name = name.strip().lower()
        if clean_name not in mcs._registry:
            raise RegistryError(
                f"Unknown signal generator '{name}'. Registered options: {list(mcs._registry.keys())}"
            )
        return mcs._registry[clean_name]

    @classmethod
    def list_generators(mcs) -> List[str]:
        """List all currently registered signal generator identifiers."""
        return list(mcs._registry.keys())


# =====================================================================
# 3. Parameter Validation Decorators
# =====================================================================

F = TypeVar('F', bound=Callable[..., Any])

def validate_params(func: F) -> F:
    """
    Decorator for generator methods that enforces physiological parameters validation.
    
    Before executing the wrapped method (typically `generate`), this decorator
    calls the instance's `validate_parameters()` method. If validation fails,
    it raises a `ParameterValidationError`.
    """
    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        is_valid, error_message = self.validate_parameters()
        if not is_valid:
            raise ParameterValidationError(
                f"Validation failed for {self.__class__.__name__}: {error_message}"
            )
        return func(self, *args, **kwargs)
    return wrapper  # type: ignore

# =====================================================================
# 4. Enums and Interpolation Options
# =====================================================================

class InterpolationMethod(Enum):
    """Supported interpolation methods for signal resampling."""
    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    NEAREST = "nearest"
    LANCZOS = "lanczos"


class BaselineRemovalMethod(Enum):
    """Supported baseline wander removal algorithms."""
    MEAN = "mean"
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    IIR_HIGHPASS = "iir_highpass"
    SAVITZKY_GOLAY = "savitzky_golay"
    DOUBLE_MEDIAN = "double_median"

# =====================================================================
# 5. Base Configuration Schemas
# =====================================================================

@dataclass
class SignalConfig:
    """
    Base configuration class for all biosignal generators.
    
    Attributes
    ----------
    fs : float
        Sampling frequency in Hz. Must be > 0.
    duration_s : float
        Signal duration in seconds. Must be > 0.
    seed : Optional[int]
        Random seed for generation reproducibility.
    """
    fs: float
    duration_s: float
    seed: Optional[int] = None


@dataclass
class NoiseConfig:
    """
    Base configuration class for all noise models.
    
    Attributes
    ----------
    seed : Optional[int]
        Random seed for generation reproducibility.
    """
    seed: Optional[int] = None

# =====================================================================
# 6. Base Signal Generator Class
# =====================================================================

class BaseSignal(ABC, metaclass=SignalRegistry):
    """
    Abstract base class for all physiological signal generators.
    
    This class manages time vectors, sampling rates, caching of generated
    arrays, basic normalization, and DSP helpers such as baseline removal.
    
    Subclasses must implement:
        - `generate()`: Returns the clean signal.
        - `validate_parameters()`: Validates config boundary conditions.
    """
    
    def __init__(
        self,
        fs: float,
        duration_s: float,
        seed: Optional[int] = None,
        multichannel: bool = False,
        n_channels: int = 1,
        cache_enabled: bool = True,
        interpolation_method: InterpolationMethod = InterpolationMethod.CUBIC
    ):
        """
        Initialize the base signal generator.
        
        Parameters
        ----------
        fs : float
            Sampling frequency in Hz (must be > 0).
        duration_s : float
            Signal duration in seconds (must be > 0).
        seed : Optional[int]
            Random seed for reproducibility (default: None).
        multichannel : bool
            Whether the signal generates multiple channels (default: False).
        n_channels : int
            Number of channels if multichannel is True (default: 1).
        cache_enabled : bool
            Whether to cache the generated signal to avoid recomputation (default: True).
        interpolation_method : InterpolationMethod
            Method to use during resampling operations (default: CUBIC).
            
        Raises
        ------
        ParameterValidationError
            If fs <= 0, duration_s <= 0, or n_channels < 1.
        """
        if fs <= 0:
            raise ParameterValidationError(
                f"Sampling frequency must be positive, got fs={fs} Hz. "
                f"Typical physiological ranges: ECG/PPG 100-500 Hz, EEG 256-1000 Hz, EMG 1000-2000 Hz."
            )
        if duration_s <= 0:
            raise ParameterValidationError(
                f"Duration must be positive, got duration_s={duration_s} s."
            )
        if n_channels < 1:
            raise ParameterValidationError(
                f"n_channels must be at least 1, got {n_channels}."
            )
            
        self.fs = float(fs)
        self.duration_s = float(duration_s)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.multichannel = multichannel
        self.n_channels = n_channels if multichannel else 1
        self.cache_enabled = cache_enabled
        self.interpolation_method = interpolation_method
        
        # Calculate sample parameters
        self._n_samples = int(np.round(self.fs * self.duration_s))
        self._t = np.arange(self._n_samples, dtype=np.float64) / self.fs
        
        # Cache management
        self._cache: Optional[np.ndarray] = None
        self._generation_count: int = 0
        
    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Generate the clean physiological signal.
        
        Must be implemented by subclasses.
        
        Returns
        -------
        np.ndarray
            A 1-D array of shape (n_samples,) or a 2-D array of shape (n_channels, n_samples)
            containing the simulated clean physiological values.
        """
        pass
        
    @abstractmethod
    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate that all configuration parameters are within physical/physiological limits.
        
        Must be implemented by subclasses.
        
        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        pass
        
    def generate_cached(self) -> np.ndarray:
        """
        Generate the signal, utilizing internal cache if enabled.
        
        Returns
        -------
        np.ndarray
            The generated physiological signal.
        """
        if self.cache_enabled and self._cache is not None:
            return self._cache.copy()
            
        signal = self.generate()
        if self.cache_enabled:
            self._cache = signal.copy()
            
        self._generation_count += 1
        return signal
        
    def clear_cache(self) -> None:
        """Clear the cached generated signal."""
        self._cache = None
        
    def reset_rng(self) -> None:
        """Reset the random number generator using the original seed."""
        self.rng = np.random.default_rng(self.seed)
        self.clear_cache()
        
    def resample(self, new_fs: float) -> np.ndarray:
        """
        Resample the generated signal to a new sampling frequency.
        
        Mathematical Formulation:
            For Lanczos resampling, the reconstruction kernel is:
                L(t) = sinc(t) * sinc(t / a)  for -a < t < a
            where 'a' is the window size parameter (typically 3).
        
        Parameters
        ----------
        new_fs : float
            New sampling frequency in Hz (must be > 0).
            
        Returns
        -------
        np.ndarray
            The resampled signal array.
            
        Raises
        ------
        ParameterValidationError
            If new_fs <= 0.
        SignalProcessingError
            If interpolation fails.
        """
        if new_fs <= 0:
            raise ParameterValidationError(f"New sampling frequency must be positive, got {new_fs} Hz.")
            
        signal = self.generate_cached()
        if np.isclose(self.fs, new_fs):
            return signal.copy()
            
        n_new_samples = int(np.round(new_fs * self.duration_s))
        t_new = np.arange(n_new_samples, dtype=np.float64) / new_fs
        
        try:
            if self.multichannel:
                resampled = np.zeros((self.n_channels, n_new_samples))
                for c in range(self.n_channels):
                    resampled[c] = self._interpolate_1d(self._t, signal[c], t_new)
                return resampled
            else:
                return self._interpolate_1d(self._t, signal, t_new)
        except Exception as e:
            raise SignalProcessingError(f"Resampling interpolation failed: {e}")
            
    def _interpolate_1d(self, t_orig: np.ndarray, x: np.ndarray, t_new: np.ndarray) -> np.ndarray:
        """Helper to interpolate 1D arrays based on configuration method."""
        if self.interpolation_method == InterpolationMethod.LINEAR:
            return np.interp(t_new, t_orig, x)
            
        if self.interpolation_method == InterpolationMethod.LANCZOS:
            return self._lanczos_interpolation(t_orig, x, t_new, a=3)
            
        from scipy.interpolate import interp1d
        method_str = "linear"
        if self.interpolation_method == InterpolationMethod.CUBIC:
            method_str = "cubic"
        elif self.interpolation_method == InterpolationMethod.SPLINE:
            method_str = "quadratic"
        elif self.interpolation_method == InterpolationMethod.NEAREST:
            method_str = "nearest"
            
        f = interp1d(t_orig, x, kind=method_str, fill_value="extrapolate")
        return f(t_new)
        
    def _lanczos_interpolation(self, t_orig: np.ndarray, x: np.ndarray, t_new: np.ndarray, a: int = 3) -> np.ndarray:
        """
        Lanczos windowed sinc interpolation for high-fidelity resampling.
        
        Parameters
        ----------
        t_orig : np.ndarray
            Original time vector.
        x : np.ndarray
            Original signal values.
        t_new : np.ndarray
            Target time vector.
        a : int
            Lanczos kernel size parameter (default: 3).
        """
        def sinc(t):
            return np.sinc(t)
            
        def lanczos_kernel(t):
            val = sinc(t) * sinc(t / a)
            # Truncate outside window boundary
            val[np.abs(t) >= a] = 0.0
            return val
            
        # Resampled output array
        y = np.zeros_like(t_new)
        # Ratio of sampling frequencies
        dt = t_orig[1] - t_orig[0] if len(t_orig) > 1 else 1.0
        
        for i, t_val in enumerate(t_new):
            # Find nearest sample indices in t_orig
            center_idx = np.searchsorted(t_orig, t_val)
            start_idx = max(0, center_idx - a - 1)
            end_idx = min(len(t_orig), center_idx + a + 1)
            
            t_slice = t_orig[start_idx:end_idx]
            x_slice = x[start_idx:end_idx]
            
            # Distance from target point scaled to sample indices
            diffs = (t_val - t_slice) / dt
            weights = lanczos_kernel(diffs)
            
            sum_w = np.sum(weights)
            if abs(sum_w) > 1e-12:
                y[i] = np.sum(x_slice * weights) / sum_w
            else:
                y[i] = 0.0
                
        return y
        
    def remove_baseline(self, method: Union[str, BaselineRemovalMethod] = 'cubic_spline') -> np.ndarray:
        """
        Estimate and remove baseline drift or DC offset from the signal.
        
        Supported Methods
        -----------------
        mean :
            Subtracts the mean value of the signal (only removes DC offset).
        linear :
            Fits a linear regression model and subtracts the trend line.
        cubic_spline :
            Fits a cubic spline using control points sampled at ~1 second intervals.
        iir_highpass :
            Applies a bidirectional 0.5 Hz highpass Butterworth filter (zero phase shift).
        savitzky_golay :
            Uses a Savitzky-Golay polynomial smoothing filter to fit the low-frequency trend.
        double_median :
            Applies consecutive median filters (e.g. 200 ms and 600 ms windows) to isolate the baseline
            (highly effective for clinical ECG baseline wander removal).
            
        Parameters
        ----------
        method : Union[str, BaselineRemovalMethod]
            The baseline removal algorithm: 'mean', 'linear', 'cubic_spline', 'iir_highpass',
            'savitzky_golay', or 'double_median' (default: 'cubic_spline').
            
        Returns
        -------
        np.ndarray
            The baseline-corrected signal array.
            
        Raises
        ------
        ValueError
            If an unsupported method is supplied.
        """
        signal = self.generate_cached()
        
        # Resolve method enum
        if isinstance(method, str):
            try:
                method_enum = BaselineRemovalMethod(method.strip().lower())
            except ValueError:
                raise ValueError(
                    f"Unknown baseline removal method: {method}. "
                    f"Available methods: {[m.value for m in BaselineRemovalMethod]}"
                )
        else:
            method_enum = method
            
        if self.multichannel:
            corrected = np.zeros_like(signal)
            for c in range(self.n_channels):
                corrected[c] = self._remove_baseline_1d(signal[c], method_enum)
            return corrected
        else:
            return self._remove_baseline_1d(signal, method_enum)
            
    def _remove_baseline_1d(self, x: np.ndarray, method: BaselineRemovalMethod) -> np.ndarray:
        """Apply baseline correction on a 1-D array."""
        
        if method == BaselineRemovalMethod.MEAN:
            return x - np.mean(x)
            
        elif method == BaselineRemovalMethod.LINEAR:
            from numpy.polynomial import Polynomial
            indices = np.arange(len(x))
            poly = Polynomial.fit(indices, x, 1)
            return x - poly(indices)
            
        elif method == BaselineRemovalMethod.CUBIC_SPLINE:
            from scipy.interpolate import UnivariateSpline
            indices = np.arange(len(x))
            
            # Subsample control points at roughly 1-second intervals
            step = max(10, int(np.round(self.fs)))
            control_indices = np.arange(0, len(x), step)
            
            if len(control_indices) < 4:
                # Fallback to mean subtraction if signal is too short for cubic spline
                return x - np.mean(x)
                
            spline = UnivariateSpline(
                control_indices, x[control_indices], k=3, s=1e6
            )
            return x - spline(indices)
            
        elif method == BaselineRemovalMethod.IIR_HIGHPASS:
            from scipy.signal import butter, filtfilt
            cutoff = 0.5
            nyq = self.fs * 0.5
            
            if cutoff >= nyq:
                # If fs is extremely low, fall back to linear baseline removal
                from numpy.polynomial import Polynomial
                indices = np.arange(len(x))
                poly = Polynomial.fit(indices, x, 1)
                return x - poly(indices)
                
            b, a = butter(4, cutoff / nyq, btype='high')
            pad_len = min(150, len(x) // 3)
            return filtfilt(b, a, x, padlen=pad_len)
            
        elif method == BaselineRemovalMethod.SAVITZKY_GOLAY:
            from scipy.signal import savgol_filter
            # Savitzky-Golay baseline estimation: large window size fits low-frequency components
            # Window size must be an odd integer, representing ~1.5 seconds of data
            win_len = int(np.round(1.5 * self.fs))
            if win_len % 2 == 0:
                win_len += 1
            win_len = max(5, min(win_len, len(x) - (1 if len(x) % 2 == 0 else 0)))
            
            if win_len < 3:
                return x - np.mean(x)
                
            baseline = savgol_filter(x, window_length=win_len, polyorder=2)
            return x - baseline
            
        elif method == BaselineRemovalMethod.DOUBLE_MEDIAN:
            # Double median filter (widely used in ECG processing)
            # Step 1: Median filter with 200 ms window to remove QRS complexes
            # Step 2: Median filter with 600 ms window to remove T-waves, leaving only baseline wander
            from scipy.signal import medfilt
            
            win_1 = int(np.round(0.2 * self.fs))
            if win_1 % 2 == 0:
                win_1 += 1
            win_1 = max(3, win_1)
            
            win_2 = int(np.round(0.6 * self.fs))
            if win_2 % 2 == 0:
                win_2 += 1
            win_2 = max(3, win_2)
            
            # Step 1
            x_filtered_1 = medfilt(x, kernel_size=win_1)
            # Step 2
            baseline = medfilt(x_filtered_1, kernel_size=win_2)
            
            return x - baseline
            
        else:
            raise ValueError(f"Unsupported baseline removal method enum: {method}")
            
    @property
    def t(self) -> np.ndarray:
        """Time vector in seconds."""
        return self._t.copy()
        
    @property
    def n_samples(self) -> int:
        """Total number of sample points."""
        return self._n_samples

    @property
    def signal_rms(self) -> float:
        """
        Calculate the root mean square (RMS) of the clean physiological signal.
        
        Formula:
            RMS = sqrt( (1 / N) * sum_{i=1}^N x_i^2 )
            
        Returns
        -------
        float
            The RMS amplitude of the generated signal.
        """
        from biosignal_simulator.core.math_utils import compute_rms
        return float(compute_rms(self.generate_cached()))

    @property
    def signal_energy(self) -> float:
        """
        Calculate the total energy of the clean physiological signal.
        
        Formula:
            Energy = sum_{i=1}^N x_i^2
            
        Returns
        -------
        float
            Total signal energy.
        """
        return float(np.sum(np.square(self.generate_cached())))

    @property
    def standard_deviation(self) -> float:
        """Calculate the standard deviation of the clean physiological signal."""
        return float(np.std(self.generate_cached()))

    @property
    def mean(self) -> float:
        """Calculate the mean value of the clean physiological signal."""
        return float(np.mean(self.generate_cached()))

    def to_record(
        self,
        noisy: Optional[np.ndarray] = None,
        noise_components: Optional[Dict[str, np.ndarray]] = None,
        snr_db: Optional[float] = None,
        notes: str = ""
    ) -> object:
        """
        Package the generated signal into a SignalRecord container.
        
        Parameters
        ----------
        noisy : Optional[np.ndarray]
            The noisy signal array. If None, it defaults to the clean signal.
        noise_components : Optional[Dict[str, np.ndarray]]
            Dictionary of individual noise component realizations.
        snr_db : Optional[float]
            The post-hoc calculated SNR in dB.
        notes : str
            Optional custom annotation string.
            
        Returns
        -------
        SignalRecord
            A fully validated and characterized SignalRecord object.
        """
        from biosignal_simulator.core.record import SignalRecord
        clean_sig = self.generate_cached()
        if noisy is None:
            noisy = clean_sig.copy()
        if noise_components is None:
            noise_components = {}
            
        # Extract configuration parameters from class instance dict
        sig_params = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and not isinstance(v, (np.random.Generator, np.ndarray, Callable)):
                if isinstance(v, Enum):
                    sig_params[k] = v.value
                else:
                    sig_params[k] = v
                
        meta = {"notes": notes}
        
        return SignalRecord(
            signal_type=self.__class__.__name__.replace("Generator", "").lower(),
            fs=self.fs,
            t=self.t,
            clean=clean_sig,
            noisy=noisy,
            noise_components=noise_components,
            signal_params=sig_params,
            snr_db=snr_db,
            metadata=meta
        )

    def plot(self, show: bool = True, filepath: Optional[str] = None) -> Any:
        """
        Plot the clean generated signal.
        
        Parameters
        ----------
        show : bool
            If True, displays the plot interactively (default: True).
        filepath : Optional[str]
            If provided, saves the plot to the specified path (default: None).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Run `pip install matplotlib`.")
            
        fig, ax = plt.subplots(figsize=(12, 4))
        signal = self.generate_cached()
        if self.multichannel:
            for ch in range(min(self.n_channels, 3)):
                ax.plot(self.t, signal[ch], label=f"Channel {ch}", linewidth=1.5)
            ax.legend(loc="upper right")
        else:
            ax.plot(self.t, signal, color="#008080", linewidth=1.5)
            
        ax.set_title(f"Simulated {self.__class__.__name__.replace('Generator', '')} Signal")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        if filepath:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
            
        return fig
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"fs={self.fs}, duration_s={self.duration_s}, "
            f"n_samples={self.n_samples}, channels={self.n_channels}, seed={self.seed})"
        )

# =====================================================================
# 7. Base Noise Model Class
# =====================================================================

class BaseNoiseModel(ABC):
    """
    Abstract base class for all physiological and environmental noise sources.
    
    Subclasses must implement:
        - `generate(n_samples, fs)`: Synthesizes raw noise values of requested size.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the noise generator.
        
        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    @abstractmethod
    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw noise array.
        
        Must be implemented by subclasses.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            A 1-D noise array of shape (n_samples,) and dtype float64.
        """
        pass
        
    def reset_rng(self) -> None:
        """Reset the random number generator using the original seed."""
        self.rng = np.random.default_rng(self.seed)
        
    def generate_scaled(self, signal: np.ndarray, snr_db: float, fs: float) -> np.ndarray:
        """
        Generate noise scaled to a target Signal-to-Noise Ratio (SNR) in decibels.
        
        Power is computed based on mean squared amplitude.
        
        Parameters
        ----------
        signal : np.ndarray
            Clean reference signal array (used to compute target noise power).
            Can be 1-D or 2-D.
        snr_db : float
            Target SNR in dB.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            Noise array matching the shape of `signal`, scaled to the target SNR.
        """
        # Generate raw noise matching the shape of signal
        if signal.ndim == 2:
            n_ch, n_samples = signal.shape
            noise_raw = np.zeros_like(signal)
            for c in range(n_ch):
                noise_raw[c] = self.generate(n_samples, fs)
        else:
            n_samples = len(signal)
            noise_raw = self.generate(n_samples, fs)
            
        # Calculate signal and noise power
        p_signal = np.mean(np.square(signal))
        p_noise_raw = np.mean(np.square(noise_raw))
        
        if p_signal <= 1e-15:
            p_signal = 1e-15
        if p_noise_raw <= 1e-15:
            return noise_raw  # Already virtually zero noise
            
        # P_noise = P_signal / (10 ** (SNR / 10))
        p_noise_target = p_signal / (10.0 ** (snr_db / 10.0))
        scale = np.sqrt(p_noise_target / p_noise_raw)
        
        return noise_raw * scale
        
    def generate_batch(self, n_samples: int, fs: float, batch_size: int) -> np.ndarray:
        """
        Generate multiple independent noise realizations.
        
        Parameters
        ----------
        n_samples : int
            Length of each realization.
        fs : float
            Sampling frequency in Hz.
        batch_size : int
            Number of realizations to generate.
            
        Returns
        -------
        np.ndarray
            Array of shape (batch_size, n_samples) containing independent noise signals.
        """
        batch = np.zeros((batch_size, n_samples))
        for i in range(batch_size):
            batch[i] = self.generate(n_samples, fs)
        return batch
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed})"

    def plot(self, n_samples: int, fs: float, show: bool = True, filepath: Optional[str] = None) -> Any:
        """
        Synthesize and plot a sample of the noise.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        fs : float
            Sampling frequency in Hz.
        show : bool
            If True, displays the plot interactively (default: True).
        filepath : Optional[str]
            If provided, saves the plot to the specified path (default: None).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Run `pip install matplotlib`.")
            
        t = np.arange(n_samples) / fs
        noise = self.generate(n_samples, fs)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, noise, color="#D2691E", linewidth=1.0)
        ax.set_title(f"Simulated {self.__class__.__name__} Noise Sample")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        if filepath:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
            
        return fig
