"""
Time-Varying Noise Level and Parameter Scheduling Engine.

This module provides classes to define time-varying parameter envelopes (schedules)
for physiological signal simulations. These schedules can govern dynamic noise levels
(e.g., standard deviation, SNR) or other time-varying parameters of generators.

Supported Schedules:
    1. `StepSchedule`: Sudden, piece-wise constant transitions at defined timestamps.
       $$S(t) = L_i \\quad \\text{for } t_i \\le t < t_{i+1}$$
    2. `RampSchedule`: Piece-wise linear interpolation between control points.
       $$S(t) = L_i + \\frac{L_{i+1} - L_i}{t_{i+1} - t_i} (t - t_i) \\quad \\text{for } t_i \\le t < t_{i+1}$$
    3. `PeriodicSchedule`: Sinusoidal modulation of parameters.
       $$S(t) = L_{\\text{base}} + A_{\\text{mod}} \\cdot \\sin\\left(2\\pi f t + \\theta\\right)$$
    4. `SigmoidSchedule`: Smooth transitions between two levels using a logistic curve.
       $$S(t) = L_{\\text{start}} + \\frac{L_{\\text{end}} - L_{\\text{start}}}{1 + \\exp\\left(-k (t - t_0)\\right)}$$
    5. `StochasticSchedule`: A parameter envelope driven by random walk (Brownian drift).
       $$S(t) = \\left| L_0 + \\int_{0}^{t} w(\\tau) d\\tau \\right|$$
    6. `CompositeSchedule`: Combination of multiple schedules via arithmetic operations.
       $$S_{\\text{comp}}(t) = S_1(t) \\oplus S_2(t)$$
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
import numpy as np
from biosignal_simulator.core.base import BaseNoiseModel
from biosignal_simulator.core.math_utils import normalize_to_rms

class BaseSchedule(ABC):
    """Abstract Base Class for all parameter scheduling envelopes."""
    
    @abstractmethod
    def get_envelope(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the schedule envelope over the target time vector.
        
        Parameters
        ----------
        t : np.ndarray
            Time vector in seconds.
            
        Returns
        -------
        np.ndarray
            The parameter envelope values, same shape as `t`.
        """
        pass
        
    def __add__(self, other: Union[float, 'BaseSchedule']) -> 'CompositeSchedule':
        return CompositeSchedule(self, other, operator='add')
        
    def __mul__(self, other: Union[float, 'BaseSchedule']) -> 'CompositeSchedule':
        return CompositeSchedule(self, other, operator='multiply')


class StepSchedule(BaseSchedule):
    """
    Step Schedule representing piece-wise constant parameter levels.
    
    Useful for simulating sudden appliance activation or state change noise.
    """
    
    def __init__(self, breakpoints: Union[List[float], np.ndarray], levels: Union[List[float], np.ndarray]):
        """
        Initialize the Step Schedule.
        
        Parameters
        ----------
        breakpoints : List[float] or np.ndarray
            Start times (in seconds) for each step segment. Must be sorted.
        levels : List[float] or np.ndarray
            Parameter value during each segment.
        """
        self.breakpoints = np.array(breakpoints, dtype=float)
        self.levels = np.array(levels, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if len(self.breakpoints) == 0:
            raise ValueError("Breakpoints list cannot be empty.")
        if len(self.levels) < len(self.breakpoints):
            raise ValueError("Levels must have at least as many elements as breakpoints.")
        if np.any(np.diff(self.breakpoints) < 0):
            raise ValueError("Breakpoints must be monotonically increasing.")

    def get_envelope(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0:
            return np.empty(0, dtype=np.float64)
            
        envelope = np.zeros_like(t)
        
        # Step through segments
        for i in range(len(self.breakpoints)):
            t_start = self.breakpoints[i]
            t_end = self.breakpoints[i + 1] if i + 1 < len(self.breakpoints) else np.inf
            level = self.levels[i]
            
            mask = (t >= t_start) & (t < t_end)
            envelope[mask] = level
            
        # Handle time points prior to the first breakpoint
        pre_mask = t < self.breakpoints[0]
        envelope[pre_mask] = self.levels[0]
        
        return envelope


class RampSchedule(BaseSchedule):
    """
    Ramp Schedule representing piece-wise linear interpolation between control points.
    
    Useful for simulating gradual drift or sensor warming trends.
    """
    
    def __init__(self, control_times: Union[List[float], np.ndarray], levels: Union[List[float], np.ndarray]):
        """
        Initialize the Ramp Schedule.
        
        Parameters
        ----------
        control_times : List[float] or np.ndarray
            Time control points in seconds. Must be sorted.
        levels : List[float] or np.ndarray
            Parameter level at each control time.
        """
        self.control_times = np.array(control_times, dtype=float)
        self.levels = np.array(levels, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if len(self.control_times) != len(self.levels):
            raise ValueError("control_times and levels lists must have equal lengths.")
        if len(self.control_times) < 2:
            raise ValueError("RampSchedule requires at least 2 control points.")
        if np.any(np.diff(self.control_times) < 0):
            raise ValueError("control_times must be monotonically increasing.")

    def get_envelope(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0:
            return np.empty(0, dtype=np.float64)
            
        # Linear interpolation with boundary holding (no extrapolation beyond limits)
        # np.interp holds left value for t < control_times[0], and right value for t > control_times[-1]
        return np.interp(t, self.control_times, self.levels)


class PeriodicSchedule(BaseSchedule):
    """
    Periodic Schedule modulating parameters sinusoidally.
    
    Useful for simulating cyclic respiration baseline changes or thermal cycles.
    """
    
    def __init__(
        self,
        base_level: float,
        modulation_amplitude: float,
        frequency_hz: float,
        phase_rad: float = 0.0
    ):
        """
        Initialize the Periodic Schedule.
        
        Parameters
        ----------
        base_level : float
            DC offset / center value of the modulation.
        modulation_amplitude : float
            Peak deviation from base level.
        frequency_hz : float
            Modulation frequency in Hz. Must be > 0.
        phase_rad : float
            Starting phase angle in radians. Default is 0.0.
        """
        self.base_level = base_level
        self.modulation_amplitude = modulation_amplitude
        self.frequency_hz = frequency_hz
        self.phase_rad = phase_rad
        self._validate()

    def _validate(self) -> None:
        if self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive.")

    def get_envelope(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0:
            return np.empty(0, dtype=np.float64)
        return self.base_level + self.modulation_amplitude * np.sin(
            2.0 * np.pi * self.frequency_hz * t + self.phase_rad
        )


class SigmoidSchedule(BaseSchedule):
    """
    Sigmoid (Logistic) transition schedule.
    
    Provides a smooth, continuous transition between two parameter levels.
    """
    
    def __init__(self, start_level: float, end_level: float, midpoint_s: float, slope: float = 1.0):
        """
        Initialize the Sigmoid Schedule.
        
        Parameters
        ----------
        start_level : float
            Initial level before the transition.
        end_level : float
            Target level after the transition.
        midpoint_s : float
             Midpoint of the transition curve (seconds).
        slope : float
            Steepness of the transition. Higher is steeper. Default is 1.0.
        """
        self.start_level = start_level
        self.end_level = end_level
        self.midpoint_s = midpoint_s
        self.slope = slope
        self._validate()

    def _validate(self) -> None:
        if self.slope <= 0.0:
            raise ValueError("Sigmoid slope must be positive.")

    def get_envelope(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0:
            return np.empty(0, dtype=np.float64)
        # S(t) = start + (end - start) / (1 + exp(-slope * (t - midpoint)))
        denom = 1.0 + np.exp(-self.slope * (t - self.midpoint_s))
        return self.start_level + (self.end_level - self.start_level) / denom


class StochasticSchedule(BaseSchedule):
    """
    Stochastic parameter schedule based on integrated Brownian walk.
    
    Simulates random fluctuations in noise parameters over time.
    """
    
    def __init__(self, initial_level: float, step_std: float, seed: Optional[int] = None):
        """
        Initialize the Stochastic Schedule.
        
        Parameters
        ----------
        initial_level : float
            Starting parameter value.
        step_std : float
            Standard deviation of random walk steps.
        seed : Optional[int]
            Random seed for walk reproducibility.
        """
        self.initial_level = initial_level
        self.step_std = step_std
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_envelope(self, t: np.ndarray) -> np.ndarray:
        if len(t) == 0:
            return np.empty(0, dtype=np.float64)
            
        n_samples = len(t)
        # Generate random walk steps
        steps = self.rng.normal(0.0, self.step_std, size=n_samples)
        # Start at 0, integrate steps
        steps[0] = 0.0
        walk = np.cumsum(steps)
        
        # Add initial offset and keep strictly positive
        return np.abs(self.initial_level + walk)


class CompositeSchedule(BaseSchedule):
    """
    Composite schedule combining two schedules (or a schedule and a float).
    
    Supports addition, multiplication, min, and max combinations.
    """
    
    def __init__(
        self,
        schedule1: BaseSchedule,
        schedule2: Union[float, BaseSchedule],
        operator: str = 'add'
    ):
        """
        Initialize the Composite Schedule.
        
        Parameters
        ----------
        schedule1 : BaseSchedule
            First schedule.
        schedule2 : BaseSchedule or float
            Second schedule or scalar offset.
        operator : str
            Combination rule: 'add', 'multiply', 'max', 'min'.
        """
        self.schedule1 = schedule1
        self.schedule2 = schedule2
        self.operator = operator.lower().strip()
        self._validate()

    def _validate(self) -> None:
        allowed = {'add', 'multiply', 'max', 'min'}
        if self.operator not in allowed:
            raise ValueError(f"Unsupported operator '{self.operator}'. Allowed: {allowed}")

    def get_envelope(self, t: np.ndarray) -> np.ndarray:
        env1 = self.schedule1.get_envelope(t)
        
        if isinstance(self.schedule2, (int, float)):
            env2 = float(self.schedule2) * np.ones_like(t)
        else:
            env2 = self.schedule2.get_envelope(t)
            
        if self.operator == 'add':
            return env1 + env2
        elif self.operator == 'multiply':
            return env1 * env2
        elif self.operator == 'max':
            return np.maximum(env1, env2)
        elif self.operator == 'min':
            return np.minimum(env1, env2)
        else:
            return env1 + env2


class NoiseScheduler(BaseNoiseModel):
    """
    Noise model wrapper that modulates noise amplitude using a Schedule envelope.
    """
    
    def __init__(self, noise_model: BaseNoiseModel, schedule: BaseSchedule, seed: Optional[int] = None):
        """
        Initialize the Noise Scheduler.
        
        Parameters
        ----------
        noise_model : BaseNoiseModel
            The underlying noise generator to modulate.
        schedule : BaseSchedule
            The schedule envelope governing noise standard deviation over time.
        seed : Optional[int]
            Random seed.
        """
        super().__init__(seed=seed)
        self.noise_model = noise_model
        self.schedule = schedule
        
        # Link RNG states for consistent seeding
        if seed is not None:
            self.noise_model.rng = np.random.default_rng(seed)

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Generate modulated noise signal.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        fs : float
            Sampling frequency.
        """
        if n_samples <= 0:
            return np.empty(0, dtype=np.float64)
            
        t = np.arange(n_samples) / fs
        
        # Generate underlying raw noise and normalize to unit standard deviation (RMS)
        raw_noise = self.noise_model.generate(n_samples, fs)
        raw_noise = raw_noise - np.mean(raw_noise)
        raw_noise = normalize_to_rms(raw_noise, 1.0)
        
        # Get modulated envelope
        envelope = self.schedule.get_envelope(t)
        
        # Modulate
        return raw_noise * envelope
