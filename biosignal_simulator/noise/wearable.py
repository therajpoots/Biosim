"""
Advanced Noise Models for Wearable Sensors and Wireless Bio-Telemetry.

This module provides high-fidelity simulations for common wearable recording artifacts:
1. `SensorDetachmentNoise`: Simulates complete sensor fall-off or electrode detachment.
2. `ElectrodeDisplacementNoise`: Simulates loose contact displacement jumps during movement.
3. `LightLeakageNoise`: Simulates ambient light leakage interference on optical sensors (PPG).
4. `PacketLossNoise`: Simulates wireless dropouts using the Gilbert-Elliott Markov model.

Physical and Physiological Context:
    1. Sensor Detachment:
       When a wet electrode (e.g., Ag/AgCl) or optical sensor loses contact with skin:
       - There is an immediate, large voltage transient spike (offset bounce) as the double-layer
         charge potential discharges.
       - The signal path transitions to a high-impedance state, where physiological waveforms
         are completely suppressed and replaced by high-amplitude amplifier thermal noise (flatline).
       $$V_{\\text{detach}}(t) = V_{\\text{bounce}} \\cdot e^{-\\frac{t - t_{\\text{det}}}{\\tau_d}} \\cdot \\mathcal{H}(t - t_{\\text{det}}) + v_{\\text{amplifier}}(t)$$

    2. Electrode Displacement:
       Loose contact during activity causes the electrode to shift position on skin:
       - This results in abrupt steps in half-cell potential offset.
       - Simultaneously, contact impedance increases, causing a temporary burst of high-frequency
         friction noise and step-wise increases in baseline thermal noise.

    3. Light Leakage (PPG Ambient Interference):
       For photoplethysmography (PPG) sensors, movement deforms the sensor-skin seal, allowing
       ambient light to reach the photodetector:
       - Indoor lighting flickers at twice the AC grid frequency ($100$ Hz or $120$ Hz) due to full-wave
         rectification of fluorescent/LED ballasts.
       - The magnitude of leakage is modulated by breathing movements or motion that changes the
         physical gap size:
         $$V_{\\text{leak}}(t) = E_{\\text{mod}}(t) \\cdot \\sum_{h=1}^{H} A_h \\sin\\left(2\\pi \\cdot h \\cdot (2f_{\\text{line}}) \\cdot t\\right)$$

    4. Wireless Packet Loss (Gilbert-Elliott Model):
       Wireless bio-telemetry (Bluetooth, Zigbee) experiences packet dropouts due to distance or
       tissue occlusion. This is modeled as a 2-state Markov chain:
       - State G (Good): No packet loss.
       - State B (Bad): Burst packet loss.
       Transition probabilities $p$ (Good -> Bad) and $q$ (Bad -> Good) govern average loss rates
       and burst lengths. Dropped packets are recovered via Zero-fill, Last-value Hold, Linear,
       or Cubic Spline interpolation.
"""

from typing import Optional, List, Tuple, Union
import numpy as np
from scipy.interpolate import interp1d
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import (
    SensorDetachmentConfig, ElectrodeDisplacementConfig, LightLeakageConfig, PacketLossConfig
)
from biosignal_simulator.utils.validation import validate_config

class SensorDetachmentNoise(BaseNoiseModel):
    """
    Sensor Detachment and Fall-off Simulator.
    
    Generates a transient voltage offset spike followed by complete signal suppression
    and high-amplitude flatline thermal/amplifier noise.
    """
    
    def __init__(self, config: Optional[SensorDetachmentConfig] = None, **kwargs):
        if config is None:
            config = SensorDetachmentConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D sensor detachment noise.
        
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
        noise = np.zeros(n_samples)
        
        det_t = self.config.detachment_time_s
        trans_d = self.config.transient_duration_s
        trans_amp = self.config.transient_amplitude
        noise_lv = self.config.noise_level_uv
        
        mask_det = t >= det_t
        if np.any(mask_det):
            t_rel = t[mask_det] - det_t
            
            # 1. Exponential decay transient discharge spike
            if trans_d > 0.0:
                noise[mask_det] = trans_amp * np.exp(-t_rel / trans_d)
            else:
                noise[mask_det] = trans_amp
                
            # 2. Amplifier thermal/flatline white noise (convert microvolts to millivolts)
            thermal = self.rng.normal(0.0, noise_lv * 1e-3, size=len(t_rel))
            noise[mask_det] += thermal
            
        return noise

    def apply(self, signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sensor detachment directly to a physiological signal.
        
        After detachment, the original physiological signal is completely suppressed
        and replaced by high-impedance flatline thermal noise.
        
        Parameters
        ----------
        signal : np.ndarray
            Clean continuous-value signal array. Can be 1-D or 2-D.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (modified_signal, detachment_noise_error)
        """
        n_samples = signal.shape[-1]
        t = np.arange(n_samples) / fs
        det_t = self.config.detachment_time_s
        trans_d = self.config.transient_duration_s
        trans_amp = self.config.transient_amplitude
        noise_lv = self.config.noise_level_uv
        
        mask_det = t >= det_t
        modified = signal.copy()
        
        if signal.ndim == 2:
            n_ch = signal.shape[0]
            for c in range(n_ch):
                if np.any(mask_det):
                    t_rel = t[mask_det] - det_t
                    # Suppress original signal
                    modified[c, mask_det] = 0.0
                    
                    # Apply transient offset spike
                    if trans_d > 0.0:
                        modified[c, mask_det] += trans_amp * np.exp(-t_rel / trans_d)
                    else:
                        modified[c, mask_det] += trans_amp
                        
                    # Apply high-impedance thermal flatline
                    thermal = self.rng.normal(0.0, noise_lv * 1e-3, size=len(t_rel))
                    modified[c, mask_det] += thermal
        else:
            if np.any(mask_det):
                t_rel = t[mask_det] - det_t
                modified[mask_det] = 0.0
                if trans_d > 0.0:
                    modified[mask_det] += trans_amp * np.exp(-t_rel / trans_d)
                else:
                    modified[mask_det] += trans_amp
                thermal = self.rng.normal(0.0, noise_lv * 1e-3, size=len(t_rel))
                modified[mask_det] += thermal
                
        error = modified - signal
        return modified, error


class ElectrodeDisplacementNoise(BaseNoiseModel):
    """
    Electrode Displacement and Loose Contact Noise Simulator.
    
    Generates sudden step changes in DC baseline offset paired with localized high-frequency
    contact friction bursts and step-wise increases in baseline thermal noise levels.
    """
    
    def __init__(
        self,
        config: Optional[ElectrodeDisplacementConfig] = None,
        burst_frequency_hz: float = 40.0,
        **kwargs
    ):
        """
        Initialize the Electrode Displacement Noise model.
        
        Parameters
        ----------
        config : Optional[ElectrodeDisplacementConfig]
            Base configuration containing displacement times, shift amplitudes, and noise increments.
        burst_frequency_hz : float
            Frictional oscillation frequency during contact shifts. Default is 40.0 Hz.
        **kwargs :
            Alternative parameters passed to ElectrodeDisplacementConfig if config is None.
        """
        if config is None:
            config = ElectrodeDisplacementConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.burst_frequency_hz = burst_frequency_hz

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D electrode displacement contact noise.
        
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
        noise = np.zeros(n_samples)
        
        times = self.config.displacement_times
        amps = self.config.shift_amplitudes
        increments = self.config.noise_increments
        
        # Sort displacement events chronologically
        sorted_idx = np.argsort(times)
        s_times = np.array([times[i] for i in sorted_idx])
        s_amps = np.array([amps[i] for i in sorted_idx])
        s_inc = np.array([increments[i] for i in sorted_idx])
        
        # Base contact noise standard deviation
        base_noise_std = 0.05
        
        for idx in range(n_samples):
            curr_t = t[idx]
            
            # Determine cumulative offset and current noise level at this timestamp
            mask = curr_t >= s_times
            shift = np.sum(s_amps[mask]) if np.any(mask) else 0.0
            
            # Incremental thermal noise multiplier due to loose contact
            noise_multiplier = np.sum(s_inc[mask]) if np.any(mask) else 0.0
            current_noise_std = base_noise_std * (1.0 + noise_multiplier)
            
            # Add baseline shift and contact thermal noise
            noise[idx] = shift + self.rng.normal(0.0, current_noise_std)
            
            # 3. Add short high-frequency frictional burst pops at the moment of shift
            # Each burst lasts ~0.3 seconds decaying exponentially
            for ev_t in s_times:
                t_diff = curr_t - ev_t
                if 0.0 <= t_diff < 0.3:
                    burst_amp = 0.8
                    decaying_burst = burst_amp * np.sin(2.0 * np.pi * self.burst_frequency_hz * curr_t) * np.exp(-t_diff / 0.05)
                    noise[idx] += decaying_burst
                    
        return noise


class LightLeakageNoise(BaseNoiseModel):
    """
    PPG Ambient Light Leakage Noise Simulator.
    
    Generates AC line frequency light flicker (100 Hz or 120 Hz) modulated by
    low-frequency movements (respiration/motion) and slow daylight drift.
    """
    
    def __init__(self, config: Optional[LightLeakageConfig] = None, **kwargs):
        if config is None:
            config = LightLeakageConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D light leakage signal.
        
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
        
        amp = self.config.leakage_amplitude
        f_mod = self.config.modulation_frequency_hz
        f_line = self.config.f_line_hz
        harm = self.config.harmonic_leakage
        
        # 1. Modulating envelope representing sensor seal deformation (respiration/motion cycles)
        envelope = 1.0 + 0.45 * np.sin(2.0 * np.pi * f_mod * t)
        
        # 2. Indoor AC grid flicker (flickers at 2 * f_line due to rectification cycles)
        f_flicker = 2.0 * f_line
        
        # Fundamental flicker component (aliased if above Nyquist)
        flicker = np.zeros(n_samples)
        if f_flicker < nyq:
            flicker += amp * np.sin(2.0 * np.pi * f_flicker * t)
            
        # First harmonic (4 * f_line)
        if 2.0 * f_flicker < nyq:
            flicker += amp * harm * np.sin(2.0 * np.pi * (2.0 * f_flicker) * t)
            
        # Second harmonic (6 * f_line)
        if 3.0 * f_flicker < nyq:
            flicker += amp * (harm * 0.3) * np.sin(2.0 * np.pi * (3.0 * f_flicker) * t)
            
        # 3. Slow diurnal baseline drift ( daylight changes or walking through shadows )
        # Frequency around 0.02 Hz
        daylight_drift = 0.3 * amp * np.sin(2.0 * np.pi * 0.02 * t)
        
        # Combine components
        return flicker * envelope + daylight_drift


class PacketLossNoise(BaseNoiseModel):
    """
    Wireless Packet Dropout and Transmission Loss Simulator.
    
    Implements a 2-state Markov chain (Gilbert-Elliott model) to simulate burst packet drops.
    Provides Zero-fill, Last-value Hold, Linear, and Cubic Spline recovery algorithms.
    """
    
    def __init__(self, config: Optional[PacketLossConfig] = None, **kwargs):
        if config is None:
            config = PacketLossConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize packet loss noise.
        
        Note: Packet loss must be applied directly to a signal vector using `apply()`.
        This method returns a flat zero array.
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            Flat zero array.
        """
        return np.zeros(n_samples)

    def apply(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply packet dropouts and packet recovery to an input digitized signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Continuous-value input signal array. Can be 1-D or 2-D.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (modified_reconstructed_signal, packet_loss_error)
        """
        loss_rate = self.config.loss_rate
        L = self.config.burst_length_samples
        mode = self.config.interpolation_mode.lower().strip()
        
        if loss_rate <= 0.0:
            return signal.copy(), np.zeros_like(signal)
            
        n_samples = signal.shape[-1]
        
        # 1. Gilbert-Elliott 2-state Markov Chain loss mask generation
        # State 0: Good (sample received), State 1: Bad (sample lost)
        # Transition probabilities:
        # q: Bad -> Good (q = 1 / L, where L is average burst length)
        # p: Good -> Bad (p = loss_rate * q / (1 - loss_rate))
        q = 1.0 / L
        p = (loss_rate * q) / (1.0 - loss_rate) if loss_rate < 1.0 else 1.0
        
        mask = np.ones(n_samples, dtype=bool)
        state = 0  # Start in Good state
        
        for idx in range(n_samples):
            if state == 0:
                if self.rng.random() < p:
                    state = 1
            else:
                if self.rng.random() < q:
                    state = 0
                    
            if state == 1:
                mask[idx] = False  # sample is dropped
                
        # 2. Reconstruct signal using selected interpolation mode
        modified = signal.copy()
        
        if signal.ndim == 2:
            n_ch = signal.shape[0]
            for c in range(n_ch):
                modified[c] = self._reconstruct_1d(signal[c], mask, mode)
        else:
            modified = self._reconstruct_1d(signal, mask, mode)
            
        error = modified - signal
        return modified, error

    def _reconstruct_1d(self, x: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
        """Apply interpolation reconstruction to a 1D channel."""
        n_samples = len(x)
        y = x.copy()
        
        # Handle complete data loss fallback
        if not np.any(mask):
            return np.zeros(n_samples)
            
        if mode == 'zero':
            y[~mask] = 0.0
            
        elif mode == 'hold':
            # Last-value hold
            last_val = 0.0
            for idx in range(n_samples):
                if mask[idx]:
                    last_val = y[idx]
                else:
                    y[idx] = last_val
                    
        elif mode == 'linear':
            # Linear interpolation across lost gaps
            x_indices = np.arange(n_samples)
            valid_x = x_indices[mask]
            valid_y = x[mask]
            
            if len(valid_x) > 1:
                f = interp1d(valid_x, valid_y, kind='linear', fill_value="extrapolate")
                y[~mask] = f(x_indices[~mask])
            else:
                y[~mask] = valid_y[0]
                
        elif mode == 'cubic' or mode == 'spline':
            # Cubic spline interpolation
            x_indices = np.arange(n_samples)
            valid_x = x_indices[mask]
            valid_y = x[mask]
            
            if len(valid_x) > 3:
                f = interp1d(valid_x, valid_y, kind='cubic', fill_value="extrapolate")
                y[~mask] = f(x_indices[~mask])
            elif len(valid_x) > 1:
                # Fall back to linear if not enough points for cubic
                f = interp1d(valid_x, valid_y, kind='linear', fill_value="extrapolate")
                y[~mask] = f(x_indices[~mask])
            else:
                y[~mask] = valid_y[0]
                
        return y
