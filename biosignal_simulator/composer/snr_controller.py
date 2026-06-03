"""
Advanced Signal-to-Noise Ratio (SNR) Control and Scaling Engine.

This module provides tools to dynamically scale individual or composite noise components
relative to clean physiological signals to achieve precise, mathematically verified SNR levels.

Mathematical Formulations:
    1. Decibel (dB) to Linear Power Relation:
       $$SNR_{\\text{dB}} = 10 \\log_{10}\\left( \\frac{P_{\\text{signal}}}{P_{\\text{noise}}} \\right)$$
       Given a target $SNR_{\\text{dB}}$, the required target noise power $P_{\\text{noise,target}}$ is:
       $$P_{\\text{noise,target}} = P_{\\text{signal}} \\cdot 10^{-\\frac{SNR_{\\text{dB}}}{10}}$$

    2. Stationary Noise Scaling:
       Given a raw noise realization $w(t)$ with power $P_{\\text{noise,raw}}$, the scaled noise $n(t)$ is:
       $$n(t) = s \\cdot w(t) \\quad \\text{where } s = \\sqrt{\\frac{P_{\\text{noise,target}}}{P_{\\text{noise,raw}}}}$$

    3. Dynamic Time-Varying SNR:
       When the target SNR is governed by a time-varying schedule $SNR_{\\text{dB}}(t)$, the target
       noise power is calculated per sample (or over short sliding window blocks of size $W$):
       $$P_{\\text{noise,target}}(t) = P_{\\text{signal,local}}(t) \\cdot 10^{-\\frac{SNR_{\\text{dB}}(t)}{10}}$$
       where $P_{\\text{signal,local}}(t)$ is the local signal power computed over a window centered at $t$:
       $$P_{\\text{signal,local}}(t) = \\frac{1}{W} \\sum_{\\tau = t - W/2}^{t + W/2} x^2(\\tau)$$
"""

from typing import List, Tuple, Union, Optional, Dict
import numpy as np
from biosignal_simulator.core.base import BaseNoiseModel
from biosignal_simulator.composer.scheduler import BaseSchedule

class SNRController:
    """
    Stationary Signal-to-Noise Ratio (SNR) Controller.
    
    Generates and scales a single noise model to achieve a target SNR (in dB)
    relative to a reference physiological signal. Supports 1-D and 2-D signals.
    """
    
    def __init__(self, noise_model: BaseNoiseModel, target_snr_db: float, channel_wise: bool = True):
        """
        Initialize the SNR Controller.
        
        Parameters
        ----------
        noise_model : BaseNoiseModel
            The noise generator.
        target_snr_db : float
            Target SNR in dB.
        channel_wise : bool
            If True and the signal is 2-D, calculates and scales noise independently
            for each channel to achieve the target SNR per channel. If False, scales
            noise globally based on average multi-channel signal power. Default is True.
        """
        self.noise_model = noise_model
        self.target_snr_db = target_snr_db
        self.channel_wise = channel_wise
        self._validate()

    def _validate(self) -> None:
        # Extreme value check, standard range usually [-20, 80] dB
        if abs(self.target_snr_db) > 150.0:
            raise ValueError(f"Extremely high/low target SNR: {self.target_snr_db} dB. Must be within [-150, 150].")

    def apply(self, clean_signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Generate noise scaled to achieve the target SNR relative to the clean signal.
        
        Parameters
        ----------
        clean_signal : np.ndarray
            Reference physiological signal (1-D or 2-D).
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            Scaled noise matching the shape of `clean_signal`.
        """
        if len(clean_signal) == 0:
            return np.empty(0, dtype=np.float64)
            
        if clean_signal.ndim == 2:
            n_ch, n_samples = clean_signal.shape
            noise = np.zeros_like(clean_signal)
            
            if self.channel_wise:
                # Scale each channel independently
                for c in range(n_ch):
                    noise[c] = self.noise_model.generate_scaled(clean_signal[c], self.target_snr_db, fs)
            else:
                # Scale globally based on aggregate multi-channel average power
                raw_noise = np.zeros_like(clean_signal)
                for c in range(n_ch):
                    raw_noise[c] = self.noise_model.generate(n_samples, fs)
                    
                p_signal = np.mean(np.square(clean_signal))
                p_noise_raw = np.mean(np.square(raw_noise))
                
                if p_signal <= 1e-15:
                    p_signal = 1e-15
                if p_noise_raw <= 1e-15:
                    return raw_noise
                    
                # B-10 FIX: clamp noise target to prevent underflow at extreme SNR values
                _MIN_NOISE_POWER = 1e-30
                p_noise_target = max(p_signal / (10.0 ** (self.target_snr_db / 10.0)), _MIN_NOISE_POWER)
                scale = np.sqrt(p_noise_target / p_noise_raw)
                noise = raw_noise * scale
                
            return noise
        else:
            return self.noise_model.generate_scaled(clean_signal, self.target_snr_db, fs)


class CompositeSNRController:
    """
    Composite SNR Controller managing multiple noise components.
    
    Independently scales and sums multiple noise generators, each at a separate target SNR level,
    relative to a reference physiological signal.
    """
    
    def __init__(self, noise_models_with_snr: List[Tuple[BaseNoiseModel, float]], channel_wise: bool = True):
        """
        Initialize the Composite SNR Controller.
        
        Parameters
        ----------
        noise_models_with_snr : List[Tuple[BaseNoiseModel, float]]
            List of tuples of (noise_model, target_snr_db).
        channel_wise : bool
            Whether scaling is calculated independently per channel. Default is True.
        """
        self.noise_models_with_snr = noise_models_with_snr
        self.channel_wise = channel_wise
        self._validate()

    def _validate(self) -> None:
        if len(self.noise_models_with_snr) == 0:
            raise ValueError("Must provide at least one noise model SNR pair.")
        for _, snr in self.noise_models_with_snr:
            if abs(snr) > 150.0:
                raise ValueError(f"Target SNR {snr} dB is outside the physically meaningful range [-150, 150].")

    def apply(self, clean_signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Generate and scale all noise components independently, then return their sum.
        
        Parameters
        ----------
        clean_signal : np.ndarray
            Reference physiological signal (1-D or 2-D).
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            Summed scaled noise matching the shape of `clean_signal`.
        """
        total_noise = np.zeros_like(clean_signal)
        for model, snr_db in self.noise_models_with_snr:
            # Instantiate an SNR controller for this individual component
            controller = SNRController(model, snr_db, channel_wise=self.channel_wise)
            total_noise += controller.apply(clean_signal, fs)
        return total_noise


class DynamicSNRController:
    """
    Dynamic Signal-to-Noise Ratio (SNR) Controller.
    
    Varies noise scaling over time based on an SNR Schedule, tracking local signal power fluctuations
    to maintain a time-varying target SNR.
    """
    
    def __init__(
        self,
        noise_model: BaseNoiseModel,
        snr_schedule: BaseSchedule,
        window_duration_s: float = 0.5,
        channel_wise: bool = True
    ):
        """
        Initialize the Dynamic SNR Controller.
        
        Parameters
        ----------
        noise_model : BaseNoiseModel
            The noise generator.
        snr_schedule : BaseSchedule
            The schedule envelope governing target SNR (in dB) over time.
        window_duration_s : float
            Duration of the sliding window (seconds) used to estimate local signal power.
            If 0.0, power is calculated per-sample (instantaneous envelope squared).
            Default is 0.5 s.
        channel_wise : bool
            Whether local scaling is calculated independently per channel. Default is True.
        """
        self.noise_model = noise_model
        self.snr_schedule = snr_schedule
        self.window_duration_s = window_duration_s
        self.channel_wise = channel_wise

    def apply(self, clean_signal: np.ndarray, fs: float) -> np.ndarray:
        """
        Generate noise scaled dynamically to track the target SNR schedule.
        
        Parameters
        ----------
        clean_signal : np.ndarray
            Reference physiological signal (1-D or 2-D).
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            Dynamically scaled noise matching the shape of `clean_signal`.
        """
        if len(clean_signal) == 0:
            return np.empty(0, dtype=np.float64)
            
        if clean_signal.ndim == 2:
            n_ch, n_samples = clean_signal.shape
            noise = np.zeros_like(clean_signal)
            
            if self.channel_wise:
                for c in range(n_ch):
                    noise[c] = self._apply_1d(clean_signal[c], fs)
            else:
                # Scale globally based on average multi-channel local power
                raw_noise = np.zeros_like(clean_signal)
                for c in range(n_ch):
                    raw_noise[c] = self.noise_model.generate(n_samples, fs)
                    
                t = np.arange(n_samples) / fs
                snr_envelope = self.snr_schedule.get_envelope(t)
                
                # Average local signal power across channels
                avg_local_p_signal = np.zeros(n_samples)
                for c in range(n_ch):
                    avg_local_p_signal += self._estimate_local_power(clean_signal[c], fs)
                avg_local_p_signal /= n_ch
                
                # Local power of raw noise
                local_p_noise_raw = np.zeros(n_samples)
                for c in range(n_ch):
                    local_p_noise_raw += self._estimate_local_power(raw_noise[c], fs)
                local_p_noise_raw /= n_ch
                
                # Bounding
                avg_local_p_signal = np.clip(avg_local_p_signal, 1e-15, None)
                local_p_noise_raw = np.clip(local_p_noise_raw, 1e-15, None)
                
                # Target local noise power
                p_noise_target = avg_local_p_signal * (10.0 ** (-snr_envelope / 10.0))
                scale_envelope = np.sqrt(p_noise_target / local_p_noise_raw)
                
                noise = raw_noise * scale_envelope
            return noise
        else:
            return self._apply_1d(clean_signal, fs)

    def _apply_1d(self, x: np.ndarray, fs: float) -> np.ndarray:
        """Apply dynamic SNR scaling to a 1D channel."""
        n_samples = len(x)
        t = np.arange(n_samples) / fs
        
        # 1. Generate raw noise and estimate its local power
        raw_noise = self.noise_model.generate(n_samples, fs)
        raw_noise = raw_noise - np.mean(raw_noise)
        
        # Estimate power profiles
        p_sig_local = self._estimate_local_power(x, fs)
        p_noise_local = self._estimate_local_power(raw_noise, fs)
        
        # Get target SNR envelope
        snr_envelope = self.snr_schedule.get_envelope(t)
        
        # Bounding to avoid division by zero
        p_sig_local = np.clip(p_sig_local, 1e-15, None)
        p_noise_local = np.clip(p_noise_local, 1e-15, None)
        
        # 2. Compute dynamic scaling envelope
        p_noise_target = p_sig_local * (10.0 ** (-snr_envelope / 10.0))
        scale_envelope = np.sqrt(p_noise_target / p_noise_local)
        
        return raw_noise * scale_envelope

    def _estimate_local_power(self, signal_1d: np.ndarray, fs: float) -> np.ndarray:
        """Estimate the running power of a 1D signal using a sliding average window."""
        n_samples = len(signal_1d)
        squared_signal = np.square(signal_1d)
        
        win_size = int(np.round(self.window_duration_s * fs))
        if win_size <= 1:
            # Return instantaneous power (per sample)
            return squared_signal
            
        # Design uniform sliding window moving average kernel
        kernel = np.ones(win_size) / win_size
        # Convolve, keeping same array length
        local_power = np.convolve(squared_signal, kernel, mode='same')
        return local_power
