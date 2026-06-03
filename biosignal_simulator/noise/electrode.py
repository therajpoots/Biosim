"""
Electrode-Skin Interface and Instrumentation Electrode Noise Simulator.

This module provides the `ElectrodeNoise` class, which models the electrical noise
generated at the boundary between biopotential electrodes (e.g. Ag/AgCl) and skin tissue.

Physical and Physiological Context:
    Electrode-skin contact noise is a complex chemical and physical phenomenon comprising:
    1. Popcorn Noise (Random Telegraph Signal, RTS):
       Sudden, discrete step changes in biopotential voltage due to charge trapping,
       corrosion, or brief contact breaks. The time between transitions follows an exponential
       distribution based on transition rates.
    2. Johnson-Nyquist Contact Thermal Noise:
       Thermal agitation of charge carriers inside the contact resistance. The skin-electrode
       interface is modeled as a parallel combination of electrode resistance $R_e$ and
       double-layer capacitance $C_e$:
       $$Z(f) = R_{\\text{contact}} \\parallel \\frac{1}{j 2\\pi f C_{\\text{double}}}$$
       The RMS thermal noise voltage over a bandwidth $B$ is:
       $$v_{\\text{rms}} = \\sqrt{4 k_B T \\cdot \\text{Re}\\{Z(f)\\} \\cdot B}$$
       where $k_B$ is the Boltzmann constant ($1.380649 \\times 10^{-23} \\text{ J/K}$), $T$ is temperature
       in Kelvin, and $\\text{Re}\\{Z(f)\\}$ is the real part of the contact impedance.
    3. Half-Cell Polarization Potential Settling:
       When electrodes are first placed on the skin, a chemical charge layer forms. This causes
       a large initial DC polarization offset that decays exponentially as the interface stabilizes.

Mathematical Formulation:
    1. Parallel RC Contact Impedance:
       $$\\text{Re}\\{Z(f)\\} = \\frac{R_c}{1 + (2\\pi f R_c C_d)^2}$$
       We generate thermal white noise and filter it to match the power spectral density
       distribution of the contact impedance real part.

    2. Polarization Settling Transient:
       $$V_{\\text{settle}}(t) = V_0 \\cdot e^{-t / \\tau_s}$$
       where $V_0$ is the initial polarization offset and $\\tau_s$ is the chemical settling time constant.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import ElectrodeNoiseConfig
from biosignal_simulator.utils.validation import validate_config

class ElectrodeNoise(BaseNoiseModel):
    """
    Biopotential Electrode and Contact Noise Simulator.
    
    Models random telegraph popcorn noise, frequency-dependent RC skin-electrode contact
    Johnson thermal noise, half-cell polarization settling curves, and multi-channel instances.
    """
    
    def __init__(
        self,
        config: Optional[ElectrodeNoiseConfig] = None,
        initial_polarization_mv: float = 15.0,
        settling_time_s: float = 3.0,
        **kwargs
    ):
        """
        Initialize the Electrode Noise generator.
        
        Parameters
        ----------
        config : Optional[ElectrodeNoiseConfig]
            Base configuration containing popcorn amplitude, rates, impedance, and temperature.
        initial_polarization_mv : float
            Initial electrochemical polarization DC offset on electrode application.
            Default is 15.0 mV.
        settling_time_s : float
            Chemical stabilization time constant in seconds. Default is 3.0 s.
        **kwargs :
            Alternative parameters passed to ElectrodeNoiseConfig if config is None.
        """
        if config is None:
            config = ElectrodeNoiseConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        self.initial_polarization_mv = initial_polarization_mv
        self.settling_time_s = settling_time_s

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D electrode noise.
        
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
        noise = np.zeros(n_samples)

        # --- 1. Popcorn Noise (Random Telegraph Signal) ---
        if self.config.enable_popcorn:
            lam = self.config.popcorn_rate_hz
            if lam > 0:
                # Estimate state transitions based on rate
                est_switches = int(lam * dur * 2.5) + 10
                intervals = self.rng.exponential(scale=1.0 / lam, size=est_switches)
                switches = np.cumsum(intervals)
                switches = switches[switches < dur]
                
                # Count cumulative transitions before each time index
                counts = np.searchsorted(switches, t)
                
                # Alternate state amplitude between +amp and -amp
                amp = self.config.popcorn_amplitude
                popcorn = amp * (1.0 - 2.0 * (counts % 2))
                noise += popcorn

        # --- 2. Skin-Electrode Parallel RC Thermal Noise ---
        if self.config.enable_impedance_noise:
            # Boltzmann constant
            kb = 1.380649e-23
            T = self.config.temperature_k
            R = self.config.impedance_ohms
            
            # If bandwidth is not defined, default to Nyquist frequency
            B = self.config.bandwidth_hz if self.config.bandwidth_hz is not None else fs / 2.0
            
            # Base Johnson-Nyquist thermal RMS voltage: V = sqrt(4 * k_B * T * R * B) in Volts
            v_rms_volts = np.sqrt(4.0 * kb * T * R * B)
            # Convert to Millivolts
            v_rms_mv = v_rms_volts * 1000.0
            
            # Generate raw thermal white noise
            white_thermal = self.rng.normal(0.0, v_rms_mv, size=n_samples)
            
            # Model skin impedance lowpass filtering (RC double-layer effect)
            # Standard electrode double-layer capacitance is around 10 uF/cm^2
            # For 1 cm^2 electrode and R = 5000 Ohms, tau = RC = 5000 * 10e-6 = 0.05 s (f_cutoff ~ 3.18 Hz)
            C_double = 10e-6  # 10 microFarads
            tau_rc = R * C_double
            f_cutoff = 1.0 / (2.0 * np.pi * tau_rc)
            
            # Filter thermal noise using a 1st order Butterworth lowpass filter matching RC cutoff
            nyq = 0.5 * fs
            if f_cutoff < nyq:
                b_rc, a_rc = sp_signal.butter(1, f_cutoff / nyq, btype='low')
                filtered_thermal = sp_signal.lfilter(b_rc, a_rc, white_thermal)
                
                # Re-scale to preserve total Johnson noise power
                p_white = np.mean(np.square(white_thermal))
                p_filtered = np.mean(np.square(filtered_thermal))
                if p_filtered > 1e-15:
                    filtered_thermal = filtered_thermal * np.sqrt(p_white / p_filtered)
                noise += filtered_thermal
            else:
                noise += white_thermal

        # --- 3. Electrode Electrochemical Polarization settling ---
        if self.initial_polarization_mv != 0.0 and self.settling_time_s > 0.0:
            settling_offset = self.initial_polarization_mv * np.exp(-t / self.settling_time_s)
            noise += settling_offset

        return noise

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel independent electrode noise instances.
        
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
        for c in range(n_channels):
            # Generate independent popcorn, thermal, and settling noise for each electrode
            noise_matrix[c] = self.generate(n_samples, fs)
            
        return noise_matrix
