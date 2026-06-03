"""
High-fidelity Physiological Crosstalk and Volume Conduction Leakage Simulator.

This module provides the `CrosstalkNoise` class, which models the electrical leakage
of signals between different physiological systems (e.g. cardiac electrical field bleeding
into EEG/EMG, or respiration movement leaking onto optical PPG).

Physiological Reference:
    Physiological crosstalk is a common recording artifact:
    1. Electrocardiographic (ECG) leakage on EMG: Since the cardiac electrical vector is
       extremely large (often millivolts at the torso), it easily propagates to thoracic,
       diaphragmatic, and abdominal EMG electrodes.
    2. ECG bleed-in on EEG: Sub-temporal or sub-occipital EEG electrodes captured relative
       to earlobe references often register R-wave spikes from the carotid arteries.
    3. Respiratory motion on PPG: Movement of the chest and tissue changes venous blood
       volume, causing breathing frequencies to modulate PPG baselines.
    4. Tissue Volume Conduction: Biological tissue behaves as a conductive volume medium
       with high capacitance. High-frequency electrical components are attenuated, making
       volume conduction behave as a lowpass filter on leaking biopotential signals.

Mathematical Formulation:
    For a multi-channel target system, the leakage noise vector $\\vec{n}_{\\text{leak}}(t)$ is
    expressed as a spatial projection of multiple physiological source generators $\\{\\vec{s}_i(t)\\}$
    filtered by tissue conduction impulse response $h_{\\text{tissue}}(t)$:
    $$\\vec{n}_{\\text{leak}}(t) = \\sum_{i} M_i \\cdot \\left( \\vec{s}_i(t) * h_{\\text{tissue},i}(t) \\right)$$

    where:
    - $\\vec{s}_i(t)$ is the multi-channel clean physiological signal vector from source $i$.
    - $M_i$ is the target-to-source coupling matrix of shape $(N_{\\text{target}}, N_{\\text{source},i})$,
      governing how source channels bleed into target channels.
    - $*$ is the convolution operator.
    - $h_{\\text{tissue},i}(t)$ is a volume conduction lowpass filter (e.g., $10$ to $50$ Hz cutoff).
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
from scipy import signal as sp_signal
from biosignal_simulator.noise.base import BaseNoiseModel
from biosignal_simulator.core.config import (
    CrosstalkNoiseConfig, ECGConfig, EEGConfig, EMGConfig, PPGConfig, EDAConfig, RespConfig
)
from biosignal_simulator.utils.validation import validate_config

class CrosstalkNoise(BaseNoiseModel):
    """
    Advanced Physiological Crosstalk and Tissue Conduction Simulator.
    
    Synthesizes biopotential leakage from other organs (e.g., ECG on EEG, EMG on ECG)
    using multi-channel coupling matrices, volume conduction lowpass filters, and time delays.
    """
    
    def __init__(
        self,
        config: Optional[CrosstalkNoiseConfig] = None,
        coupling_matrix: Optional[Union[List[List[float]], np.ndarray]] = None,
        enable_volume_conduction: bool = True,
        conduction_cutoff_hz: float = 30.0,
        **kwargs
    ):
        """
        Initialize the Crosstalk Noise generator.
        
        Parameters
        ----------
        config : Optional[CrosstalkNoiseConfig]
            Base configuration containing coupling factor and source generator type.
        coupling_matrix : Optional[Union[List[List[float]], np.ndarray]]
            A 2D spatial mixing matrix of shape (n_target_channels, n_source_channels)
            defining how individual channels of the source signal map to target channels.
            If None, the first source channel is leaked to all target channels.
        enable_volume_conduction : bool
            If True, applies a tissue volume conduction lowpass filter to the leaked signals.
            Default is True.
        conduction_cutoff_hz : float
            Lowpass filter cutoff frequency in Hz representing tissue conduction impedance.
            Default is 30.0 Hz.
        **kwargs :
            Alternative parameters passed to CrosstalkNoiseConfig if config is None.
        """
        if config is None:
            config = CrosstalkNoiseConfig(**kwargs)
        else:
            validate_config(config)
            
        super().__init__(seed=config.seed)
        self.config = config
        
        self.coupling_matrix = np.array(coupling_matrix) if coupling_matrix is not None else None
        self.enable_volume_conduction = enable_volume_conduction
        self.conduction_cutoff_hz = conduction_cutoff_hz
        
        self._validate_crosstalk_parameters()

    def _validate_crosstalk_parameters(self) -> None:
        """Validate input boundaries."""
        if self.conduction_cutoff_hz <= 0.0:
            raise ValueError("Conduction filter cutoff frequency 'conduction_cutoff_hz' must be positive.")

    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize raw 1-D physiological leakage noise.
        
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
            
        source = self.config.source_type.lower().strip()
        coupling = self.config.coupling_factor
        duration = n_samples / fs
        
        # 1. Resolve and copy/align configuration parameters
        cfg = self.config.source_config
        if cfg is None:
            if source == 'ecg':
                cfg = ECGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'eeg':
                cfg = EEGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'emg':
                cfg = EMGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'ppg':
                cfg = PPGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'eda':
                cfg = EDAConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'resp':
                cfg = RespConfig(fs=fs, duration_s=duration, seed=self.seed)
            else:
                raise ValueError(f"Unknown crosstalk source type: {source}")
        else:
            import copy
            cfg = copy.deepcopy(cfg)
            
        if hasattr(cfg, 'fs'):
            cfg.fs = fs
        if hasattr(cfg, 'duration_s'):
            cfg.duration_s = duration
            
        # 2. Instantiate the source generator
        if source == 'ecg':
            from biosignal_simulator.signals.ecg import ECGGenerator
            gen = ECGGenerator(cfg)
        elif source == 'eeg':
            from biosignal_simulator.signals.eeg import EEGGenerator
            gen = EEGGenerator(cfg)
        elif source == 'emg':
            from biosignal_simulator.signals.emg import EMGGenerator
            gen = EMGGenerator(cfg)
        elif source == 'ppg':
            from biosignal_simulator.signals.ppg import PPGGenerator
            gen = PPGGenerator(cfg)
        elif source == 'eda':
            from biosignal_simulator.signals.eda import EDAGenerator
            gen = EDAGenerator(cfg)
        elif source == 'resp':
            from biosignal_simulator.signals.resp import RespGenerator
            gen = RespGenerator(cfg)
            
        # Synthesize source signal
        leakage_raw = gen.generate()
        
        # If source is multi-channel, extract the first channel for 1D output
        if leakage_raw.ndim > 1:
            leakage_raw = leakage_raw[0]
            
        # 2. Apply tissue volume conduction lowpass filter if enabled
        if self.enable_volume_conduction:
            nyq = 0.5 * fs
            cutoff = min(self.conduction_cutoff_hz, nyq - 0.1)
            if cutoff > 0.1:
                b_con, a_con = sp_signal.butter(2, cutoff / nyq, btype='low')
                leakage_raw = sp_signal.filtfilt(b_con, a_con, leakage_raw)
                
        # 3. Center and apply coupling amplitude scaling
        leakage_centered = leakage_raw - np.mean(leakage_raw)
        return leakage_centered * coupling

    def generate_multichannel(self, n_channels: int, n_samples: int, fs: float) -> np.ndarray:
        """
        Synthesize multi-channel physiological leakage using coupling matrices.
        
        Parameters
        ----------
        n_channels : int
            Number of target channels.
        n_samples : int
            Number of samples.
        fs : float
            Sampling frequency in Hz.
            
        Returns
        -------
        np.ndarray
            2-D noise array of shape (n_target_channels, n_samples).
        """
        if n_channels <= 0 or n_samples <= 0:
            return np.empty((n_channels, 0), dtype=np.float64)
            
        source = self.config.source_type.lower().strip()
        coupling = self.config.coupling_factor
        duration = n_samples / fs
        
        # 1. Determine configurations and copy/align parameters
        cfg = self.config.source_config
        if cfg is None:
            if source == 'ecg':
                cfg = ECGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'eeg':
                cfg = EEGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'emg':
                cfg = EMGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'ppg':
                cfg = PPGConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'eda':
                cfg = EDAConfig(fs=fs, duration_s=duration, seed=self.seed)
            elif source == 'resp':
                cfg = RespConfig(fs=fs, duration_s=duration, seed=self.seed)
            else:
                raise ValueError(f"Unknown crosstalk source: {source}")
        else:
            import copy
            cfg = copy.deepcopy(cfg)
            
        if hasattr(cfg, 'fs'):
            cfg.fs = fs
        if hasattr(cfg, 'duration_s'):
            cfg.duration_s = duration
            
        # 2. Instantiate source generator
        if source == 'ecg':
            from biosignal_simulator.signals.ecg import ECGGenerator
            gen = ECGGenerator(cfg)
        elif source == 'eeg':
            from biosignal_simulator.signals.eeg import EEGGenerator
            gen = EEGGenerator(cfg)
        elif source == 'emg':
            from biosignal_simulator.signals.emg import EMGGenerator
            gen = EMGGenerator(cfg)
        elif source == 'ppg':
            from biosignal_simulator.signals.ppg import PPGGenerator
            gen = PPGGenerator(cfg)
        elif source == 'eda':
            from biosignal_simulator.signals.eda import EDAGenerator
            gen = EDAGenerator(cfg)
        elif source == 'resp':
            from biosignal_simulator.signals.resp import RespGenerator
            gen = RespGenerator(cfg)
            
        # Synthesize multi-channel source signals
        source_signals = gen.generate()
        if source_signals.ndim == 1:
            source_signals = np.atleast_2d(source_signals)
            
        n_source_ch = source_signals.shape[0]
        
        # Apply tissue volume conduction lowpass filter to all source channels
        if self.enable_volume_conduction:
            nyq = 0.5 * fs
            cutoff = min(self.conduction_cutoff_hz, nyq - 0.1)
            if cutoff > 0.1:
                b_con, a_con = sp_signal.butter(2, cutoff / nyq, btype='low')
                for c in range(n_source_ch):
                    source_signals[c] = sp_signal.filtfilt(b_con, a_con, source_signals[c])
                    
        # Apply coupling matrix multiplication: Y = M * X
        # where X is shape (n_source_ch, n_samples) and M is (n_target_ch, n_source_ch)
        matrix = self.coupling_matrix
        if matrix is None:
            # Fallback: Leak the first source channel to all target channels scaled by coupling
            leaked_signal = source_signals[0] - np.mean(source_signals[0])
            noise_matrix = np.zeros((n_channels, n_samples))
            for c in range(n_channels):
                noise_matrix[c] = leaked_signal * coupling
        else:
            if matrix.shape != (n_channels, n_source_ch):
                raise ValueError(
                    f"Coupling matrix dimensions {matrix.shape} must match target/source "
                    f"channels {n_channels}x{n_source_ch}."
                )
            # Center source signals
            centered_sources = np.zeros_like(source_signals)
            for c in range(n_source_ch):
                centered_sources[c] = source_signals[c] - np.mean(source_signals[c])
                
            # Mix sources and scale by coupling factor
            noise_matrix = np.dot(matrix, centered_sources) * coupling
            
        return noise_matrix
