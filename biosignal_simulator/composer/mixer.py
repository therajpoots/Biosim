"""
High-Fidelity Physiological and Additive Noise Composition Engine.

This module provides the `SignalMixer` class, which orchestrates the final stage of the
simulation pipeline: mixing clean physiological signals (single or multi-channel) with
additive noise processes, transient injected artifacts, digitizer quantization errors,
and wireless dropped frames.

Physical and Physiological Context:
    Recorded biosignals are never isolated. For example, a single chest patch records
    not only cardiac activity (ECG), but also breathing chest movements (respiration)
    and chest muscle contractions (EMG). The `SignalMixer` supports composite clean
    simulations to mimic these multi-signal mixtures.

Mathematical Formulations:
    1. Composite clean signal formulation:
       $$\\vec{x}_{\\text{clean}}(t) = \\sum_{j} \\beta_j \\cdot \\mathbf{P}_j \\cdot \\vec{s}_j(t)$$
       where $\\vec{s}_j(t)$ is the $j$-th clean physiological source, $\\beta_j$ is its mixing
       coefficient, and $\\mathbf{P}_j$ is the channel projection matrix.

    2. Total Additive Noise:
       $$\\vec{n}_{\\text{additive}}(t) = \\sum_{k} \\vec{w}_k(t)$$
       where $\\vec{w}_k(t)$ are individual additive noise realizations.

    3. Non-Linear & Discontinuous Artifact Cascades:
       Wearable artifacts (such as sensor detachment and packet dropouts) and discretization
       quantization are applied sequentially in a physical cascade:
       $$\\vec{x}_{\\text{noisy}}(t) = \\mathcal{Q}\\left( \\mathcal{D}\\left( \\mathcal{P}\\left( \\vec{x}_{\\text{clean}}(t) + \\vec{n}_{\\text{additive}}(t) \\right) \\right) \\right)$$
       where:
         - $\\mathcal{P}$ represents Sensor Detachment suppression.
         - $\\mathcal{D}$ represents wireless Packet Loss dropouts.
         - $\\mathcal{Q}$ represents ADC Quantization roundoff and noise shaping.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from biosignal_simulator.core.base import BaseSignal, BaseNoiseModel
from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.noise.quantization import QuantizationNoise
from biosignal_simulator.noise.wearable import SensorDetachmentNoise, PacketLossNoise
from biosignal_simulator.composer.scheduler import NoiseScheduler, BaseSchedule
from biosignal_simulator.composer.snr_controller import SNRController, DynamicSNRController

class SignalMixer:
    """
    Physiological Signal and Noise Composition Orchestrator.
    
    Manages composition of clean single-source or multi-source composite signals,
    scales and modulates additive noises (stationary or scheduled), injects transient
    artifacts, applies wearable dropouts/detachments, and processes ADC discretization.
    """
    
    def __init__(
        self,
        signal_generator: BaseSignal,
        noise_models: List[BaseNoiseModel],
        target_snr_db: Optional[Union[float, BaseSchedule]] = None,
        composite_signals: Optional[List[Tuple[BaseSignal, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Signal Mixer.
        
        Parameters
        ----------
        signal_generator : BaseSignal
            The primary physiological signal generator.
        noise_models : List[BaseNoiseModel]
            List of noise models to apply during composition.
        target_snr_db : Optional[float or BaseSchedule]
            Target SNR in dB. Can be a static float or a time-varying Schedule.
        composite_signals : Optional[List[Tuple[BaseSignal, float]]]
            Additional clean signals to mix with the primary signal.
            Each element is a tuple of (signal_generator, coupling_factor).
            Useful for simulating multi-signal recordings (e.g. ECG + Respiration).
        metadata : Optional[Dict[str, Any]]
            Additional custom annotations or experimental metadata.
        """
        self.signal_generator = signal_generator
        self.noise_models = noise_models
        self.target_snr_db = target_snr_db
        self.composite_signals = composite_signals or []
        self.metadata = metadata or {}

    def mix(self) -> SignalRecord:
        """
        Synthesize, scale, compose, and package the physiological signal and noise components.
        
        Returns
        -------
        SignalRecord
            The completed, fully validated and characterized simulation record.
        """
        # 1. Generate primary clean signal
        clean_primary = self.signal_generator.generate_cached()
        fs = self.signal_generator.fs
        t = self.signal_generator.t
        n_samples = clean_primary.shape[-1]
        
        # 2. Build composite clean signal if extra sources are provided
        clean = clean_primary.copy()
        for extra_gen, coupling in self.composite_signals:
            extra_clean = extra_gen.generate_cached()
            
            # Resample if sampling rates are mismatched
            if not np.isclose(extra_gen.fs, fs):
                extra_clean = extra_gen.resample(fs)
                
            # Truncate or pad to match primary length
            if extra_clean.shape[-1] != n_samples:
                if extra_clean.ndim == 2:
                    extra_clean = extra_clean[:, :n_samples]
                    if extra_clean.shape[-1] < n_samples:
                        pad = np.zeros((extra_clean.shape[0], n_samples - extra_clean.shape[-1]))
                        extra_clean = np.hstack((extra_clean, pad))
                else:
                    extra_clean = extra_clean[:n_samples]
                    if len(extra_clean) < n_samples:
                        extra_clean = np.pad(extra_clean, (0, n_samples - len(extra_clean)))
                        
            # Match channel dimensions
            if clean.ndim == 2 and extra_clean.ndim == 1:
                # Project 1D composite clean to all target channels
                for c in range(clean.shape[0]):
                    clean[c] += coupling * extra_clean
            elif clean.ndim == 2 and extra_clean.ndim == 2:
                # Mix channel-wise, wrapping around if channels mismatch
                n_ch_target = clean.shape[0]
                n_ch_extra = extra_clean.shape[0]
                for c in range(n_ch_target):
                    clean[c] += coupling * extra_clean[c % n_ch_extra]
            else:
                # Target is 1D, take first channel of extra if it's 2D
                if extra_clean.ndim == 2:
                    clean += coupling * extra_clean[0]
                else:
                    clean += coupling * extra_clean

        # --- Categorize Noise Models ---
        additive_models = []
        quant_model: Optional[QuantizationNoise] = None
        detachment_model: Optional[SensorDetachmentNoise] = None
        packet_loss_model: Optional[PacketLossNoise] = None
        
        for m in self.noise_models:
            if isinstance(m, QuantizationNoise):
                quant_model = m
            elif isinstance(m, SensorDetachmentNoise):
                detachment_model = m
            elif isinstance(m, PacketLossNoise):
                packet_loss_model = m
            else:
                additive_models.append(m)

        # 3. Generate additive noise components
        noise_components: Dict[str, np.ndarray] = {}
        total_additive_noise = np.zeros_like(clean)
        
        for model in additive_models:
            name = model.__class__.__name__
            base_name = name
            counter = 1
            while name in noise_components:
                name = f"{base_name}_{counter}"
                counter += 1
                
            # Synthesize noise matching shape of composite clean signal
            if clean.ndim == 2:
                n_ch = clean.shape[0]
                noise_val = np.zeros_like(clean)
                for c in range(n_ch):
                    noise_val[c] = model.generate(n_samples, fs)
            else:
                noise_val = model.generate(n_samples, fs)
                
            noise_components[name] = noise_val
            total_additive_noise += noise_val

        # 4. Apply SNR Scaling
        if self.target_snr_db is not None:
            if isinstance(self.target_snr_db, BaseSchedule):
                # Apply Dynamic SNR scheduling
                # Wrap noise in a dummy generator to use DynamicSNRController
                # B-06 FIX: DummyNoise must return a 1D slice for the channel
                # requested by DynamicSNRController, not the full 2D array.
                class DummyNoise(BaseNoiseModel):
                    def __init__(self, raw_n):
                        super().__init__()
                        self.raw_n = raw_n
                        self._channel_idx = 0  # current channel being processed
                    def generate(self, n_s, f_s):
                        if self.raw_n.ndim == 2:
                            return self.raw_n[self._channel_idx]
                        return self.raw_n
                        
                dummy = DummyNoise(total_additive_noise)
                controller = DynamicSNRController(dummy, self.target_snr_db, window_duration_s=0.4)
                
                if clean.ndim == 2:
                    scaled_noise = np.zeros_like(total_additive_noise)
                    for c in range(clean.shape[0]):
                        dummy._channel_idx = c
                        scaled_noise[c] = controller.apply(clean[c], fs)
                        scaled_noise[c] -= clean[c]  # controller returns clean+noise; extract noise
                    # Apportion scaling to individual components
                    p_total_raw = np.mean(np.square(total_additive_noise))
                    if p_total_raw > 1e-15:
                        scale_envelope = scaled_noise / (total_additive_noise + 1e-15)
                        for name in noise_components:
                            noise_components[name] *= scale_envelope
                        total_additive_noise = scaled_noise
                else:
                    scaled_noise = controller.apply(clean, fs)
                    # Apportion scaling to individual components
                    p_total_raw = np.mean(np.square(total_additive_noise))
                    if p_total_raw > 1e-15:
                        scale_envelope = scaled_noise / (total_additive_noise + 1e-15)
                        for name in noise_components:
                            noise_components[name] *= scale_envelope
                        total_additive_noise = scaled_noise
            else:
                # Apply static SNR scaling
                snr = float(self.target_snr_db)
                p_signal = np.mean(np.square(clean))
                p_noise_raw = np.mean(np.square(total_additive_noise))
                
                if p_signal <= 1e-15:
                    p_signal = 1e-15
                    
                p_noise_target = p_signal * (10.0 ** (-snr / 10.0))
                
                if p_noise_raw > 1e-15:
                    scale = np.sqrt(p_noise_target / p_noise_raw)
                else:
                    scale = 1.0
                    
                for name in noise_components:
                    noise_components[name] *= scale
                total_additive_noise *= scale

        # Mix clean signal and additive noise
        noisy = clean + total_additive_noise

        # 5. Apply Sensor Detachment (Suppression)
        if detachment_model is not None:
            if clean.ndim == 2:
                det_errors = np.zeros_like(clean)
                for c in range(clean.shape[0]):
                    noisy[c], det_err = detachment_model.apply(noisy[c], fs)
                    det_errors[c] = det_err
                noise_components['SensorDetachmentNoise'] = det_errors
            else:
                noisy, det_err = detachment_model.apply(noisy, fs)
                noise_components['SensorDetachmentNoise'] = det_err

        # 6. Apply Packet Loss
        if packet_loss_model is not None:
            if clean.ndim == 2:
                loss_errors = np.zeros_like(clean)
                for c in range(clean.shape[0]):
                    noisy[c], loss_err = packet_loss_model.apply(noisy[c])
                    loss_errors[c] = loss_err
                noise_components['PacketLossNoise'] = loss_errors
            else:
                noisy, loss_err = packet_loss_model.apply(noisy)
                noise_components['PacketLossNoise'] = loss_err

        # 7. Apply Quantization (ADC)
        if quant_model is not None:
            if clean.ndim == 2:
                q_errors = np.zeros_like(clean)
                for c in range(clean.shape[0]):
                    noisy[c], q_err = quant_model.apply(noisy[c])
                    q_errors[c] = q_err
                noise_components['QuantizationNoise'] = q_errors
            else:
                noisy, q_err = quant_model.apply(noisy)
                noise_components['QuantizationNoise'] = q_err

        # 8. Compute achieved SNR statistics
        p_noise_actual = np.mean(np.square(noisy - clean))
        p_sig_actual = np.mean(np.square(clean))
        if p_noise_actual > 1e-15 and p_sig_actual > 1e-15:
            achieved_snr = 10.0 * np.log10(p_sig_actual / p_noise_actual)
        else:
            achieved_snr = None

        # 9. Extract Diagnostics and Parameters
        signal_params = {}
        if hasattr(self.signal_generator, 'config') and self.signal_generator.config:
            from dataclasses import asdict
            signal_params = asdict(self.signal_generator.config)
            
        noise_params = {}
        for m in self.noise_models:
            name = m.__class__.__name__
            if hasattr(m, 'config') and m.config:
                from dataclasses import asdict
                noise_params[name] = asdict(m.config)

        # Run clinical diagnostics (stored in metadata)
        diagnostics = self._run_signal_diagnostics(clean, noisy, fs)
        self.metadata['diagnostics'] = diagnostics
        
        # Primary signal identifier
        sig_type_name = self.signal_generator.__class__.__name__.replace('Generator', '').lower()

        # B-14 FIX: BaseSchedule objects are not JSON-serialisable.
        # Store a safe scalar or string representation instead.
        if isinstance(self.target_snr_db, BaseSchedule):
            serialisable_target_snr = repr(self.target_snr_db)
        elif self.target_snr_db is None:
            serialisable_target_snr = None
        else:
            serialisable_target_snr = float(self.target_snr_db)

        return SignalRecord(
            signal_type=sig_type_name,
            fs=fs,
            t=t,
            clean=clean,
            noisy=noisy,
            noise_components=noise_components,
            signal_params=signal_params,
            noise_params=noise_params,
            snr_db=achieved_snr,
            target_snr_db=serialisable_target_snr,
            metadata=self.metadata
        )

    def _run_signal_diagnostics(self, clean: np.ndarray, noisy: np.ndarray, fs: float) -> Dict[str, Any]:
        """Compute structural signal diagnostics and data quality indicators."""
        from scipy.stats import skew, kurtosis
        
        def calculate_metrics_1d(c_arr, n_arr):
            error = n_arr - c_arr
            p_sig = np.mean(c_arr ** 2)
            p_err = np.mean(error ** 2)
            
            # Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(n_arr)) != 0) / (len(n_arr) / fs)
            
            # Clipping detection (amplitude saturation)
            sig_max = np.max(np.abs(n_arr))
            clipping_threshold = 0.99 * sig_max
            clip_count = int(np.sum(np.abs(n_arr) >= clipping_threshold))
            clip_ratio = clip_count / len(n_arr)
            
            return {
                'skewness': float(skew(n_arr)),
                'kurtosis': float(kurtosis(n_arr)),
                'zero_crossing_rate_hz': float(zero_crossings),
                'clipping_ratio': float(clip_ratio),
                'rms_amplitude': float(np.sqrt(p_sig)),
                'error_rms': float(np.sqrt(p_err))
            }
            
        if clean.ndim == 2:
            n_ch = clean.shape[0]
            metrics = []
            for c in range(n_ch):
                metrics.append(calculate_metrics_1d(clean[c], noisy[c]))
            return {'channels': metrics}
        else:
            return calculate_metrics_1d(clean, noisy)
