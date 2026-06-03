import os
import tempfile
import numpy as np
import pytest
from typing import Tuple

from biosignal_simulator.core.base import BaseSignal, BaseNoiseModel
from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.core.config import (
    ECGConfig, EEGConfig, EMGConfig, RespConfig,
    ClinicalPresets, sweep_config
)

# Create dummy implementations for testing ABC classes
class DummySignalGenerator(BaseSignal):
    def generate(self) -> np.ndarray:
        # A simple sine wave of 10 Hz
        return np.sin(2 * np.pi * 10.0 * self.t)
        
    def validate_parameters(self) -> Tuple[bool, str]:
        return True, ""


class DummyNoiseModel(BaseNoiseModel):
    def generate(self, n_samples: int, fs: float) -> np.ndarray:
        return self.rng.normal(0, 0.1, n_samples)


def test_dummy_signal_properties():
    gen = DummySignalGenerator(fs=100.0, duration_s=2.0, seed=42)
    
    assert gen.n_samples == 200
    assert len(gen.t) == 200
    assert np.isclose(gen.signal_rms, 1.0 / np.sqrt(2.0), atol=1e-2)
    assert gen.signal_energy > 0
    assert np.isclose(gen.mean, 0.0, atol=1e-2)
    
    # Test to_record packaging
    record = gen.to_record(notes="Dummy test")
    assert isinstance(record, SignalRecord)
    assert record.signal_type == "dummysignal"
    assert record.fs == 100.0
    assert record.metadata["notes"] == "Dummy test"
    assert np.array_equal(record.clean, gen.generate_cached())
    
    # Test plotting execution (mock show to prevent blocking)
    try:
        import matplotlib
        matplotlib.use('Agg') # Non-blocking backend
        fig = gen.plot(show=False)
        assert fig is not None
    except ImportError:
        pass


def test_dummy_noise_properties():
    noise_model = DummyNoiseModel(seed=42)
    signal = np.ones(500)
    
    # Test scale generation
    noisy_arr = noise_model.generate_scaled(signal, snr_db=10.0, fs=100.0)
    assert noisy_arr.shape == signal.shape
    
    # SNR should be approx 10 dB
    p_signal = np.mean(np.square(signal))
    p_noise = np.mean(np.square(noisy_arr))
    snr_actual = 10.0 * np.log10(p_signal / p_noise)
    assert np.isclose(snr_actual, 10.0, atol=0.5)
    
    # Test plotting
    try:
        import matplotlib
        matplotlib.use('Agg')
        fig = noise_model.plot(n_samples=200, fs=100.0, show=False)
        assert fig is not None
    except ImportError:
        pass


def test_signal_record_diagnostic_report(tmp_path):
    gen = DummySignalGenerator(fs=100.0, duration_s=2.0, seed=42)
    clean = gen.generate_cached()
    # Add dummy noise components
    noise_components = {
        "white_noise": np.random.default_rng(42).normal(0, 0.1, 200),
        "line_noise": 0.05 * np.sin(2 * np.pi * 50.0 * gen.t)
    }
    noisy = clean + noise_components["white_noise"] + noise_components["line_noise"]
    
    # Calculate target SNR
    p_sig = np.mean(np.square(clean))
    p_noise = np.mean(np.square(noisy - clean))
    snr = 10.0 * np.log10(p_sig / p_noise)
    
    record = SignalRecord(
        signal_type="ecg",
        fs=100.0,
        t=gen.t,
        clean=clean,
        noisy=noisy,
        noise_components=noise_components,
        signal_params={"heart_rate": 75.0, "target_snr_db": 15.0},
        snr_db=snr
    )
    
    # Test basic properties
    assert record.duration_s == 1.99 # t[-1]
    assert record.n_channels == 1
    assert record.n_samples == 200
    assert record.sampling_rate == 100.0
    assert not record.quality_flags["has_nan"]
    assert not record.quality_flags["has_inf"]
    assert record.snr_linear > 0
    
    # Check that quality metrics are populated
    assert "clean_rms" in record.quality_metrics
    assert "noisy_rms" in record.quality_metrics
    assert "rmse" in record.quality_metrics
    assert "ssim_index" in record.quality_metrics
    assert "noise_snr_breakdown" in record.quality_metrics
    
    # Test console summary
    summary_str = record.summary()
    assert "BIOSIGNAL RECORD SUMMARY" in summary_str
    assert "white_noise" in summary_str
    
    # Test markdown quality report generation
    report_file = tmp_path / "quality_report.md"
    record.export_quality_report(str(report_file))
    
    assert report_file.exists()
    report_content = report_file.read_text(encoding='utf-8')
    assert "# Physiological Signal Simulation Quality Report" in report_content
    assert "General Metadata" in report_content
    assert "Quality Check" in report_content
    assert "white_noise" in report_content


def test_signal_record_convenience_exporters(tmp_path):
    gen = DummySignalGenerator(fs=100.0, duration_s=2.0, seed=42)
    record = gen.to_record(notes="test export")
    
    # pandas DataFrame export
    try:
        df = record.to_dataframe()
        assert df is not None
        assert "clean" in df.columns
        assert "noisy" in df.columns
    except ImportError:
        pass
        
    # CSV export
    csv_file = tmp_path / "signal.csv"
    try:
        record.to_csv(str(csv_file))
        assert csv_file.exists()
    except ImportError:
        pass
        
    # HDF5 export
    h5_file = tmp_path / "signal.h5"
    try:
        record.to_hdf5(str(h5_file))
        assert h5_file.exists()
    except ImportError:
        pass
        
    # EDF-lite export
    edf_file = tmp_path / "signal.edf"
    record.to_edf_lite(str(edf_file))
    assert edf_file.exists()
    
    # NumPy NPZ export
    npz_file = tmp_path / "signal.npz"
    record.to_numpy(str(npz_file))
    assert npz_file.exists()


def test_clinical_presets():
    # 1. ECG Presets
    normal_ecg = ClinicalPresets.get_normal_ecg()
    assert isinstance(normal_ecg, ECGConfig)
    assert normal_ecg.rhythm_type == "normal"
    
    afib_ecg = ClinicalPresets.get_afib_ecg()
    assert isinstance(afib_ecg, ECGConfig)
    assert afib_ecg.rhythm_type == "afib"
    assert afib_ecg.p_amplitude == 0.0 # P-wave is absent in AFib
    
    pvc_ecg = ClinicalPresets.get_pvc_ecg()
    assert isinstance(pvc_ecg, ECGConfig)
    assert pvc_ecg.rhythm_type == "pvc"
    
    vtach_ecg = ClinicalPresets.get_vtach_ecg()
    assert isinstance(vtach_ecg, ECGConfig)
    assert vtach_ecg.rhythm_type == "vtach"
    
    # 2. EEG Presets
    active_eeg = ClinicalPresets.get_sleep_eeg('awake_active')
    assert isinstance(active_eeg, EEGConfig)
    assert active_eeg.state == "active"
    
    relaxed_eeg = ClinicalPresets.get_sleep_eeg('awake_relaxed')
    assert isinstance(relaxed_eeg, EEGConfig)
    assert relaxed_eeg.state == "relaxed"
    
    n2_eeg = ClinicalPresets.get_sleep_eeg('n2_sleep')
    assert isinstance(n2_eeg, EEGConfig)
    assert n2_eeg.state == "n2_sleep"
    
    n3_eeg = ClinicalPresets.get_sleep_eeg('n3_deep_sleep')
    assert isinstance(n3_eeg, EEGConfig)
    assert n3_eeg.state == "n3_sleep"
    
    with pytest.raises(ValueError):
        ClinicalPresets.get_sleep_eeg('invalid_stage')
        
    seizure_eeg = ClinicalPresets.get_seizure_eeg()
    assert isinstance(seizure_eeg, EEGConfig)
    assert seizure_eeg.state == "tonic_clonic"
    
    # 3. EMG Presets
    als_emg = ClinicalPresets.get_emg_als()
    assert isinstance(als_emg, EMGConfig)
    assert als_emg.pathology == "als"
    
    myo_emg = ClinicalPresets.get_emg_myopathy()
    assert isinstance(myo_emg, EMGConfig)
    assert myo_emg.pathology == "myopathic"
    
    # 4. Respiration Presets
    resting_resp = ClinicalPresets.get_resting_resp()
    assert isinstance(resting_resp, RespConfig)
    assert resting_resp.resp_rate_hz == pytest.approx(0.20, abs=0.05)  # ~12 breaths/min
    
    tachy_resp = ClinicalPresets.get_tachypnea_resp()
    assert isinstance(tachy_resp, RespConfig)
    assert tachy_resp.resp_rate_hz == pytest.approx(0.47, abs=0.10)  # ~28 breaths/min


def test_sweep_config():
    base_cfg = ClinicalPresets.get_normal_ecg()
    
    # Sweep heart rate
    hr_vals = [60.0, 80.0, 100.0]
    swept_configs = sweep_config(base_cfg, "heart_rate", hr_vals)
    
    assert len(swept_configs) == 3
    for cfg, val in zip(swept_configs, hr_vals):
        assert isinstance(cfg, ECGConfig)
        assert cfg.heart_rate == val
        
    # Sweep invalid attribute
    with pytest.raises(ValueError):
        sweep_config(base_cfg, "invalid_param_name_xyz", [1, 2, 3])
