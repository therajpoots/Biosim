"""
End-to-End Integration Tests for BioSignal Simulator Platform.

This module validates the complete biopotential simulation pipeline:
1. Signal generation with clinical configurations.
2. Mixing with multiple noise models using the SNR controller.
3. Exporting records to clinical formats (HDF5, EDF, WFDB, CSV, JSON).
4. Re-importing records and validating signal integrity.
5. Evaluating quality metrics (RMSE, SSIM) across SNR sweeps.
6. Generating HTML validation reports.
"""

import os
import tempfile
import numpy as np
import pytest

from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.core.config import (
    ECGConfig,
    EEGConfig,
    GaussianNoiseConfig,
    PowerlineNoiseConfig,
    MotionArtifactConfig
)
from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.signals.eeg import EEGGenerator
from biosignal_simulator.noise.gaussian import GaussianNoise
from biosignal_simulator.noise.powerline import PowerlineNoise
from biosignal_simulator.noise.motion import MotionArtifact
from biosignal_simulator.composer.mixer import SignalMixer
from biosignal_simulator.composer.snr_controller import SNRController
from biosignal_simulator.io.exporters import BiosignalExporter, BiosignalImporter
from biosignal_simulator.utils.validation import validate_signal, generate_validation_report_html


# =====================================================================
# 1. Full Pipeline Simulation & Roundtrip Verification
# =====================================================================

def test_full_pipeline_ecg_roundtrip():
    """Verify standard ECG simulation, mixing, file roundtrip, and report generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Configure & Generate Signal
        cfg_sig = ECGConfig(
            fs=250.0,
            duration_s=5.0,
            heart_rate=80.0,
            seed=42
        )
        gen = ECGGenerator(cfg_sig)

        # Step 2: Configure & Generate Noise Models
        noise_g = GaussianNoise(std=0.05, seed=123)
        noise_p = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=456)

        # Step 3: Mix Signal and Noise to Target SNR (e.g. 12 dB)
        target_snr_db = 12.0
        
        # Use SignalMixer to do the mixing
        mixer = SignalMixer(
            signal_generator=gen,
            noise_models=[noise_g, noise_p],
            target_snr_db=target_snr_db
        )
        composite_record = mixer.mix()
        
        assert isinstance(composite_record, SignalRecord)
        assert composite_record.signal_type == 'ecg'
        assert len(composite_record.t) == 1250

        # Step 4: Export to Multiple Formats
        # 4a. Export to CSV
        csv_path = os.path.join(tmpdir, "record.csv")
        BiosignalExporter.export_csv(composite_record, csv_path)
        assert os.path.exists(csv_path)

        # 4b. Export to JSON
        json_path = os.path.join(tmpdir, "record.json")
        BiosignalExporter.export_json(composite_record, json_path)
        assert os.path.exists(json_path)

        # 4c. Export to HDF5
        try:
            import h5py
            h5_path = os.path.join(tmpdir, "record.h5")
            BiosignalExporter.export_hdf5(composite_record, h5_path)
            assert os.path.exists(h5_path)
        except ImportError:
            h5_path = None

        # 4d. Export to EDF
        edf_path = os.path.join(tmpdir, "record.edf")
        BiosignalExporter.export_edf_lite(composite_record, edf_path)
        assert os.path.exists(edf_path)

        # 4e. Export to WFDB
        wfdb_base = os.path.join(tmpdir, "record_wfdb")
        BiosignalExporter.export_wfdb(composite_record, wfdb_base, format_code=16)
        assert os.path.exists(f"{wfdb_base}.hea")
        assert os.path.exists(f"{wfdb_base}.dat")

        # Step 5: Import and Re-validate
        # 5a. CSV Import
        imported_csv = BiosignalImporter.import_csv(csv_path)
        assert np.allclose(imported_csv.clean, composite_record.clean, atol=1e-5)
        
        # 5b. JSON Import
        imported_json = BiosignalImporter.import_json(json_path)
        assert np.allclose(imported_json.noisy, composite_record.noisy, atol=1e-5)

        # 5c. HDF5 Import
        if h5_path:
            imported_h5 = BiosignalImporter.import_hdf5(h5_path)
            assert np.allclose(imported_h5.clean, composite_record.clean, atol=1e-5)

        # 5d. EDF Import
        imported_edf = BiosignalImporter.import_edf(edf_path)
        # EDF is 16-bit integer quantized, check bounds
        span = np.max(composite_record.noisy) - np.min(composite_record.noisy)
        assert np.allclose(imported_edf.noisy, composite_record.noisy, atol=0.005 * span)

        # 5e. WFDB Import
        imported_wfdb = BiosignalImporter.import_wfdb(wfdb_base)
        assert np.allclose(imported_wfdb.clean, composite_record.clean, atol=1e-4)

        # Step 6: Physiological Validation & Report
        report = validate_signal(composite_record.noisy, composite_record.fs, 'ecg')
        assert report.metrics['heart_rate_bpm'] > 0.0
        
        report_path = os.path.join(tmpdir, "report.html")
        generate_validation_report_html(composite_record, report, report_path)
        assert os.path.exists(report_path)


# =====================================================================
# 2. Sweeping Target SNRs and Metrics Validation
# =====================================================================

def test_snr_controller_sweep():
    """Verify that sweeping target SNR scales noise power and changes similarity metrics monotonically."""
    fs = 200.0
    t = np.arange(1000) / fs
    clean_signal = np.sin(2.0 * np.pi * 1.5 * t)
    
    # Instantiate raw noise (Gaussian)
    noise_model = GaussianNoise(std=1.0, seed=42)
    
    # We will sweep target SNRs from -10 dB to +30 dB
    snrs_to_test = [-10.0, 0.0, 10.0, 20.0, 30.0]
    
    rmses = []
    ssims = []
    
    from biosignal_simulator.core.math_utils import compute_rmse, compute_ssim_1d
    
    for snr in snrs_to_test:
        # Scale noise to target SNR
        ctrl = SNRController(noise_model, target_snr_db=snr)
        noise_scaled = ctrl.apply(clean_signal, fs)
        mixed = clean_signal + noise_scaled
        
        # Calculate actual mixed SNR
        p_sig = np.mean(np.square(clean_signal))
        p_noise = np.mean(np.square(noise_scaled))
        actual_snr = 10.0 * np.log10(p_sig / p_noise)
        
        # Assert exact SNR matching
        assert np.isclose(actual_snr, snr, rtol=1e-5)
        
        # Calculate quality metrics
        rmse = compute_rmse(clean_signal, mixed)
        ssim = compute_ssim_1d(clean_signal, mixed)
        
        rmses.append(rmse)
        ssims.append(ssim)
        
    # Check monotonic trends:
    # 1. As target SNR increases (less noise), RMSE must decrease monotonically
    for i in range(len(rmses) - 1):
        assert rmses[i] > rmses[i+1], f"RMSE did not decrease: {rmses}"
        
    # 2. As target SNR increases (higher similarity), SSIM must increase monotonically
    for i in range(len(ssims) - 1):
        assert ssims[i] < ssims[i+1], f"SSIM did not increase: {ssims}"


# =====================================================================
# 3. Multichannel High-Density EEG Mixed Simulation
# =====================================================================

def test_multichannel_eeg_mixed_simulation():
    """Verify end-to-end multi-channel EEG simulation with motion bursts and crosstalk."""
    fs = 128.0
    duration_s = 4.0
    n_channels = 2
    
    # 1. Generate 2-channel clean EEG
    cfg_eeg = EEGConfig(
        fs=fs,
        duration_s=duration_s,
        n_channels=n_channels,
        seed=100
    )
    gen = EEGGenerator(cfg_eeg)
    
    # 2. Generate multi-channel motion artifacts
    motion_model = MotionArtifact(lf_amplitude=0.3, seed=200)
    
    # 3. Mix using SignalMixer
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[motion_model],
        target_snr_db=10.0
    )
    mixed_record = mixer.mix()
    
    assert mixed_record.clean.shape == (2, 512)
    assert mixed_record.noisy.shape == (2, 512)
    assert 'MotionArtifact' in mixed_record.noise_components
