"""
Exhaustive Round-Trip and Edge-Case Tests for BioSignal Exporters and Importers.

This module validates that signal records can be serialized and deserialized across
all supported formats (NumPy, CSV, JSON, HDF5, EDF, WFDB, MAT) while maintaining
numerical precision, metadata integrity, and system-level folder structures.
"""

import os
import tempfile
import datetime
import warnings
import json
import numpy as np
import pytest

from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.io.exporters import BiosignalExporter, BiosignalImporter

@pytest.fixture
def clean_record_params():
    """Returns baseline parameters for generating test records."""
    return {
        'signal_type': 'ecg',
        'fs': 100.0,
        't': np.arange(500) / 100.0,
        'clean': np.sin(2.0 * np.pi * 5.0 * (np.arange(500) / 100.0)),
        'noisy': np.sin(2.0 * np.pi * 5.0 * (np.arange(500) / 100.0)) + 0.1 * np.sin(2.0 * np.pi * 50.0 * (np.arange(500) / 100.0)),
        'noise_components': {
            'powerline': 0.1 * np.sin(2.0 * np.pi * 50.0 * (np.arange(500) / 100.0)),
            'gaussian': 0.02 * np.random.default_rng(42).normal(size=500)
        },
        'signal_params': {'heart_rate': 75.0, 'amplitude': 1.0},
        'noise_params': {'powerline': {'frequency': 50.0}, 'gaussian': {'std': 0.02}},
        'snr_db': 15.4,
        'metadata': {'subject_id': 'SUBJ-001', 'lead': 'II', 'notes': 'Test run'}
    }

@pytest.fixture
def single_channel_record(clean_record_params):
    """Fixture providing a standard 1-D SignalRecord."""
    return SignalRecord(**clean_record_params)

@pytest.fixture
def multi_channel_record(clean_record_params):
    """Fixture providing a 3-channel SignalRecord."""
    params = clean_record_params.copy()
    
    # Expand to 3 channels
    params['clean'] = np.stack([params['clean'], params['clean'] * 0.8, -params['clean']])
    params['noisy'] = np.stack([params['noisy'], params['noisy'] * 0.85, -params['noisy'] + 0.05])
    
    # Expand noise components
    params['noise_components'] = {
        'powerline': np.stack([params['noise_components']['powerline'], params['noise_components']['powerline'] * 1.1, -params['noise_components']['powerline']]),
        'gaussian': np.stack([params['noise_components']['gaussian'], params['noise_components']['gaussian'] * 0.9, params['noise_components']['gaussian'] * 1.2])
    }
    
    return SignalRecord(**params)


# =====================================================================
# NumPy Exporter Roundtrip Tests
# =====================================================================

def test_numpy_roundtrip_single_channel(single_channel_record):
    """Verify 1D compressed NumPy roundtrip preserves all values and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "subdir", "record.npz")
        
        # Test directory auto-creation and export
        BiosignalExporter.export_numpy(single_channel_record, path)
        assert os.path.exists(path)
        
        # Import and verify
        imported = BiosignalImporter.import_numpy(path)
        
        assert imported.signal_type == single_channel_record.signal_type
        assert np.isclose(imported.fs, single_channel_record.fs)
        assert np.isclose(imported.snr_db, single_channel_record.snr_db)
        
        assert np.allclose(imported.t, single_channel_record.t)
        assert np.allclose(imported.clean, single_channel_record.clean)
        assert np.allclose(imported.noisy, single_channel_record.noisy)
        
        assert set(imported.noise_components.keys()) == set(single_channel_record.noise_components.keys())
        for k in imported.noise_components:
            assert np.allclose(imported.noise_components[k], single_channel_record.noise_components[k])
            
        assert imported.signal_params == single_channel_record.signal_params
        assert imported.noise_params == single_channel_record.noise_params
        assert imported.metadata == single_channel_record.metadata

def test_numpy_roundtrip_multi_channel(multi_channel_record):
    """Verify multi-channel compressed NumPy roundtrip preserves multi-dimensional arrays."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.npz")
        
        BiosignalExporter.export_numpy(multi_channel_record, path)
        imported = BiosignalImporter.import_numpy(path)
        
        assert imported.clean.shape == (3, 500)
        assert imported.noisy.shape == (3, 500)
        assert np.allclose(imported.clean, multi_channel_record.clean)
        assert np.allclose(imported.noisy, multi_channel_record.noisy)
        
        for k in imported.noise_components:
            assert imported.noise_components[k].shape == (3, 500)
            assert np.allclose(imported.noise_components[k], multi_channel_record.noise_components[k])

def test_numpy_export_empty_record():
    """Verify that attempting to create an empty record raises ValueError."""
    with pytest.raises(ValueError):
        SignalRecord(
            signal_type='eeg', fs=100.0, t=np.array([]), clean=np.array([]), noisy=np.array([]), noise_components={}
        )


# =====================================================================
# CSV Exporter Roundtrip Tests
# =====================================================================

def test_csv_roundtrip_single_channel_relative(single_channel_record):
    """Verify single-channel CSV roundtrip with relative timestamps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.csv")
        
        # Export
        BiosignalExporter.export_csv(
            record=single_channel_record,
            path=path,
            delimiter=",",
            timestamp_format="relative",
            write_header_comments=True
        )
        assert os.path.exists(path)
        
        # Import and check
        imported = BiosignalImporter.import_csv(path, delimiter=",")
        
        assert imported.signal_type == single_channel_record.signal_type
        assert np.isclose(imported.fs, single_channel_record.fs)
        assert np.allclose(imported.t, single_channel_record.t)
        assert np.allclose(imported.clean, single_channel_record.clean)
        assert np.allclose(imported.noisy, single_channel_record.noisy)
        
        for k in imported.noise_components:
            assert np.allclose(imported.noise_components[k], single_channel_record.noise_components[k])
            
        assert imported.metadata['subject_id'] == 'SUBJ-001'

def test_csv_roundtrip_multi_channel_absolute(multi_channel_record):
    """Verify multi-channel CSV roundtrip with absolute datetime timestamps and custom delimiter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.csv")
        ref_date = datetime.datetime(2026, 6, 2, 12, 0, 0, tzinfo=datetime.timezone.utc)
        
        # Export
        BiosignalExporter.export_csv(
            record=multi_channel_record,
            path=path,
            delimiter=";",
            timestamp_format="absolute",
            start_datetime=ref_date,
            write_header_comments=True
        )
        
        # Import
        imported = BiosignalImporter.import_csv(path, delimiter=";")
        
        assert imported.clean.shape == (3, 500)
        assert np.allclose(imported.clean, multi_channel_record.clean)
        assert np.allclose(imported.noisy, multi_channel_record.noisy)
        
        # Time indices will be reconstructed from samples because pandas parses string timestamps back to seconds
        assert len(imported.t) == 500
        assert np.isclose(imported.fs, multi_channel_record.fs)

def test_csv_no_header_comments(single_channel_record):
    """Verify that importing a CSV without comments uses default fallback values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record_no_header.csv")
        
        # Export without comments
        BiosignalExporter.export_csv(
            record=single_channel_record,
            path=path,
            write_header_comments=False
        )
        
        # Import
        imported = BiosignalImporter.import_csv(path)
        
        # Reconstructed signal should still have correct arrays, but defaults for metadata
        assert np.allclose(imported.clean, single_channel_record.clean)
        assert imported.signal_type == 'unknown' # Default fallback
        assert imported.fs == 250.0 # Default fallback


# =====================================================================
# HDF5 Exporter Roundtrip Tests
# =====================================================================

def test_hdf5_roundtrip_single_channel(single_channel_record):
    """Verify structured HDF5 roundtrip with compression settings."""
    pytest.importorskip("h5py")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.h5")
        
        # Export
        BiosignalExporter.export_hdf5(
            record=single_channel_record,
            path=path,
            compression="gzip",
            compression_opts=5
        )
        assert os.path.exists(path)
        
        # Import
        imported = BiosignalImporter.import_hdf5(path)
        
        assert imported.signal_type == single_channel_record.signal_type
        assert np.isclose(imported.fs, single_channel_record.fs)
        assert np.isclose(imported.snr_db, single_channel_record.snr_db)
        
        assert np.allclose(imported.t, single_channel_record.t)
        assert np.allclose(imported.clean, single_channel_record.clean)
        assert np.allclose(imported.noisy, single_channel_record.noisy)
        
        for k in imported.noise_components:
            assert np.allclose(imported.noise_components[k], single_channel_record.noise_components[k])
            
        assert imported.signal_params == single_channel_record.signal_params
        assert imported.metadata == single_channel_record.metadata

def test_hdf5_roundtrip_multi_channel(multi_channel_record):
    """Verify multi-channel HDF5 roundtrip."""
    pytest.importorskip("h5py")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.h5")
        
        # Export
        BiosignalExporter.export_hdf5(multi_channel_record, path)
        
        # Import
        imported = BiosignalImporter.import_hdf5(path)
        
        assert imported.clean.shape == (3, 500)
        assert imported.noisy.shape == (3, 500)
        assert np.allclose(imported.clean, multi_channel_record.clean)
        assert np.allclose(imported.noisy, multi_channel_record.noisy)


# =====================================================================
# European Data Format (EDF) Roundtrip Tests
# =====================================================================

def test_edf_roundtrip_legacy_single_channel_noisy(single_channel_record):
    """Verify legacy export_edf_lite and import_edf roundtrip for single-channel noisy data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy_record.edf")
        
        # Export noisy signal
        BiosignalExporter.export_edf_lite(single_channel_record, path, use_clean=False)
        assert os.path.exists(path)
        
        # Import back
        imported = BiosignalImporter.import_edf(path)
        
        assert imported.fs == single_channel_record.fs
        assert len(imported.t) == len(single_channel_record.t)
        
        # Precision: EDF scales to 16-bit integer (max error should be < 0.1% of signal span)
        span = np.max(single_channel_record.noisy) - np.min(single_channel_record.noisy)
        max_error = 0.002 * span
        
        # The imported record clean and noisy channels will map to the single channel loaded
        assert np.allclose(imported.noisy, single_channel_record.noisy, atol=max_error)

def test_edf_roundtrip_multi_channel_rich(multi_channel_record):
    """Verify advanced export_edf multi-channel structure parses clean, noisy, and noise components."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rich_record.edf")
        
        # Export
        BiosignalExporter.export_edf(
            record=multi_channel_record,
            path=path,
            record_duration_s=1.0,
            patient_id="SUBJ-002",
            recording_id="REC-002"
        )
        assert os.path.exists(path)
        
        # Import
        imported = BiosignalImporter.import_edf(path)
        
        assert imported.fs == multi_channel_record.fs
        assert len(imported.t) == len(multi_channel_record.t)
        
        # Check shapes (should reconstruct 3 channels of clean and noisy)
        assert imported.clean.shape == (3, 500)
        assert imported.noisy.shape == (3, 500)
        
        # Quantization check (atol)
        clean_span = np.max(multi_channel_record.clean) - np.min(multi_channel_record.clean)
        noisy_span = np.max(multi_channel_record.noisy) - np.min(multi_channel_record.noisy)
        
        assert np.allclose(imported.clean, multi_channel_record.clean, atol=0.002 * clean_span)
        assert np.allclose(imported.noisy, multi_channel_record.noisy, atol=0.002 * noisy_span)
        
        # Verify noise components reconstruction
        assert 'powerline' in imported.noise_components
        assert imported.noise_components['powerline'].shape == (3, 500)


# =====================================================================
# PhysioNet WFDB Roundtrip Tests
# =====================================================================

def test_wfdb_format_16_roundtrip(multi_channel_record):
    """Verify PhysioNet Format 16 (16-bit) exporter and importer roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "wfdb_f16")
        
        # Export
        BiosignalExporter.export_wfdb(multi_channel_record, base_path, format_code=16)
        
        assert os.path.exists(f"{base_path}.hea")
        assert os.path.exists(f"{base_path}.dat")
        
        # Import
        imported = BiosignalImporter.import_wfdb(base_path)
        
        assert imported.fs == multi_channel_record.fs
        assert len(imported.t) == len(multi_channel_record.t)
        
        # Check multi-channel signals (clean and noisy should be restored)
        # Verify channels shapes
        assert imported.clean.shape == (3, 500)
        assert imported.noisy.shape == (3, 500)
        
        # Quantization limits: Format 16 offers 16-bit range -> very high precision
        assert np.allclose(imported.clean, multi_channel_record.clean, atol=1e-4)
        assert np.allclose(imported.noisy, multi_channel_record.noisy, atol=1e-4)

def test_wfdb_format_212_roundtrip(multi_channel_record):
    """Verify PhysioNet Format 212 (packed 12-bit) exporter and importer roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "wfdb_f212")
        
        # Export
        BiosignalExporter.export_wfdb(multi_channel_record, base_path, format_code=212)
        
        assert os.path.exists(f"{base_path}.hea")
        assert os.path.exists(f"{base_path}.dat")
        
        # Import
        imported = BiosignalImporter.import_wfdb(base_path)
        
        assert imported.fs == multi_channel_record.fs
        assert len(imported.t) == len(multi_channel_record.t)
        
        assert imported.clean.shape == (3, 500)
        assert imported.noisy.shape == (3, 500)
        
        # Quantization limit: 12-bit range maps to 4096 bins
        clean_span = np.max(multi_channel_record.clean) - np.min(multi_channel_record.clean)
        noisy_span = np.max(multi_channel_record.noisy) - np.min(multi_channel_record.noisy)
        
        assert np.allclose(imported.clean, multi_channel_record.clean, atol=0.002 * clean_span)
        assert np.allclose(imported.noisy, multi_channel_record.noisy, atol=0.002 * noisy_span)

def test_wfdb_invalid_format_code(single_channel_record):
    """Verify that providing an unsupported format code raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "wfdb_invalid")
        with pytest.raises(ValueError, match="Supported WFDB formats: 16 and 212"):
            BiosignalExporter.export_wfdb(single_channel_record, base_path, format_code=8)


# =====================================================================
# MATLAB Exporter Roundtrip Tests
# =====================================================================

def test_matlab_roundtrip(single_channel_record):
    """Verify MATLAB workspace structure roundtrip."""
    pytest.importorskip("scipy.io")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.mat")
        
        # Export
        BiosignalExporter.export_mat(single_channel_record, path)
        assert os.path.exists(path)
        
        # Import
        imported = BiosignalImporter.import_mat(path)
        
        assert imported.signal_type == single_channel_record.signal_type
        assert np.isclose(imported.fs, single_channel_record.fs)
        
        assert np.allclose(imported.t, single_channel_record.t)
        assert np.allclose(imported.clean, single_channel_record.clean)
        assert np.allclose(imported.noisy, single_channel_record.noisy)
        
        for k in imported.noise_components:
            assert np.allclose(imported.noise_components[k], single_channel_record.noise_components[k])


# =====================================================================
# JSON Exporter Roundtrip Tests
# =====================================================================

def test_json_roundtrip_raw(single_channel_record):
    """Verify raw JSON serialization roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.json")
        
        # Export
        BiosignalExporter.export_json(single_channel_record, path, compress=False)
        assert os.path.exists(path)
        
        # Import
        imported = BiosignalImporter.import_json(path, compressed=False)
        
        assert imported.signal_type == single_channel_record.signal_type
        assert np.isclose(imported.fs, single_channel_record.fs)
        assert np.allclose(imported.t, single_channel_record.t)
        assert np.allclose(imported.clean, single_channel_record.clean)
        assert np.allclose(imported.noisy, single_channel_record.noisy)

def test_json_roundtrip_compressed(single_channel_record):
    """Verify compressed gzip JSON serialization roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "record.json.gz")
        
        # Export
        BiosignalExporter.export_json(single_channel_record, path, compress=True)
        assert os.path.exists(path)
        
        # Import
        imported = BiosignalImporter.import_json(path, compressed=True)
        
        assert imported.signal_type == single_channel_record.signal_type
        assert np.allclose(imported.clean, single_channel_record.clean)


# =====================================================================
# Precision and Boundary Edge Cases
# =====================================================================

def test_flatline_scaling_safety():
    """Verify that exporting flatline signals (which can cause divide-by-zero) handles bounds gracefully."""
    fs = 100.0
    t = np.arange(100) / fs
    clean = np.ones(100) * 1.5 # Constant flatline
    noisy = np.ones(100) * 1.5
    
    flat_record = SignalRecord(
        signal_type='eeg',
        fs=fs,
        t=t,
        clean=clean,
        noisy=noisy,
        noise_components={}
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test HDF5
        try:
            import h5py
            h5_path = os.path.join(tmpdir, "flat.h5")
            BiosignalExporter.export_hdf5(flat_record, h5_path)
            assert os.path.exists(h5_path)
        except ImportError:
            pass
            
        # Test EDF (EDF relies heavily on physical bounds division)
        edf_path = os.path.join(tmpdir, "flat.edf")
        # Should not crash with division by zero!
        BiosignalExporter.export_edf_lite(flat_record, edf_path)
        assert os.path.exists(edf_path)
        
        # Verify readback
        imported = BiosignalImporter.import_edf(edf_path)
        # Should reconstruct with constant values
        assert np.allclose(imported.noisy, 1.5, atol=0.01)
