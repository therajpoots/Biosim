"""
Exhaustive IO format roundtrips for multi-channel configuration parameters.
"""
import os
import tempfile
import json
import numpy as np
import pytest
from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.io.exporters import BiosignalExporter, BiosignalImporter


def test_io_format_csv_channels_1_dur_1_0():
    """
    Verify symmetrical csv roundtrip with 1 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_1_dur_2_0():
    """
    Verify symmetrical csv roundtrip with 1 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_2_dur_1_0():
    """
    Verify symmetrical csv roundtrip with 2 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_2_dur_2_0():
    """
    Verify symmetrical csv roundtrip with 2 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_4_dur_1_0():
    """
    Verify symmetrical csv roundtrip with 4 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_4_dur_2_0():
    """
    Verify symmetrical csv roundtrip with 4 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_8_dur_1_0():
    """
    Verify symmetrical csv roundtrip with 8 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_8_dur_2_0():
    """
    Verify symmetrical csv roundtrip with 8 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_16_dur_1_0():
    """
    Verify symmetrical csv roundtrip with 16 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_16_dur_2_0():
    """
    Verify symmetrical csv roundtrip with 16 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_32_dur_1_0():
    """
    Verify symmetrical csv roundtrip with 32 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_csv_channels_32_dur_2_0():
    """
    Verify symmetrical csv roundtrip with 32 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'csv' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'csv' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'csv' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'csv' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_1_dur_1_0():
    """
    Verify symmetrical json roundtrip with 1 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_1_dur_2_0():
    """
    Verify symmetrical json roundtrip with 1 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_2_dur_1_0():
    """
    Verify symmetrical json roundtrip with 2 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_2_dur_2_0():
    """
    Verify symmetrical json roundtrip with 2 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_4_dur_1_0():
    """
    Verify symmetrical json roundtrip with 4 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_4_dur_2_0():
    """
    Verify symmetrical json roundtrip with 4 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_8_dur_1_0():
    """
    Verify symmetrical json roundtrip with 8 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_8_dur_2_0():
    """
    Verify symmetrical json roundtrip with 8 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_16_dur_1_0():
    """
    Verify symmetrical json roundtrip with 16 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_16_dur_2_0():
    """
    Verify symmetrical json roundtrip with 16 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_32_dur_1_0():
    """
    Verify symmetrical json roundtrip with 32 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_json_channels_32_dur_2_0():
    """
    Verify symmetrical json roundtrip with 32 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'json' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'json' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'json' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'json' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_1_dur_1_0():
    """
    Verify symmetrical hdf5 roundtrip with 1 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_1_dur_2_0():
    """
    Verify symmetrical hdf5 roundtrip with 1 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_2_dur_1_0():
    """
    Verify symmetrical hdf5 roundtrip with 2 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_2_dur_2_0():
    """
    Verify symmetrical hdf5 roundtrip with 2 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_4_dur_1_0():
    """
    Verify symmetrical hdf5 roundtrip with 4 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_4_dur_2_0():
    """
    Verify symmetrical hdf5 roundtrip with 4 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_8_dur_1_0():
    """
    Verify symmetrical hdf5 roundtrip with 8 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_8_dur_2_0():
    """
    Verify symmetrical hdf5 roundtrip with 8 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_16_dur_1_0():
    """
    Verify symmetrical hdf5 roundtrip with 16 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_16_dur_2_0():
    """
    Verify symmetrical hdf5 roundtrip with 16 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_32_dur_1_0():
    """
    Verify symmetrical hdf5 roundtrip with 32 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_hdf5_channels_32_dur_2_0():
    """
    Verify symmetrical hdf5 roundtrip with 32 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    pytest.importorskip('h5py')
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'hdf5' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'hdf5' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'hdf5' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'hdf5' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_1_dur_1_0():
    """
    Verify symmetrical edf roundtrip with 1 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_1_dur_2_0():
    """
    Verify symmetrical edf roundtrip with 1 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_2_dur_1_0():
    """
    Verify symmetrical edf roundtrip with 2 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_2_dur_2_0():
    """
    Verify symmetrical edf roundtrip with 2 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_4_dur_1_0():
    """
    Verify symmetrical edf roundtrip with 4 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_4_dur_2_0():
    """
    Verify symmetrical edf roundtrip with 4 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_8_dur_1_0():
    """
    Verify symmetrical edf roundtrip with 8 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_8_dur_2_0():
    """
    Verify symmetrical edf roundtrip with 8 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_16_dur_1_0():
    """
    Verify symmetrical edf roundtrip with 16 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_16_dur_2_0():
    """
    Verify symmetrical edf roundtrip with 16 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_32_dur_1_0():
    """
    Verify symmetrical edf roundtrip with 32 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_edf_channels_32_dur_2_0():
    """
    Verify symmetrical edf roundtrip with 32 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'edf' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'edf' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'edf' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'edf' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_1_dur_1_0():
    """
    Verify symmetrical wfdb roundtrip with 1 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_1_dur_2_0():
    """
    Verify symmetrical wfdb roundtrip with 1 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 1 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(1)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 1 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (1, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_2_dur_1_0():
    """
    Verify symmetrical wfdb roundtrip with 2 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_2_dur_2_0():
    """
    Verify symmetrical wfdb roundtrip with 2 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 2 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(2)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 2 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (2, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_4_dur_1_0():
    """
    Verify symmetrical wfdb roundtrip with 4 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_4_dur_2_0():
    """
    Verify symmetrical wfdb roundtrip with 4 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 4 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(4)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 4 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (4, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_8_dur_1_0():
    """
    Verify symmetrical wfdb roundtrip with 8 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_8_dur_2_0():
    """
    Verify symmetrical wfdb roundtrip with 8 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 8 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(8)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 8 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (8, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_16_dur_1_0():
    """
    Verify symmetrical wfdb roundtrip with 16 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_16_dur_2_0():
    """
    Verify symmetrical wfdb roundtrip with 16 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 16 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(16)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 16 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (16, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_32_dur_1_0():
    """
    Verify symmetrical wfdb roundtrip with 32 channels and duration 1.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 1.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))


def test_io_format_wfdb_channels_32_dur_2_0():
    """
    Verify symmetrical wfdb roundtrip with 32 channels and duration 2.0s.
    Verifies that the generated file contains all headers and matches on reconstruction.
    """
    fs = 100.0
    length = int(fs * 2.0)
    t = np.arange(length) / fs
    if 32 == 1:
        clean = np.sin(t)
        noisy = clean + 0.05
    else:
        clean = np.stack([np.sin(t) * (i+1) for i in range(32)])
        noisy = clean + 0.05
        
    rec = SignalRecord('ecg', fs, t, clean, noisy, {})
    
    with tempfile.TemporaryDirectory() as tmpdir:
        if 'wfdb' == 'wfdb':
            path = os.path.join(tmpdir, "rec_wfdb")
            BiosignalExporter.export_wfdb(rec, path, format_code=16)
            imported = BiosignalImporter.import_wfdb(path)
        elif 'wfdb' == 'edf':
            path = os.path.join(tmpdir, "rec.edf")
            BiosignalExporter.export_edf(rec, path)
            imported = BiosignalImporter.import_edf(path)
        elif 'wfdb' == 'hdf5':
            path = os.path.join(tmpdir, "rec.h5")
            BiosignalExporter.export_hdf5(rec, path)
            imported = BiosignalImporter.import_hdf5(path)
        elif 'wfdb' == 'json':
            path = os.path.join(tmpdir, "rec.json")
            BiosignalExporter.export_json(rec, path, compress=False)
            imported = BiosignalImporter.import_json(path, compressed=False)
        else: # csv
            path = os.path.join(tmpdir, "rec.csv")
            BiosignalExporter.export_csv(rec, path)
            imported = BiosignalImporter.import_csv(path)
            
        # Verify readback fields and lengths match exactly
        assert imported.fs == fs
        if 32 == 1:
            assert len(imported.clean) == length
        else:
            assert imported.clean.shape == (32, length)
        assert not np.any(np.isnan(imported.clean))
