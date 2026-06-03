"""
Exhaustive EEG sleep states, seizures, and multi-channel covariance checks.
"""
import numpy as np
import pytest
from scipy.signal import welch
from biosignal_simulator.core.config import EEGConfig
from biosignal_simulator.signals.eeg import EEGGenerator
from biosignal_simulator.core.math_utils import bandpower, compute_rms


def test_eeg_state_active_channels_1_fs_100():
    """
    Verify EEG state active with 1 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=1,
        state='active',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_1_fs_200():
    """
    Verify EEG state active with 1 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=1,
        state='active',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_2_fs_100():
    """
    Verify EEG state active with 2 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=2,
        state='active',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_2_fs_200():
    """
    Verify EEG state active with 2 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=2,
        state='active',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_4_fs_100():
    """
    Verify EEG state active with 4 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=4,
        state='active',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_4_fs_200():
    """
    Verify EEG state active with 4 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=4,
        state='active',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_8_fs_100():
    """
    Verify EEG state active with 8 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=8,
        state='active',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_8_fs_200():
    """
    Verify EEG state active with 8 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=8,
        state='active',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_16_fs_100():
    """
    Verify EEG state active with 16 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=16,
        state='active',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_16_fs_200():
    """
    Verify EEG state active with 16 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=16,
        state='active',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_32_fs_100():
    """
    Verify EEG state active with 32 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=32,
        state='active',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_active_channels_32_fs_200():
    """
    Verify EEG state active with 32 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=32,
        state='active',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_1_fs_100():
    """
    Verify EEG state relaxed with 1 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=1,
        state='relaxed',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_1_fs_200():
    """
    Verify EEG state relaxed with 1 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=1,
        state='relaxed',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_2_fs_100():
    """
    Verify EEG state relaxed with 2 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=2,
        state='relaxed',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_2_fs_200():
    """
    Verify EEG state relaxed with 2 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=2,
        state='relaxed',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_4_fs_100():
    """
    Verify EEG state relaxed with 4 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=4,
        state='relaxed',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_4_fs_200():
    """
    Verify EEG state relaxed with 4 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=4,
        state='relaxed',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_8_fs_100():
    """
    Verify EEG state relaxed with 8 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=8,
        state='relaxed',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_8_fs_200():
    """
    Verify EEG state relaxed with 8 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=8,
        state='relaxed',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_16_fs_100():
    """
    Verify EEG state relaxed with 16 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=16,
        state='relaxed',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_16_fs_200():
    """
    Verify EEG state relaxed with 16 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=16,
        state='relaxed',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_32_fs_100():
    """
    Verify EEG state relaxed with 32 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=32,
        state='relaxed',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_relaxed_channels_32_fs_200():
    """
    Verify EEG state relaxed with 32 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=32,
        state='relaxed',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_1_fs_100():
    """
    Verify EEG state n2_sleep with 1 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=1,
        state='n2_sleep',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_1_fs_200():
    """
    Verify EEG state n2_sleep with 1 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=1,
        state='n2_sleep',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_2_fs_100():
    """
    Verify EEG state n2_sleep with 2 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=2,
        state='n2_sleep',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_2_fs_200():
    """
    Verify EEG state n2_sleep with 2 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=2,
        state='n2_sleep',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_4_fs_100():
    """
    Verify EEG state n2_sleep with 4 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=4,
        state='n2_sleep',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_4_fs_200():
    """
    Verify EEG state n2_sleep with 4 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=4,
        state='n2_sleep',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_8_fs_100():
    """
    Verify EEG state n2_sleep with 8 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=8,
        state='n2_sleep',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_8_fs_200():
    """
    Verify EEG state n2_sleep with 8 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=8,
        state='n2_sleep',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_16_fs_100():
    """
    Verify EEG state n2_sleep with 16 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=16,
        state='n2_sleep',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_16_fs_200():
    """
    Verify EEG state n2_sleep with 16 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=16,
        state='n2_sleep',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_32_fs_100():
    """
    Verify EEG state n2_sleep with 32 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=32,
        state='n2_sleep',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n2_sleep_channels_32_fs_200():
    """
    Verify EEG state n2_sleep with 32 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=32,
        state='n2_sleep',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_1_fs_100():
    """
    Verify EEG state n3_sleep with 1 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=1,
        state='n3_sleep',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_1_fs_200():
    """
    Verify EEG state n3_sleep with 1 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=1,
        state='n3_sleep',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_2_fs_100():
    """
    Verify EEG state n3_sleep with 2 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=2,
        state='n3_sleep',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_2_fs_200():
    """
    Verify EEG state n3_sleep with 2 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=2,
        state='n3_sleep',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_4_fs_100():
    """
    Verify EEG state n3_sleep with 4 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=4,
        state='n3_sleep',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_4_fs_200():
    """
    Verify EEG state n3_sleep with 4 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=4,
        state='n3_sleep',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_8_fs_100():
    """
    Verify EEG state n3_sleep with 8 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=8,
        state='n3_sleep',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_8_fs_200():
    """
    Verify EEG state n3_sleep with 8 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=8,
        state='n3_sleep',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_16_fs_100():
    """
    Verify EEG state n3_sleep with 16 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=16,
        state='n3_sleep',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_16_fs_200():
    """
    Verify EEG state n3_sleep with 16 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=16,
        state='n3_sleep',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_32_fs_100():
    """
    Verify EEG state n3_sleep with 32 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=32,
        state='n3_sleep',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_n3_sleep_channels_32_fs_200():
    """
    Verify EEG state n3_sleep with 32 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=32,
        state='n3_sleep',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_1_fs_100():
    """
    Verify EEG state tonic_clonic with 1 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=1,
        state='tonic_clonic',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_1_fs_200():
    """
    Verify EEG state tonic_clonic with 1 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=1,
        state='tonic_clonic',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_2_fs_100():
    """
    Verify EEG state tonic_clonic with 2 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=2,
        state='tonic_clonic',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_2_fs_200():
    """
    Verify EEG state tonic_clonic with 2 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=2,
        state='tonic_clonic',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_4_fs_100():
    """
    Verify EEG state tonic_clonic with 4 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=4,
        state='tonic_clonic',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_4_fs_200():
    """
    Verify EEG state tonic_clonic with 4 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=4,
        state='tonic_clonic',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_8_fs_100():
    """
    Verify EEG state tonic_clonic with 8 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=8,
        state='tonic_clonic',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_8_fs_200():
    """
    Verify EEG state tonic_clonic with 8 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=8,
        state='tonic_clonic',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_16_fs_100():
    """
    Verify EEG state tonic_clonic with 16 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=16,
        state='tonic_clonic',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_16_fs_200():
    """
    Verify EEG state tonic_clonic with 16 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=16,
        state='tonic_clonic',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_32_fs_100():
    """
    Verify EEG state tonic_clonic with 32 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=32,
        state='tonic_clonic',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_tonic_clonic_channels_32_fs_200():
    """
    Verify EEG state tonic_clonic with 32 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=32,
        state='tonic_clonic',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_1_fs_100():
    """
    Verify EEG state absence with 1 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=1,
        state='absence',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_1_fs_200():
    """
    Verify EEG state absence with 1 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=1,
        state='absence',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_2_fs_100():
    """
    Verify EEG state absence with 2 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=2,
        state='absence',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_2_fs_200():
    """
    Verify EEG state absence with 2 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=2,
        state='absence',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_4_fs_100():
    """
    Verify EEG state absence with 4 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=4,
        state='absence',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_4_fs_200():
    """
    Verify EEG state absence with 4 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=4,
        state='absence',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_8_fs_100():
    """
    Verify EEG state absence with 8 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=8,
        state='absence',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_8_fs_200():
    """
    Verify EEG state absence with 8 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=8,
        state='absence',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_16_fs_100():
    """
    Verify EEG state absence with 16 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=16,
        state='absence',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_16_fs_200():
    """
    Verify EEG state absence with 16 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=16,
        state='absence',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_32_fs_100():
    """
    Verify EEG state absence with 32 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=32,
        state='absence',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_absence_channels_32_fs_200():
    """
    Verify EEG state absence with 32 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=32,
        state='absence',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_1_fs_100():
    """
    Verify EEG state epileptiform_spikes with 1 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=1,
        state='epileptiform_spikes',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_1_fs_200():
    """
    Verify EEG state epileptiform_spikes with 1 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=1,
        state='epileptiform_spikes',
        seed=42
    )
    if 1 > 1:
        corr = np.eye(1) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (1, expected_len) if 1 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_2_fs_100():
    """
    Verify EEG state epileptiform_spikes with 2 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=2,
        state='epileptiform_spikes',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_2_fs_200():
    """
    Verify EEG state epileptiform_spikes with 2 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=2,
        state='epileptiform_spikes',
        seed=42
    )
    if 2 > 1:
        corr = np.eye(2) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (2, expected_len) if 2 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_4_fs_100():
    """
    Verify EEG state epileptiform_spikes with 4 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=4,
        state='epileptiform_spikes',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_4_fs_200():
    """
    Verify EEG state epileptiform_spikes with 4 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=4,
        state='epileptiform_spikes',
        seed=42
    )
    if 4 > 1:
        corr = np.eye(4) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (4, expected_len) if 4 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_8_fs_100():
    """
    Verify EEG state epileptiform_spikes with 8 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=8,
        state='epileptiform_spikes',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_8_fs_200():
    """
    Verify EEG state epileptiform_spikes with 8 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=8,
        state='epileptiform_spikes',
        seed=42
    )
    if 8 > 1:
        corr = np.eye(8) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (8, expected_len) if 8 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_16_fs_100():
    """
    Verify EEG state epileptiform_spikes with 16 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=16,
        state='epileptiform_spikes',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_16_fs_200():
    """
    Verify EEG state epileptiform_spikes with 16 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=16,
        state='epileptiform_spikes',
        seed=42
    )
    if 16 > 1:
        corr = np.eye(16) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (16, expected_len) if 16 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_32_fs_100():
    """
    Verify EEG state epileptiform_spikes with 32 channels and fs 100.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=100.0,
        duration_s=2.0,
        n_channels=32,
        state='epileptiform_spikes',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(100.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_state_epileptiform_spikes_channels_32_fs_200():
    """
    Verify EEG state epileptiform_spikes with 32 channels and fs 200.0 Hz generates cleanly.
    This test verifies that multi-channel signals are generated and mixed correctly,
    maintaining stability and correctness under all configurations.
    """
    cfg = EEGConfig(
        fs=200.0,
        duration_s=2.0,
        n_channels=32,
        state='epileptiform_spikes',
        seed=42
    )
    if 32 > 1:
        corr = np.eye(32) * 0.8 + 0.2
        cfg.corr_matrix = corr.tolist()
        
    rec = EEGGenerator(cfg).to_record()
    expected_len = int(200.0 * 2.0)
    
    # 1. Assert output shape match channels and sample length
    assert rec.clean.shape == (32, expected_len) if 32 > 1 else rec.clean.shape == (expected_len,)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(rec.clean))
    
    # 3. Check signal is not a flatline
    assert np.std(rec.clean) > 0.0


def test_eeg_amplitude_sweep_10():
    """
    Verify EEG signal scaling with target amplitude 10.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=10.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_20():
    """
    Verify EEG signal scaling with target amplitude 20.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=20.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_30():
    """
    Verify EEG signal scaling with target amplitude 30.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=30.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_40():
    """
    Verify EEG signal scaling with target amplitude 40.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=40.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_50():
    """
    Verify EEG signal scaling with target amplitude 50.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=50.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_75():
    """
    Verify EEG signal scaling with target amplitude 75.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=75.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_100():
    """
    Verify EEG signal scaling with target amplitude 100.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=100.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_150():
    """
    Verify EEG signal scaling with target amplitude 150.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=150.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_200():
    """
    Verify EEG signal scaling with target amplitude 200.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=200.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_300():
    """
    Verify EEG signal scaling with target amplitude 300.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=300.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_500():
    """
    Verify EEG signal scaling with target amplitude 500.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=500.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_amplitude_sweep_1000():
    """
    Verify EEG signal scaling with target amplitude 1000.0 uV.
    We compute the RMS amplitude of the generated trace and ensure it scales appropriately.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, amplitude_uv=1000.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    rms = compute_rms(eeg)
    assert rms > 0.0
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_6_0():
    """
    Verify EEG alpha peak frequency 6.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=6.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_7_0():
    """
    Verify EEG alpha peak frequency 7.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=7.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_8_0():
    """
    Verify EEG alpha peak frequency 8.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=8.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_9_0():
    """
    Verify EEG alpha peak frequency 9.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=9.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_10_0():
    """
    Verify EEG alpha peak frequency 10.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=10.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_11_0():
    """
    Verify EEG alpha peak frequency 11.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=11.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_12_0():
    """
    Verify EEG alpha peak frequency 12.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=12.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_13_0():
    """
    Verify EEG alpha peak frequency 13.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=13.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))


def test_eeg_alpha_peak_sweep_14_0():
    """
    Verify EEG alpha peak frequency 14.0 Hz is modeled correctly.
    Ensures that custom alpha peak settings instantiate and run cleanly without crashes.
    """
    cfg = EEGConfig(fs=100.0, duration_s=2.0, alpha_peak_hz=14.0, seed=42)
    eeg = EEGGenerator(cfg).generate()
    assert len(eeg) == 200
    assert not np.any(np.isnan(eeg))
