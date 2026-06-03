import numpy as np
import pytest
from biosignal_simulator.signals.eeg import EEGGenerator
from biosignal_simulator.signals.emg import EMGGenerator
from biosignal_simulator.signals.ppg import PPGGenerator
from biosignal_simulator.signals.eda import EDAGenerator
from biosignal_simulator.signals.resp import RespGenerator

from biosignal_simulator.core.config import (
    EEGConfig,
    EMGConfig,
    PPGConfig,
    EDAConfig,
    RespConfig
)

def test_eeg_generator():
    cfg = EEGConfig(fs=128, duration_s=4, n_channels=1, seed=10)
    gen = EEGGenerator(cfg)
    eeg = gen.generate()
    assert len(eeg) == 128 * 4
    
    # Test multi-channel
    cfg_multi = EEGConfig(fs=128, duration_s=4, n_channels=3, seed=10)
    gen_multi = EEGGenerator(cfg_multi)
    eeg_multi = gen_multi.generate()
    assert eeg_multi.shape == (3, 128 * 4)
    
    # Test reproducibility
    gen_multi2 = EEGGenerator(cfg_multi)
    assert np.array_equal(eeg_multi, gen_multi2.generate())
    
    # Test brain states
    for state in ['active', 'n2_sleep', 'n3_sleep']:
        cfg_state = EEGConfig(fs=100, duration_s=6, state=state, seed=42)
        eeg_state = EEGGenerator(cfg_state).generate()
        assert len(eeg_state) == 100 * 6
        assert not np.all(eeg_state == 0.0)

def test_emg_generator():
    cfg = EMGConfig(fs=500, duration_s=3, envelope_type='constant', seed=20)
    gen = EMGGenerator(cfg)
    emg = gen.generate()
    assert len(emg) == 500 * 3
    
    # Test ramp
    cfg_ramp = EMGConfig(fs=500, duration_s=3, envelope_type='ramp', seed=20)
    emg_ramp = EMGGenerator(cfg_ramp).generate()
    assert len(emg_ramp) == 500 * 3
    
    # Test burst
    cfg_burst = EMGConfig(fs=500, duration_s=3, envelope_type='burst', seed=20)
    emg_burst = EMGGenerator(cfg_burst).generate()
    assert len(emg_burst) == 500 * 3
    
    # Test intramuscular pathology modes
    for path in ['normal', 'neuropathic', 'myopathic']:
        cfg_path = EMGConfig(fs=1000, duration_s=2, emg_type='intramuscular', pathology=path, seed=42)
        emg_path = EMGGenerator(cfg_path).generate()
        assert len(emg_path) == 1000 * 2
        assert not np.all(emg_path == 0.0)

def test_ppg_generator():
    cfg = PPGConfig(fs=100, duration_s=10, heart_rate=80, seed=30)
    gen = PPGGenerator(cfg)
    ppg = gen.generate()
    assert len(ppg) == 100 * 10
    
    # Test reproducibility
    gen2 = PPGGenerator(cfg)
    assert np.array_equal(ppg, gen2.generate())

def test_eda_generator():
    cfg = EDAConfig(fs=32, duration_s=30, seed=40)
    gen = EDAGenerator(cfg)
    eda = gen.generate()
    assert len(eda) == 32 * 30
    # EDA skin conductance values should be positive
    assert np.all(eda >= 0.0)

def test_resp_generator():
    cfg = RespConfig(fs=32, duration_s=30, seed=50)
    gen = RespGenerator(cfg)
    resp = gen.generate()
    assert len(resp) == 32 * 30
    
    # Reproducibility
    gen2 = RespGenerator(cfg)
    assert np.array_equal(resp, gen2.generate())
