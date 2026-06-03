import numpy as np
import pytest
from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.core.config import ECGConfig

def test_ecg_generator_shape_and_seed():
    # Verify shape
    config = ECGConfig(fs=250, duration_s=5, heart_rate=60, hr_variability_std=0.0)
    gen = ECGGenerator(config)
    ecg = gen.generate()
    assert len(ecg) == 250 * 5
    assert len(gen.t) == len(ecg)
    
    # Verify seed reproducibility (bit-identical outputs)
    gen1 = ECGGenerator(ECGConfig(fs=250, duration_s=5, seed=42))
    gen2 = ECGGenerator(ECGConfig(fs=250, duration_s=5, seed=42))
    assert np.array_equal(gen1.generate(), gen2.generate())
    
    # Different seeds should give different outputs
    gen3 = ECGGenerator(ECGConfig(fs=250, duration_s=5, seed=43))
    assert not np.array_equal(gen1.generate(), gen3.generate())

def test_ecg_invalid_params():
    with pytest.raises(ValueError):
        ECGGenerator(ECGConfig(heart_rate=30.0)) # Out of range [40, 200]
        
    with pytest.raises(ValueError):
        ECGGenerator(ECGConfig(qrs_amplitude=0.1)) # Out of range [0.3, 3.0]

def test_ecg_baseline_subtraction():
    # Verify that a baseline trend is not present (spline correction works)
    config = ECGConfig(fs=200, duration_s=6, heart_rate=70)
    gen = ECGGenerator(config)
    ecg = gen.generate()
    
    # Average of signal should be close to 0.0 after baseline subtraction
    assert np.isclose(np.mean(ecg), 0.0, atol=0.05)

def test_ecg_12lead_and_vcg():
    # 12-Lead projection test
    cfg_12 = ECGConfig(fs=250, duration_s=4, lead_type='12lead')
    gen_12 = ECGGenerator(cfg_12)
    ecg_12 = gen_12.generate()
    assert ecg_12.shape == (12, 250 * 4)
    
    # VCG trajectory test
    cfg_vcg = ECGConfig(fs=200, duration_s=4, lead_type='vcg')
    gen_vcg = ECGGenerator(cfg_vcg)
    vcg = gen_vcg.generate()
    assert vcg.shape == (3, 200 * 4)
    
    # ST elevation test
    cfg_st = ECGConfig(fs=200, duration_s=4, st_elevation=0.15, lead_type='single', lead_name='II')
    gen_st = ECGGenerator(cfg_st)
    ecg_st = gen_st.generate()
    assert len(ecg_st) == 200 * 4
