"""
Exhaustive Parameter Bounds and Edge-Case Tests for BioSignal Configurations.

This module validates that all 20 config dataclasses enforce strict physiological
boundaries, clinical limits, and mathematical constraints, raising ValueErrors
on invalid parameter inputs while accepting valid ranges.
"""

import pytest
import numpy as np
from biosignal_simulator.core.config import (
    ECGConfig,
    EEGConfig,
    EMGConfig,
    PPGConfig,
    EDAConfig,
    RespConfig,
    GaussianNoiseConfig,
    ColoredNoiseConfig,
    PowerlineNoiseConfig,
    BaselineWanderConfig,
    MotionArtifactConfig,
    ElectrodeNoiseConfig,
    EMGArtifactConfig,
    ImpulseNoiseConfig,
    QuantizationNoiseConfig,
    CrosstalkNoiseConfig,
    SensorDetachmentConfig,
    ElectrodeDisplacementConfig,
    LightLeakageConfig,
    PacketLossConfig
)

# =====================================================================
# 1. ECGConfig Parameter Bounds
# =====================================================================

def test_ecg_config_defaults():
    """Verify ECGConfig default values are stable and instantiate correctly."""
    cfg = ECGConfig()
    assert cfg.fs == 500.0
    assert cfg.duration_s == 10.0
    assert cfg.heart_rate == 75.0
    assert cfg.hr_variability_std == 0.05
    assert cfg.p_amplitude == 0.15
    assert cfg.qrs_amplitude == 1.0
    assert cfg.t_amplitude == 0.35
    assert cfg.qrs_width == 0.08
    assert cfg.pr_interval == 0.16
    assert cfg.st_elevation == 0.0
    assert cfg.lead_type == 'single'
    assert cfg.lead_name == 'II'
    assert cfg.rhythm_type == 'normal'

def test_ecg_config_valid_bounds():
    """Verify ECGConfig accepts extreme but valid physical inputs."""
    # Min bounds
    cfg_min = ECGConfig(
        fs=50.0,
        duration_s=0.1,
        heart_rate=40.0,
        hr_variability_std=0.0,
        p_amplitude=0.0,
        qrs_amplitude=0.3,
        t_amplitude=0.0,
        qrs_width=0.03,
        pr_interval=0.08,
        st_elevation=-2.0,
        lead_type='vcg',
        rhythm_type='afib'
    )
    assert cfg_min.fs == 50.0

    # Max bounds
    cfg_max = ECGConfig(
        fs=5000.0,
        duration_s=3600.0,
        heart_rate=200.0,
        hr_variability_std=0.5,
        p_amplitude=2.0,
        qrs_amplitude=3.0,
        t_amplitude=2.0,
        qrs_width=0.25,
        pr_interval=0.4,
        st_elevation=2.0,
        lead_type='12lead',
        lead_name='V6',
        rhythm_type='pvc'
    )
    assert cfg_max.fs == 5000.0

def test_ecg_config_invalid_bounds():
    """Verify ECGConfig raises ValueError for invalid parameter combinations."""
    # fs below bounds
    with pytest.raises(ValueError, match="fs"):
        ECGConfig(fs=49.9)

    # fs above bounds
    with pytest.raises(ValueError, match="fs"):
        ECGConfig(fs=5000.1)

    # duration non-positive
    with pytest.raises(ValueError, match="duration_s"):
        ECGConfig(duration_s=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        ECGConfig(duration_s=-5.0)

    # heart rate below bounds
    with pytest.raises(ValueError, match="heart_rate"):
        ECGConfig(heart_rate=39.0)

    # heart rate above bounds
    with pytest.raises(ValueError, match="heart_rate"):
        ECGConfig(heart_rate=200.1)

    # HRV variance negative
    with pytest.raises(ValueError, match="hr_variability_std"):
        ECGConfig(hr_variability_std=-0.01)

    # HRV variance too high
    with pytest.raises(ValueError, match="hr_variability_std"):
        ECGConfig(hr_variability_std=0.51)

    # p_amplitude out of range
    with pytest.raises(ValueError, match="p_amplitude"):
        ECGConfig(p_amplitude=-0.1)
    with pytest.raises(ValueError, match="p_amplitude"):
        ECGConfig(p_amplitude=2.1)

    # qrs_amplitude out of range
    with pytest.raises(ValueError, match="qrs_amplitude"):
        ECGConfig(qrs_amplitude=0.2)
    with pytest.raises(ValueError, match="qrs_amplitude"):
        ECGConfig(qrs_amplitude=3.1)

    # t_amplitude out of range
    with pytest.raises(ValueError, match="t_amplitude"):
        ECGConfig(t_amplitude=-0.05)
    with pytest.raises(ValueError, match="t_amplitude"):
        ECGConfig(t_amplitude=2.05)

    # qrs_width out of range
    with pytest.raises(ValueError, match="qrs_width"):
        ECGConfig(qrs_width=0.02)
    with pytest.raises(ValueError, match="qrs_width"):
        ECGConfig(qrs_width=0.26)

    # pr_interval out of range
    with pytest.raises(ValueError, match="pr_interval"):
        ECGConfig(pr_interval=0.07)
    with pytest.raises(ValueError, match="pr_interval"):
        ECGConfig(pr_interval=0.41)

    # st_elevation out of range
    with pytest.raises(ValueError, match="ST elevation"):
        ECGConfig(st_elevation=2.1)
    with pytest.raises(ValueError, match="ST elevation"):
        ECGConfig(st_elevation=-2.1)

    # Invalid lead name
    with pytest.raises(ValueError, match="lead_name"):
        ECGConfig(lead_name='XYZ')

    # Invalid lead type
    with pytest.raises(ValueError, match="lead_type"):
        ECGConfig(lead_type='custom_lead')

    # Invalid rhythm type
    with pytest.raises(ValueError, match="rhythm_type"):
        ECGConfig(rhythm_type='dead')


# =====================================================================
# 2. EEGConfig Parameter Bounds
# =====================================================================

def test_eeg_config_defaults():
    """Verify EEGConfig default values."""
    cfg = EEGConfig()
    assert cfg.fs == 256.0
    assert cfg.duration_s == 10.0
    assert 'alpha' in cfg.band_powers
    assert cfg.background_1f_power == 0.3
    assert cfg.alpha_peak_hz == 10.0
    assert cfg.n_channels == 1
    assert cfg.amplitude_uv == 50.0
    assert cfg.state == 'relaxed'

def test_eeg_config_valid_bounds():
    """Verify EEGConfig valid range extremes."""
    cfg_min = EEGConfig(
        fs=32.0,
        duration_s=0.5,
        background_1f_power=0.0,
        alpha_peak_hz=6.0,
        n_channels=1,
        amplitude_uv=0.1,
        state='active'
    )
    assert cfg_min.fs == 32.0

    cfg_max = EEGConfig(
        fs=4000.0,
        duration_s=1200.0,
        background_1f_power=1.0,
        alpha_peak_hz=14.0,
        n_channels=3,
        corr_matrix=[[1.0, 0.5, 0.2], [0.5, 1.0, 0.4], [0.2, 0.4, 1.0]],
        amplitude_uv=1000.0,
        state='epileptiform_spikes'
    )
    assert cfg_max.fs == 4000.0

def test_eeg_config_invalid_bounds():
    """Verify EEGConfig invalid parameters."""
    with pytest.raises(ValueError, match="fs"):
        EEGConfig(fs=31.9)
    with pytest.raises(ValueError, match="fs"):
        EEGConfig(fs=4000.1)

    with pytest.raises(ValueError, match="duration_s"):
        EEGConfig(duration_s=0.0)

    with pytest.raises(ValueError, match="background_1f_power"):
        EEGConfig(background_1f_power=-0.05)
    with pytest.raises(ValueError, match="background_1f_power"):
        EEGConfig(background_1f_power=1.05)

    with pytest.raises(ValueError, match="alpha_peak_hz"):
        EEGConfig(alpha_peak_hz=5.9)
    with pytest.raises(ValueError, match="alpha_peak_hz"):
        EEGConfig(alpha_peak_hz=14.1)

    with pytest.raises(ValueError, match="n_channels"):
        EEGConfig(n_channels=0)

    with pytest.raises(ValueError, match="amplitude_uv"):
        EEGConfig(amplitude_uv=0.0)

    with pytest.raises(ValueError, match="state"):
        EEGConfig(state='coma')

    # Correlation matrix shape mismatch
    with pytest.raises(ValueError, match="corr_matrix shape"):
        EEGConfig(n_channels=2, corr_matrix=[[1.0]])

    # Correlation matrix asymmetric
    with pytest.raises(ValueError, match="symmetric"):
        EEGConfig(n_channels=2, corr_matrix=[[1.0, 0.5], [0.6, 1.0]])

    # Correlation matrix not positive-definite
    with pytest.raises(ValueError, match="positive-definite"):
        EEGConfig(n_channels=2, corr_matrix=[[1.0, 1.5], [1.5, 1.0]])


# =====================================================================
# 3. EMGConfig Parameter Bounds
# =====================================================================

def test_emg_config_defaults():
    """Verify EMGConfig default values."""
    cfg = EMGConfig()
    assert cfg.fs == 2000.0
    assert cfg.duration_s == 10.0
    assert cfg.fmin_hz == 20.0
    assert cfg.fmax_hz == 500.0
    assert cfg.envelope_type == 'constant'
    assert cfg.contraction_level == 1.0
    assert cfg.amplitude_uv == 500.0
    assert cfg.emg_type == 'surface'
    assert cfg.pathology == 'normal'

def test_emg_config_valid_bounds():
    """Verify EMGConfig valid range extremes."""
    cfg_min = EMGConfig(
        fs=100.0,
        duration_s=0.5,
        fmin_hz=5.0,
        fmax_hz=45.0,
        envelope_type='burst',
        contraction_level=0.0,
        amplitude_uv=1.0,
        emg_type='intramuscular',
        pathology='als'
    )
    assert cfg_min.fs == 100.0

    cfg_max = EMGConfig(
        fs=10000.0,
        duration_s=600.0,
        fmin_hz=100.0,
        fmax_hz=4000.0,
        envelope_type='ramp',
        contraction_level=1.0,
        amplitude_uv=10000.0,
        pathology='parkinsons_tremor'
    )
    assert cfg_max.fs == 10000.0

def test_emg_config_invalid_bounds():
    """Verify EMGConfig invalid parameters."""
    with pytest.raises(ValueError, match="fs"):
        EMGConfig(fs=99.0)
    with pytest.raises(ValueError, match="fs"):
        EMGConfig(fs=10001.0)

    with pytest.raises(ValueError, match="duration_s"):
        EMGConfig(duration_s=-1.0)

    with pytest.raises(ValueError, match="fmin_hz"):
        EMGConfig(fmin_hz=0.5)

    with pytest.raises(ValueError, match="fmax_hz"):
        EMGConfig(fs=500.0, fmax_hz=250.0) # Equal to Nyquist (fmax must be strictly below Nyquist)

    with pytest.raises(ValueError, match="contraction_level"):
        EMGConfig(contraction_level=-0.1)
    with pytest.raises(ValueError, match="contraction_level"):
        EMGConfig(contraction_level=1.1)

    with pytest.raises(ValueError, match="amplitude_uv"):
        EMGConfig(amplitude_uv=0.0)

    with pytest.raises(ValueError, match="envelope_type"):
        EMGConfig(envelope_type='noise')

    with pytest.raises(ValueError, match="emg_type"):
        EMGConfig(emg_type='cardiac')

    with pytest.raises(ValueError, match="pathology"):
        EMGConfig(pathology='healthy') # 'normal' is correct, not 'healthy'


# =====================================================================
# 4. PPGConfig Parameter Bounds
# =====================================================================

def test_ppg_config_valid_invalid():
    """Verify PPGConfig parameters validation."""
    cfg = PPGConfig()
    assert cfg.fs == 100.0
    assert cfg.heart_rate == 75.0
    
    # Valid extreme
    cfg_extreme = PPGConfig(fs=50.0, heart_rate=40.0, resp_modulation=0.5)
    assert cfg_extreme.resp_modulation == 0.5
    
    # Invalid fs
    with pytest.raises(ValueError):
        PPGConfig(fs=9.0)
        
    # Invalid heart_rate
    with pytest.raises(ValueError):
        PPGConfig(heart_rate=25.0)
    with pytest.raises(ValueError):
        PPGConfig(heart_rate=225.0)


# =====================================================================
# 5. EDAConfig Parameter Bounds
# =====================================================================

def test_eda_config_valid_invalid():
    """Verify EDAConfig parameters validation."""
    cfg = EDAConfig()
    assert cfg.fs == 32.0
    
    # Valid extreme
    cfg_extreme = EDAConfig(fs=2.0, scl_amplitude_us=0.1, event_rate_hz=0.0)
    assert cfg_extreme.scl_amplitude_us == 0.1
    
    # Invalid fs
    with pytest.raises(ValueError):
        EDAConfig(fs=0.5)
        
    # Invalid baseline conductance
    with pytest.raises(ValueError):
        EDAConfig(scl_amplitude_us=-0.1)


# =====================================================================
# 6. RespConfig Parameter Bounds
# =====================================================================

def test_resp_config_valid_invalid():
    """Verify RespConfig parameters validation."""
    cfg = RespConfig()
    assert cfg.fs == 32.0
    assert cfg.resp_rate_hz == 0.25
    
    # Valid extreme
    cfg_extreme = RespConfig(fs=5.0, resp_rate_hz=0.05, amplitude=1.0)
    assert cfg_extreme.resp_rate_hz == 0.05
    
    # Invalid respiration rate
    with pytest.raises(ValueError):
        RespConfig(resp_rate_hz=0.04)
    with pytest.raises(ValueError):
        RespConfig(resp_rate_hz=2.1)


# =====================================================================
# 7. Noise Configurations Parameter Bounds
# =====================================================================

def test_gaussian_noise_config():
    """Verify GaussianNoiseConfig boundaries."""
    cfg = GaussianNoiseConfig()
    assert cfg.std == 1.0
    
    with pytest.raises(ValueError):
        GaussianNoiseConfig(std=-0.01)

def test_colored_noise_config():
    """Verify ColoredNoiseConfig boundaries."""
    cfg = ColoredNoiseConfig()
    assert cfg.exponent == 1.0 # Pink noise
    
    # Valid exponent range [-2.0, 2.0]
    cfg_val = ColoredNoiseConfig(exponent=-1.5)
    assert cfg_val.exponent == -1.5
    
    with pytest.raises(ValueError):
        ColoredNoiseConfig(method='non-existent')

def test_powerline_noise_config():
    """Verify PowerlineNoiseConfig boundaries."""
    cfg = PowerlineNoiseConfig()
    assert cfg.f_line_hz == 50.0
    
    cfg_60 = PowerlineNoiseConfig(f_line_hz=60.0, n_harmonics=5)
    assert cfg_60.f_line_hz == 60.0
    
    with pytest.raises(ValueError):
        PowerlineNoiseConfig(amplitude=-0.1)
    with pytest.raises(ValueError):
        PowerlineNoiseConfig(n_harmonics=0)

def test_baseline_wander_config():
    """Verify BaselineWanderConfig boundaries."""
    cfg = BaselineWanderConfig()
    assert cfg.amplitude == 0.1
    
    with pytest.raises(ValueError):
        BaselineWanderConfig(amplitude=-0.1)
    with pytest.raises(ValueError):
        BaselineWanderConfig(f_resp_hz=0.0)

def test_motion_artifact_config():
    """Verify MotionArtifactConfig boundaries."""
    cfg = MotionArtifactConfig()
    assert cfg.lf_amplitude == 0.2
    
    with pytest.raises(ValueError):
        MotionArtifactConfig(lf_amplitude=-0.05)

def test_electrode_noise_config():
    """Verify ElectrodeNoiseConfig boundaries."""
    cfg = ElectrodeNoiseConfig()
    assert cfg.popcorn_amplitude == 0.05
    
    with pytest.raises(ValueError):
        ElectrodeNoiseConfig(popcorn_amplitude=-0.1)

def test_emg_artifact_config():
    """Verify EMGArtifactConfig boundaries."""
    cfg = EMGArtifactConfig()
    assert cfg.amplitude_fraction == 0.1
    
    with pytest.raises(ValueError):
        EMGArtifactConfig(amplitude_fraction=-0.5)
    with pytest.raises(ValueError):
        EMGArtifactConfig(amplitude_fraction=1.5)

def test_impulse_noise_config():
    """Verify ImpulseNoiseConfig boundaries."""
    cfg = ImpulseNoiseConfig()
    assert cfg.rate_hz == 1.0
    
    with pytest.raises(ValueError):
        ImpulseNoiseConfig(rate_hz=-0.1)

def test_quantization_noise_config():
    """Verify QuantizationNoiseConfig boundaries."""
    cfg = QuantizationNoiseConfig()
    assert cfg.n_bits == 12
    
    cfg_8 = QuantizationNoiseConfig(n_bits=8, v_range=5.0)
    assert cfg_8.n_bits == 8
    
    with pytest.raises(ValueError):
        QuantizationNoiseConfig(n_bits=3)
    with pytest.raises(ValueError):
        QuantizationNoiseConfig(n_bits=33)

def test_crosstalk_noise_config():
    """Verify CrosstalkNoiseConfig boundaries."""
    cfg = CrosstalkNoiseConfig()
    assert cfg.coupling_factor == 0.1
    
    with pytest.raises(ValueError):
        CrosstalkNoiseConfig(coupling_factor=-0.01)
    with pytest.raises(ValueError):
        CrosstalkNoiseConfig(source_type='invalid_source')


# =====================================================================
# 8. Wearable and Sensor Conditions Parameter Bounds
# =====================================================================

def test_sensor_detachment_config():
    """Verify SensorDetachmentConfig boundaries."""
    cfg = SensorDetachmentConfig()
    assert cfg.detachment_time_s == 5.0
    
    with pytest.raises(ValueError):
        SensorDetachmentConfig(detachment_time_s=-0.1)

def test_electrode_displacement_config():
    """Verify ElectrodeDisplacementConfig boundaries."""
    cfg = ElectrodeDisplacementConfig()
    assert cfg.displacement_times == [3.0, 7.0]
    
    with pytest.raises(ValueError):
        ElectrodeDisplacementConfig(displacement_times=[1.0], shift_amplitudes=[])

def test_light_leakage_config():
    """Verify LightLeakageConfig boundaries."""
    cfg = LightLeakageConfig()
    assert cfg.leakage_amplitude == 0.2
    
    with pytest.raises(ValueError):
        LightLeakageConfig(leakage_amplitude=-0.01)

def test_packet_loss_config():
    """Verify PacketLossConfig boundaries."""
    cfg = PacketLossConfig()
    assert cfg.loss_rate == 0.05
    
    # Loss rate is a fraction [0.0, 1.0]
    cfg_val = PacketLossConfig(loss_rate=1.0)
    assert cfg_val.loss_rate == 1.0
    
    with pytest.raises(ValueError):
        PacketLossConfig(loss_rate=-0.01)
    with pytest.raises(ValueError):
        PacketLossConfig(loss_rate=1.01)
