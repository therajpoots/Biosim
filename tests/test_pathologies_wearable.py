"""
Automated unit tests for advanced pathologies, arrhythmias, crosstalk, and wearable sensor noise.
"""

import numpy as np
import pytest
from biosignal_simulator.core.config import (
    ECGConfig, EEGConfig, EMGConfig, CrosstalkNoiseConfig,
    SensorDetachmentConfig, ElectrodeDisplacementConfig, LightLeakageConfig, PacketLossConfig
)
from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.signals.eeg import EEGGenerator
from biosignal_simulator.signals.emg import EMGGenerator
from biosignal_simulator.noise.crosstalk import CrosstalkNoise
from biosignal_simulator.noise.wearable import (
    SensorDetachmentNoise, ElectrodeDisplacementNoise, LightLeakageNoise, PacketLossNoise
)
from biosignal_simulator.composer.mixer import SignalMixer

def test_ecg_arrhythmias():
    fs = 200.0
    duration = 5.0
    
    # Test AFib (no P-wave, f-waves added)
    cfg_af = ECGConfig(fs=fs, duration_s=duration, rhythm_type='afib', lead_type='single')
    gen_af = ECGGenerator(cfg_af)
    ecg_af = gen_af.generate()
    assert len(ecg_af) == int(fs * duration)
    
    # Test PVC (ectopic beats)
    cfg_pvc = ECGConfig(fs=fs, duration_s=duration, rhythm_type='pvc', lead_type='single')
    gen_pvc = ECGGenerator(cfg_pvc)
    ecg_pvc = gen_pvc.generate()
    assert len(ecg_pvc) == int(fs * duration)

    # Test VTach (rapid regular)
    cfg_vt = ECGConfig(fs=fs, duration_s=duration, rhythm_type='vtach', lead_type='single')
    gen_vt = ECGGenerator(cfg_vt)
    ecg_vt = gen_vt.generate()
    assert len(ecg_vt) == int(fs * duration)
    
    # Test AV Block (dropped beats)
    cfg_av = ECGConfig(fs=fs, duration_s=duration, rhythm_type='av_block', lead_type='single')
    gen_av = ECGGenerator(cfg_av)
    ecg_av = gen_av.generate()
    assert len(ecg_av) == int(fs * duration)


def test_eeg_seizures():
    fs = 128.0
    duration = 8.0
    
    # Test Tonic-Clonic Seizure
    cfg_tc = EEGConfig(fs=fs, duration_s=duration, state='tonic_clonic', n_channels=2)
    gen_tc = EEGGenerator(cfg_tc)
    eeg_tc = gen_tc.generate()
    assert eeg_tc.shape == (2, int(fs * duration))
    
    # Test Absence Seizure (rhythmic 3Hz)
    cfg_ab = EEGConfig(fs=fs, duration_s=duration, state='absence', n_channels=1)
    gen_ab = EEGGenerator(cfg_ab)
    eeg_ab = gen_ab.generate()
    assert len(eeg_ab) == int(fs * duration)
    
    # Test Epileptiform Spikes
    cfg_sp = EEGConfig(fs=fs, duration_s=duration, state='epileptiform_spikes', n_channels=1)
    gen_sp = EEGGenerator(cfg_sp)
    eeg_sp = gen_sp.generate()
    assert len(eeg_sp) == int(fs * duration)


def test_emg_disorders():
    fs = 1000.0
    duration = 4.0
    
    # Test ALS in intramuscular EMG
    cfg_als = EMGConfig(fs=fs, duration_s=duration, emg_type='intramuscular', pathology='als')
    gen_als = EMGGenerator(cfg_als)
    emg_als = gen_als.generate()
    assert len(emg_als) == int(fs * duration)
    
    # Test Myasthenia Gravis in surface EMG
    cfg_mg = EMGConfig(fs=fs, duration_s=duration, emg_type='surface', pathology='myasthenia_gravis')
    gen_mg = EMGGenerator(cfg_mg)
    emg_mg = gen_mg.generate()
    assert len(emg_mg) == int(fs * duration)
    
    # Test Parkinson's Tremor
    cfg_pk = EMGConfig(fs=fs, duration_s=duration, emg_type='surface', pathology='parkinsons_tremor')
    gen_pk = EMGGenerator(cfg_pk)
    emg_pk = gen_pk.generate()
    assert len(emg_pk) == int(fs * duration)


def test_crosstalk_noise():
    fs = 250.0
    duration = 5.0
    gen = EEGGenerator(EEGConfig(fs=fs, duration_s=duration, state='relaxed'))
    
    # ECG Crosstalk Leakage on EEG
    crosstalk_cfg = CrosstalkNoiseConfig(coupling_factor=0.15, source_type='ecg')
    crosstalk_noise = CrosstalkNoise(crosstalk_cfg)
    
    mixer = SignalMixer(signal_generator=gen, noise_models=[crosstalk_noise])
    record = mixer.mix()
    
    assert 'CrosstalkNoise' in record.noise_components
    assert record.noise_components['CrosstalkNoise'].shape == record.clean.shape
    # Check scaling was applied
    p_clean_leak = np.mean(record.noise_components['CrosstalkNoise'] ** 2)
    assert p_clean_leak > 0.0


def test_wearable_conditions():
    fs = 200.0
    duration = 10.0
    gen = ECGGenerator(ECGConfig(fs=fs, duration_s=duration))
    
    # 1. Sensor Detachment
    det_cfg = SensorDetachmentConfig(detachment_time_s=4.0, transient_amplitude=2.0, noise_level_uv=15.0)
    det_noise = SensorDetachmentNoise(det_cfg)
    
    mixer_det = SignalMixer(signal_generator=gen, noise_models=[det_noise])
    rec_det = mixer_det.mix()
    
    assert 'SensorDetachmentNoise' in rec_det.noise_components
    # After t=4s, the signal should be suppressed and overridden by flatline white noise
    # We check that the variance of the noisy signal after t=5s is small (around noise level)
    # compared to the clean signal amplitude before t=4s
    t_arr = rec_det.t
    post_det = t_arr >= 5.0
    pre_det = t_arr < 4.0
    
    assert np.std(rec_det.noisy[post_det]) < 0.2  # Mostly flatline noise
    assert np.std(rec_det.noisy[pre_det]) > 0.05  # ECG signal present
    
    # 2. Electrode Displacement
    disp_cfg = ElectrodeDisplacementConfig(
        displacement_times=[3.0, 7.0],
        shift_amplitudes=[1.0, -1.0],
        noise_increments=[2.0, 3.0]
    )
    disp_noise = ElectrodeDisplacementNoise(disp_cfg)
    ecg_disp = disp_noise.generate(len(t_arr), fs)
    assert len(ecg_disp) == len(t_arr)
    # Shifts check
    assert abs(np.mean(ecg_disp[t_arr < 3.0])) < 0.1
    assert abs(np.mean(ecg_disp[(t_arr > 3.1) & (t_arr < 6.9)])) > 0.3
    
    # 3. Light Leakage
    light_cfg = LightLeakageConfig(leakage_amplitude=0.3, modulation_frequency_hz=0.2, f_line_hz=50.0)
    light_noise = LightLeakageNoise(light_cfg)
    leak = light_noise.generate(len(t_arr), fs)
    assert len(leak) == len(t_arr)
    
    # 4. Packet Loss
    loss_cfg = PacketLossConfig(loss_rate=0.1, burst_length_samples=4, interpolation_mode='zero')
    loss_noise = PacketLossNoise(loss_cfg)
    
    mixer_loss = SignalMixer(signal_generator=gen, noise_models=[loss_noise])
    rec_loss = mixer_loss.mix()
    
    assert 'PacketLossNoise' in rec_loss.noise_components
    # Check that some values in noisy were set to zero (zero interpolation)
    # The clean signal has virtually no exact zero values
    assert np.any(rec_loss.noisy == 0.0)
