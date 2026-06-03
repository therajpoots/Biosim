"""
Exhaustive mixing matrices sweeps and target SNR mappings.
"""
import numpy as np
import pytest
from biosignal_simulator.core.config import ECGConfig, EEGConfig, EMGConfig, PPGConfig, EDAConfig, RespConfig
from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.signals.eeg import EEGGenerator
from biosignal_simulator.signals.emg import EMGGenerator
from biosignal_simulator.signals.ppg import PPGGenerator
from biosignal_simulator.signals.eda import EDAGenerator
from biosignal_simulator.signals.resp import RespGenerator
from biosignal_simulator.noise.gaussian import GaussianNoise
from biosignal_simulator.noise.colored import PinkNoise, BrownNoise
from biosignal_simulator.noise.powerline import PowerlineNoise
from biosignal_simulator.noise.baseline import BaselineWander
from biosignal_simulator.noise.motion import MotionArtifact
from biosignal_simulator.composer import SignalMixer
from biosignal_simulator.metrics.snr import compute_snr_wideband


def test_mix_signal_ecg_noise_gaussian_snr_minus_20_0():
    """Verify mixing clean ecg with gaussian noise at target SNR -20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_gaussian_snr_minus_10_0():
    """Verify mixing clean ecg with gaussian noise at target SNR -10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_gaussian_snr_0_0():
    """Verify mixing clean ecg with gaussian noise at target SNR 0.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_gaussian_snr_10_0():
    """Verify mixing clean ecg with gaussian noise at target SNR 10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_gaussian_snr_20_0():
    """Verify mixing clean ecg with gaussian noise at target SNR 20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_gaussian_snr_30_0():
    """Verify mixing clean ecg with gaussian noise at target SNR 30.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_gaussian_snr_40_0():
    """Verify mixing clean ecg with gaussian noise at target SNR 40.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_colored_snr_minus_20_0():
    """Verify mixing clean ecg with colored noise at target SNR -20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_colored_snr_minus_10_0():
    """Verify mixing clean ecg with colored noise at target SNR -10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_colored_snr_0_0():
    """Verify mixing clean ecg with colored noise at target SNR 0.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_colored_snr_10_0():
    """Verify mixing clean ecg with colored noise at target SNR 10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_colored_snr_20_0():
    """Verify mixing clean ecg with colored noise at target SNR 20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_colored_snr_30_0():
    """Verify mixing clean ecg with colored noise at target SNR 30.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_colored_snr_40_0():
    """Verify mixing clean ecg with colored noise at target SNR 40.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_powerline_snr_minus_20_0():
    """Verify mixing clean ecg with powerline noise at target SNR -20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_powerline_snr_minus_10_0():
    """Verify mixing clean ecg with powerline noise at target SNR -10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_powerline_snr_0_0():
    """Verify mixing clean ecg with powerline noise at target SNR 0.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_powerline_snr_10_0():
    """Verify mixing clean ecg with powerline noise at target SNR 10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_powerline_snr_20_0():
    """Verify mixing clean ecg with powerline noise at target SNR 20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_powerline_snr_30_0():
    """Verify mixing clean ecg with powerline noise at target SNR 30.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_powerline_snr_40_0():
    """Verify mixing clean ecg with powerline noise at target SNR 40.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_baseline_snr_minus_20_0():
    """Verify mixing clean ecg with baseline noise at target SNR -20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_baseline_snr_minus_10_0():
    """Verify mixing clean ecg with baseline noise at target SNR -10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_baseline_snr_0_0():
    """Verify mixing clean ecg with baseline noise at target SNR 0.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_baseline_snr_10_0():
    """Verify mixing clean ecg with baseline noise at target SNR 10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_baseline_snr_20_0():
    """Verify mixing clean ecg with baseline noise at target SNR 20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_baseline_snr_30_0():
    """Verify mixing clean ecg with baseline noise at target SNR 30.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_baseline_snr_40_0():
    """Verify mixing clean ecg with baseline noise at target SNR 40.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_motion_snr_minus_20_0():
    """Verify mixing clean ecg with motion noise at target SNR -20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_motion_snr_minus_10_0():
    """Verify mixing clean ecg with motion noise at target SNR -10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_motion_snr_0_0():
    """Verify mixing clean ecg with motion noise at target SNR 0.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_motion_snr_10_0():
    """Verify mixing clean ecg with motion noise at target SNR 10.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_motion_snr_20_0():
    """Verify mixing clean ecg with motion noise at target SNR 20.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_motion_snr_30_0():
    """Verify mixing clean ecg with motion noise at target SNR 30.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ecg_noise_motion_snr_40_0():
    """Verify mixing clean ecg with motion noise at target SNR 40.0 dB."""
    gen = ECGGenerator(ECGConfig(fs=200.0, duration_s=2.0, heart_rate=80.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_gaussian_snr_minus_20_0():
    """Verify mixing clean eeg with gaussian noise at target SNR -20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_gaussian_snr_minus_10_0():
    """Verify mixing clean eeg with gaussian noise at target SNR -10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_gaussian_snr_0_0():
    """Verify mixing clean eeg with gaussian noise at target SNR 0.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_gaussian_snr_10_0():
    """Verify mixing clean eeg with gaussian noise at target SNR 10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_gaussian_snr_20_0():
    """Verify mixing clean eeg with gaussian noise at target SNR 20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_gaussian_snr_30_0():
    """Verify mixing clean eeg with gaussian noise at target SNR 30.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_gaussian_snr_40_0():
    """Verify mixing clean eeg with gaussian noise at target SNR 40.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_colored_snr_minus_20_0():
    """Verify mixing clean eeg with colored noise at target SNR -20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_colored_snr_minus_10_0():
    """Verify mixing clean eeg with colored noise at target SNR -10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_colored_snr_0_0():
    """Verify mixing clean eeg with colored noise at target SNR 0.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_colored_snr_10_0():
    """Verify mixing clean eeg with colored noise at target SNR 10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_colored_snr_20_0():
    """Verify mixing clean eeg with colored noise at target SNR 20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_colored_snr_30_0():
    """Verify mixing clean eeg with colored noise at target SNR 30.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_colored_snr_40_0():
    """Verify mixing clean eeg with colored noise at target SNR 40.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_powerline_snr_minus_20_0():
    """Verify mixing clean eeg with powerline noise at target SNR -20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_powerline_snr_minus_10_0():
    """Verify mixing clean eeg with powerline noise at target SNR -10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_powerline_snr_0_0():
    """Verify mixing clean eeg with powerline noise at target SNR 0.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_powerline_snr_10_0():
    """Verify mixing clean eeg with powerline noise at target SNR 10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_powerline_snr_20_0():
    """Verify mixing clean eeg with powerline noise at target SNR 20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_powerline_snr_30_0():
    """Verify mixing clean eeg with powerline noise at target SNR 30.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_powerline_snr_40_0():
    """Verify mixing clean eeg with powerline noise at target SNR 40.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_baseline_snr_minus_20_0():
    """Verify mixing clean eeg with baseline noise at target SNR -20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_baseline_snr_minus_10_0():
    """Verify mixing clean eeg with baseline noise at target SNR -10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_baseline_snr_0_0():
    """Verify mixing clean eeg with baseline noise at target SNR 0.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_baseline_snr_10_0():
    """Verify mixing clean eeg with baseline noise at target SNR 10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_baseline_snr_20_0():
    """Verify mixing clean eeg with baseline noise at target SNR 20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_baseline_snr_30_0():
    """Verify mixing clean eeg with baseline noise at target SNR 30.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_baseline_snr_40_0():
    """Verify mixing clean eeg with baseline noise at target SNR 40.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_motion_snr_minus_20_0():
    """Verify mixing clean eeg with motion noise at target SNR -20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_motion_snr_minus_10_0():
    """Verify mixing clean eeg with motion noise at target SNR -10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_motion_snr_0_0():
    """Verify mixing clean eeg with motion noise at target SNR 0.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_motion_snr_10_0():
    """Verify mixing clean eeg with motion noise at target SNR 10.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_motion_snr_20_0():
    """Verify mixing clean eeg with motion noise at target SNR 20.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_motion_snr_30_0():
    """Verify mixing clean eeg with motion noise at target SNR 30.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eeg_noise_motion_snr_40_0():
    """Verify mixing clean eeg with motion noise at target SNR 40.0 dB."""
    gen = EEGGenerator(EEGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_gaussian_snr_minus_20_0():
    """Verify mixing clean emg with gaussian noise at target SNR -20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_gaussian_snr_minus_10_0():
    """Verify mixing clean emg with gaussian noise at target SNR -10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_gaussian_snr_0_0():
    """Verify mixing clean emg with gaussian noise at target SNR 0.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_gaussian_snr_10_0():
    """Verify mixing clean emg with gaussian noise at target SNR 10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_gaussian_snr_20_0():
    """Verify mixing clean emg with gaussian noise at target SNR 20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_gaussian_snr_30_0():
    """Verify mixing clean emg with gaussian noise at target SNR 30.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_gaussian_snr_40_0():
    """Verify mixing clean emg with gaussian noise at target SNR 40.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_colored_snr_minus_20_0():
    """Verify mixing clean emg with colored noise at target SNR -20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_colored_snr_minus_10_0():
    """Verify mixing clean emg with colored noise at target SNR -10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_colored_snr_0_0():
    """Verify mixing clean emg with colored noise at target SNR 0.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_colored_snr_10_0():
    """Verify mixing clean emg with colored noise at target SNR 10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_colored_snr_20_0():
    """Verify mixing clean emg with colored noise at target SNR 20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_colored_snr_30_0():
    """Verify mixing clean emg with colored noise at target SNR 30.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_colored_snr_40_0():
    """Verify mixing clean emg with colored noise at target SNR 40.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_powerline_snr_minus_20_0():
    """Verify mixing clean emg with powerline noise at target SNR -20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_powerline_snr_minus_10_0():
    """Verify mixing clean emg with powerline noise at target SNR -10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_powerline_snr_0_0():
    """Verify mixing clean emg with powerline noise at target SNR 0.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_powerline_snr_10_0():
    """Verify mixing clean emg with powerline noise at target SNR 10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_powerline_snr_20_0():
    """Verify mixing clean emg with powerline noise at target SNR 20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_powerline_snr_30_0():
    """Verify mixing clean emg with powerline noise at target SNR 30.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_powerline_snr_40_0():
    """Verify mixing clean emg with powerline noise at target SNR 40.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_baseline_snr_minus_20_0():
    """Verify mixing clean emg with baseline noise at target SNR -20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_baseline_snr_minus_10_0():
    """Verify mixing clean emg with baseline noise at target SNR -10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_baseline_snr_0_0():
    """Verify mixing clean emg with baseline noise at target SNR 0.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_baseline_snr_10_0():
    """Verify mixing clean emg with baseline noise at target SNR 10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_baseline_snr_20_0():
    """Verify mixing clean emg with baseline noise at target SNR 20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_baseline_snr_30_0():
    """Verify mixing clean emg with baseline noise at target SNR 30.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_baseline_snr_40_0():
    """Verify mixing clean emg with baseline noise at target SNR 40.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_motion_snr_minus_20_0():
    """Verify mixing clean emg with motion noise at target SNR -20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_motion_snr_minus_10_0():
    """Verify mixing clean emg with motion noise at target SNR -10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_motion_snr_0_0():
    """Verify mixing clean emg with motion noise at target SNR 0.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_motion_snr_10_0():
    """Verify mixing clean emg with motion noise at target SNR 10.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_motion_snr_20_0():
    """Verify mixing clean emg with motion noise at target SNR 20.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_motion_snr_30_0():
    """Verify mixing clean emg with motion noise at target SNR 30.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_emg_noise_motion_snr_40_0():
    """Verify mixing clean emg with motion noise at target SNR 40.0 dB."""
    gen = EMGGenerator(EMGConfig(fs=500.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_gaussian_snr_minus_20_0():
    """Verify mixing clean ppg with gaussian noise at target SNR -20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_gaussian_snr_minus_10_0():
    """Verify mixing clean ppg with gaussian noise at target SNR -10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_gaussian_snr_0_0():
    """Verify mixing clean ppg with gaussian noise at target SNR 0.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_gaussian_snr_10_0():
    """Verify mixing clean ppg with gaussian noise at target SNR 10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_gaussian_snr_20_0():
    """Verify mixing clean ppg with gaussian noise at target SNR 20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_gaussian_snr_30_0():
    """Verify mixing clean ppg with gaussian noise at target SNR 30.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_gaussian_snr_40_0():
    """Verify mixing clean ppg with gaussian noise at target SNR 40.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_colored_snr_minus_20_0():
    """Verify mixing clean ppg with colored noise at target SNR -20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_colored_snr_minus_10_0():
    """Verify mixing clean ppg with colored noise at target SNR -10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_colored_snr_0_0():
    """Verify mixing clean ppg with colored noise at target SNR 0.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_colored_snr_10_0():
    """Verify mixing clean ppg with colored noise at target SNR 10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_colored_snr_20_0():
    """Verify mixing clean ppg with colored noise at target SNR 20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_colored_snr_30_0():
    """Verify mixing clean ppg with colored noise at target SNR 30.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_colored_snr_40_0():
    """Verify mixing clean ppg with colored noise at target SNR 40.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_powerline_snr_minus_20_0():
    """Verify mixing clean ppg with powerline noise at target SNR -20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_powerline_snr_minus_10_0():
    """Verify mixing clean ppg with powerline noise at target SNR -10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_powerline_snr_0_0():
    """Verify mixing clean ppg with powerline noise at target SNR 0.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_powerline_snr_10_0():
    """Verify mixing clean ppg with powerline noise at target SNR 10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_powerline_snr_20_0():
    """Verify mixing clean ppg with powerline noise at target SNR 20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_powerline_snr_30_0():
    """Verify mixing clean ppg with powerline noise at target SNR 30.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_powerline_snr_40_0():
    """Verify mixing clean ppg with powerline noise at target SNR 40.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_baseline_snr_minus_20_0():
    """Verify mixing clean ppg with baseline noise at target SNR -20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_baseline_snr_minus_10_0():
    """Verify mixing clean ppg with baseline noise at target SNR -10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_baseline_snr_0_0():
    """Verify mixing clean ppg with baseline noise at target SNR 0.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_baseline_snr_10_0():
    """Verify mixing clean ppg with baseline noise at target SNR 10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_baseline_snr_20_0():
    """Verify mixing clean ppg with baseline noise at target SNR 20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_baseline_snr_30_0():
    """Verify mixing clean ppg with baseline noise at target SNR 30.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_baseline_snr_40_0():
    """Verify mixing clean ppg with baseline noise at target SNR 40.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_motion_snr_minus_20_0():
    """Verify mixing clean ppg with motion noise at target SNR -20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_motion_snr_minus_10_0():
    """Verify mixing clean ppg with motion noise at target SNR -10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_motion_snr_0_0():
    """Verify mixing clean ppg with motion noise at target SNR 0.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_motion_snr_10_0():
    """Verify mixing clean ppg with motion noise at target SNR 10.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_motion_snr_20_0():
    """Verify mixing clean ppg with motion noise at target SNR 20.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_motion_snr_30_0():
    """Verify mixing clean ppg with motion noise at target SNR 30.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_ppg_noise_motion_snr_40_0():
    """Verify mixing clean ppg with motion noise at target SNR 40.0 dB."""
    gen = PPGGenerator(PPGConfig(fs=200.0, duration_s=2.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_gaussian_snr_minus_20_0():
    """Verify mixing clean eda with gaussian noise at target SNR -20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_gaussian_snr_minus_10_0():
    """Verify mixing clean eda with gaussian noise at target SNR -10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_gaussian_snr_0_0():
    """Verify mixing clean eda with gaussian noise at target SNR 0.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_gaussian_snr_10_0():
    """Verify mixing clean eda with gaussian noise at target SNR 10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_gaussian_snr_20_0():
    """Verify mixing clean eda with gaussian noise at target SNR 20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_gaussian_snr_30_0():
    """Verify mixing clean eda with gaussian noise at target SNR 30.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_gaussian_snr_40_0():
    """Verify mixing clean eda with gaussian noise at target SNR 40.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_colored_snr_minus_20_0():
    """Verify mixing clean eda with colored noise at target SNR -20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_colored_snr_minus_10_0():
    """Verify mixing clean eda with colored noise at target SNR -10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_colored_snr_0_0():
    """Verify mixing clean eda with colored noise at target SNR 0.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_colored_snr_10_0():
    """Verify mixing clean eda with colored noise at target SNR 10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_colored_snr_20_0():
    """Verify mixing clean eda with colored noise at target SNR 20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_colored_snr_30_0():
    """Verify mixing clean eda with colored noise at target SNR 30.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_colored_snr_40_0():
    """Verify mixing clean eda with colored noise at target SNR 40.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_powerline_snr_minus_20_0():
    """Verify mixing clean eda with powerline noise at target SNR -20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_powerline_snr_minus_10_0():
    """Verify mixing clean eda with powerline noise at target SNR -10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_powerline_snr_0_0():
    """Verify mixing clean eda with powerline noise at target SNR 0.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_powerline_snr_10_0():
    """Verify mixing clean eda with powerline noise at target SNR 10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_powerline_snr_20_0():
    """Verify mixing clean eda with powerline noise at target SNR 20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_powerline_snr_30_0():
    """Verify mixing clean eda with powerline noise at target SNR 30.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_powerline_snr_40_0():
    """Verify mixing clean eda with powerline noise at target SNR 40.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_baseline_snr_minus_20_0():
    """Verify mixing clean eda with baseline noise at target SNR -20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_baseline_snr_minus_10_0():
    """Verify mixing clean eda with baseline noise at target SNR -10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_baseline_snr_0_0():
    """Verify mixing clean eda with baseline noise at target SNR 0.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_baseline_snr_10_0():
    """Verify mixing clean eda with baseline noise at target SNR 10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_baseline_snr_20_0():
    """Verify mixing clean eda with baseline noise at target SNR 20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_baseline_snr_30_0():
    """Verify mixing clean eda with baseline noise at target SNR 30.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_baseline_snr_40_0():
    """Verify mixing clean eda with baseline noise at target SNR 40.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_motion_snr_minus_20_0():
    """Verify mixing clean eda with motion noise at target SNR -20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_motion_snr_minus_10_0():
    """Verify mixing clean eda with motion noise at target SNR -10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_motion_snr_0_0():
    """Verify mixing clean eda with motion noise at target SNR 0.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_motion_snr_10_0():
    """Verify mixing clean eda with motion noise at target SNR 10.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_motion_snr_20_0():
    """Verify mixing clean eda with motion noise at target SNR 20.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_motion_snr_30_0():
    """Verify mixing clean eda with motion noise at target SNR 30.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_eda_noise_motion_snr_40_0():
    """Verify mixing clean eda with motion noise at target SNR 40.0 dB."""
    gen = EDAGenerator(EDAConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_gaussian_snr_minus_20_0():
    """Verify mixing clean resp with gaussian noise at target SNR -20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_gaussian_snr_minus_10_0():
    """Verify mixing clean resp with gaussian noise at target SNR -10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_gaussian_snr_0_0():
    """Verify mixing clean resp with gaussian noise at target SNR 0.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_gaussian_snr_10_0():
    """Verify mixing clean resp with gaussian noise at target SNR 10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_gaussian_snr_20_0():
    """Verify mixing clean resp with gaussian noise at target SNR 20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_gaussian_snr_30_0():
    """Verify mixing clean resp with gaussian noise at target SNR 30.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_gaussian_snr_40_0():
    """Verify mixing clean resp with gaussian noise at target SNR 40.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_colored_snr_minus_20_0():
    """Verify mixing clean resp with colored noise at target SNR -20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_colored_snr_minus_10_0():
    """Verify mixing clean resp with colored noise at target SNR -10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_colored_snr_0_0():
    """Verify mixing clean resp with colored noise at target SNR 0.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_colored_snr_10_0():
    """Verify mixing clean resp with colored noise at target SNR 10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_colored_snr_20_0():
    """Verify mixing clean resp with colored noise at target SNR 20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_colored_snr_30_0():
    """Verify mixing clean resp with colored noise at target SNR 30.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_colored_snr_40_0():
    """Verify mixing clean resp with colored noise at target SNR 40.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PinkNoise(std=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_powerline_snr_minus_20_0():
    """Verify mixing clean resp with powerline noise at target SNR -20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_powerline_snr_minus_10_0():
    """Verify mixing clean resp with powerline noise at target SNR -10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_powerline_snr_0_0():
    """Verify mixing clean resp with powerline noise at target SNR 0.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_powerline_snr_10_0():
    """Verify mixing clean resp with powerline noise at target SNR 10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_powerline_snr_20_0():
    """Verify mixing clean resp with powerline noise at target SNR 20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_powerline_snr_30_0():
    """Verify mixing clean resp with powerline noise at target SNR 30.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_powerline_snr_40_0():
    """Verify mixing clean resp with powerline noise at target SNR 40.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = PowerlineNoise(amplitude=0.1, f_line_hz=50.0, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_baseline_snr_minus_20_0():
    """Verify mixing clean resp with baseline noise at target SNR -20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_baseline_snr_minus_10_0():
    """Verify mixing clean resp with baseline noise at target SNR -10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_baseline_snr_0_0():
    """Verify mixing clean resp with baseline noise at target SNR 0.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_baseline_snr_10_0():
    """Verify mixing clean resp with baseline noise at target SNR 10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_baseline_snr_20_0():
    """Verify mixing clean resp with baseline noise at target SNR 20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_baseline_snr_30_0():
    """Verify mixing clean resp with baseline noise at target SNR 30.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_baseline_snr_40_0():
    """Verify mixing clean resp with baseline noise at target SNR 40.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = BaselineWander(amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_motion_snr_minus_20_0():
    """Verify mixing clean resp with motion noise at target SNR -20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_motion_snr_minus_10_0():
    """Verify mixing clean resp with motion noise at target SNR -10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=-10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, -10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_motion_snr_0_0():
    """Verify mixing clean resp with motion noise at target SNR 0.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=0.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 0.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_motion_snr_10_0():
    """Verify mixing clean resp with motion noise at target SNR 10.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=10.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 10.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_motion_snr_20_0():
    """Verify mixing clean resp with motion noise at target SNR 20.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=20.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 20.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_motion_snr_30_0():
    """Verify mixing clean resp with motion noise at target SNR 30.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=30.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 30.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))


def test_mix_signal_resp_noise_motion_snr_40_0():
    """Verify mixing clean resp with motion noise at target SNR 40.0 dB."""
    gen = RespGenerator(RespConfig(fs=200.0, duration_s=5.0, seed=42))
    noise_model = MotionArtifact(lf_amplitude=0.1, seed=42)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=40.0
    )
    rec = mixer.mix()
    
    # Check SNR
    actual_snr = compute_snr_wideband(rec.clean, rec.noisy, gen.fs)
    assert np.isclose(actual_snr, 40.0, atol=0.25)
    assert not np.any(np.isnan(rec.noisy))
