import numpy as np
import pytest
from biosignal_simulator.composer import (
    SignalMixer,
    SNRController,
    CompositeSNRController,
    DynamicSNRController,
    NoiseScheduler,
    StepSchedule,
    RampSchedule,
    PeriodicSchedule,
    SigmoidSchedule,
    StochasticSchedule,
    CompositeSchedule,
    ArtifactInjector
)
from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.signals.resp import RespGenerator
from biosignal_simulator.core.config import ECGConfig, RespConfig, GaussianNoiseConfig
from biosignal_simulator.noise.gaussian import GaussianNoise
from biosignal_simulator.noise.quantization import QuantizationNoise

def test_signal_mixer_and_snr():
    fs = 250.0
    duration = 5.0
    gen = ECGGenerator(ECGConfig(fs=fs, duration_s=duration, heart_rate=75, seed=42))
    
    # Mixer with Gaussian noise at target SNR = 15 dB
    noise_model = GaussianNoise(std=1.0, seed=42)
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise_model],
        target_snr_db=15.0
    )
    record = mixer.mix()
    
    # Check dimensions
    assert len(record.clean) == int(fs * duration)
    assert len(record.noisy) == len(record.clean)
    assert 'GaussianNoise' in record.noise_components
    
    # Check that achieved SNR is close to 15.0 dB
    assert record.snr_db is not None
    assert np.isclose(record.snr_db, 15.0, atol=0.1)
    
    # Check sum of components
    total_noise_comp = record.noise_components['GaussianNoise']
    assert np.allclose(record.noisy - record.clean, total_noise_comp)

def test_mixer_with_quantization():
    fs = 200.0
    gen = ECGGenerator(ECGConfig(fs=fs, duration_s=4, heart_rate=80, seed=42))
    noise = GaussianNoise(std=0.01, seed=42)
    quant = QuantizationNoise(n_bits=12, v_range=5.0)
    
    mixer = SignalMixer(
        signal_generator=gen,
        noise_models=[noise, quant]
    )
    record = mixer.mix()
    
    assert 'QuantizationNoise' in record.noise_components
    # The noisy signal is clean + gaussian + quantization
    total_expected = record.clean + record.noise_components['GaussianNoise'] + record.noise_components['QuantizationNoise']
    assert np.allclose(record.noisy, total_expected)

def test_snr_controllers():
    fs = 200.0
    t = np.arange(1000) / fs
    clean = np.sin(2 * np.pi * 5 * t)
    
    noise_model = GaussianNoise(std=1.0, seed=42)
    ctrl = SNRController(noise_model, target_snr_db=20.0)
    noise_scaled = ctrl.apply(clean, fs)
    
    p_sig = np.mean(clean ** 2)
    p_noise = np.mean(noise_scaled ** 2)
    snr_db = 10 * np.log10(p_sig / p_noise)
    assert np.isclose(snr_db, 20.0, atol=0.1)

def test_noise_scheduler():
    fs = 100.0
    n_samples = 500
    noise_model = GaussianNoise(std=1.0, seed=42)
    
    # Step Schedule: 0-2s is level 0.1, 2-5s is level 2.0
    sched = StepSchedule(breakpoints=[0.0, 2.0], levels=[0.1, 2.0])
    scheduler = NoiseScheduler(noise_model, sched)
    
    noise = scheduler.generate(n_samples, fs)
    
    # First 2 seconds (200 samples) should have small variance
    assert np.std(noise[:200]) < 0.2
    # Next 3 seconds should have large variance
    assert np.std(noise[200:]) > 1.0

def test_artifact_injector():
    fs = 100.0
    clean = np.zeros(500)
    
    noise_model = GaussianNoise(std=1.5, seed=42)
    injector = ArtifactInjector()
    
    # Inject at t=1.0s and t=3.0s, duration 0.5s
    injector.add(noise_model, timestamps_s=[1.0, 3.0], duration_s=0.5, fade_in_s=0.01, fade_out_s=0.01)
    noisy = injector.apply(clean, fs)
    
    # Outside injection windows: should be zero
    # t=0 to 1.0s (indices 0 to 100) -> zero
    assert np.all(noisy[:100] == 0.0)
    # t=1.0s to 1.5s (indices 100 to 150) -> non-zero
    assert not np.all(noisy[100:150] == 0.0)
    # t=1.5s to 3.0s (indices 150 to 300) -> zero
    assert np.all(noisy[150:300] == 0.0)

# =====================================================================
# Advanced Composer Feature Tests
# =====================================================================

def test_sigmoid_and_stochastic_schedules():
    fs = 100.0
    t = np.arange(500) / fs
    
    # 1. Sigmoid transition: smooth rise from 0.2 to 2.5
    sig_sched = SigmoidSchedule(start_level=0.2, end_level=2.5, midpoint_s=2.5, slope=2.0)
    env_sig = sig_sched.get_envelope(t)
    assert env_sig[0] == pytest.approx(0.2, abs=0.02)
    assert env_sig[-1] == pytest.approx(2.5, abs=0.02)
    assert env_sig[250] == pytest.approx(1.35, abs=0.05)  # halfway midpoint
    
    # 2. Stochastic schedule
    stoch_sched = StochasticSchedule(initial_level=1.0, step_std=0.05, seed=42)
    env_stoch = stoch_sched.get_envelope(t)
    assert len(env_stoch) == len(t)
    assert env_stoch[0] == 1.0
    assert np.all(env_stoch >= 0.0)  # absolute value clipping holds positive
    
    # 3. Composite Schedule: multiply a periodic swing by a linear ramp
    ramp = RampSchedule(control_times=[0.0, 5.0], levels=[1.0, 3.0])
    periodic = PeriodicSchedule(base_level=1.0, modulation_amplitude=0.5, frequency_hz=1.0)
    composite = ramp * periodic
    env_comp = composite.get_envelope(t)
    assert len(env_comp) == len(t)


def test_dynamic_snr_controller():
    fs = 200.0
    t = np.arange(1000) / fs  # 5 seconds
    clean = np.sin(2.0 * np.pi * 8.0 * t)
    
    noise_model = GaussianNoise(std=1.0, seed=42)
    # Dynamic SNR schedule transitioning smoothly from 30dB (very clean) to 5dB (very noisy)
    snr_sched = SigmoidSchedule(start_level=30.0, end_level=5.0, midpoint_s=2.5, slope=2.0)
    
    dyn_ctrl = DynamicSNRController(noise_model, snr_sched, window_duration_s=0.5)
    noise_dyn = dyn_ctrl.apply(clean, fs)
    
    assert len(noise_dyn) == len(clean)
    # Local noise envelope should be much smaller at the beginning than at the end
    assert np.std(noise_dyn[:200]) < np.std(noise_dyn[-200:]) * 0.1


def test_composite_clean_mixer_and_diagnostics():
    fs = 250.0
    duration = 6.0
    
    # Primary signal ECG
    primary_gen = ECGGenerator(ECGConfig(fs=fs, duration_s=duration, seed=42))
    
    # Secondary composite signal: respiration chest movement leaking in
    resp_gen = RespGenerator(RespConfig(fs=fs, duration_s=duration, resp_rate_hz=0.2, seed=42))
    
    noise_model = GaussianNoise(std=0.1, seed=42)
    
    # Mix ECG with 0.3 * Respiration
    mixer = SignalMixer(
        signal_generator=primary_gen,
        noise_models=[noise_model],
        composite_signals=[(resp_gen, 0.3)],
        target_snr_db=20.0
    )
    record = mixer.mix()
    
    assert record.clean.shape == record.noisy.shape
    # Diagnostics check
    assert 'diagnostics' in record.metadata
    diag = record.metadata['diagnostics']
    assert 'skewness' in diag
    assert 'kurtosis' in diag
    assert 'zero_crossing_rate_hz' in diag
    assert 'clipping_ratio' in diag


def test_artifact_injector_overlap_and_leakage():
    fs = 100.0
    clean = np.zeros((2, 500))  # 2-channel clean baseline
    
    noise_model = GaussianNoise(std=1.0, seed=42)
    
    # 2x2 spatial coupling matrix (channel 0 leaks 40% to channel 1, channel 1 leaks 20% to channel 0)
    leakage = np.array([
        [1.0, 0.2],
        [0.4, 1.0]
    ])
    
    # Injector with spatial coupling enabled
    injector = ArtifactInjector(spatial_coupling=leakage)
    
    # Inject event into channel 0 at t=2.0s, duration 1.0s, with 0.05s Tukey fade transitions
    injector.add(
        noise_model,
        timestamps_s=[2.0],
        duration_s=1.0,
        amplitude_scale=1.5,
        target_channels=[0],
        fade_in_s=0.05,
        fade_out_s=0.05
    )
    noisy = injector.apply(clean, fs)
    
    assert noisy.shape == (2, 500)
    # Outside the injection window, both should be zero
    assert np.all(noisy[:, :100] == 0.0)
    # Inside the injection window, channel 0 is directly corrupted, channel 1 is corrupted via spatial leakage
    assert not np.all(noisy[0, 200:300] == 0.0)
    assert not np.all(noisy[1, 200:300] == 0.0)
    # Verification of leakage scaling: channel 1 should be scaled by ~40% of channel 0
    assert np.std(noisy[1, 200:300]) < np.std(noisy[0, 200:300])
