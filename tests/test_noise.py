import numpy as np
import pytest
from scipy import signal as sp_signal
from biosignal_simulator.noise.gaussian import GaussianNoise
from biosignal_simulator.noise.colored import ColoredNoise, PinkNoise, BrownNoise, BlueNoise, VioletNoise
from biosignal_simulator.noise.baseline import BaselineWander
from biosignal_simulator.noise.powerline import PowerlineNoise
from biosignal_simulator.noise.motion import MotionArtifact
from biosignal_simulator.noise.electrode import ElectrodeNoise
from biosignal_simulator.noise.emg_artifact import EMGArtifact
from biosignal_simulator.noise.impulse import ImpulseNoise
from biosignal_simulator.noise.quantization import QuantizationNoise
from biosignal_simulator.noise.crosstalk import CrosstalkNoise
from biosignal_simulator.noise.wearable import (
    SensorDetachmentNoise, ElectrodeDisplacementNoise, LightLeakageNoise, PacketLossNoise
)

# =====================================================================
# 1. Gaussian Noise Tests
# =====================================================================

def test_gaussian_noise_distributions():
    n_samples = 1000
    fs = 100.0
    
    # 1. Normal/Gaussian
    model_g = GaussianNoise(distribution="gaussian", std=1.0, mean=0.0, seed=42)
    noise_g = model_g.generate(n_samples, fs)
    assert len(noise_g) == n_samples
    assert np.isclose(np.mean(noise_g), 0.0, atol=0.15)
    assert np.isclose(np.std(noise_g), 1.0, atol=0.15)
    
    # 2. Uniform
    model_u = GaussianNoise(distribution="uniform", std=1.0, mean=0.0, seed=42)
    noise_u = model_u.generate(n_samples, fs)
    assert len(noise_u) == n_samples
    assert np.max(noise_u) <= np.sqrt(3.0) + 0.05
    assert np.min(noise_u) >= -np.sqrt(3.0) - 0.05
    
    # 3. Laplacian
    model_l = GaussianNoise(distribution="laplacian", std=1.0, mean=0.0, seed=42)
    noise_l = model_l.generate(n_samples, fs)
    assert len(noise_l) == n_samples
    
    # 4. Student-t
    model_t = GaussianNoise(distribution="student_t", distribution_params={"df": 4.0}, std=1.0, seed=42)
    noise_t = model_t.generate(n_samples, fs)
    assert len(noise_t) == n_samples
    
    # 5. Cauchy
    model_c = GaussianNoise(distribution="cauchy", distribution_params={"scale": 0.5}, std=1.0, seed=42)
    noise_c = model_c.generate(n_samples, fs)
    assert len(noise_c) == n_samples


def test_gaussian_noise_envelopes():
    n_samples = 1000
    fs = 100.0
    
    # Sinusoidal
    model_sin = GaussianNoise(envelope_type="sinusoidal", envelope_params={"freq_hz": 0.5, "depth": 0.5}, seed=42)
    noise_sin = model_sin.generate(n_samples, fs)
    assert len(noise_sin) == n_samples
    
    # Exponential decay
    model_exp = GaussianNoise(envelope_type="exponential", envelope_params={"decay_rate": 2.0}, seed=42)
    noise_exp = model_exp.generate(n_samples, fs)
    # The amplitude should decay exponentially: check first half vs second half std
    assert np.std(noise_exp[:200]) > np.std(noise_exp[-200:])
    
    # Random walk envelope
    model_rw = GaussianNoise(envelope_type="random_walk", envelope_params={"step_std": 0.02}, seed=42)
    noise_rw = model_rw.generate(n_samples, fs)
    assert len(noise_rw) == n_samples


def test_gaussian_noise_spatial_correlation():
    n_samples = 1000
    fs = 100.0
    n_channels = 3
    
    # 3x3 positive-definite correlation matrix
    R = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    
    model = GaussianNoise(std=1.0, mean=0.0, spatial_correlation=R, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)
    
    # Check spatial correlation is approximately matched
    sample_R = np.corrcoef(noise)
    assert np.allclose(sample_R, R, atol=0.25)


# =====================================================================
# 2. Colored Noise Tests
# =====================================================================

def test_colored_noise_methods():
    n_samples = 1000
    fs = 200.0
    
    # FFT White
    model_w = ColoredNoise(exponent=0.0, std=1.0, seed=42)
    noise_w = model_w.generate(n_samples, fs)
    assert np.isclose(np.std(noise_w), 1.0, atol=0.05)
    
    # FFT Pink
    model_p = PinkNoise(std=1.0, method='fft', seed=42)
    noise_p = model_p.generate(n_samples, fs)
    assert np.isclose(np.std(noise_p), 1.0, atol=0.05)
    
    # IIR Pink
    model_p_iir = PinkNoise(std=1.0, method='iir', seed=42)
    noise_p_iir = model_p_iir.generate(n_samples, fs)
    assert np.isclose(np.std(noise_p_iir), 1.0, atol=0.1)
    
    # Voss-McCartney Pink
    model_p_voss = PinkNoise(std=1.0, method='voss', seed=42)
    noise_p_voss = model_p_voss.generate(n_samples, fs)
    assert np.isclose(np.std(noise_p_voss), 1.0, atol=0.1)
    
    # Standard colors
    assert len(BrownNoise(seed=42).generate(n_samples, fs)) == n_samples
    assert len(BlueNoise(seed=42).generate(n_samples, fs)) == n_samples
    assert len(VioletNoise(seed=42).generate(n_samples, fs)) == n_samples


def test_colored_noise_spatial_correlation():
    n_samples = 800
    fs = 100.0
    n_channels = 2
    R = np.array([
        [1.0, 0.7],
        [0.7, 1.0]
    ])
    model = PinkNoise(std=1.0, spatial_correlation=R, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)
    
    sample_R = np.corrcoef(noise)
    assert np.isclose(sample_R[0, 1], R[0, 1], atol=0.25)


# =====================================================================
# 3. Baseline Wander Tests
# =====================================================================

def test_baseline_wander_components():
    n_samples = 1000
    fs = 100.0
    
    # Sine breathing + trend
    model = BaselineWander(
        amplitude=0.5,
        f_resp_hz=0.2,
        resp_fraction=0.5,
        drift_fraction=0.0,
        trend_fraction=0.5,
        trend_degree=2,
        asymmetric_breathing=True,
        seed=42
    )
    noise = model.generate(n_samples, fs)
    assert len(noise) == n_samples
    assert np.isclose(np.std(noise), 0.5, atol=0.05)


def test_baseline_wander_multichannel():
    n_samples = 1000
    fs = 100.0
    n_channels = 3
    R = np.array([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.6],
        [0.5, 0.6, 1.0]
    ])
    model = BaselineWander(amplitude=0.3, spatial_correlation=R, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)


# =====================================================================
# 4. Powerline Noise Tests
# =====================================================================

def test_powerline_noise_harmonics():
    n_samples = 1000
    fs = 1000.0
    
    model = PowerlineNoise(
        f_line_hz=50.0,
        n_harmonics=4,
        amplitude=1.2,
        harmonic_decay=1.2,
        freq_std_hz=0.2,
        amplitude_mod_depth=0.15,
        seed=42
    )
    noise = model.generate(n_samples, fs)
    assert len(noise) == n_samples
    assert np.isclose(np.std(noise), 1.2, atol=0.1)


def test_powerline_noise_multichannel():
    n_samples = 1000
    fs = 500.0
    n_channels = 2
    phases = [0.0, np.pi/2.0]
    model = PowerlineNoise(f_line_hz=60.0, amplitude=0.8, channel_phases=phases, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)
    
    # Phase offset between channels should cause a high degree of differences
    assert not np.allclose(noise[0], noise[1])


# =====================================================================
# 5. Motion Artifact Tests
# =====================================================================

def test_motion_artifact_types():
    n_samples = 1000
    fs = 200.0
    
    # Impacts enabled
    model = MotionArtifact(
        lf_amplitude=0.2,
        enable_lf=True,
        enable_impacts=True,
        impact_rate_hz=1.0,
        impact_amplitude=1.5,
        impact_decay_s=0.3,
        impact_freq_hz=15.0,
        enable_cable=True,
        cable_amplitude=0.2,
        seed=42
    )
    noise = model.generate(n_samples, fs)
    assert len(noise) == n_samples
    assert not np.all(noise == 0.0)


def test_motion_artifact_projection():
    n_samples = 500
    fs = 100.0
    n_channels = 2
    proj_vec = [1.0, -0.5]
    model = MotionArtifact(enable_lf=True, lf_amplitude=1.0, motion_direction_vector=proj_vec, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    
    assert noise.shape == (n_channels, n_samples)
    # Check that they are anti-phase correlated roughly matching -0.5 vs 1.0 projection ratio
    corr = np.corrcoef(noise[0], noise[1])[0, 1]
    assert corr < 0.0  # Should be negatively correlated


# =====================================================================
# 6. Electrode Noise Tests
# =====================================================================

def test_electrode_noise_thermal_and_polarization():
    n_samples = 1000
    fs = 500.0
    
    model = ElectrodeNoise(
        enable_popcorn=True,
        popcorn_amplitude=0.1,
        popcorn_rate_hz=5.0,
        enable_impedance_noise=True,
        impedance_ohms=10000.0,
        temperature_k=310.0,
        initial_polarization_mv=5.0,
        settling_time_s=1.0,
        seed=42
    )
    noise = model.generate(n_samples, fs)
    assert len(noise) == n_samples
    
    # Settling check: mean of first 100 samples should capture high polarization decay
    assert np.mean(noise[:100]) > np.mean(noise[-100:])


def test_electrode_noise_multichannel():
    n_samples = 500
    fs = 200.0
    n_channels = 2
    model = ElectrodeNoise(enable_popcorn=True, enable_impedance_noise=True, initial_polarization_mv=0.0, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)
    # Channels should be fully independent
    assert abs(np.corrcoef(noise[0], noise[1])[0, 1]) < 0.25


# =====================================================================
# 7. EMG Artifact Tests
# =====================================================================

def test_emg_artifact_fatigue():
    n_samples = 2000
    fs = 1000.0
    
    # Dynamic fatigue filter enabled
    model = EMGArtifact(
        amplitude_fraction=0.8,
        fmin_hz=50.0,
        fmax_hz=400.0,
        burst_rate_hz=0.0,  # Continuous tonic EMG
        tonic_amplitude=1.0,
        fatigue_rate=1.0,   # Strong fatigue compression
        seed=42
    )
    noise = model.generate(n_samples, fs)
    assert len(noise) == n_samples
    
    # Fatigue should compress frequency towards lower bands. We check this
    # by counting zero-crossing rates in the first half vs the second half.
    # Higher frequency = more zero crossings.
    zero_crossings = lambda x: np.sum(np.diff(np.sign(x)) != 0)
    zc_start = zero_crossings(noise[:1000])
    zc_end = zero_crossings(noise[-1000:])
    assert zc_start > zc_end


def test_emg_artifact_multichannel():
    n_samples = 1000
    fs = 500.0
    n_channels = 2
    model = EMGArtifact(amplitude_fraction=0.2, burst_rate_hz=1.0, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)


# =====================================================================
# 8. Impulse Noise Tests
# =====================================================================

def test_impulse_noise_shapes():
    n_samples = 1000
    fs = 200.0
    
    # 1. Double Exponential
    model_de = ImpulseNoise(
        rate_hz=5.0,
        amplitude_scale=2.0,
        pulse_width_s=0.1,
        pulse_shape="double_exponential",
        rise_time_s=0.01,
        seed=42
    )
    noise_de = model_de.generate(n_samples, fs)
    assert len(noise_de) == n_samples
    
    # 2. Triangular
    model_tri = ImpulseNoise(rate_hz=4.0, pulse_width_s=0.05, pulse_shape="triangular", seed=42)
    noise_tri = model_tri.generate(n_samples, fs)
    assert len(noise_tri) == n_samples
    
    # 3. Rectangular
    model_rect = ImpulseNoise(rate_hz=4.0, pulse_width_s=0.05, pulse_shape="rectangular", seed=42)
    noise_rect = model_rect.generate(n_samples, fs)
    assert len(noise_rect) == n_samples


def test_impulse_noise_multichannel():
    n_samples = 1000
    fs = 200.0
    n_channels = 3
    model = ImpulseNoise(rate_hz=3.0, spatial_leakage_factor=0.4, seed=42)
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)


# =====================================================================
# 9. Quantization Noise Tests
# =====================================================================

def test_quantization_companding_and_shaping():
    t = np.linspace(0, 1, 1000)
    sig = 1.0 * np.sin(2.0 * np.pi * 10.0 * t)
    
    # 1. mu-law companding
    model_mu = QuantizationNoise(n_bits=8, v_range=2.5, companding="mu_law", companding_factor=255.0, seed=42)
    q_mu, err_mu = model_mu.apply(sig)
    assert q_mu.shape == sig.shape
    
    # 2. A-law companding
    model_a = QuantizationNoise(n_bits=8, v_range=2.5, companding="a_law", companding_factor=87.6, seed=42)
    q_a, err_a = model_a.apply(sig)
    assert q_a.shape == sig.shape
    
    # 3. Error Feedback Noise Shaping (2nd order)
    model_shape = QuantizationNoise(n_bits=10, v_range=3.0, noise_shaping_order=2, seed=42)
    q_sh, err_sh = model_shape.apply(sig)
    assert q_sh.shape == sig.shape


# =====================================================================
# 10. Crosstalk Noise Tests
# =====================================================================

def test_crosstalk_volume_conduction():
    n_samples = 500
    fs = 250.0
    
    # Conduction lowpass filtering enabled
    model = CrosstalkNoise(
        coupling_factor=0.2,
        source_type="ecg",
        enable_volume_conduction=True,
        conduction_cutoff_hz=15.0,
        seed=42
    )
    noise = model.generate(n_samples, fs)
    assert len(noise) == n_samples
    
    # Since lowpass filtered at 15Hz, high frequency elements should be attenuated
    # (Checking that power at 100Hz is extremely low)
    f, psd = sp_signal.welch(noise, fs=fs, nperseg=256)
    idx_100 = np.searchsorted(f, 100.0)
    assert psd[idx_100] < 1e-4


def test_crosstalk_multichannel_matrix():
    n_samples = 500
    fs = 250.0
    n_channels = 2  # EEG target channels
    
    # Source is VCG ECG (has 3 source channels: X, Y, Z)
    # Target is 2 channels. Mixing matrix is 2x3
    M = np.array([
        [0.8, -0.4, 0.1],
        [0.2, 0.9, -0.3]
    ])
    
    from biosignal_simulator.core.config import ECGConfig, CrosstalkNoiseConfig
    # Use VCG source type to generate 3 channels
    ecg_vcg = ECGConfig(fs=fs, lead_type='vcg', seed=42)
    
    crosstalk_cfg = CrosstalkNoiseConfig(coupling_factor=0.25, source_type='ecg', source_config=ecg_vcg)
    model = CrosstalkNoise(config=crosstalk_cfg, coupling_matrix=M, seed=42)
    
    noise = model.generate_multichannel(n_channels, n_samples, fs)
    assert noise.shape == (n_channels, n_samples)
