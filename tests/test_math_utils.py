import numpy as np
import pytest
from biosignal_simulator.core.math_utils import (
    compute_rms,
    normalize_to_rms,
    db_to_linear,
    linear_to_db,
    bandpower,
    spectral_shape
)

def test_compute_rms():
    # Empty array
    assert compute_rms(np.array([])) == 0.0
    # Zero array
    assert compute_rms(np.zeros(10)) == 0.0
    # Sine wave RMS should be approx amplitude / sqrt(2)
    t = np.linspace(0, 1, 1000)
    sine = np.sin(2 * np.pi * 10 * t)
    assert np.isclose(compute_rms(sine), 1.0 / np.sqrt(2.0), atol=1e-3)

def test_normalize_to_rms():
    t = np.linspace(0, 1, 1000)
    x = np.sin(2 * np.pi * 5 * t)
    target = 2.5
    x_norm = normalize_to_rms(x, target)
    assert np.isclose(compute_rms(x_norm), target)
    
    # Zero array handling
    zero = np.zeros(10)
    assert np.array_equal(normalize_to_rms(zero, target), zero)

def test_db_linear_conversion():
    assert db_to_linear(0.0) == 1.0
    assert db_to_linear(10.0) == 10.0
    assert db_to_linear(20.0) == 100.0
    
    assert linear_to_db(1.0) == 0.0
    assert linear_to_db(10.0) == 10.0
    assert linear_to_db(100.0) == 20.0
    assert linear_to_db(0.0) == -150.0

def test_bandpower():
    fs = 100.0
    t = np.arange(1000) / fs
    # Sine wave of 10 Hz with amplitude sqrt(2) has power = 1.0
    sig = np.sqrt(2) * np.sin(2 * np.pi * 10.0 * t)
    
    # Integrate around 10 Hz
    power = bandpower(sig, fs, 8.0, 12.0)
    assert np.isclose(power, 1.0, atol=0.15)
    
    # Out of band power should be low
    power_out = bandpower(sig, fs, 20.0, 30.0)
    assert power_out < 0.05

def test_spectral_shape():
    n = 1024
    fs = 250.0
    
    # White noise exponent=0
    shape_white = spectral_shape(n, fs, 0.0)
    assert np.all(shape_white[1:] == 1.0)
    assert shape_white[0] == 0.0 # DC is zeroed
    
    # Pink noise exponent=1
    shape_pink = spectral_shape(n, fs, 1.0)
    freqs = np.fft.rfftfreq(n, 1/fs)
    # Shape should be 1/sqrt(f) for f > 0
    assert np.isclose(shape_pink[1], freqs[1] ** (-0.5))


def test_advanced_statistical_moments():
    from biosignal_simulator.core.math_utils import (
        compute_skewness,
        compute_kurtosis,
        compute_zcr,
        compute_shannon_entropy,
        compute_cross_correlation
    )
    
    # 1. Skewness test
    # Normal distribution should have skewness approx 0.0
    rng = np.random.default_rng(42)
    normal_data = rng.normal(0, 1, 1000)
    assert abs(compute_skewness(normal_data)) < 0.15
    
    # Log-normal distribution is right-skewed (skewness > 0)
    skewed_data = rng.lognormal(0, 0.5, 1000)
    assert compute_skewness(skewed_data) > 0.3
    
    # Empty or short list fallbacks
    assert compute_skewness(np.array([1, 2])) == 0.0
    assert compute_skewness(np.zeros(10)) == 0.0

    # 2. Kurtosis test
    # Normal distribution excess kurtosis should be approx 0.0
    assert abs(compute_kurtosis(normal_data, fisher=True)) < 0.3
    # Pearson kurtosis of normal should be approx 3.0
    assert abs(compute_kurtosis(normal_data, fisher=False) - 3.0) < 0.3
    
    # Short list fallbacks
    assert compute_kurtosis(np.array([1, 2, 3])) == 0.0
    assert compute_kurtosis(np.zeros(10)) == 0.0

    # 3. ZCR test
    # Sine wave ZCR should match 2 * frequency / fs
    fs = 1000.0
    t = np.arange(1000) / fs
    sig_10hz = np.sin(2 * np.pi * 10.0 * t)
    # 10 Hz over 1 second has 20 zero crossings
    zcr_val = compute_zcr(sig_10hz)
    assert np.isclose(zcr_val, 19.0 / 999.0, atol=1e-3)
    
    # ZCR for flat line
    assert compute_zcr(np.ones(10)) == 0.0

    # 4. Shannon Entropy
    # Constant signal should have 0 entropy
    assert compute_shannon_entropy(np.ones(100)) == 0.0
    # Uniform distribution should have higher entropy
    uniform_data = rng.uniform(0, 10, 1000)
    assert compute_shannon_entropy(uniform_data) > 3.0

    # 5. Cross Correlation
    assert np.isclose(compute_cross_correlation(sig_10hz, sig_10hz), 1.0)
    assert np.isclose(compute_cross_correlation(sig_10hz, -sig_10hz), -1.0)
    assert abs(compute_cross_correlation(sig_10hz, np.sin(2 * np.pi * 10.0 * t + np.pi / 2))) < 0.05
    assert compute_cross_correlation(sig_10hz, np.zeros(1000)) == 0.0


def test_classical_iir_filters():
    from biosignal_simulator.core.math_utils import (
        chebyshev1_lowpass,
        chebyshev2_lowpass,
        ellip_lowpass,
        bessel_lowpass,
        chebyshev1_highpass,
        chebyshev2_highpass,
        ellip_highpass,
        bessel_highpass
    )
    
    fs = 200.0
    t = np.arange(1000) / fs
    # Composite signal: 2 Hz (low frequency) + 60 Hz (high frequency noise)
    sig_low = np.sin(2 * np.pi * 2.0 * t)
    sig_high = 0.5 * np.sin(2 * np.pi * 60.0 * t)
    sig = sig_low + sig_high
    
    # Filter out 60 Hz with Chebyshev Type I lowpass (cutoff = 10 Hz)
    filtered_cheby1 = chebyshev1_lowpass(sig, fs, 10.0)
    assert compute_rms(filtered_cheby1 - sig_low) < 0.1
    
    # Filter with Chebyshev Type II lowpass
    filtered_cheby2 = chebyshev2_lowpass(sig, fs, 10.0)
    assert compute_rms(filtered_cheby2 - sig_low) < 0.1
    
    # Filter with Elliptic lowpass
    filtered_ellip = ellip_lowpass(sig, fs, 10.0)
    assert compute_rms(filtered_ellip - sig_low) < 0.1
    
    # Filter with Bessel lowpass
    filtered_bessel = bessel_lowpass(sig, fs, 10.0)
    assert compute_rms(filtered_bessel - sig_low) < 0.1

    # Highpass test (cutoff = 20 Hz, preserves 60 Hz, filters 2 Hz)
    filtered_hp_cheby1 = chebyshev1_highpass(sig, fs, 20.0)
    assert compute_rms(filtered_hp_cheby1 - sig_high) < 0.15


def test_distortion_and_similarity():
    from biosignal_simulator.core.math_utils import compute_rmse, compute_ssim_1d
    
    x = np.sin(np.linspace(0, 10, 1000))
    y = x + 0.1 * np.random.default_rng(42).normal(0, 1, 1000)
    
    rmse_val = compute_rmse(x, y)
    assert 0.08 < rmse_val < 0.12
    
    # SSIM between identical signals should be exactly 1.0
    assert np.isclose(compute_ssim_1d(x, x), 1.0)
    
    # SSIM for perturbed signal should be slightly less than 1.0
    assert 0.8 < compute_ssim_1d(x, y) < 1.0
    
    # Error checking
    with pytest.raises(ValueError):
        compute_rmse(x, x[:-1])
    with pytest.raises(ValueError):
        compute_ssim_1d(x, x[:-1])


def test_detrend_and_smoothing():
    from biosignal_simulator.core.math_utils import detrend_polynomial, moving_average
    
    t = np.linspace(0, 10, 1000)
    trend = 0.5 * t - 1.2
    sig_clean = np.sin(t)
    sig = sig_clean + trend
    
    # Polynomial detrending should remove a pure linear trend
    detrended_trend = detrend_polynomial(trend, deg=1)
    assert np.allclose(detrended_trend, 0.0, atol=1e-10)
    
    # Moving average smoothing
    noisy_sig = sig_clean + 0.2 * np.random.default_rng(42).normal(0, 1, 1000)
    smoothed = moving_average(noisy_sig, window_size=15)
    assert len(smoothed) == len(sig_clean)
    assert compute_rms(smoothed - sig_clean) < compute_rms(noisy_sig - sig_clean)

