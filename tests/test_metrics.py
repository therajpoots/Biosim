import numpy as np
import pytest
from biosignal_simulator.metrics.snr import (
    compute_snr_wideband,
    compute_snr_segmental,
    compute_snr_narrowband,
    compute_snr_spectral,
    compute_snr_adaptive,
    compute_snr_wavelet
)
from biosignal_simulator.metrics.spectral import (
    compute_psd,
    compute_band_power,
    compute_spectral_flatness,
    compute_thd,
    compute_spectral_entropy,
    compute_median_frequency,
    compute_mean_frequency,
    compute_spectral_edge_frequency
)
from biosignal_simulator.metrics.distortion import (
    compute_mse,
    compute_rmse,
    compute_psnr,
    compute_correlation,
    compute_ste,
    compute_qrs_correlation,
    compute_prd,
    compute_prdn,
    compute_max_absolute_error,
    compute_ssim_1d
)

def test_snr_metrics():
    fs = 100.0
    t = np.arange(1000) / fs
    clean = np.sin(2 * np.pi * 5.0 * t)
    
    # 20 dB noise
    noise = clean * 0.1
    noisy = clean + noise
    
    snr_wb = compute_snr_wideband(clean, noisy, fs)
    assert np.isclose(snr_wb, 20.0, atol=0.1)
    
    # Segmental SNR
    snr_seg = compute_snr_segmental(clean, noisy, fs, segment_s=2.0)
    assert len(snr_seg) == 5
    assert np.allclose(snr_seg, 20.0, atol=0.1)
    
    # Narrowband SNR
    snr_nb = compute_snr_narrowband(clean, noisy, fs, 4.0, 6.0)
    assert np.isclose(snr_nb, 20.0, atol=0.2)

def test_spectral_metrics():
    fs = 100.0
    t = np.arange(1000) / fs
    sig = np.sin(2 * np.pi * 10.0 * t)
    
    f, psd = compute_psd(sig, fs, method='welch')
    assert len(f) == len(psd)
    
    bp = compute_band_power(sig, fs, 9.0, 11.0)
    assert bp > 0.4 # Sine power is 0.5
    
    # Flatness of white noise should be high (~1.0), flatness of sine wave should be low (~0)
    white = np.random.default_rng(42).normal(0.0, 1.0, size=5000)
    flatness_white = compute_spectral_flatness(white, fs)
    flatness_sine = compute_spectral_flatness(sig, fs)
    assert flatness_white > 0.7
    assert flatness_sine < 0.1
    
    # THD of sine wave + 2nd harmonic (10%) + 3rd harmonic (5%)
    # Power ratio of harmonics to fundamental = (0.1**2 + 0.05**2) / 1.0 = 0.0125
    fundamental = np.sin(2 * np.pi * 10.0 * t)
    harm2 = 0.1 * np.sin(2 * np.pi * 20.0 * t)
    harm3 = 0.05 * np.sin(2 * np.pi * 30.0 * t)
    combined = fundamental + harm2 + harm3
    
    thd = compute_thd(combined, fs, fundamental_hz=10.0, n_harmonics=3)
    assert np.isclose(thd, 0.0125, atol=0.005)

def test_distortion_metrics():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    filt = np.array([1.1, 1.9, 3.0, 4.2, 4.8])
    
    mse = compute_mse(ref, filt)
    rmse = compute_rmse(ref, filt)
    psnr = compute_psnr(ref, filt)
    corr = compute_correlation(ref, filt)
    
    assert mse == pytest.approx(0.02)
    assert rmse == pytest.approx(np.sqrt(0.02))
    assert psnr == pytest.approx(20.0 * np.log10(5.0 / np.sqrt(0.02)))
    assert corr > 0.99
    
    # QRS correlation - simple verification with identical signals
    fs = 250.0
    t = np.arange(1000) / fs
    clean_ecg = np.sin(2 * np.pi * 1.5 * t) ** 4
    filtered_ecg = clean_ecg.copy()
    
    qrs_corr = compute_qrs_correlation(clean_ecg, filtered_ecg, fs)
    assert np.isclose(qrs_corr['mean'], 1.0)
    assert np.isclose(qrs_corr['std'], 0.0, atol=1e-7)

# =====================================================================
# Advanced Metrics Feature Tests
# =====================================================================

def test_spectral_metrics_advanced():
    fs = 200.0
    t = np.arange(2000) / fs  # 10 seconds
    
    # 10 Hz sine wave
    sine = np.sin(2.0 * np.pi * 10.0 * t)
    
    # 1. Median frequency of 10Hz sine should be close to 10 Hz
    mdf = compute_median_frequency(sine, fs)
    assert np.isclose(mdf, 10.0, atol=1.0)
    
    # 2. Mean frequency of 10Hz sine should be close to 10 Hz
    mnf = compute_mean_frequency(sine, fs)
    assert np.isclose(mnf, 10.0, atol=1.0)
    
    # 3. Spectral edge frequency (SEF95) of 10Hz sine should be slightly above 10 Hz
    sef = compute_spectral_edge_frequency(sine, fs, percentile=95.0)
    assert sef >= 9.5 and sef <= 15.0
    
    # 4. Spectral entropy: White noise should be high, sine should be low
    white = np.random.default_rng(42).normal(0.0, 1.0, size=2000)
    entropy_white = compute_spectral_entropy(white, fs)
    entropy_sine = compute_spectral_entropy(sine, fs)
    assert entropy_white > 0.7
    assert entropy_sine < 0.2


def test_snr_metrics_advanced_and_wavelet():
    fs = 200.0
    t = np.arange(1000) / fs  # 5 seconds
    clean_sine = np.sin(2.0 * np.pi * 6.0 * t)
    gate = (t >= 1.0) & (t <= 2.5)
    clean = clean_sine * gate
    
    # Add noise with 15dB wideband SNR
    # p_sig = 0.5 * 0.3 = 0.15
    # p_noise for 15dB SNR = 0.15 / 10**1.5 = 0.00474
    # std = sqrt(0.00474) = 0.0689
    noise = np.random.default_rng(42).normal(0.0, 0.0689, size=1000)
    noisy = clean + noise
    
    # 1. Spectral SNR
    snr_spec_wide = compute_snr_spectral(clean, noisy, fs)
    assert np.isclose(snr_spec_wide, 15.0, atol=4.0)
    
    snr_spec_narrow = compute_snr_spectral(clean, noisy, fs, fmin=4.0, fmax=8.0)
    assert snr_spec_narrow > 20.0
    
    # 2. Adaptive SNR (Blind estimation without clean reference)
    snr_adapt = compute_snr_adaptive(noisy, fs)
    assert np.isclose(snr_adapt, 15.0, atol=4.0)
    
    # 3. Wavelet Subband SNR
    wavelet_snrs = compute_snr_wavelet(clean, noisy, fs, level=3)
    assert 'D1' in wavelet_snrs
    assert 'D2' in wavelet_snrs
    assert 'D3' in wavelet_snrs
    assert 'A3' in wavelet_snrs
    assert np.all([isinstance(v, float) for v in wavelet_snrs.values()])


def test_distortion_similarity_metrics_advanced():
    fs = 100.0
    t = np.arange(500) / fs
    clean = np.sin(2.0 * np.pi * 5.0 * t)
    
    # Add 20% offset and scale distortion
    corrupted = 0.8 * clean + 0.1
    
    # 1. PRD (Percent Residual Difference)
    prd = compute_prd(clean, corrupted)
    assert prd > 10.0
    
    # 2. PRDN (Normalized PRD)
    prdn = compute_prdn(clean, corrupted)
    assert prdn > 10.0
    
    # 3. Maximum Absolute Error (MAE)
    mae = compute_max_absolute_error(clean, corrupted)
    assert np.isclose(mae, 0.3, atol=0.01)  # max absolute error is at peaks
    
    # 4. 1D SSIM (Structural Similarity)
    ssim_self = compute_ssim_1d(clean, clean, window_size=15)
    assert np.isclose(ssim_self, 1.0, atol=1e-7)  # identical signal SSIM = 1.0
    
    ssim_corr = compute_ssim_1d(clean, corrupted, window_size=15)
    assert ssim_corr < 0.99  # distorted signal SSIM should be lower than 1.0
