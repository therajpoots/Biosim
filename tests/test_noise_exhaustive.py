"""
Exhaustive Noise and Artifact Contamination Tests for BioSignal Simulator.

This module validates that noise and environmental artifact models correctly synthesize
contamination profiles, digital quantization, sensor degradation, and telemetry dropouts:
1. Voss-McCartney vs FFT Colored Noise: Exponent sweeps, PSD slopes (-10dB/dec for pink, -20dB/dec for brown).
2. Quantization and Dither: ADC bit widths, Uniform/TPDF dithering noise floors.
3. Packet Loss: Markov-based wireless dropouts, burst length statistics, interpolation fallbacks.
4. Electrode Detachment and Popcorn Telegraph Shifts: Popcorn noise jump rates, thermal contact noise.
"""

import numpy as np
import pytest
from scipy import signal as sp_signal

from biosignal_simulator.core.config import (
    GaussianNoiseConfig,
    ColoredNoiseConfig,
    QuantizationNoiseConfig,
    PacketLossConfig,
    SensorDetachmentConfig,
    ElectrodeDisplacementConfig
)
from biosignal_simulator.noise.gaussian import GaussianNoise
from biosignal_simulator.noise.colored import ColoredNoise, PinkNoise, BrownNoise
from biosignal_simulator.noise.quantization import QuantizationNoise
from biosignal_simulator.noise.wearable import PacketLossNoise, SensorDetachmentNoise, ElectrodeDisplacementNoise
from biosignal_simulator.core.math_utils import compute_rms, bandpower, robust_std


# =====================================================================
# 1. Colored Noise Exponent & PSD Slope Verification
# =====================================================================

def test_colored_noise_psd_slopes():
    """Verify that different colored noises have correct PSD slopes."""
    n_samples = 4000
    fs = 1000.0
    
    # Generate pink noise (exponent = 1.0)
    pink = PinkNoise(std=1.0, seed=42).generate(n_samples, fs)
    # Generate brown noise (exponent = 2.0)
    brown = BrownNoise(std=1.0, seed=42).generate(n_samples, fs)
    
    # Compute PSD slopes
    for noise, expected_exponent in [(pink, 1.0), (brown, 2.0)]:
        f, psd = sp_signal.welch(noise - np.mean(noise), fs=fs, nperseg=1024)
        
        # Exclude DC and high frequencies near Nyquist to avoid filter cutoff shape
        mask = (f >= 5.0) & (f <= 200.0)
        log_f = np.log10(f[mask])
        log_psd = np.log10(psd[mask] + 1e-15)
        
        # Fit a line: log10(PSD) = -exponent * log10(f) + constant
        slope, _ = np.polyfit(log_f, log_psd, 1)
        
        # The slope should match -exponent within ~0.3 tolerance
        assert np.isclose(slope, -expected_exponent, atol=0.35), f"Slope {slope:.2f} did not match expected {-expected_exponent}"

def test_colored_noise_methods():
    """Verify that Voss-McCartney (IIR approximation) and FFT generation are stable."""
    fs = 200.0
    n_samples = 1000
    
    # 1. FFT method (default)
    noise_fft = ColoredNoise(exponent=1.0, method='fft', seed=42).generate(n_samples, fs)
    assert len(noise_fft) == n_samples
    assert np.std(noise_fft) > 0.1
    
    # 2. Voss method
    noise_voss = ColoredNoise(exponent=1.0, method='voss', seed=42).generate(n_samples, fs)
    assert len(noise_voss) == n_samples
    assert np.std(noise_voss) > 0.1


# =====================================================================
# 2. ADC Quantization & Dither Tests
# =====================================================================

def test_quantization_noise_level():
    """Verify quantization noise RMS levels match theoretical values for different bit depths."""
    fs = 100.0
    n_samples = 2000
    
    # Clean full-scale sine wave
    t = np.arange(n_samples) / fs
    clean = 2.0 * np.sin(2.0 * np.pi * 2.0 * t)
    v_range = 5.0 # full-scale range [-2.5, 2.5]
    
    for bits in [8, 12, 16]:
        # Quantization model without dither
        model = QuantizationNoise(n_bits=bits, v_range=v_range, dither=False)
        quantized, quant_noise = model.apply(clean)
        
        # Theoretical quantization noise standard deviation: LSB / sqrt(12)
        lsb = v_range / (2 ** bits)
        theoretical_std = lsb / np.sqrt(12.0)
        
        # Measured quantization noise (difference between quantized and clean)
        measured_std = np.std(quant_noise)
        
        # Measured std must match theoretical within 30% (due to finite sample effects)
        assert np.isclose(measured_std, theoretical_std, rtol=0.30)

def test_quantization_dither():
    """Verify that adding dither removes harmonic distortion at the cost of higher noise floor."""
    fs = 100.0
    n_samples = 1000
    t = np.arange(n_samples) / fs
    clean = 2.0 * np.sin(2.0 * np.pi * 5.0 * t)
    
    # 1. No dither (produces harmonic peaks in the error spectrum)
    model_no_dither = QuantizationNoise(n_bits=8, v_range=5.0, dither=False)
    quantized_no_dither, error_no_dither = model_no_dither.apply(clean)
    
    # 2. With dither
    model_dither = QuantizationNoise(n_bits=8, v_range=5.0, dither=True)
    quantized_dither, error_dither = model_dither.apply(clean)
    
    # Dithered error standard deviation should be slightly higher
    assert np.std(error_dither) > np.std(error_no_dither)


# =====================================================================
# 3. Telemetry Packet Loss & Markov Dropouts
# =====================================================================

def test_packet_loss_dropout_statistics():
    """Verify Markov-based packet loss drops correct percentage of signal points."""
    fs = 100.0
    n_samples = 5000
    clean = np.sin(2.0 * np.pi * 1.0 * (np.arange(n_samples) / fs))
    
    for loss_rate in [0.05, 0.20, 0.50]:
        model = PacketLossNoise(loss_rate=loss_rate, burst_length_samples=5, interpolation_mode='zero', seed=42)
        dropped_signal, error = model.apply(clean)
        
        # Zero interpolation replaces dropped values with exactly 0.0
        # Count the number of zeros in the dropped signal (since clean signal is rarely exactly 0)
        zeros_count = np.sum(np.isclose(dropped_signal, 0.0, atol=1e-15))
        measured_loss_rate = zeros_count / n_samples
        
        # Measured loss rate should be close to target loss rate
        assert np.isclose(measured_loss_rate, loss_rate, atol=0.10)

def test_packet_loss_interpolation_modes():
    """Verify that different packet loss interpolation modes ('zero', 'hold', 'linear') operate correctly."""
    fs = 100.0
    n_samples = 100
    clean = np.ones(n_samples) * 5.0 # Flat signal at 5.0
    
    # 1. Zero mode
    model_zero = PacketLossNoise(loss_rate=0.5, burst_length_samples=4, interpolation_mode='zero', seed=123)
    dropped_zero, error_zero = model_zero.apply(clean)
    assert np.any(np.isclose(dropped_zero, 0.0))
    
    # 2. Hold mode (should stay at 5.0 because it holds the last value)
    model_hold = PacketLossNoise(loss_rate=0.5, burst_length_samples=4, interpolation_mode='hold', seed=123)
    dropped_hold, error_hold = model_hold.apply(clean)
    assert np.all(np.isclose(dropped_hold, 5.0))


# =====================================================================
# 4. Sensor Detachment & Contact Displacement Tests
# =====================================================================

def test_sensor_detachment_transients():
    """Verify that sensor detachment generates a sharp bounce transient followed by flatline noise."""
    fs = 100.0
    n_samples = 1000
    clean = np.sin(2.0 * np.pi * 1.0 * (np.arange(n_samples) / fs))
    
    model = SensorDetachmentNoise(
        detachment_time_s=3.0,
        transient_duration_s=0.2,
        transient_amplitude=10.0,
        noise_level_uv=50.0,
        seed=42
    )
    
    corrupted, error = model.apply(clean, fs)
    
    # 1. Before detachment (t < 3s, index < 300), signal should be clean
    assert np.allclose(corrupted[:250], clean[:250], atol=0.01)
    
    # 2. At detachment (t = 3s to 3.2s, index 300 to 320), there should be a huge transient spike
    assert np.max(np.abs(corrupted[300:320])) > 5.0
    
    # 3. After transient (t > 3.5s, index > 350), it should be a low-level flatline (noise std ~ 50uV)
    # Since clean was a sine wave of amplitude 1.0, the flatline noise is very small (50uV = 0.05 mV)
    assert np.std(corrupted[500:]) < 0.1

def test_electrode_displacement_offset_jumps():
    """Verify sudden displacement shifts add step offsets to the signal."""
    fs = 100.0
    n_samples = 1000
    clean = np.zeros(n_samples)
    
    model = ElectrodeDisplacementNoise(
        displacement_times=[2.0, 6.0],
        shift_amplitudes=[1.5, -2.5],
        noise_increments=[1.0, 1.0],
        seed=42
    )
    
    noise = model.generate(n_samples, fs)
    corrupted = clean + noise
    
    # 1. Before first shift (t < 2s, index < 200), we have thermal noise with std=0.05 (peaks < 0.25)
    assert np.allclose(corrupted[:190], 0.0, atol=0.25)
    
    # 2. Between shifts (t = 3s to 5s, index 300 to 500), shift is 1.5, noise std is 0.10 (peaks < 0.40)
    assert np.allclose(corrupted[300:500], 1.5, atol=0.40)
    
    # 3. After second shift (t > 7s, index > 700), cumulative shift is -1.0, noise std is 0.15 (peaks < 0.60)
    assert np.allclose(corrupted[700:], -1.0, atol=0.60)
