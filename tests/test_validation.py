"""
Exhaustive Tests for BioSignal Platform Validation and Quality Assessment Engine.

This module validates the SignalIntegrityChecker (flatlines, clipping, DC offsets,
powerline line noise, motion bursts) and the PhysiologicalValidator (Pan-Tompkins
QRS peak detector, respiration rate zero-crossings, PPG pulse rate, EEG spectral
bands, EMG mean/median frequencies).
"""

import os
import tempfile
import numpy as np
import pytest
from scipy import signal as sp_signal

from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.utils.validation import (
    validate_signal,
    validate_physiological_bounds,
    generate_validation_report_html,
    SignalIntegrityChecker,
    PhysiologicalValidator,
    ValidationReport
)

@pytest.fixture
def clean_record_params():
    """Returns baseline parameters for generating test records."""
    return {
        'signal_type': 'ecg',
        'fs': 100.0,
        't': np.arange(500) / 100.0,
        'clean': np.sin(2.0 * np.pi * 5.0 * (np.arange(500) / 100.0)),
        'noisy': np.sin(2.0 * np.pi * 5.0 * (np.arange(500) / 100.0)) + 0.1 * np.sin(2.0 * np.pi * 50.0 * (np.arange(500) / 100.0)),
        'noise_components': {
            'powerline': 0.1 * np.sin(2.0 * np.pi * 50.0 * (np.arange(500) / 100.0)),
            'gaussian': 0.02 * np.random.default_rng(42).normal(size=500)
        },
        'signal_params': {'heart_rate': 75.0, 'amplitude': 1.0},
        'noise_params': {'powerline': {'frequency': 50.0}, 'gaussian': {'std': 0.02}},
        'snr_db': 15.4,
        'metadata': {'subject_id': 'SUBJ-001', 'lead': 'II', 'notes': 'Test run'}
    }

@pytest.fixture
def single_channel_record(clean_record_params):
    """Fixture providing a standard 1-D SignalRecord."""
    return SignalRecord(**clean_record_params)


# =====================================================================
# Engineering Integrity Checker Tests
# =====================================================================

def test_flatline_detection():
    """Verify SignalIntegrityChecker.detect_flatline identifies constant regions."""
    fs = 100.0
    t = np.arange(1000) / fs # 10 seconds
    
    # Create signal with flatline from 2.0s to 4.0s (index 200 to 400)
    # and another flatline from 7.5s to 9.0s (index 750 to 900)
    sig = np.sin(2.0 * np.pi * 1.0 * t)
    sig[200:400] = 0.5
    sig[750:900] = -0.2
    
    # 1. Standard detection
    flat_segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-5, min_duration_s=0.5)
    assert len(flat_segments) == 2
    
    # Verify segment 1: ~2.0s to ~4.0s
    assert np.isclose(flat_segments[0][0], 2.0, atol=0.05)
    assert np.isclose(flat_segments[0][1], 4.0, atol=0.05)
    
    # Verify segment 2: ~7.5s to ~9.0s
    assert np.isclose(flat_segments[1][0], 7.5, atol=0.05)
    assert np.isclose(flat_segments[1][1], 9.0, atol=0.05)
    
    # 2. Test below min_duration threshold
    flat_short = SignalIntegrityChecker.detect_flatline(sig, fs, min_duration_s=3.0)
    assert len(flat_short) == 0 # no flatline is longer than 3s
    
    # 3. Test empty/clean signal has no flatlines
    clean_sig = np.sin(2.0 * np.pi * 1.0 * t)
    flat_clean = SignalIntegrityChecker.detect_flatline(clean_sig, fs)
    assert len(flat_clean) == 0

def test_clipping_detection():
    """Verify SignalIntegrityChecker.detect_clipping identifies signal saturation."""
    # Create highly clipped signal: sine wave clipped at 0.8 and -0.8
    t = np.arange(1000) / 100.0
    sig = np.sin(2.0 * np.pi * 2.0 * t)
    
    sig_clipped = np.clip(sig, -0.8, 0.8)
    
    # 1. Verify clipped signal detection
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.95)
    assert is_clipped is True
    # Theoretical clipped ratio: arcsin(0.8) is ~0.927 rad, which is ~30% of wave clipped
    assert ratio > 0.1
    
    # 2. Verify clean sine is not flagged as clipped
    is_clipped_clean, ratio_clean = SignalIntegrityChecker.detect_clipping(sig, threshold_ratio=0.99)
    assert is_clipped_clean is False
    assert np.isclose(ratio_clean, 0.0, atol=0.02) # only endpoints

def test_dc_offset_detection():
    """Verify SignalIntegrityChecker.detect_dc_offset flags high baseline offsets."""
    t = np.arange(500) / 100.0
    
    # 1. Clean zero-mean signal
    sig_clean = np.sin(2.0 * np.pi * 5.0 * t)
    is_dc, offset = SignalIntegrityChecker.detect_dc_offset(sig_clean)
    assert is_dc is False
    assert np.isclose(offset, 0.0, atol=1e-5)
    
    # 2. Signal with moderate DC offset (mean = 0.5, rms = sqrt(0.5**2 + 0.5) = sqrt(0.75) = 0.866)
    # Ratio = 0.5 / 0.866 = 0.577 (below default 0.8)
    sig_mod = np.sin(2.0 * np.pi * 5.0 * t) + 0.5
    is_dc_mod, offset_mod = SignalIntegrityChecker.detect_dc_offset(sig_mod, max_offset_ratio=0.5)
    assert is_dc_mod is True
    assert np.isclose(offset_mod, 0.5, atol=0.01)
    
    # 3. Signal with extreme DC offset (mean = 5.0, rms = ~5.02, ratio = 0.99)
    sig_extreme = np.sin(2.0 * np.pi * 5.0 * t) + 5.0
    is_dc_ext, offset_ext = SignalIntegrityChecker.detect_dc_offset(sig_extreme, max_offset_ratio=0.8)
    assert is_dc_ext is True

def test_powerline_interference_detection():
    """Verify SignalIntegrityChecker.detect_powerline_interference flags line leakage."""
    fs = 200.0
    t = np.arange(1000) / fs # 5 seconds
    
    # 1. Clean signal: 10 Hz sine
    clean = np.sin(2.0 * np.pi * 10.0 * t)
    has_noise, ratio = SignalIntegrityChecker.detect_powerline_interference(clean, fs, powerline_freq=50.0)
    assert has_noise is False
    assert ratio < 0.01
    
    # 2. Corrupted signal: 10 Hz sine + 50 Hz powerline line noise (amplitude 0.5, power ratio ~ 20%)
    noisy = clean + 0.5 * np.sin(2.0 * np.pi * 50.0 * t)
    has_noise_noisy, ratio_noisy = SignalIntegrityChecker.detect_powerline_interference(noisy, fs, powerline_freq=50.0)
    assert has_noise_noisy is True
    assert ratio_noisy > 0.05

def test_motion_burst_detection():
    """Verify SignalIntegrityChecker.detect_motion_bursts identifies sudden movement artifacts."""
    fs = 100.0
    t = np.arange(1000) / fs # 10 seconds
    
    # Base signal
    sig = np.sin(2.0 * np.pi * 2.0 * t)
    
    # Inject large amplitude burst from 4.0s to 5.5s (index 400 to 550)
    sig[400:550] += 8.0 * np.sin(2.0 * np.pi * 20.0 * (t[400:550] - 4.0))
    
    burst_segments = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.5, threshold_std=3.0)
    
    assert len(burst_segments) >= 1
    # Verify overlap with injected region
    assert burst_segments[0][0] <= 4.5
    assert burst_segments[0][1] >= 5.0


# =====================================================================
# Physiological Validator Tests
# =====================================================================

def test_pan_tompkins_qrs_detector():
    """Verify Pan-Tompkins peak detector accurately extracts heart rate locations."""
    fs = 250.0
    t = np.arange(1000) / fs # 4 seconds
    
    # Simulate a stylized ECG with R peaks every 0.8 seconds (75 bpm)
    # Peak indexes: 0.4s (100), 1.2s (300), 2.0s (500), 2.8s (700), 3.6s (900)
    ecg = np.zeros(1000)
    r_peaks_true = [100, 300, 500, 700, 900]
    
    for peak in r_peaks_true:
        # Styled QRS triangle
        ecg[peak-5:peak+5] = np.array([0.1, 0.3, 0.6, 1.0, 0.7, 0.4, 0.2, 0.0, -0.1, -0.05])
        # Styled T-wave (slow bulge)
        t_start = peak + 40
        r_len = 30
        ecg[t_start:t_start+r_len] = 0.2 * np.sin(np.pi * np.arange(r_len) / r_len)
        
    # Detect peaks
    peaks_detected = PhysiologicalValidator.pan_tompkins_qrs_detector(ecg, fs)
    
    assert len(peaks_detected) == 5
    assert np.allclose(peaks_detected, r_peaks_true, atol=2)

def test_validate_ecg():
    """Verify ECG validator extracts heart rate and flags abnormal conditions."""
    fs = 200.0
    t = np.arange(1000) / fs # 5 seconds
    
    # 1. Healthy ECG (75 bpm, RR = 0.8s)
    # R peaks at indexes: 100, 260, 420, 580, 740, 900
    ecg = np.zeros(1000)
    r_peaks = [100, 260, 420, 580, 740, 900]
    for p in r_peaks:
        ecg[p] = 1.5
        ecg[p-2:p+3] = [0.5, 1.0, 1.5, 0.8, 0.2]
        
    warnings_h, metrics_h = PhysiologicalValidator.validate_ecg(ecg, fs)
    assert len(warnings_h) == 0
    assert np.isclose(metrics_h['heart_rate_bpm'], 75.0, atol=1.0)
    assert np.isclose(metrics_h['rr_mean_ms'], 800.0, atol=10.0)
    
    # 2. Extreme Bradycardia ECG (30 bpm, RR = 2.0s)
    # R peaks at: 100, 500, 900
    ecg_brady = np.zeros(1000)
    r_peaks_b = [100, 500, 900]
    for p in r_peaks_b:
        ecg_brady[p-2:p+3] = [0.5, 1.0, 1.5, 0.8, 0.2]
        
    warnings_b, metrics_b = PhysiologicalValidator.validate_ecg(ecg_brady, fs)
    assert any("Bradycardia" in w for w in warnings_b)
    assert np.isclose(metrics_b['heart_rate_bpm'], 30.0, atol=1.0)
    
    # 3. Arrhythmic ECG (High SDNN variance)
    # Irregular R peaks: 100, 220, 420, 550, 740, 850
    ecg_arr = np.zeros(1000)
    r_peaks_a = [100, 220, 420, 550, 740, 850]
    for p in r_peaks_a:
        ecg_arr[p-2:p+3] = [0.5, 1.0, 1.5, 0.8, 0.2]
        
    warnings_a, metrics_a = PhysiologicalValidator.validate_ecg(ecg_arr, fs)
    assert any("variability" in w or "arrhythmia" in w for w in warnings_a)
    assert metrics_a['rr_sdnn_ms'] > 50.0

def test_validate_eeg():
    """Verify EEG spectral band power integration and warnings."""
    fs = 100.0
    t = np.arange(1000) / fs # 10 seconds
    
    # 1. Healthy alert state: high Beta oscillation (18 Hz)
    eeg_alert = np.sin(2.0 * np.pi * 18.0 * t)
    warnings_a, metrics_a = PhysiologicalValidator.validate_eeg(eeg_alert, fs)
    assert metrics_a['rel_beta'] > 0.5
    
    # 2. Deep sleep state: dominant Delta oscillation (2.0 Hz)
    eeg_sleep = np.sin(2.0 * np.pi * 2.0 * t)
    warnings_s, metrics_s = PhysiologicalValidator.validate_eeg(eeg_sleep, fs)
    assert metrics_s['rel_delta'] > 0.7
    assert any("Delta" in w for w in warnings_s)

def test_validate_emg():
    """Verify EMG mean and median spectral frequency tracking."""
    fs = 500.0
    t = np.arange(1000) / fs # 2 seconds
    
    # 1. Healthy active EMG: random noise filtered between 50 and 200 Hz
    # We will generate a colored noise signal peaking at 100 Hz
    rng = np.random.default_rng(42)
    noise = rng.normal(size=1000)
    
    # Apply bandpass Butterworth: 70 - 150 Hz
    nyq = 0.5 * fs
    b, a = sp_signal.butter(4, [70.0 / nyq, 150.0 / nyq], btype='bandpass')
    emg_active = sp_signal.filtfilt(b, a, noise)
    
    warnings_h, metrics_h = PhysiologicalValidator.validate_emg(emg_active, fs)
    assert len(warnings_h) == 0
    # Median frequency should fall within [70, 150] Hz band
    assert 70.0 <= metrics_h['median_frequency_hz'] <= 150.0
    
    # 2. Fatigued EMG (filtered with very low cutoff, e.g. 30 - 50 Hz)
    b_fat, a_fat = sp_signal.butter(4, [25.0 / nyq, 55.0 / nyq], btype='bandpass')
    emg_fatigue = sp_signal.filtfilt(b_fat, a_fat, noise)
    
    warnings_f, metrics_f = PhysiologicalValidator.validate_emg(emg_fatigue, fs)
    assert any("fatigue" in w.lower() for w in warnings_f)
    assert metrics_f['median_frequency_hz'] < 60.0

def test_validate_ppg():
    """Verify PPG pulse validator extracts correct rate."""
    fs = 100.0
    t = np.arange(1000) / fs # 10 seconds
    
    # Create PPG-like signal with pulses every 0.85s (70.6 bpm)
    ppg = np.zeros(1000)
    pulse_centers = np.arange(50, 950, 85)
    for center in pulse_centers:
        # Systolic peak (fast rise, slow decay)
        for i in range(-15, 30):
            idx = center + i
            if 0 <= idx < 1000:
                if i < 0:
                    ppg[idx] = (i + 15) / 15.0
                else:
                    ppg[idx] = np.exp(-i / 10.0)
                    
    warnings, metrics = PhysiologicalValidator.validate_ppg(ppg, fs)
    assert len(warnings) == 0
    assert np.isclose(metrics['pulse_rate_bpm'], 70.6, atol=2.0)

def test_validate_eda():
    """Verify EDA conductance bounds and low amplitude warnings."""
    fs = 50.0
    t = np.arange(500) / fs
    
    # 1. Normal EDA signal (mean = 5 uS)
    eda_norm = np.ones(500) * 5.0 + 0.1 * np.sin(2.0 * np.pi * 0.1 * t)
    warnings_n, metrics_n = PhysiologicalValidator.validate_eda(eda_norm, fs)
    assert len(warnings_n) == 0
    assert np.isclose(metrics_n['eda_mean_microsiemens'], 5.0, atol=0.1)
    
    # 2. Critically low EDA (0.002 uS)
    eda_low = np.ones(500) * 0.002
    warnings_l, metrics_l = PhysiologicalValidator.validate_eda(eda_low, fs)
    assert any("detachment" in w or "zero" in w for w in warnings_l)

def test_validate_resp():
    """Verify respiration rate zero-crossings and breathing warnings."""
    fs = 50.0
    t = np.arange(1000) / fs # 20 seconds
    
    # 1. Normal breathing: 15 breaths per minute (RR = 4.0s)
    # Cycles at 0.25 Hz
    resp_norm = np.sin(2.0 * np.pi * 0.25 * t)
    warnings_n, metrics_n = PhysiologicalValidator.validate_resp(resp_norm, fs)
    assert len(warnings_n) == 0
    assert np.isclose(metrics_n['respiration_rate_cpm'], 15.0, atol=1.0)
    
    # 2. Tachypnea (fast breathing): 45 cpm (0.75 Hz)
    resp_fast = np.sin(2.0 * np.pi * 0.75 * t)
    warnings_f, metrics_f = PhysiologicalValidator.validate_resp(resp_fast, fs)
    assert any("Tachypnea" in w for w in warnings_f)
    assert np.isclose(metrics_f['respiration_rate_cpm'], 45.0, atol=2.0)


# =====================================================================
# Standalone validate_physiological_bounds Tests
# =====================================================================

def test_validate_physiological_bounds_wrapper():
    """Verify validate_physiological_bounds routes correctly across signal types."""
    fs = 100.0
    t = np.arange(500) / fs
    
    # Test ECG routing
    ecg = np.sin(2.0 * np.pi * 1.2 * t)
    w_ecg, m_ecg = validate_physiological_bounds(ecg, fs, 'ecg')
    assert 'detected_peaks_count' in m_ecg
    
    # Test Respiration routing
    resp = np.sin(2.0 * np.pi * 0.2 * t)
    w_resp, m_resp = validate_physiological_bounds(resp, fs, 'resp')
    assert 'respiration_rate_cpm' in m_resp
    
    # Test Unknown routing
    w_unk, m_unk = validate_physiological_bounds(ecg, fs, 'unknown')
    assert len(w_unk) == 0
    assert len(m_unk) == 0


# =====================================================================
# validate_signal Joint Function Tests
# =====================================================================

def test_validate_signal_clean():
    """Verify validate_signal returns an all-clear report for a clean simulated ECG."""
    fs = 200.0
    t = np.arange(1000) / fs
    
    # Standard clean ECG (simulated peaks at 1s intervals -> 60 bpm)
    ecg = np.zeros(1000)
    peaks = [200, 400, 600, 800]
    for p in peaks:
        ecg[p-2:p+3] = [0.4, 0.8, 1.5, 0.6, 0.1]
        
    report = validate_signal(ecg, fs, 'ecg')
    
    assert isinstance(report, ValidationReport)
    assert report.is_valid is True
    assert len(report.warnings) == 0
    assert np.isclose(report.metrics['heart_rate_bpm'], 60.0, atol=1.0)
    assert report.metrics['clipping_ratio'] == 0.0

def test_validate_signal_corrupted():
    """Verify validate_signal identifies multiple integrity and physiological anomalies."""
    fs = 200.0
    t = np.arange(1000) / fs
    
    # Corrupted ECG:
    # 1. Constant flatline from 1.0 to 2.0s (index 200 to 400)
    # 2. Large clipping saturation at index 600 to 700
    # 3. High DC offset of 2.0 mV
    ecg = np.sin(2.0 * np.pi * 1.0 * t) + 2.0
    ecg[200:400] = 2.0
    ecg[600:700] = 3.5 # clipped limit
    
    report = validate_signal(ecg, fs, 'ecg')
    
    assert report.is_valid is False
    assert len(report.warnings) > 0
    # Verify warnings capture flatline, clipping, and DC offset
    warnings_joined = " ".join(report.warnings).lower()
    assert "flatline" in warnings_joined
    assert "clipped" in warnings_joined
    assert "dc offset" in warnings_joined


# =====================================================================
# HTML Validation Report Generator Tests
# =====================================================================

def test_generate_validation_report_html(single_channel_record):
    """Verify generate_validation_report_html writes structured, readable HTML pages."""
    report = validate_signal(single_channel_record.noisy, single_channel_record.fs, 'ecg')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "reports", "verification_log.html")
        
        # Generate
        generate_validation_report_html(single_channel_record, report, path)
        assert os.path.exists(path)
        
        # Read back and check contents
        with open(path, 'r', encoding='utf-8') as f:
            html = f.read()
            
        assert "<title>BioSignal Quality Validation Report</title>" in html
        assert "ECG" in html
        assert f"{single_channel_record.fs:.1f} Hz" in html
        assert "Measured Value" in html
        
        # Check that metrics are rendered inside table cells
        for k in report.metrics:
            assert k in html
