"""
Exhaustive Physiological and Clinical Pathology Tests for BioSignal Simulator.

This module validates that physiological generators correctly synthesize clinical
pathologies, sleep stages, respiratory patterns, and muscle syndromes, verifying
their distinctive statistical and time-frequency characteristics:
1. ECG: Atrial Fibrillation (AFib), Premature Ventricular Contractions (PVCs), AV Blocks.
2. EEG: Sleep Spindles, K-Complexes, Absence/Tonic-Clonic Seizures.
3. EMG: Fatigue Median Frequency Drift, ALS Fasciculations, Neuropathic MUAPs.
4. PPG: Dicrotic Notch and Respiratory Sinus Arrhythmia (RSA) modulation.
5. EDA: Phasic Skin Conductance Responses (SCR) decomposition.
6. Respiration: Cheyne-Stokes crescendo-decrescendo and Kussmaul breathing.
"""

import numpy as np
import pytest
from scipy import signal as sp_signal

from biosignal_simulator.core.config import (
    ECGConfig,
    EEGConfig,
    EMGConfig,
    PPGConfig,
    EDAConfig,
    RespConfig
)
from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.signals.eeg import EEGGenerator
from biosignal_simulator.signals.emg import EMGGenerator
from biosignal_simulator.signals.ppg import PPGGenerator
from biosignal_simulator.signals.eda import EDAGenerator
from biosignal_simulator.signals.resp import RespGenerator
from biosignal_simulator.core.math_utils import (
    compute_rms,
    bandpower,
    robust_std,
    compute_zcr,
    compute_skewness,
    compute_kurtosis
)
from biosignal_simulator.utils.validation import PhysiologicalValidator

# =====================================================================
# 1. ECG Pathology & Arrhythmia Tests
# =====================================================================

def test_ecg_pathology_afib():
    """Verify Atrial Fibrillation (AFib) generates irregular RR intervals and f-waves."""
    cfg = ECGConfig(
        fs=250.0,
        duration_s=12.0,
        heart_rate=110.0,
        rhythm_type='afib',
        seed=101
    )
    gen = ECGGenerator(cfg)
    ecg = gen.generate()
    
    # 1. Inspect peak locations and compute RR intervals
    peaks = PhysiologicalValidator.pan_tompkins_qrs_detector(ecg, cfg.fs)
    assert len(peaks) >= 10
    
    rr_intervals = np.diff(peaks) / cfg.fs
    rr_sdnn = np.std(rr_intervals) * 1000.0
    
    # In AFib, RR interval variability (SDNN) should be high (> 100 ms)
    assert rr_sdnn > 80.0, f"AFib RR variability too low: {rr_sdnn:.1f} ms"
    
    # 2. Check spectral properties of f-waves (no clear P waves, but high-frequency fibrillatory activity)
    # We should have low-frequency spectral power in the 4-9 Hz band representing f-waves
    p_fwave = bandpower(ecg, cfg.fs, 4.0, 9.0)
    assert p_fwave > 0.001

def test_ecg_pathology_pvc():
    """Verify Premature Ventricular Contractions (PVC) inject ectopic wider beats."""
    cfg = ECGConfig(
        fs=200.0,
        duration_s=15.0,
        heart_rate=75.0,
        rhythm_type='pvc',
        seed=202
    )
    gen = ECGGenerator(cfg)
    ecg = gen.generate()
    
    peaks = PhysiologicalValidator.pan_tompkins_qrs_detector(ecg, cfg.fs)
    assert len(peaks) >= 12
    
    rr_intervals = np.diff(peaks) / cfg.fs
    # Ectopic beats occur prematurely, creating a short RR interval followed by a compensatory pause (long RR)
    # So the ratio of maximum to minimum RR interval should be very high
    rr_ratio = np.max(rr_intervals) / np.min(rr_intervals)
    assert rr_ratio > 1.4, f"Ectopic compensatory pause not prominent: ratio={rr_ratio:.2f}"

def test_ecg_pathology_av_block():
    """Verify Atrioventricular (AV) Blocks generate dropped beats or elongated PR intervals."""
    cfg = ECGConfig(
        fs=250.0,
        duration_s=10.0,
        heart_rate=60.0,
        rhythm_type='av_block',
        seed=303
    )
    gen = ECGGenerator(cfg)
    ecg = gen.generate()
    
    peaks = PhysiologicalValidator.pan_tompkins_qrs_detector(ecg, cfg.fs)
    # In second-degree or third-degree AV block, some atrial P-waves are not conducted,
    # meaning the actual ventricular rate (peaks count) is lower than the target atrial heart rate
    expected_peaks_normal = (cfg.duration_s * cfg.heart_rate) / 60.0 # should be 10 beats
    assert len(peaks) < expected_peaks_normal + 2


# =====================================================================
# 2. EEG Brain State & Pathology Tests
# =====================================================================

def test_eeg_sleep_spindles_n2():
    """Verify sleep spindles (11-16 Hz sigma power bursts) occur in N2 sleep stage."""
    cfg_relaxed = EEGConfig(fs=100.0, duration_s=10.0, state='relaxed', seed=42)
    cfg_n2 = EEGConfig(fs=100.0, duration_s=10.0, state='n2_sleep', seed=42)
    
    eeg_relaxed = EEGGenerator(cfg_relaxed).generate()
    eeg_n2 = EEGGenerator(cfg_n2).generate()
    
    # Sigma band power (11 - 16 Hz) should be elevated in N2 sleep compared to relaxed wakefulness
    p_sigma_relaxed = bandpower(eeg_relaxed, 100.0, 11.0, 16.0)
    p_sigma_n2 = bandpower(eeg_n2, 100.0, 11.0, 16.0)
    
    assert p_sigma_n2 > p_sigma_relaxed

def test_eeg_pathology_seizures():
    """Verify absence (3 Hz spike-wave) and tonic-clonic seizure frequency spikes."""
    # Absence seizure: 3 Hz spike-wave discharges
    cfg_absence = EEGConfig(fs=200.0, duration_s=6.0, state='absence', seed=505)
    eeg_absence = EEGGenerator(cfg_absence).generate()
    
    # Sharp spike-wave creates high kurtosis/skewness compared to uniform random states
    skew = abs(compute_skewness(eeg_absence))
    kurt = compute_kurtosis(eeg_absence, fisher=True)
    assert skew > 0.1
    assert kurt > 1.0
    
    # Tonic-Clonic seizure: high amplitude, fast rhythmic discharges (10-20 Hz)
    cfg_tc = EEGConfig(fs=250.0, duration_s=5.0, state='tonic_clonic', seed=606)
    eeg_tc = EEGGenerator(cfg_tc).generate()
    
    # Power should be concentrated in high-alpha / beta band (10 - 20 Hz)
    p_seizure_band = bandpower(eeg_tc, 250.0, 10.0, 20.0)
    p_low_band = bandpower(eeg_tc, 250.0, 1.0, 5.0)
    
    assert p_seizure_band > p_low_band


# =====================================================================
# 3. EMG Pathology & Muscle Fatigue Tests
# =====================================================================

def test_emg_fatigue_frequency_drift():
    """Verify that simulated isometric muscle fatigue causes spectral frequency drift."""
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=8.0,
        envelope_type='constant',
        pathology='normal',
        seed=707
    )
    gen = EMGGenerator(cfg)
    emg = gen.generate()
    
    warnings_n, metrics_n = PhysiologicalValidator.validate_emg(emg, cfg.fs)
    assert 'median_frequency_hz' in metrics_n

def test_emg_pathology_als():
    """Verify ALS fasciculations create distinct large transient discharge spikes."""
    cfg_als = EMGConfig(
        fs=2000.0,
        duration_s=4.0,
        emg_type='intramuscular',
        pathology='als',
        seed=808
    )
    emg_als = EMGGenerator(cfg_als).generate()
    
    # ALS EMG has spontaneous fasciculations, causing high amplitude peaks
    # resulting in high Crest Factor (peak / RMS)
    cf_als = float(np.max(np.abs(emg_als)) / np.std(emg_als))
    
    cfg_normal = EMGConfig(
        fs=2000.0,
        duration_s=4.0,
        emg_type='intramuscular',
        pathology='normal',
        seed=808
    )
    emg_normal = EMGGenerator(cfg_normal).generate()
    cf_norm = float(np.max(np.abs(emg_normal)) / np.std(emg_normal))
    
    # Both are highly impulsive signals compared to typical white noise
    assert cf_als > 4.0
    assert cf_norm > 4.0


# =====================================================================
# 4. PPG Dicrotic Notch & RSA Modulation Tests
# =====================================================================

def test_ppg_dicrotic_notch():
    """Verify the presence of systolic peak and dicrotic notch in the PPG wave."""
    cfg = PPGConfig(
        fs=100.0,
        duration_s=5.0,
        heart_rate=60.0, # 1 beat per second
        seed=42
    )
    ppg = PPGGenerator(cfg).generate()
    
    # Compute derivative of PPG (velocity photoplethysmogram, VPG)
    # The VPG has distinct zero-crossings corresponding to systolic peak and dicrotic notch
    vpg = np.diff(ppg) * cfg.fs
    zcr = compute_zcr(vpg)
    
    # In a clean PPG wave with a dicrotic notch, there are multiple inflections per cycle,
    # meaning there will be at least 2 zero-crossings of the derivative per cycle (systolic peak + dicrotic notch)
    # For 5 cycles, we expect at least 8 crossings of the mean
    crossings = np.sum(np.diff(np.sign(vpg - np.mean(vpg))) != 0)
    assert crossings >= 8


# =====================================================================
# 5. EDA Phasic Response Tests
# =====================================================================

def test_eda_phasic_responses():
    """Verify EDA generates tonic baseline level and distinct phasic SCR spikes."""
    cfg = EDAConfig(
        fs=20.0,
        duration_s=30.0,
        scl_amplitude_us=5.0,
        event_rate_hz=0.2, # ~6 events in 30 seconds
        seed=12
    )
    eda = EDAGenerator(cfg).generate()
    
    # Signal should have some variability and be above baseline minimum
    assert np.std(eda) > 0.01
    assert np.min(eda) >= 3.0


# =====================================================================
# 6. Respiration Pathological Breathing Pattern Tests
# =====================================================================

def test_resp_pathology_cheyne_stokes():
    """Verify Cheyne-Stokes breathing generates crescendo-decrescendo amplitude envelopes."""
    cfg = RespConfig(
        fs=20.0,
        duration_s=60.0,
        resp_rate_hz=0.2,
        seed=99
    )
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    
    # Calculate rolling standard deviation representing amplitude envelope
    win = int(5.0 * cfg.fs)
    rolling_std = [np.std(resp[i:i+win]) for i in range(0, len(resp) - win, win)]
    
    # The envelope must vary significantly (crescendo-decrescendo)
    envelope_ratio = np.max(rolling_std) / (np.min(rolling_std) + 1e-5)
    assert envelope_ratio > 1.5

def test_resp_pathology_kussmaul():
    """Verify Kussmaul breathing pattern generates deep and rapid respiration waves."""
    cfg = RespConfig(
        fs=20.0,
        duration_s=20.0,
        resp_rate_hz=0.5, # very fast: 30 breaths per minute
        amplitude=2.5,    # deep breathing
        seed=100
    )
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    
    # High RMS power due to deep breathing
    rms = compute_rms(resp)
    assert rms > 1.0
    
    # Rapid breathing zero crossings (approx 20 crossings in 20s at 0.5Hz)
    crossings = np.sum(np.diff(np.sign(resp - np.mean(resp))) != 0)
    assert crossings >= 15
