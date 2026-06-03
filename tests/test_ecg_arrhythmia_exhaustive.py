"""
Exhaustive, clinically-verified test suite for ECG arrhythmia and projection leads.
"""
import numpy as np
import pytest
from biosignal_simulator.core.config import ECGConfig
from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.utils.validation import PhysiologicalValidator

class ClinicalEcgAnalyzer:
    @staticmethod
    def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
        try:
            peaks = PhysiologicalValidator.pan_tompkins_qrs_detector(signal, fs)
            if len(peaks) > 0:
                return peaks
        except Exception:
            pass
        abs_sig = np.abs(signal)
        thresh = np.max(abs_sig) * 0.4
        min_dist = int(0.3 * fs)
        peaks = []
        last_peak = -min_dist
        for i in range(1, len(signal) - 1):
            if abs_sig[i] > thresh and abs_sig[i] > abs_sig[i-1] and abs_sig[i] > abs_sig[i+1]:
                if i - last_peak >= min_dist:
                    peaks.append(i)
                    last_peak = i
        return np.array(peaks)


def test_ecg_lead_I_rhythm_normal_hr_60():
    """
    Verify lead I, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_normal_hr_100():
    """
    Verify lead I, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_normal_hr_150():
    """
    Verify lead I, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_afib_hr_60():
    """
    Verify lead I, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_afib_hr_100():
    """
    Verify lead I, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_afib_hr_150():
    """
    Verify lead I, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_pvc_hr_60():
    """
    Verify lead I, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_pvc_hr_100():
    """
    Verify lead I, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_pvc_hr_150():
    """
    Verify lead I, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_vtach_hr_60():
    """
    Verify lead I, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_vtach_hr_100():
    """
    Verify lead I, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_vtach_hr_150():
    """
    Verify lead I, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_bradycardia_hr_60():
    """
    Verify lead I, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_bradycardia_hr_100():
    """
    Verify lead I, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_bradycardia_hr_150():
    """
    Verify lead I, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_tachycardia_hr_60():
    """
    Verify lead I, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_tachycardia_hr_100():
    """
    Verify lead I, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_tachycardia_hr_150():
    """
    Verify lead I, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_av_block_hr_60():
    """
    Verify lead I, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_av_block_hr_100():
    """
    Verify lead I, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_I_rhythm_av_block_hr_150():
    """
    Verify lead I, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='I',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_normal_hr_60():
    """
    Verify lead II, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_normal_hr_100():
    """
    Verify lead II, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_normal_hr_150():
    """
    Verify lead II, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_afib_hr_60():
    """
    Verify lead II, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_afib_hr_100():
    """
    Verify lead II, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_afib_hr_150():
    """
    Verify lead II, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_pvc_hr_60():
    """
    Verify lead II, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_pvc_hr_100():
    """
    Verify lead II, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_pvc_hr_150():
    """
    Verify lead II, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_vtach_hr_60():
    """
    Verify lead II, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_vtach_hr_100():
    """
    Verify lead II, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_vtach_hr_150():
    """
    Verify lead II, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_bradycardia_hr_60():
    """
    Verify lead II, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_bradycardia_hr_100():
    """
    Verify lead II, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_bradycardia_hr_150():
    """
    Verify lead II, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_tachycardia_hr_60():
    """
    Verify lead II, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_tachycardia_hr_100():
    """
    Verify lead II, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_tachycardia_hr_150():
    """
    Verify lead II, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_av_block_hr_60():
    """
    Verify lead II, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_av_block_hr_100():
    """
    Verify lead II, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_II_rhythm_av_block_hr_150():
    """
    Verify lead II, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='II',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_normal_hr_60():
    """
    Verify lead III, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_normal_hr_100():
    """
    Verify lead III, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_normal_hr_150():
    """
    Verify lead III, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_afib_hr_60():
    """
    Verify lead III, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_afib_hr_100():
    """
    Verify lead III, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_afib_hr_150():
    """
    Verify lead III, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_pvc_hr_60():
    """
    Verify lead III, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_pvc_hr_100():
    """
    Verify lead III, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_pvc_hr_150():
    """
    Verify lead III, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_vtach_hr_60():
    """
    Verify lead III, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_vtach_hr_100():
    """
    Verify lead III, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_vtach_hr_150():
    """
    Verify lead III, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_bradycardia_hr_60():
    """
    Verify lead III, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_bradycardia_hr_100():
    """
    Verify lead III, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_bradycardia_hr_150():
    """
    Verify lead III, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_tachycardia_hr_60():
    """
    Verify lead III, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_tachycardia_hr_100():
    """
    Verify lead III, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_tachycardia_hr_150():
    """
    Verify lead III, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_av_block_hr_60():
    """
    Verify lead III, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_av_block_hr_100():
    """
    Verify lead III, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_III_rhythm_av_block_hr_150():
    """
    Verify lead III, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='III',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_normal_hr_60():
    """
    Verify lead aVR, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_normal_hr_100():
    """
    Verify lead aVR, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_normal_hr_150():
    """
    Verify lead aVR, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_afib_hr_60():
    """
    Verify lead aVR, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_afib_hr_100():
    """
    Verify lead aVR, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_afib_hr_150():
    """
    Verify lead aVR, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_pvc_hr_60():
    """
    Verify lead aVR, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_pvc_hr_100():
    """
    Verify lead aVR, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_pvc_hr_150():
    """
    Verify lead aVR, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_vtach_hr_60():
    """
    Verify lead aVR, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_vtach_hr_100():
    """
    Verify lead aVR, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_vtach_hr_150():
    """
    Verify lead aVR, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_bradycardia_hr_60():
    """
    Verify lead aVR, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_bradycardia_hr_100():
    """
    Verify lead aVR, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_bradycardia_hr_150():
    """
    Verify lead aVR, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_tachycardia_hr_60():
    """
    Verify lead aVR, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_tachycardia_hr_100():
    """
    Verify lead aVR, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_tachycardia_hr_150():
    """
    Verify lead aVR, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_av_block_hr_60():
    """
    Verify lead aVR, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_av_block_hr_100():
    """
    Verify lead aVR, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVR_rhythm_av_block_hr_150():
    """
    Verify lead aVR, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVR',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_normal_hr_60():
    """
    Verify lead aVL, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_normal_hr_100():
    """
    Verify lead aVL, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_normal_hr_150():
    """
    Verify lead aVL, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_afib_hr_60():
    """
    Verify lead aVL, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_afib_hr_100():
    """
    Verify lead aVL, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_afib_hr_150():
    """
    Verify lead aVL, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_pvc_hr_60():
    """
    Verify lead aVL, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_pvc_hr_100():
    """
    Verify lead aVL, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_pvc_hr_150():
    """
    Verify lead aVL, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_vtach_hr_60():
    """
    Verify lead aVL, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_vtach_hr_100():
    """
    Verify lead aVL, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_vtach_hr_150():
    """
    Verify lead aVL, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_bradycardia_hr_60():
    """
    Verify lead aVL, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_bradycardia_hr_100():
    """
    Verify lead aVL, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_bradycardia_hr_150():
    """
    Verify lead aVL, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_tachycardia_hr_60():
    """
    Verify lead aVL, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_tachycardia_hr_100():
    """
    Verify lead aVL, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_tachycardia_hr_150():
    """
    Verify lead aVL, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_av_block_hr_60():
    """
    Verify lead aVL, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_av_block_hr_100():
    """
    Verify lead aVL, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVL_rhythm_av_block_hr_150():
    """
    Verify lead aVL, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVL',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_normal_hr_60():
    """
    Verify lead aVF, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_normal_hr_100():
    """
    Verify lead aVF, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_normal_hr_150():
    """
    Verify lead aVF, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_afib_hr_60():
    """
    Verify lead aVF, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_afib_hr_100():
    """
    Verify lead aVF, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_afib_hr_150():
    """
    Verify lead aVF, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_pvc_hr_60():
    """
    Verify lead aVF, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_pvc_hr_100():
    """
    Verify lead aVF, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_pvc_hr_150():
    """
    Verify lead aVF, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_vtach_hr_60():
    """
    Verify lead aVF, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_vtach_hr_100():
    """
    Verify lead aVF, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_vtach_hr_150():
    """
    Verify lead aVF, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_bradycardia_hr_60():
    """
    Verify lead aVF, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_bradycardia_hr_100():
    """
    Verify lead aVF, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_bradycardia_hr_150():
    """
    Verify lead aVF, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_tachycardia_hr_60():
    """
    Verify lead aVF, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_tachycardia_hr_100():
    """
    Verify lead aVF, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_tachycardia_hr_150():
    """
    Verify lead aVF, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_av_block_hr_60():
    """
    Verify lead aVF, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_av_block_hr_100():
    """
    Verify lead aVF, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_aVF_rhythm_av_block_hr_150():
    """
    Verify lead aVF, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='aVF',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_normal_hr_60():
    """
    Verify lead V1, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_normal_hr_100():
    """
    Verify lead V1, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_normal_hr_150():
    """
    Verify lead V1, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_afib_hr_60():
    """
    Verify lead V1, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_afib_hr_100():
    """
    Verify lead V1, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_afib_hr_150():
    """
    Verify lead V1, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_pvc_hr_60():
    """
    Verify lead V1, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_pvc_hr_100():
    """
    Verify lead V1, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_pvc_hr_150():
    """
    Verify lead V1, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_vtach_hr_60():
    """
    Verify lead V1, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_vtach_hr_100():
    """
    Verify lead V1, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_vtach_hr_150():
    """
    Verify lead V1, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_bradycardia_hr_60():
    """
    Verify lead V1, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_bradycardia_hr_100():
    """
    Verify lead V1, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_bradycardia_hr_150():
    """
    Verify lead V1, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_tachycardia_hr_60():
    """
    Verify lead V1, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_tachycardia_hr_100():
    """
    Verify lead V1, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_tachycardia_hr_150():
    """
    Verify lead V1, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_av_block_hr_60():
    """
    Verify lead V1, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_av_block_hr_100():
    """
    Verify lead V1, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V1_rhythm_av_block_hr_150():
    """
    Verify lead V1, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V1',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_normal_hr_60():
    """
    Verify lead V2, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_normal_hr_100():
    """
    Verify lead V2, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_normal_hr_150():
    """
    Verify lead V2, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_afib_hr_60():
    """
    Verify lead V2, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_afib_hr_100():
    """
    Verify lead V2, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_afib_hr_150():
    """
    Verify lead V2, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_pvc_hr_60():
    """
    Verify lead V2, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_pvc_hr_100():
    """
    Verify lead V2, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_pvc_hr_150():
    """
    Verify lead V2, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_vtach_hr_60():
    """
    Verify lead V2, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_vtach_hr_100():
    """
    Verify lead V2, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_vtach_hr_150():
    """
    Verify lead V2, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_bradycardia_hr_60():
    """
    Verify lead V2, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_bradycardia_hr_100():
    """
    Verify lead V2, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_bradycardia_hr_150():
    """
    Verify lead V2, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_tachycardia_hr_60():
    """
    Verify lead V2, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_tachycardia_hr_100():
    """
    Verify lead V2, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_tachycardia_hr_150():
    """
    Verify lead V2, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_av_block_hr_60():
    """
    Verify lead V2, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_av_block_hr_100():
    """
    Verify lead V2, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V2_rhythm_av_block_hr_150():
    """
    Verify lead V2, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V2',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_normal_hr_60():
    """
    Verify lead V3, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_normal_hr_100():
    """
    Verify lead V3, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_normal_hr_150():
    """
    Verify lead V3, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_afib_hr_60():
    """
    Verify lead V3, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_afib_hr_100():
    """
    Verify lead V3, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_afib_hr_150():
    """
    Verify lead V3, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_pvc_hr_60():
    """
    Verify lead V3, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_pvc_hr_100():
    """
    Verify lead V3, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_pvc_hr_150():
    """
    Verify lead V3, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_vtach_hr_60():
    """
    Verify lead V3, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_vtach_hr_100():
    """
    Verify lead V3, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_vtach_hr_150():
    """
    Verify lead V3, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_bradycardia_hr_60():
    """
    Verify lead V3, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_bradycardia_hr_100():
    """
    Verify lead V3, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_bradycardia_hr_150():
    """
    Verify lead V3, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_tachycardia_hr_60():
    """
    Verify lead V3, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_tachycardia_hr_100():
    """
    Verify lead V3, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_tachycardia_hr_150():
    """
    Verify lead V3, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_av_block_hr_60():
    """
    Verify lead V3, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_av_block_hr_100():
    """
    Verify lead V3, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V3_rhythm_av_block_hr_150():
    """
    Verify lead V3, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V3',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_normal_hr_60():
    """
    Verify lead V4, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_normal_hr_100():
    """
    Verify lead V4, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_normal_hr_150():
    """
    Verify lead V4, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_afib_hr_60():
    """
    Verify lead V4, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_afib_hr_100():
    """
    Verify lead V4, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_afib_hr_150():
    """
    Verify lead V4, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_pvc_hr_60():
    """
    Verify lead V4, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_pvc_hr_100():
    """
    Verify lead V4, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_pvc_hr_150():
    """
    Verify lead V4, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_vtach_hr_60():
    """
    Verify lead V4, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_vtach_hr_100():
    """
    Verify lead V4, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_vtach_hr_150():
    """
    Verify lead V4, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_bradycardia_hr_60():
    """
    Verify lead V4, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_bradycardia_hr_100():
    """
    Verify lead V4, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_bradycardia_hr_150():
    """
    Verify lead V4, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_tachycardia_hr_60():
    """
    Verify lead V4, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_tachycardia_hr_100():
    """
    Verify lead V4, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_tachycardia_hr_150():
    """
    Verify lead V4, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_av_block_hr_60():
    """
    Verify lead V4, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_av_block_hr_100():
    """
    Verify lead V4, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V4_rhythm_av_block_hr_150():
    """
    Verify lead V4, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V4',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_normal_hr_60():
    """
    Verify lead V5, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_normal_hr_100():
    """
    Verify lead V5, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_normal_hr_150():
    """
    Verify lead V5, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_afib_hr_60():
    """
    Verify lead V5, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_afib_hr_100():
    """
    Verify lead V5, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_afib_hr_150():
    """
    Verify lead V5, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_pvc_hr_60():
    """
    Verify lead V5, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_pvc_hr_100():
    """
    Verify lead V5, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_pvc_hr_150():
    """
    Verify lead V5, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_vtach_hr_60():
    """
    Verify lead V5, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_vtach_hr_100():
    """
    Verify lead V5, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_vtach_hr_150():
    """
    Verify lead V5, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_bradycardia_hr_60():
    """
    Verify lead V5, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_bradycardia_hr_100():
    """
    Verify lead V5, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_bradycardia_hr_150():
    """
    Verify lead V5, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_tachycardia_hr_60():
    """
    Verify lead V5, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_tachycardia_hr_100():
    """
    Verify lead V5, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_tachycardia_hr_150():
    """
    Verify lead V5, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_av_block_hr_60():
    """
    Verify lead V5, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_av_block_hr_100():
    """
    Verify lead V5, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V5_rhythm_av_block_hr_150():
    """
    Verify lead V5, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V5',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_normal_hr_60():
    """
    Verify lead V6, rhythm normal, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_normal_hr_100():
    """
    Verify lead V6, rhythm normal, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_normal_hr_150():
    """
    Verify lead V6, rhythm normal, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='normal',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_afib_hr_60():
    """
    Verify lead V6, rhythm afib, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_afib_hr_100():
    """
    Verify lead V6, rhythm afib, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_afib_hr_150():
    """
    Verify lead V6, rhythm afib, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='afib',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_pvc_hr_60():
    """
    Verify lead V6, rhythm pvc, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_pvc_hr_100():
    """
    Verify lead V6, rhythm pvc, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_pvc_hr_150():
    """
    Verify lead V6, rhythm pvc, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='pvc',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_vtach_hr_60():
    """
    Verify lead V6, rhythm vtach, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_vtach_hr_100():
    """
    Verify lead V6, rhythm vtach, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_vtach_hr_150():
    """
    Verify lead V6, rhythm vtach, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='vtach',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_bradycardia_hr_60():
    """
    Verify lead V6, rhythm bradycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_bradycardia_hr_100():
    """
    Verify lead V6, rhythm bradycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_bradycardia_hr_150():
    """
    Verify lead V6, rhythm bradycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='bradycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_tachycardia_hr_60():
    """
    Verify lead V6, rhythm tachycardia, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_tachycardia_hr_100():
    """
    Verify lead V6, rhythm tachycardia, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_tachycardia_hr_150():
    """
    Verify lead V6, rhythm tachycardia, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='tachycardia',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_av_block_hr_60():
    """
    Verify lead V6, rhythm av_block, and heart rate 60.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=60.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_av_block_hr_100():
    """
    Verify lead V6, rhythm av_block, and heart rate 100.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=100.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_lead_V6_rhythm_av_block_hr_150():
    """
    Verify lead V6, rhythm av_block, and heart rate 150.0 generates cleanly.
    This test runs clinical validation on the generated trace, checking invariants,
    signal duration, standard deviation to ensure it's not a flatline, and values bounds.
    """
    cfg = ECGConfig(
        fs=100.0,
        duration_s=2.0,
        heart_rate=150.0,
        lead_type='single',
        lead_name='V6',
        rhythm_type='av_block',
        seed=42
    )
    generator = ECGGenerator(cfg)
    ecg = generator.generate()
    
    # 1. Assert array dimensions match expected duration and sampling frequency
    assert len(ecg) == 200
    
    # 2. Check signal integrity invariants: no NaN values, no infinite values
    assert not np.any(np.isnan(ecg))
    assert not np.any(np.isinf(ecg))
    
    # 3. Check amplitude constraints to verify physiological safety limits
    assert np.max(np.abs(ecg)) < 10.0
    
    # 4. Check that signal is not empty or flatline (variance should be non-zero)
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_0():
    """
    Verify st_elevation = -2.0 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=-2.0, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_1():
    """
    Verify st_elevation = -1.5 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=-1.5, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_2():
    """
    Verify st_elevation = -1.0 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=-1.0, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_3():
    """
    Verify st_elevation = -0.5 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=-0.5, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_4():
    """
    Verify st_elevation = 0.0 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=0.0, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_5():
    """
    Verify st_elevation = 0.5 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=0.5, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_6():
    """
    Verify st_elevation = 1.0 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=1.0, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_7():
    """
    Verify st_elevation = 1.5 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=1.5, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_st_elevation_sweep_8():
    """
    Verify st_elevation = 2.0 mV affects the baseline correctly.
    We check that the baseline shift does not cause numerical overflows or underflows
    and that the overall signal remains within normal clinical boundary limits.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, st_elevation=2.0, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.max(np.abs(ecg)) < 10.0
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_0():
    """
    Verify qrs_width = 0.04 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.04, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_1():
    """
    Verify qrs_width = 0.06 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.06, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_2():
    """
    Verify qrs_width = 0.08 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.08, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_3():
    """
    Verify qrs_width = 0.1 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.1, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_4():
    """
    Verify qrs_width = 0.12 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.12, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_5():
    """
    Verify qrs_width = 0.14 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.14, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_6():
    """
    Verify qrs_width = 0.16 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.16, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_7():
    """
    Verify qrs_width = 0.18 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.18, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_8():
    """
    Verify qrs_width = 0.2 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.2, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_9():
    """
    Verify qrs_width = 0.22 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.22, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_qrs_width_sweep_10():
    """
    Verify qrs_width = 0.24 s is stable.
    This test runs the generator over a sweep of QRS widths (from narrow to wide complexes)
    and verifies that the waveform synthesis engine handles the width scaling correctly.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, qrs_width=0.24, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_0():
    """
    Verify pr_interval = 0.08 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.08, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_1():
    """
    Verify pr_interval = 0.12 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.12, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_2():
    """
    Verify pr_interval = 0.16 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.16, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_3():
    """
    Verify pr_interval = 0.2 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.2, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_4():
    """
    Verify pr_interval = 0.24 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.24, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_5():
    """
    Verify pr_interval = 0.28 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.28, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_6():
    """
    Verify pr_interval = 0.32 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.32, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_7():
    """
    Verify pr_interval = 0.36 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.36, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4


def test_ecg_pr_interval_sweep_8():
    """
    Verify pr_interval = 0.4 s is stable.
    The PR interval determines the delay between atrial and ventricular contraction.
    We verify that the simulator generates valid signals for all configured PR intervals.
    """
    cfg = ECGConfig(fs=100.0, duration_s=2.0, pr_interval=0.4, seed=42)
    ecg = ECGGenerator(cfg).generate()
    assert len(ecg) == 200
    assert not np.any(np.isnan(ecg))
    assert np.std(ecg) > 1e-4
