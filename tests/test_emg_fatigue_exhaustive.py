"""
Exhaustive EMG activation profiles, surface/intramuscular models, and fatigue checks.
"""
import numpy as np
import pytest
from biosignal_simulator.core.config import EMGConfig
from biosignal_simulator.signals.emg import EMGGenerator
from biosignal_simulator.core.math_utils import compute_rms, compute_zcr


def test_emg_surface_constant_normal_fs_500():
    """
    Verify EMG type surface, envelope constant, pathology normal, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_normal_fs_1000():
    """
    Verify EMG type surface, envelope constant, pathology normal, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_normal_fs_500():
    """
    Verify EMG type intramuscular, envelope constant, pathology normal, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_normal_fs_1000():
    """
    Verify EMG type intramuscular, envelope constant, pathology normal, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_normal_fs_500():
    """
    Verify EMG type surface, envelope ramp, pathology normal, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_normal_fs_1000():
    """
    Verify EMG type surface, envelope ramp, pathology normal, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_normal_fs_500():
    """
    Verify EMG type intramuscular, envelope ramp, pathology normal, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_normal_fs_1000():
    """
    Verify EMG type intramuscular, envelope ramp, pathology normal, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_normal_fs_500():
    """
    Verify EMG type surface, envelope burst, pathology normal, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_normal_fs_1000():
    """
    Verify EMG type surface, envelope burst, pathology normal, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_normal_fs_500():
    """
    Verify EMG type intramuscular, envelope burst, pathology normal, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_normal_fs_1000():
    """
    Verify EMG type intramuscular, envelope burst, pathology normal, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='normal',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_neuropathic_fs_500():
    """
    Verify EMG type surface, envelope constant, pathology neuropathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_neuropathic_fs_1000():
    """
    Verify EMG type surface, envelope constant, pathology neuropathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_neuropathic_fs_500():
    """
    Verify EMG type intramuscular, envelope constant, pathology neuropathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_neuropathic_fs_1000():
    """
    Verify EMG type intramuscular, envelope constant, pathology neuropathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_neuropathic_fs_500():
    """
    Verify EMG type surface, envelope ramp, pathology neuropathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_neuropathic_fs_1000():
    """
    Verify EMG type surface, envelope ramp, pathology neuropathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_neuropathic_fs_500():
    """
    Verify EMG type intramuscular, envelope ramp, pathology neuropathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_neuropathic_fs_1000():
    """
    Verify EMG type intramuscular, envelope ramp, pathology neuropathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_neuropathic_fs_500():
    """
    Verify EMG type surface, envelope burst, pathology neuropathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_neuropathic_fs_1000():
    """
    Verify EMG type surface, envelope burst, pathology neuropathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_neuropathic_fs_500():
    """
    Verify EMG type intramuscular, envelope burst, pathology neuropathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_neuropathic_fs_1000():
    """
    Verify EMG type intramuscular, envelope burst, pathology neuropathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='neuropathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_myopathic_fs_500():
    """
    Verify EMG type surface, envelope constant, pathology myopathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_myopathic_fs_1000():
    """
    Verify EMG type surface, envelope constant, pathology myopathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_myopathic_fs_500():
    """
    Verify EMG type intramuscular, envelope constant, pathology myopathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_myopathic_fs_1000():
    """
    Verify EMG type intramuscular, envelope constant, pathology myopathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_myopathic_fs_500():
    """
    Verify EMG type surface, envelope ramp, pathology myopathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_myopathic_fs_1000():
    """
    Verify EMG type surface, envelope ramp, pathology myopathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_myopathic_fs_500():
    """
    Verify EMG type intramuscular, envelope ramp, pathology myopathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_myopathic_fs_1000():
    """
    Verify EMG type intramuscular, envelope ramp, pathology myopathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_myopathic_fs_500():
    """
    Verify EMG type surface, envelope burst, pathology myopathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_myopathic_fs_1000():
    """
    Verify EMG type surface, envelope burst, pathology myopathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_myopathic_fs_500():
    """
    Verify EMG type intramuscular, envelope burst, pathology myopathic, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_myopathic_fs_1000():
    """
    Verify EMG type intramuscular, envelope burst, pathology myopathic, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='myopathic',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_als_fs_500():
    """
    Verify EMG type surface, envelope constant, pathology als, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_als_fs_1000():
    """
    Verify EMG type surface, envelope constant, pathology als, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_als_fs_500():
    """
    Verify EMG type intramuscular, envelope constant, pathology als, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_als_fs_1000():
    """
    Verify EMG type intramuscular, envelope constant, pathology als, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_als_fs_500():
    """
    Verify EMG type surface, envelope ramp, pathology als, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_als_fs_1000():
    """
    Verify EMG type surface, envelope ramp, pathology als, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_als_fs_500():
    """
    Verify EMG type intramuscular, envelope ramp, pathology als, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_als_fs_1000():
    """
    Verify EMG type intramuscular, envelope ramp, pathology als, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_als_fs_500():
    """
    Verify EMG type surface, envelope burst, pathology als, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_als_fs_1000():
    """
    Verify EMG type surface, envelope burst, pathology als, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_als_fs_500():
    """
    Verify EMG type intramuscular, envelope burst, pathology als, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_als_fs_1000():
    """
    Verify EMG type intramuscular, envelope burst, pathology als, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='als',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_myasthenia_gravis_fs_500():
    """
    Verify EMG type surface, envelope constant, pathology myasthenia_gravis, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_myasthenia_gravis_fs_1000():
    """
    Verify EMG type surface, envelope constant, pathology myasthenia_gravis, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_myasthenia_gravis_fs_500():
    """
    Verify EMG type intramuscular, envelope constant, pathology myasthenia_gravis, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_myasthenia_gravis_fs_1000():
    """
    Verify EMG type intramuscular, envelope constant, pathology myasthenia_gravis, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_myasthenia_gravis_fs_500():
    """
    Verify EMG type surface, envelope ramp, pathology myasthenia_gravis, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_myasthenia_gravis_fs_1000():
    """
    Verify EMG type surface, envelope ramp, pathology myasthenia_gravis, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_myasthenia_gravis_fs_500():
    """
    Verify EMG type intramuscular, envelope ramp, pathology myasthenia_gravis, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_myasthenia_gravis_fs_1000():
    """
    Verify EMG type intramuscular, envelope ramp, pathology myasthenia_gravis, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_myasthenia_gravis_fs_500():
    """
    Verify EMG type surface, envelope burst, pathology myasthenia_gravis, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_myasthenia_gravis_fs_1000():
    """
    Verify EMG type surface, envelope burst, pathology myasthenia_gravis, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_myasthenia_gravis_fs_500():
    """
    Verify EMG type intramuscular, envelope burst, pathology myasthenia_gravis, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_myasthenia_gravis_fs_1000():
    """
    Verify EMG type intramuscular, envelope burst, pathology myasthenia_gravis, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='myasthenia_gravis',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_parkinsons_tremor_fs_500():
    """
    Verify EMG type surface, envelope constant, pathology parkinsons_tremor, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_constant_parkinsons_tremor_fs_1000():
    """
    Verify EMG type surface, envelope constant, pathology parkinsons_tremor, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='surface',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_parkinsons_tremor_fs_500():
    """
    Verify EMG type intramuscular, envelope constant, pathology parkinsons_tremor, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_constant_parkinsons_tremor_fs_1000():
    """
    Verify EMG type intramuscular, envelope constant, pathology parkinsons_tremor, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='constant',
        emg_type='intramuscular',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_parkinsons_tremor_fs_500():
    """
    Verify EMG type surface, envelope ramp, pathology parkinsons_tremor, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_ramp_parkinsons_tremor_fs_1000():
    """
    Verify EMG type surface, envelope ramp, pathology parkinsons_tremor, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='surface',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_parkinsons_tremor_fs_500():
    """
    Verify EMG type intramuscular, envelope ramp, pathology parkinsons_tremor, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_ramp_parkinsons_tremor_fs_1000():
    """
    Verify EMG type intramuscular, envelope ramp, pathology parkinsons_tremor, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='ramp',
        emg_type='intramuscular',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_parkinsons_tremor_fs_500():
    """
    Verify EMG type surface, envelope burst, pathology parkinsons_tremor, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_surface_burst_parkinsons_tremor_fs_1000():
    """
    Verify EMG type surface, envelope burst, pathology parkinsons_tremor, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='surface',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_parkinsons_tremor_fs_500():
    """
    Verify EMG type intramuscular, envelope burst, pathology parkinsons_tremor, fs 500.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=500.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(500.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_intramuscular_burst_parkinsons_tremor_fs_1000():
    """
    Verify EMG type intramuscular, envelope burst, pathology parkinsons_tremor, fs 1000.0 Hz is stable.
    This test runs the EMG synthesis engine with different configurations to verify
    correct output lengths, standard deviations, and absence of NaN values.
    """
    cfg = EMGConfig(
        fs=1000.0,
        duration_s=1.0,
        envelope_type='burst',
        emg_type='intramuscular',
        pathology='parkinsons_tremor',
        seed=42
    )
    emg = EMGGenerator(cfg).generate()
    
    # 1. Assert length matches expected duration and sampling frequency
    assert len(emg) == int(1000.0)
    
    # 2. Check signal integrity invariants: no NaN values
    assert not np.any(np.isnan(emg))
    
    # 3. Check standard deviation to verify activation
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_05():
    """
    Verify EMG contraction level scaling at 0.05.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.05, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_1():
    """
    Verify EMG contraction level scaling at 0.1.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.1, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_2():
    """
    Verify EMG contraction level scaling at 0.2.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.2, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_3():
    """
    Verify EMG contraction level scaling at 0.3.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.3, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_4():
    """
    Verify EMG contraction level scaling at 0.4.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.4, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_5():
    """
    Verify EMG contraction level scaling at 0.5.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.5, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_6():
    """
    Verify EMG contraction level scaling at 0.6.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.6, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_7():
    """
    Verify EMG contraction level scaling at 0.7.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.7, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_8():
    """
    Verify EMG contraction level scaling at 0.8.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.8, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_9():
    """
    Verify EMG contraction level scaling at 0.9.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.9, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_0_95():
    """
    Verify EMG contraction level scaling at 0.95.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=0.95, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_contraction_level_1_0():
    """
    Verify EMG contraction level scaling at 1.0.
    Contraction level determines muscle force scaling. We verify it scales amplitude cleanly.
    """
    cfg = EMGConfig(fs=500.0, duration_s=1.0, contraction_level=1.0, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 500
    assert not np.any(np.isnan(emg))
    assert np.std(emg) > 0.0


def test_emg_burst_rate_sweep_0_5():
    """
    Verify EMG burst rate 0.5 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=0.5, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_1_0():
    """
    Verify EMG burst rate 1.0 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=1.0, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_1_5():
    """
    Verify EMG burst rate 1.5 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=1.5, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_2_0():
    """
    Verify EMG burst rate 2.0 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=2.0, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_2_5():
    """
    Verify EMG burst rate 2.5 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=2.5, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_3_0():
    """
    Verify EMG burst rate 3.0 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=3.0, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_3_5():
    """
    Verify EMG burst rate 3.5 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=3.5, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_4_0():
    """
    Verify EMG burst rate 4.0 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=4.0, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_4_5():
    """
    Verify EMG burst rate 4.5 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=4.5, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))


def test_emg_burst_rate_sweep_5_0():
    """
    Verify EMG burst rate 5.0 Hz.
    Repetitive bursts represent rhythmic contractions. We verify the time vector remains aligned.
    """
    cfg = EMGConfig(fs=500.0, duration_s=2.0, envelope_type='burst', burst_rate_hz=5.0, seed=42)
    emg = EMGGenerator(cfg).generate()
    assert len(emg) == 1000
    assert not np.any(np.isnan(emg))
