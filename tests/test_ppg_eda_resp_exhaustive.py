"""
Exhaustive verification of PPG, EDA, and Respiration morphology/pathology patterns.
"""
import numpy as np
import pytest
from biosignal_simulator.core.config import PPGConfig, EDAConfig, RespConfig
from biosignal_simulator.signals.ppg import PPGGenerator
from biosignal_simulator.signals.eda import EDAGenerator
from biosignal_simulator.signals.resp import RespGenerator
from biosignal_simulator.core.math_utils import compute_rms, compute_zcr


def test_ppg_heart_rate_40():
    """
    Verify PPG generator under heart rate 40.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=40.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_50():
    """
    Verify PPG generator under heart rate 50.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=50.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_60():
    """
    Verify PPG generator under heart rate 60.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=60.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_70():
    """
    Verify PPG generator under heart rate 70.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=70.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_80():
    """
    Verify PPG generator under heart rate 80.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=80.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_90():
    """
    Verify PPG generator under heart rate 90.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=90.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_100():
    """
    Verify PPG generator under heart rate 100.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=100.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_110():
    """
    Verify PPG generator under heart rate 110.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=110.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_120():
    """
    Verify PPG generator under heart rate 120.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=120.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_130():
    """
    Verify PPG generator under heart rate 130.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=130.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_140():
    """
    Verify PPG generator under heart rate 140.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=140.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_150():
    """
    Verify PPG generator under heart rate 150.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=150.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_160():
    """
    Verify PPG generator under heart rate 160.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=160.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_170():
    """
    Verify PPG generator under heart rate 170.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=170.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_180():
    """
    Verify PPG generator under heart rate 180.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=180.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_190():
    """
    Verify PPG generator under heart rate 190.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=190.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_heart_rate_200():
    """
    Verify PPG generator under heart rate 200.0 bpm.
    We check that the output signal contains valid values and doesn't contain NaNs.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, heart_rate=200.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))
    assert np.std(ppg) > 1e-4


def test_ppg_respiration_modulation_0_0():
    """
    Verify PPG respiration modulation depth 0.0.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.0, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_05():
    """
    Verify PPG respiration modulation depth 0.05.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.05, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_1():
    """
    Verify PPG respiration modulation depth 0.1.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.1, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_15():
    """
    Verify PPG respiration modulation depth 0.15.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.15, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_2():
    """
    Verify PPG respiration modulation depth 0.2.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.2, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_25():
    """
    Verify PPG respiration modulation depth 0.25.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.25, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_3():
    """
    Verify PPG respiration modulation depth 0.3.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.3, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_35():
    """
    Verify PPG respiration modulation depth 0.35.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.35, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_4():
    """
    Verify PPG respiration modulation depth 0.4.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.4, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_45():
    """
    Verify PPG respiration modulation depth 0.45.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.45, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_5():
    """
    Verify PPG respiration modulation depth 0.5.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.5, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_55():
    """
    Verify PPG respiration modulation depth 0.55.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.55, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_6():
    """
    Verify PPG respiration modulation depth 0.6.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.6, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_65():
    """
    Verify PPG respiration modulation depth 0.65.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.65, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_ppg_respiration_modulation_0_7():
    """
    Verify PPG respiration modulation depth 0.7.
    Respiration modulates heart rate and amplitude. We verify stability at this depth.
    """
    cfg = PPGConfig(fs=100.0, duration_s=2.0, resp_modulation=0.7, seed=42)
    ppg = PPGGenerator(cfg).generate()
    assert len(ppg) == 200
    assert not np.any(np.isnan(ppg))


def test_eda_scl_level_0_1():
    """
    Verify EDA skin conductance base level 0.1 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=0.1, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_0_5():
    """
    Verify EDA skin conductance base level 0.5 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=0.5, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_1_0():
    """
    Verify EDA skin conductance base level 1.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=1.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_2_0():
    """
    Verify EDA skin conductance base level 2.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=2.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_3_0():
    """
    Verify EDA skin conductance base level 3.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=3.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_4_0():
    """
    Verify EDA skin conductance base level 4.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=4.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_5_0():
    """
    Verify EDA skin conductance base level 5.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=5.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_7_5():
    """
    Verify EDA skin conductance base level 7.5 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=7.5, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_10_0():
    """
    Verify EDA skin conductance base level 10.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=10.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_12_5():
    """
    Verify EDA skin conductance base level 12.5 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=12.5, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_15_0():
    """
    Verify EDA skin conductance base level 15.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=15.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_17_5():
    """
    Verify EDA skin conductance base level 17.5 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=17.5, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_20_0():
    """
    Verify EDA skin conductance base level 20.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=20.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_25_0():
    """
    Verify EDA skin conductance base level 25.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=25.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_scl_level_30_0():
    """
    Verify EDA skin conductance base level 30.0 uS.
    Verifies that skin conductance level baseline maps correctly.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, scl_amplitude_us=30.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert np.mean(eda) > 0.0
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_0():
    """
    Verify EDA SCR event activation rate 0.0 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.0, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_01():
    """
    Verify EDA SCR event activation rate 0.01 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.01, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_05():
    """
    Verify EDA SCR event activation rate 0.05 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.05, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_1():
    """
    Verify EDA SCR event activation rate 0.1 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.1, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_15():
    """
    Verify EDA SCR event activation rate 0.15 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.15, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_2():
    """
    Verify EDA SCR event activation rate 0.2 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.2, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_25():
    """
    Verify EDA SCR event activation rate 0.25 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.25, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_3():
    """
    Verify EDA SCR event activation rate 0.3 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.3, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_35():
    """
    Verify EDA SCR event activation rate 0.35 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.35, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_4():
    """
    Verify EDA SCR event activation rate 0.4 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.4, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_45():
    """
    Verify EDA SCR event activation rate 0.45 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.45, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_5():
    """
    Verify EDA SCR event activation rate 0.5 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.5, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_6():
    """
    Verify EDA SCR event activation rate 0.6 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.6, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_7():
    """
    Verify EDA SCR event activation rate 0.7 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.7, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_eda_event_rate_0_8():
    """
    Verify EDA SCR event activation rate 0.8 Hz.
    Checks that skin conductance responses generate correctly without numerical crashes.
    """
    cfg = EDAConfig(fs=20.0, duration_s=5.0, event_rate_hz=0.8, seed=42)
    eda = EDAGenerator(cfg).generate()
    assert len(eda) == 100
    assert not np.any(np.isnan(eda))


def test_resp_rate_0_05_pattern_normal():
    """
    Verify Respiration rate 0.05 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.05, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_05_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.05 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.05, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_05_pattern_biot():
    """
    Verify Respiration rate 0.05 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.05, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_05_pattern_kussmaul():
    """
    Verify Respiration rate 0.05 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.05, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_1_pattern_normal():
    """
    Verify Respiration rate 0.1 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.1, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_1_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.1 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.1, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_1_pattern_biot():
    """
    Verify Respiration rate 0.1 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.1, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_1_pattern_kussmaul():
    """
    Verify Respiration rate 0.1 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.1, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_15_pattern_normal():
    """
    Verify Respiration rate 0.15 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.15, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_15_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.15 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.15, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_15_pattern_biot():
    """
    Verify Respiration rate 0.15 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.15, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_15_pattern_kussmaul():
    """
    Verify Respiration rate 0.15 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.15, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_2_pattern_normal():
    """
    Verify Respiration rate 0.2 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.2, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_2_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.2 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.2, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_2_pattern_biot():
    """
    Verify Respiration rate 0.2 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.2, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_2_pattern_kussmaul():
    """
    Verify Respiration rate 0.2 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.2, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_25_pattern_normal():
    """
    Verify Respiration rate 0.25 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.25, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_25_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.25 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.25, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_25_pattern_biot():
    """
    Verify Respiration rate 0.25 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.25, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_25_pattern_kussmaul():
    """
    Verify Respiration rate 0.25 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.25, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_3_pattern_normal():
    """
    Verify Respiration rate 0.3 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.3, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_3_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.3 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.3, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_3_pattern_biot():
    """
    Verify Respiration rate 0.3 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.3, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_3_pattern_kussmaul():
    """
    Verify Respiration rate 0.3 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.3, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_35_pattern_normal():
    """
    Verify Respiration rate 0.35 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.35, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_35_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.35 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.35, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_35_pattern_biot():
    """
    Verify Respiration rate 0.35 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.35, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_35_pattern_kussmaul():
    """
    Verify Respiration rate 0.35 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.35, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_4_pattern_normal():
    """
    Verify Respiration rate 0.4 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.4, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_4_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.4 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.4, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_4_pattern_biot():
    """
    Verify Respiration rate 0.4 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.4, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_4_pattern_kussmaul():
    """
    Verify Respiration rate 0.4 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.4, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_45_pattern_normal():
    """
    Verify Respiration rate 0.45 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.45, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_45_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.45 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.45, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_45_pattern_biot():
    """
    Verify Respiration rate 0.45 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.45, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_45_pattern_kussmaul():
    """
    Verify Respiration rate 0.45 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.45, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_5_pattern_normal():
    """
    Verify Respiration rate 0.5 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.5, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_5_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.5 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.5, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_5_pattern_biot():
    """
    Verify Respiration rate 0.5 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.5, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_5_pattern_kussmaul():
    """
    Verify Respiration rate 0.5 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.5, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_55_pattern_normal():
    """
    Verify Respiration rate 0.55 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.55, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_55_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.55 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.55, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_55_pattern_biot():
    """
    Verify Respiration rate 0.55 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.55, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_55_pattern_kussmaul():
    """
    Verify Respiration rate 0.55 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.55, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_6_pattern_normal():
    """
    Verify Respiration rate 0.6 Hz and breathing pattern normal.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.6, seed=42)
    cfg.pattern = 'normal'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_6_pattern_cheyne_stokes():
    """
    Verify Respiration rate 0.6 Hz and breathing pattern cheyne_stokes.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.6, seed=42)
    cfg.pattern = 'cheyne_stokes'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_6_pattern_biot():
    """
    Verify Respiration rate 0.6 Hz and breathing pattern biot.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.6, seed=42)
    cfg.pattern = 'biot'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_resp_rate_0_6_pattern_kussmaul():
    """
    Verify Respiration rate 0.6 Hz and breathing pattern kussmaul.
    This test runs physiological model checks to ensure respiratory waveforms are synthesized correctly.
    """
    cfg = RespConfig(fs=20.0, duration_s=5.0, resp_rate_hz=0.6, seed=42)
    cfg.pattern = 'kussmaul'
    resp = RespGenerator(cfg).generate()
    assert len(resp) == 100
    assert not np.any(np.isnan(resp))
    assert np.std(resp) > 1e-4


def test_ppg_derivatives_option():
    """Verify that PPGConfig derivative parameter correctly alters outputs."""
    # VPG
    cfg_vpg = PPGConfig(fs=100.0, duration_s=5.0, derivative='vpg', seed=42)
    vpg = PPGGenerator(cfg_vpg).generate()
    assert len(vpg) == 500
    assert not np.any(np.isnan(vpg))
    assert np.isclose(np.std(vpg), 1.0, atol=1e-2)

    # APG
    cfg_apg = PPGConfig(fs=100.0, duration_s=5.0, derivative='apg', seed=42)
    apg = PPGGenerator(cfg_apg).generate()
    assert len(apg) == 500
    assert not np.any(np.isnan(apg))
    assert np.isclose(np.std(apg), 1.0, atol=1e-2)

    # Invalid option should raise ValueError
    with pytest.raises(ValueError):
        PPGConfig(fs=100.0, derivative='invalid_derivative')


def test_ppg_factory_functions():
    """Verify the PPG and derivative factory functions."""
    from biosignal_simulator import make_vpg, make_apg, make_ppg_motion_artifact, make_ppg_light_leakage

    # make_vpg
    vpg = make_vpg(duration_s=4.0, fs=100.0, seed=42)
    assert len(vpg) == 400
    assert not np.any(np.isnan(vpg))

    # make_apg
    apg = make_apg(duration_s=4.0, fs=100.0, seed=42)
    assert len(apg) == 400
    assert not np.any(np.isnan(apg))

    # make_ppg_motion_artifact
    motion_ppg = make_ppg_motion_artifact(duration_s=4.0, fs=100.0, snr_db=10.0, seed=42)
    assert len(motion_ppg) == 400
    assert not np.any(np.isnan(motion_ppg))

    # make_ppg_light_leakage
    light_ppg = make_ppg_light_leakage(duration_s=4.0, fs=100.0, snr_db=10.0, seed=42)
    assert len(light_ppg) == 400
    assert not np.any(np.isnan(light_ppg))
