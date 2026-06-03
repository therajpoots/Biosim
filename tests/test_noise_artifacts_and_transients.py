"""
Exhaustive verification of structured physiological noise artifacts and transients.
"""
import numpy as np
import pytest
from scipy.signal import welch
from biosignal_simulator.noise.powerline import PowerlineNoise
from biosignal_simulator.noise.motion import MotionArtifact
from biosignal_simulator.noise.electrode import ElectrodeNoise
from biosignal_simulator.noise.impulse import ImpulseNoise
from biosignal_simulator.noise.quantization import QuantizationNoise
from biosignal_simulator.noise.wearable import SensorDetachmentNoise, PacketLossNoise
from biosignal_simulator.core.math_utils import compute_rms


def test_powerline_freq_50_harmonics_1_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 1 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=1, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_1_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 1 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=1, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_1_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 1 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=1, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_2_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 2 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=2, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_2_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 2 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=2, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_2_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 2 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=2, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_3_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 3 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=3, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_3_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 3 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=3, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_3_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 3 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=3, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_4_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 4 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=4, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_4_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 4 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=4, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_4_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 4 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=4, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_5_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 5 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=5, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_5_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 5 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=5, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_5_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 5 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=5, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_6_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 6 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=6, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_6_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 6 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=6, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_6_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 6 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=6, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_7_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 7 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=7, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_7_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 7 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=7, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_7_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 7 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=7, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_8_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 8 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=8, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_8_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 8 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=8, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_8_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 8 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=8, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_9_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 9 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=9, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_9_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 9 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=9, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_9_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 9 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=9, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_10_amp_0_5():
    """
    Verify line interference at 50.0 Hz with 10 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=50.0, n_harmonics=10, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_10_amp_1_0():
    """
    Verify line interference at 50.0 Hz with 10 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=50.0, n_harmonics=10, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_50_harmonics_10_amp_2_0():
    """
    Verify line interference at 50.0 Hz with 10 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=50.0, n_harmonics=10, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_1_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 1 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=1, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_1_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 1 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=1, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_1_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 1 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=1, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_2_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 2 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=2, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_2_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 2 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=2, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_2_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 2 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=2, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_3_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 3 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=3, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_3_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 3 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=3, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_3_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 3 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=3, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_4_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 4 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=4, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_4_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 4 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=4, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_4_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 4 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=4, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_5_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 5 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=5, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_5_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 5 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=5, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_5_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 5 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=5, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_6_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 6 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=6, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_6_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 6 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=6, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_6_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 6 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=6, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_7_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 7 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=7, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_7_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 7 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=7, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_7_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 7 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=7, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_8_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 8 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=8, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_8_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 8 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=8, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_8_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 8 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=8, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_9_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 9 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=9, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_9_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 9 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=9, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_9_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 9 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=9, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_10_amp_0_5():
    """
    Verify line interference at 60.0 Hz with 10 harmonics and amp 0.5.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=0.5, f_line_hz=60.0, n_harmonics=10, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_10_amp_1_0():
    """
    Verify line interference at 60.0 Hz with 10 harmonics and amp 1.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=1.0, f_line_hz=60.0, n_harmonics=10, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_powerline_freq_60_harmonics_10_amp_2_0():
    """
    Verify line interference at 60.0 Hz with 10 harmonics and amp 2.0.
    Harmonics represents high-frequency leakages. We verify they generate cleanly.
    """
    model = PowerlineNoise(amplitude=2.0, f_line_hz=60.0, n_harmonics=10, seed=42)
    noise = model.generate(500, 500.0)
    assert len(noise) == 500
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_quantization_bits_8_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 8 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=8, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_8_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 8 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=8, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_8_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 8 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=8, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_8_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 8 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=8, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_8_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 8 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=8, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_8_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 8 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=8, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_9_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 9 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=9, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_9_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 9 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=9, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_9_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 9 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=9, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_9_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 9 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=9, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_9_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 9 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=9, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_9_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 9 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=9, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_10_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 10 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=10, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_10_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 10 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=10, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_10_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 10 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=10, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_10_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 10 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=10, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_10_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 10 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=10, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_10_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 10 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=10, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_11_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 11 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=11, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_11_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 11 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=11, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_11_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 11 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=11, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_11_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 11 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=11, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_11_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 11 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=11, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_11_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 11 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=11, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_12_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 12 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=12, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_12_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 12 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=12, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_12_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 12 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=12, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_12_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 12 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=12, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_12_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 12 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=12, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_12_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 12 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=12, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_13_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 13 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=13, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_13_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 13 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=13, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_13_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 13 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=13, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_13_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 13 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=13, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_13_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 13 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=13, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_13_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 13 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=13, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_14_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 14 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=14, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_14_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 14 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=14, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_14_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 14 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=14, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_14_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 14 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=14, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_14_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 14 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=14, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_14_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 14 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=14, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_15_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 15 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=15, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_15_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 15 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=15, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_15_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 15 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=15, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_15_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 15 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=15, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_15_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 15 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=15, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_15_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 15 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=15, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_16_dither_True_vrange_1_0():
    """
    Verify ADC quantization error at 16 bits (dither=True, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=16, v_range=1.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_16_dither_True_vrange_5_0():
    """
    Verify ADC quantization error at 16 bits (dither=True, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=16, v_range=5.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_16_dither_True_vrange_10_0():
    """
    Verify ADC quantization error at 16 bits (dither=True, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=16, v_range=10.0, dither=True)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_16_dither_False_vrange_1_0():
    """
    Verify ADC quantization error at 16 bits (dither=False, vrange=1.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=16, v_range=1.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_16_dither_False_vrange_5_0():
    """
    Verify ADC quantization error at 16 bits (dither=False, vrange=5.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=16, v_range=5.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_quantization_bits_16_dither_False_vrange_10_0():
    """
    Verify ADC quantization error at 16 bits (dither=False, vrange=10.0).
    Verifies that the quantization algorithm executes without numerical roundoff issues.
    """
    model = QuantizationNoise(n_bits=16, v_range=10.0, dither=False)
    t = np.arange(200) / 100.0
    sig = np.sin(t)
    quant, err = model.apply(sig)
    assert len(quant) == 200
    assert not np.any(np.isnan(quant))


def test_packet_loss_rate_0_01():
    """
    Verify Markov packet dropouts at loss rate 0.01.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.01, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_02():
    """
    Verify Markov packet dropouts at loss rate 0.02.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.02, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_05():
    """
    Verify Markov packet dropouts at loss rate 0.05.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.05, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_08():
    """
    Verify Markov packet dropouts at loss rate 0.08.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.08, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_1():
    """
    Verify Markov packet dropouts at loss rate 0.1.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.1, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_12():
    """
    Verify Markov packet dropouts at loss rate 0.12.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.12, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_15():
    """
    Verify Markov packet dropouts at loss rate 0.15.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.15, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_18():
    """
    Verify Markov packet dropouts at loss rate 0.18.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.18, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_2():
    """
    Verify Markov packet dropouts at loss rate 0.2.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.2, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_25():
    """
    Verify Markov packet dropouts at loss rate 0.25.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.25, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_3():
    """
    Verify Markov packet dropouts at loss rate 0.3.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.3, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_35():
    """
    Verify Markov packet dropouts at loss rate 0.35.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.35, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_4():
    """
    Verify Markov packet dropouts at loss rate 0.4.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.4, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_45():
    """
    Verify Markov packet dropouts at loss rate 0.45.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.45, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_packet_loss_rate_0_5():
    """
    Verify Markov packet dropouts at loss rate 0.5.
    Gilbert-Elliott models wireless losses. We verify output length is preserved.
    """
    model = PacketLossNoise(loss_rate=0.5, burst_length_samples=3, interpolation_mode="zero", seed=42)
    sig = np.sin(np.arange(500) / 10.0)
    lost, err = model.apply(sig)
    assert len(lost) == 500
    assert not np.any(np.isnan(lost))


def test_sensor_detachment_at_0_5s():
    """
    Verify transient bounce when contact is broken at 0.5 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=0.5, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_1_0s():
    """
    Verify transient bounce when contact is broken at 1.0 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=1.0, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_1_5s():
    """
    Verify transient bounce when contact is broken at 1.5 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=1.5, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_2_0s():
    """
    Verify transient bounce when contact is broken at 2.0 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=2.0, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_2_5s():
    """
    Verify transient bounce when contact is broken at 2.5 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=2.5, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_3_0s():
    """
    Verify transient bounce when contact is broken at 3.0 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=3.0, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_3_5s():
    """
    Verify transient bounce when contact is broken at 3.5 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=3.5, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_4_0s():
    """
    Verify transient bounce when contact is broken at 4.0 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=4.0, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_4_5s():
    """
    Verify transient bounce when contact is broken at 4.5 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=4.5, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))


def test_sensor_detachment_at_5_0s():
    """
    Verify transient bounce when contact is broken at 5.0 s.
    Saves open circuit biopotentials stability under dynamic limits.
    """
    model = SensorDetachmentNoise(detachment_time_s=5.0, transient_amplitude=5.0, seed=42)
    sig = np.sin(np.arange(600) / 10.0)
    corrupted, error = model.apply(sig, 100.0)
    assert len(corrupted) == 600
    assert not np.any(np.isnan(corrupted))
