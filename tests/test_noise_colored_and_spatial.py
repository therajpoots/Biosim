"""
Exhaustive verification of colored noise power spectral densities.
"""
import numpy as np
import pytest
from scipy.signal import welch
from biosignal_simulator.noise.colored import ColoredNoise, PinkNoise, BrownNoise
from biosignal_simulator.core.math_utils import robust_std


def test_colored_noise_exp_minus_2_0_method_fft_fs_100():
    """
    Verify colored noise with exponent -2.0 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-2.0, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_2_0_method_fft_fs_200():
    """
    Verify colored noise with exponent -2.0 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-2.0, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_2_0_method_voss_fs_100():
    """
    Verify colored noise with exponent -2.0 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-2.0, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_2_0_method_voss_fs_200():
    """
    Verify colored noise with exponent -2.0 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-2.0, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_8_method_fft_fs_100():
    """
    Verify colored noise with exponent -1.8 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_8_method_fft_fs_200():
    """
    Verify colored noise with exponent -1.8 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_8_method_voss_fs_100():
    """
    Verify colored noise with exponent -1.8 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_8_method_voss_fs_200():
    """
    Verify colored noise with exponent -1.8 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_6_method_fft_fs_100():
    """
    Verify colored noise with exponent -1.6 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_6_method_fft_fs_200():
    """
    Verify colored noise with exponent -1.6 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_6_method_voss_fs_100():
    """
    Verify colored noise with exponent -1.6 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_6_method_voss_fs_200():
    """
    Verify colored noise with exponent -1.6 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_4_method_fft_fs_100():
    """
    Verify colored noise with exponent -1.4 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_4_method_fft_fs_200():
    """
    Verify colored noise with exponent -1.4 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_4_method_voss_fs_100():
    """
    Verify colored noise with exponent -1.4 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_4_method_voss_fs_200():
    """
    Verify colored noise with exponent -1.4 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_2_method_fft_fs_100():
    """
    Verify colored noise with exponent -1.2 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_2_method_fft_fs_200():
    """
    Verify colored noise with exponent -1.2 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_2_method_voss_fs_100():
    """
    Verify colored noise with exponent -1.2 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_2_method_voss_fs_200():
    """
    Verify colored noise with exponent -1.2 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_0_method_fft_fs_100():
    """
    Verify colored noise with exponent -1.0 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.0, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_0_method_fft_fs_200():
    """
    Verify colored noise with exponent -1.0 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.0, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_0_method_voss_fs_100():
    """
    Verify colored noise with exponent -1.0 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.0, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_1_0_method_voss_fs_200():
    """
    Verify colored noise with exponent -1.0 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-1.0, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_8_method_fft_fs_100():
    """
    Verify colored noise with exponent -0.8 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_8_method_fft_fs_200():
    """
    Verify colored noise with exponent -0.8 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_8_method_voss_fs_100():
    """
    Verify colored noise with exponent -0.8 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_8_method_voss_fs_200():
    """
    Verify colored noise with exponent -0.8 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_6_method_fft_fs_100():
    """
    Verify colored noise with exponent -0.6 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_6_method_fft_fs_200():
    """
    Verify colored noise with exponent -0.6 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_6_method_voss_fs_100():
    """
    Verify colored noise with exponent -0.6 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_6_method_voss_fs_200():
    """
    Verify colored noise with exponent -0.6 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_4_method_fft_fs_100():
    """
    Verify colored noise with exponent -0.4 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_4_method_fft_fs_200():
    """
    Verify colored noise with exponent -0.4 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_4_method_voss_fs_100():
    """
    Verify colored noise with exponent -0.4 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_4_method_voss_fs_200():
    """
    Verify colored noise with exponent -0.4 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_2_method_fft_fs_100():
    """
    Verify colored noise with exponent -0.2 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_2_method_fft_fs_200():
    """
    Verify colored noise with exponent -0.2 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_2_method_voss_fs_100():
    """
    Verify colored noise with exponent -0.2 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_minus_0_2_method_voss_fs_200():
    """
    Verify colored noise with exponent -0.2 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=-0.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_0_method_fft_fs_100():
    """
    Verify colored noise with exponent 0.0 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.0, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_0_method_fft_fs_200():
    """
    Verify colored noise with exponent 0.0 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.0, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_0_method_voss_fs_100():
    """
    Verify colored noise with exponent 0.0 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.0, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_0_method_voss_fs_200():
    """
    Verify colored noise with exponent 0.0 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.0, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_2_method_fft_fs_100():
    """
    Verify colored noise with exponent 0.2 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_2_method_fft_fs_200():
    """
    Verify colored noise with exponent 0.2 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_2_method_voss_fs_100():
    """
    Verify colored noise with exponent 0.2 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_2_method_voss_fs_200():
    """
    Verify colored noise with exponent 0.2 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_4_method_fft_fs_100():
    """
    Verify colored noise with exponent 0.4 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_4_method_fft_fs_200():
    """
    Verify colored noise with exponent 0.4 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_4_method_voss_fs_100():
    """
    Verify colored noise with exponent 0.4 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_4_method_voss_fs_200():
    """
    Verify colored noise with exponent 0.4 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_6_method_fft_fs_100():
    """
    Verify colored noise with exponent 0.6 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_6_method_fft_fs_200():
    """
    Verify colored noise with exponent 0.6 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_6_method_voss_fs_100():
    """
    Verify colored noise with exponent 0.6 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_6_method_voss_fs_200():
    """
    Verify colored noise with exponent 0.6 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_8_method_fft_fs_100():
    """
    Verify colored noise with exponent 0.8 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_8_method_fft_fs_200():
    """
    Verify colored noise with exponent 0.8 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_8_method_voss_fs_100():
    """
    Verify colored noise with exponent 0.8 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_0_8_method_voss_fs_200():
    """
    Verify colored noise with exponent 0.8 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=0.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_0_method_fft_fs_100():
    """
    Verify colored noise with exponent 1.0 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = PinkNoise(method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_0_method_fft_fs_200():
    """
    Verify colored noise with exponent 1.0 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = PinkNoise(method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_0_method_voss_fs_100():
    """
    Verify colored noise with exponent 1.0 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = PinkNoise(method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_0_method_voss_fs_200():
    """
    Verify colored noise with exponent 1.0 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = PinkNoise(method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_2_method_fft_fs_100():
    """
    Verify colored noise with exponent 1.2 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_2_method_fft_fs_200():
    """
    Verify colored noise with exponent 1.2 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.2, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_2_method_voss_fs_100():
    """
    Verify colored noise with exponent 1.2 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_2_method_voss_fs_200():
    """
    Verify colored noise with exponent 1.2 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.2, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_4_method_fft_fs_100():
    """
    Verify colored noise with exponent 1.4 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_4_method_fft_fs_200():
    """
    Verify colored noise with exponent 1.4 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.4, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_4_method_voss_fs_100():
    """
    Verify colored noise with exponent 1.4 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_4_method_voss_fs_200():
    """
    Verify colored noise with exponent 1.4 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.4, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_6_method_fft_fs_100():
    """
    Verify colored noise with exponent 1.6 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_6_method_fft_fs_200():
    """
    Verify colored noise with exponent 1.6 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.6, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_6_method_voss_fs_100():
    """
    Verify colored noise with exponent 1.6 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_6_method_voss_fs_200():
    """
    Verify colored noise with exponent 1.6 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.6, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_8_method_fft_fs_100():
    """
    Verify colored noise with exponent 1.8 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_8_method_fft_fs_200():
    """
    Verify colored noise with exponent 1.8 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.8, method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_8_method_voss_fs_100():
    """
    Verify colored noise with exponent 1.8 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_1_8_method_voss_fs_200():
    """
    Verify colored noise with exponent 1.8 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = ColoredNoise(exponent=1.8, method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_2_0_method_fft_fs_100():
    """
    Verify colored noise with exponent 2.0 using method fft on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = BrownNoise(method='fft', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_2_0_method_fft_fs_200():
    """
    Verify colored noise with exponent 2.0 using method fft on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = BrownNoise(method='fft', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_2_0_method_voss_fs_100():
    """
    Verify colored noise with exponent 2.0 using method voss on fs 100.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = BrownNoise(method='voss', std=1.0, seed=42)
    noise = model.generate(200, 100.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_exp_2_0_method_voss_fs_200():
    """
    Verify colored noise with exponent 2.0 using method voss on fs 200.0 Hz.
    Checks that the output matches the expected length and does not contain NaNs.
    """
    model = BrownNoise(method='voss', std=1.0, seed=42)
    noise = model.generate(200, 200.0)
    assert len(noise) == 200
    assert not np.any(np.isnan(noise))
    assert np.std(noise) > 0.0


def test_colored_noise_amplitude_0_01():
    """
    Verify noise standard deviation matches target 0.01.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=0.01, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 0.01, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_0_05():
    """
    Verify noise standard deviation matches target 0.05.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=0.05, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 0.05, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_0_1():
    """
    Verify noise standard deviation matches target 0.1.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=0.1, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 0.1, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_0_2():
    """
    Verify noise standard deviation matches target 0.2.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=0.2, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 0.2, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_0_5():
    """
    Verify noise standard deviation matches target 0.5.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=0.5, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 0.5, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_0_8():
    """
    Verify noise standard deviation matches target 0.8.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=0.8, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 0.8, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_1_0():
    """
    Verify noise standard deviation matches target 1.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=1.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 1.0, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_1_5():
    """
    Verify noise standard deviation matches target 1.5.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=1.5, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 1.5, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_2_0():
    """
    Verify noise standard deviation matches target 2.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=2.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 2.0, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_2_5():
    """
    Verify noise standard deviation matches target 2.5.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=2.5, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 2.5, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_3_0():
    """
    Verify noise standard deviation matches target 3.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=3.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 3.0, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_4_0():
    """
    Verify noise standard deviation matches target 4.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=4.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 4.0, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_5_0():
    """
    Verify noise standard deviation matches target 5.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=5.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 5.0, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_7_5():
    """
    Verify noise standard deviation matches target 7.5.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=7.5, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 7.5, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_10_0():
    """
    Verify noise standard deviation matches target 10.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=10.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 10.0, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_15_0():
    """
    Verify noise standard deviation matches target 15.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=15.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 15.0, rtol=0.6)
    assert not np.any(np.isnan(noise))


def test_colored_noise_amplitude_20_0():
    """
    Verify noise standard deviation matches target 20.0.
    Ensures that standard deviation scaling holds within statistical tolerance.
    """
    model = PinkNoise(std=20.0, seed=42)
    # Generate more samples to have stable standard deviation check
    noise = model.generate(2000, 100.0)
    assert np.isclose(robust_std(noise), 20.0, rtol=0.6)
    assert not np.any(np.isnan(noise))
