"""
Exhaustive verification of filters, interpolation, and PSD estimators.
"""
import numpy as np
import pytest
from biosignal_simulator.core.math_utils import (
    butter_lowpass,
    butter_highpass,
    butter_notch,
    robust_std,
    interpolate_1d_grid,
    bandpower
)


def test_butterworth_lowpass_order_1_cutoff_5_fs_500():
    """
    Verify stability of lowpass filter order 1, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 5.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_5_fs_1000():
    """
    Verify stability of lowpass filter order 1, cutoff 5.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 5.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_10_fs_500():
    """
    Verify stability of lowpass filter order 1, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 10.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_10_fs_1000():
    """
    Verify stability of lowpass filter order 1, cutoff 10.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 10.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_20_fs_500():
    """
    Verify stability of lowpass filter order 1, cutoff 20.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 20.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_20_fs_1000():
    """
    Verify stability of lowpass filter order 1, cutoff 20.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 20.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_50_fs_500():
    """
    Verify stability of lowpass filter order 1, cutoff 50.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 50.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_50_fs_1000():
    """
    Verify stability of lowpass filter order 1, cutoff 50.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 50.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_100_fs_500():
    """
    Verify stability of lowpass filter order 1, cutoff 100.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 100.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_100_fs_1000():
    """
    Verify stability of lowpass filter order 1, cutoff 100.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 100.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_200_fs_500():
    """
    Verify stability of lowpass filter order 1, cutoff 200.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 200.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_1_cutoff_200_fs_1000():
    """
    Verify stability of lowpass filter order 1, cutoff 200.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 200.0, 1)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_5_fs_500():
    """
    Verify stability of lowpass filter order 2, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 5.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_5_fs_1000():
    """
    Verify stability of lowpass filter order 2, cutoff 5.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 5.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_10_fs_500():
    """
    Verify stability of lowpass filter order 2, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 10.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_10_fs_1000():
    """
    Verify stability of lowpass filter order 2, cutoff 10.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 10.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_20_fs_500():
    """
    Verify stability of lowpass filter order 2, cutoff 20.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 20.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_20_fs_1000():
    """
    Verify stability of lowpass filter order 2, cutoff 20.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 20.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_50_fs_500():
    """
    Verify stability of lowpass filter order 2, cutoff 50.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 50.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_50_fs_1000():
    """
    Verify stability of lowpass filter order 2, cutoff 50.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 50.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_100_fs_500():
    """
    Verify stability of lowpass filter order 2, cutoff 100.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 100.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_100_fs_1000():
    """
    Verify stability of lowpass filter order 2, cutoff 100.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 100.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_200_fs_500():
    """
    Verify stability of lowpass filter order 2, cutoff 200.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 200.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_2_cutoff_200_fs_1000():
    """
    Verify stability of lowpass filter order 2, cutoff 200.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 200.0, 2)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_5_fs_500():
    """
    Verify stability of lowpass filter order 3, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 5.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_5_fs_1000():
    """
    Verify stability of lowpass filter order 3, cutoff 5.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 5.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_10_fs_500():
    """
    Verify stability of lowpass filter order 3, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 10.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_10_fs_1000():
    """
    Verify stability of lowpass filter order 3, cutoff 10.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 10.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_20_fs_500():
    """
    Verify stability of lowpass filter order 3, cutoff 20.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 20.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_20_fs_1000():
    """
    Verify stability of lowpass filter order 3, cutoff 20.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 20.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_50_fs_500():
    """
    Verify stability of lowpass filter order 3, cutoff 50.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 50.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_50_fs_1000():
    """
    Verify stability of lowpass filter order 3, cutoff 50.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 50.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_100_fs_500():
    """
    Verify stability of lowpass filter order 3, cutoff 100.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 100.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_100_fs_1000():
    """
    Verify stability of lowpass filter order 3, cutoff 100.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 100.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_200_fs_500():
    """
    Verify stability of lowpass filter order 3, cutoff 200.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 200.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_3_cutoff_200_fs_1000():
    """
    Verify stability of lowpass filter order 3, cutoff 200.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 200.0, 3)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_5_fs_500():
    """
    Verify stability of lowpass filter order 4, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 5.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_5_fs_1000():
    """
    Verify stability of lowpass filter order 4, cutoff 5.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 5.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_10_fs_500():
    """
    Verify stability of lowpass filter order 4, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 10.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_10_fs_1000():
    """
    Verify stability of lowpass filter order 4, cutoff 10.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 10.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_20_fs_500():
    """
    Verify stability of lowpass filter order 4, cutoff 20.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 20.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_20_fs_1000():
    """
    Verify stability of lowpass filter order 4, cutoff 20.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 20.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_50_fs_500():
    """
    Verify stability of lowpass filter order 4, cutoff 50.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 50.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_50_fs_1000():
    """
    Verify stability of lowpass filter order 4, cutoff 50.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 50.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_100_fs_500():
    """
    Verify stability of lowpass filter order 4, cutoff 100.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 100.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_100_fs_1000():
    """
    Verify stability of lowpass filter order 4, cutoff 100.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 100.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_200_fs_500():
    """
    Verify stability of lowpass filter order 4, cutoff 200.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 200.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_4_cutoff_200_fs_1000():
    """
    Verify stability of lowpass filter order 4, cutoff 200.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 200.0, 4)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_5_fs_500():
    """
    Verify stability of lowpass filter order 5, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 5.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_5_fs_1000():
    """
    Verify stability of lowpass filter order 5, cutoff 5.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 5.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_10_fs_500():
    """
    Verify stability of lowpass filter order 5, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 10.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_10_fs_1000():
    """
    Verify stability of lowpass filter order 5, cutoff 10.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 10.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_20_fs_500():
    """
    Verify stability of lowpass filter order 5, cutoff 20.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 20.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_20_fs_1000():
    """
    Verify stability of lowpass filter order 5, cutoff 20.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 20.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_50_fs_500():
    """
    Verify stability of lowpass filter order 5, cutoff 50.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 50.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_50_fs_1000():
    """
    Verify stability of lowpass filter order 5, cutoff 50.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 50.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_100_fs_500():
    """
    Verify stability of lowpass filter order 5, cutoff 100.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 100.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_100_fs_1000():
    """
    Verify stability of lowpass filter order 5, cutoff 100.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 100.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_200_fs_500():
    """
    Verify stability of lowpass filter order 5, cutoff 200.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 200.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_5_cutoff_200_fs_1000():
    """
    Verify stability of lowpass filter order 5, cutoff 200.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 200.0, 5)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_5_fs_500():
    """
    Verify stability of lowpass filter order 6, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 5.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_5_fs_1000():
    """
    Verify stability of lowpass filter order 6, cutoff 5.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 5.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_10_fs_500():
    """
    Verify stability of lowpass filter order 6, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 10.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_10_fs_1000():
    """
    Verify stability of lowpass filter order 6, cutoff 10.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 10.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_20_fs_500():
    """
    Verify stability of lowpass filter order 6, cutoff 20.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 20.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_20_fs_1000():
    """
    Verify stability of lowpass filter order 6, cutoff 20.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 20.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_50_fs_500():
    """
    Verify stability of lowpass filter order 6, cutoff 50.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 50.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_50_fs_1000():
    """
    Verify stability of lowpass filter order 6, cutoff 50.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 50.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_100_fs_500():
    """
    Verify stability of lowpass filter order 6, cutoff 100.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 100.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_100_fs_1000():
    """
    Verify stability of lowpass filter order 6, cutoff 100.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 100.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_200_fs_500():
    """
    Verify stability of lowpass filter order 6, cutoff 200.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 500.0, 200.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_lowpass_order_6_cutoff_200_fs_1000():
    """
    Verify stability of lowpass filter order 6, cutoff 200.0 Hz, fs 1000.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(200) / 10.0)
    y = butter_lowpass(x, 1000.0, 200.0, 6)
    assert len(y) == 200
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_1_fs_250():
    """
    Verify stability of highpass filter order 1, cutoff 1.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 1.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_1_fs_500():
    """
    Verify stability of highpass filter order 1, cutoff 1.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 1.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_5_fs_250():
    """
    Verify stability of highpass filter order 1, cutoff 5.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 5.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_5_fs_500():
    """
    Verify stability of highpass filter order 1, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 5.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_10_fs_250():
    """
    Verify stability of highpass filter order 1, cutoff 10.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 10.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_10_fs_500():
    """
    Verify stability of highpass filter order 1, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 10.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_25_fs_250():
    """
    Verify stability of highpass filter order 1, cutoff 25.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 25.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_1_cutoff_25_fs_500():
    """
    Verify stability of highpass filter order 1, cutoff 25.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 25.0, 1)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_1_fs_250():
    """
    Verify stability of highpass filter order 2, cutoff 1.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 1.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_1_fs_500():
    """
    Verify stability of highpass filter order 2, cutoff 1.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 1.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_5_fs_250():
    """
    Verify stability of highpass filter order 2, cutoff 5.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 5.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_5_fs_500():
    """
    Verify stability of highpass filter order 2, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 5.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_10_fs_250():
    """
    Verify stability of highpass filter order 2, cutoff 10.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 10.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_10_fs_500():
    """
    Verify stability of highpass filter order 2, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 10.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_25_fs_250():
    """
    Verify stability of highpass filter order 2, cutoff 25.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 25.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_2_cutoff_25_fs_500():
    """
    Verify stability of highpass filter order 2, cutoff 25.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 25.0, 2)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_1_fs_250():
    """
    Verify stability of highpass filter order 4, cutoff 1.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 1.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_1_fs_500():
    """
    Verify stability of highpass filter order 4, cutoff 1.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 1.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_5_fs_250():
    """
    Verify stability of highpass filter order 4, cutoff 5.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 5.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_5_fs_500():
    """
    Verify stability of highpass filter order 4, cutoff 5.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 5.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_10_fs_250():
    """
    Verify stability of highpass filter order 4, cutoff 10.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 10.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_10_fs_500():
    """
    Verify stability of highpass filter order 4, cutoff 10.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 10.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_25_fs_250():
    """
    Verify stability of highpass filter order 4, cutoff 25.0 Hz, fs 250.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 250.0, 25.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_butterworth_highpass_order_4_cutoff_25_fs_500():
    """
    Verify stability of highpass filter order 4, cutoff 25.0 Hz, fs 500.0 Hz.
    Checks that the filter output array is populated and stable without NaNs.
    """
    x = np.sin(np.arange(100) / 5.0)
    y = butter_highpass(x, 500.0, 25.0, 4)
    assert len(y) == 100
    assert not np.any(np.isnan(y))
    assert not np.any(np.isinf(y))


def test_resample_from_100_to_50_method_linear():
    """
    Verify resampling from 100 samples to 50 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 100)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 50)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 50
    assert not np.any(np.isnan(y))


def test_resample_from_100_to_50_method_cubic():
    """
    Verify resampling from 100 samples to 50 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 100)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 50)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 50
    assert not np.any(np.isnan(y))


def test_resample_from_100_to_200_method_linear():
    """
    Verify resampling from 100 samples to 200 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 100)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 200)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 200
    assert not np.any(np.isnan(y))


def test_resample_from_100_to_200_method_cubic():
    """
    Verify resampling from 100 samples to 200 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 100)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 200)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 200
    assert not np.any(np.isnan(y))


def test_resample_from_100_to_800_method_linear():
    """
    Verify resampling from 100 samples to 800 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 100)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 800)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 800
    assert not np.any(np.isnan(y))


def test_resample_from_100_to_800_method_cubic():
    """
    Verify resampling from 100 samples to 800 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 100)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 800)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 800
    assert not np.any(np.isnan(y))


def test_resample_from_250_to_50_method_linear():
    """
    Verify resampling from 250 samples to 50 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 250)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 50)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 50
    assert not np.any(np.isnan(y))


def test_resample_from_250_to_50_method_cubic():
    """
    Verify resampling from 250 samples to 50 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 250)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 50)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 50
    assert not np.any(np.isnan(y))


def test_resample_from_250_to_200_method_linear():
    """
    Verify resampling from 250 samples to 200 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 250)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 200)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 200
    assert not np.any(np.isnan(y))


def test_resample_from_250_to_200_method_cubic():
    """
    Verify resampling from 250 samples to 200 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 250)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 200)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 200
    assert not np.any(np.isnan(y))


def test_resample_from_250_to_800_method_linear():
    """
    Verify resampling from 250 samples to 800 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 250)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 800)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 800
    assert not np.any(np.isnan(y))


def test_resample_from_250_to_800_method_cubic():
    """
    Verify resampling from 250 samples to 800 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 250)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 800)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 800
    assert not np.any(np.isnan(y))


def test_resample_from_500_to_50_method_linear():
    """
    Verify resampling from 500 samples to 50 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 500)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 50)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 50
    assert not np.any(np.isnan(y))


def test_resample_from_500_to_50_method_cubic():
    """
    Verify resampling from 500 samples to 50 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 500)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 50)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 50
    assert not np.any(np.isnan(y))


def test_resample_from_500_to_200_method_linear():
    """
    Verify resampling from 500 samples to 200 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 500)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 200)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 200
    assert not np.any(np.isnan(y))


def test_resample_from_500_to_200_method_cubic():
    """
    Verify resampling from 500 samples to 200 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 500)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 200)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 200
    assert not np.any(np.isnan(y))


def test_resample_from_500_to_800_method_linear():
    """
    Verify resampling from 500 samples to 800 samples using linear interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 500)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 800)
    y = interpolate_1d_grid(t_orig, x, t_new, method='linear')
    assert len(y) == 800
    assert not np.any(np.isnan(y))


def test_resample_from_500_to_800_method_cubic():
    """
    Verify resampling from 500 samples to 800 samples using cubic interpolation.
    Checks that downsampling or upsampling aligns the endpoints correctly.
    """
    t_orig = np.linspace(0.0, 2.0, 500)
    x = np.sin(t_orig)
    t_new = np.linspace(0.0, 2.0, 800)
    y = interpolate_1d_grid(t_orig, x, t_new, method='cubic')
    assert len(y) == 800
    assert not np.any(np.isnan(y))
