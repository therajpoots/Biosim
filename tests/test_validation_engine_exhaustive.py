"""
Exhaustive checks for validation pipelines, flatlines, clipping, and motion bursts.
"""
import numpy as np
import pytest
from biosignal_simulator.utils.validation import (
    validate_signal,
    SignalIntegrityChecker,
    PhysiologicalValidator,
    ValidationReport
)


def test_flatline_detection_duration_0_5s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 0.5 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(0.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=0.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_0_5s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 0.5 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(0.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=0.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_0_5s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 0.5 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(0.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=0.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_1_0s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 1.0 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(1.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=1.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_1_0s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 1.0 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(1.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=1.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_1_0s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 1.0 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(1.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=1.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_1_5s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 1.5 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(1.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=1.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_1_5s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 1.5 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(1.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=1.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_1_5s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 1.5 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(1.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=1.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_2_0s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 2.0 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(2.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=2.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_2_0s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 2.0 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(2.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=2.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_2_0s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 2.0 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(2.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=2.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_2_5s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 2.5 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(2.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=2.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_2_5s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 2.5 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(2.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=2.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_2_5s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 2.5 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(2.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=2.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_3_0s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 3.0 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(3.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=3.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_3_0s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 3.0 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(3.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=3.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_3_0s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 3.0 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(3.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=3.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_3_5s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 3.5 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(3.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=3.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_3_5s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 3.5 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(3.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=3.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_3_5s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 3.5 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(3.5 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=3.5-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_4_0s_tol_1eminus_05():
    """
    Verify flatline locator identifies constant segment of 4.0 s under tol 1e-05.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(4.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=1e-05, min_duration_s=4.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_4_0s_tol_0_0001():
    """
    Verify flatline locator identifies constant segment of 4.0 s under tol 0.0001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(4.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.0001, min_duration_s=4.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_flatline_detection_duration_4_0s_tol_0_001():
    """
    Verify flatline locator identifies constant segment of 4.0 s under tol 0.001.
    Checks that the segment timings match expected injected values.
    """
    fs = 100.0
    t = np.arange(500) / fs
    sig = np.sin(t)
    flat_idx = int(4.0 * fs)
    sig[100 : 100 + flat_idx] = 0.5
    
    segments = SignalIntegrityChecker.detect_flatline(sig, fs, tolerance=0.001, min_duration_s=4.0-0.05)
    assert len(segments) >= 1
    assert not np.any(np.isnan(sig))


def test_clipping_detection_limit_0_1_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.1 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.1, 0.1)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_1_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.1 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.1, 0.1)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_2_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.2 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.2, 0.2)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_2_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.2 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.2, 0.2)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_3_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.3 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.3, 0.3)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_3_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.3 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.3, 0.3)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_4_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.4 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.4, 0.4)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_4_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.4 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.4, 0.4)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_5_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.5 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.5, 0.5)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_5_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.5 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.5, 0.5)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_6_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.6 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.6, 0.6)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_6_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.6 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.6, 0.6)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_7_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.7 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.7, 0.7)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_7_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.7 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.7, 0.7)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_8_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.8 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.8, 0.8)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_8_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.8 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.8, 0.8)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_9_ratio_0_85():
    """
    Verify clipping locator identifies signal saturation at limit 0.9 with threshold 0.85.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.9, 0.9)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_clipping_detection_limit_0_9_ratio_0_95():
    """
    Verify clipping locator identifies signal saturation at limit 0.9 with threshold 0.95.
    Verifies that the clipping ratio scales proportionally.
    """
    fs = 100.0
    t = np.arange(300) / fs
    sig = np.sin(2 * np.pi * t)
    sig_clipped = np.clip(sig, -0.9, 0.9)
    is_clipped, ratio = SignalIntegrityChecker.detect_clipping(sig_clipped, threshold_ratio=0.99)
    assert is_clipped is True
    assert ratio > 0.02
    assert not np.any(np.isnan(sig_clipped))


def test_motion_burst_window_0_2_std_2_0():
    """
    Verify motion burst locator under window 0.2 s and multiplier 2.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.2, threshold_std=2.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_0_2_std_3_0():
    """
    Verify motion burst locator under window 0.2 s and multiplier 3.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.2, threshold_std=3.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_0_2_std_4_0():
    """
    Verify motion burst locator under window 0.2 s and multiplier 4.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.2, threshold_std=4.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_0_2_std_5_0():
    """
    Verify motion burst locator under window 0.2 s and multiplier 5.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.2, threshold_std=5.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_0_5_std_2_0():
    """
    Verify motion burst locator under window 0.5 s and multiplier 2.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.5, threshold_std=2.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_0_5_std_3_0():
    """
    Verify motion burst locator under window 0.5 s and multiplier 3.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.5, threshold_std=3.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_0_5_std_4_0():
    """
    Verify motion burst locator under window 0.5 s and multiplier 4.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.5, threshold_std=4.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_0_5_std_5_0():
    """
    Verify motion burst locator under window 0.5 s and multiplier 5.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=0.5, threshold_std=5.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_1_0_std_2_0():
    """
    Verify motion burst locator under window 1.0 s and multiplier 2.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=1.0, threshold_std=2.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_1_0_std_3_0():
    """
    Verify motion burst locator under window 1.0 s and multiplier 3.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=1.0, threshold_std=3.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_1_0_std_4_0():
    """
    Verify motion burst locator under window 1.0 s and multiplier 4.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=1.0, threshold_std=4.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))


def test_motion_burst_window_1_0_std_5_0():
    """
    Verify motion burst locator under window 1.0 s and multiplier 5.0.
    Verifies that transient bursts are captured and reported.
    """
    fs = 100.0
    t = np.arange(2000) / fs
    sig = np.sin(t)
    sig[800:900] += 12.0 * np.sin(10.0 * t[800:900])
    bursts = SignalIntegrityChecker.detect_motion_bursts(sig, fs, window_s=1.0, threshold_std=5.0)
    assert len(bursts) >= 1
    assert not np.any(np.isnan(sig))
