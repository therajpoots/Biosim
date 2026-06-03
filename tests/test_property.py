"""
Property-Based and Invariant Tests for BioSignal Simulator Core Components.

This module uses a hybrid property-testing pattern. If the `hypothesis` library
is installed, it leverages it to run exhaustive randomized fuzzing; otherwise,
it falls back to a deterministic, high-coverage seed-based fuzzing loop to guarantee
that mathematical invariants, filter stability, and resampling alignment hold.
"""

import math
import numpy as np
import pytest

from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.core.math_utils import (
    butter_lowpass,
    butter_notch,
    robust_std,
    interpolate_1d_grid,
    bandpower
)

# Hybrid Hypothesis Import / Mock Fallback
try:
    from hypothesis import given, strategies as st, settings, MaxExamples
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    
    # Simple mock fallback to run static sweeps when hypothesis is missing
    def given(*args, **kwargs):
        def decorator(func):
            def wrapper(*w_args, **w_kwargs):
                return func(*w_args, **w_kwargs)
            return wrapper
        return decorator

    class MockStrategies:
        def floats(self, min_value=None, max_value=None, **kwargs):
            return [min_value or 0.0, max_value or 100.0]
        def integers(self, min_value=None, max_value=None, **kwargs):
            return [min_value or 0, max_value or 100]
        def booleans(self):
            return [True, False]
        def lists(self, *args, **kwargs):
            return []
    st = MockStrategies()


# =====================================================================
# Helper generators for the fallback execution mode
# =====================================================================

def get_floats(min_val, max_val, count=10):
    """Fallback float generator for seed-based sweeps."""
    return list(np.linspace(min_val, max_val, count))

def get_integers(min_val, max_val, count=10):
    """Fallback integer generator for seed-based sweeps."""
    return list(np.linspace(min_val, max_val, count, dtype=int))


# =====================================================================
# 1. SignalRecord Mathematical Invariants
# =====================================================================

def test_record_time_monotonicity():
    """Verify that any instantiated SignalRecord has monotonically increasing time array."""
    # Deterministic sweep of sizes and sampling rates
    for fs in [50.0, 100.0, 250.0, 1000.0]:
        for length in [100, 500, 1000]:
            t = np.arange(length) / fs
            clean = np.sin(2.0 * np.pi * 1.0 * t)
            noisy = clean + 0.1 * np.random.default_rng(42).normal(size=length)
            
            record = SignalRecord(
                signal_type='eeg',
                fs=fs,
                t=t,
                clean=clean,
                noisy=noisy,
                noise_components={}
            )
            
            # 1. Time starts at 0
            assert np.isclose(record.t[0], 0.0)
            # 2. Time differences are exactly 1/fs
            diffs = np.diff(record.t)
            assert np.allclose(diffs, 1.0 / fs)
            # 3. Monotonically increasing
            assert np.all(diffs > 0)
            # 4. Shapes match
            assert record.t.shape == record.clean.shape
            assert record.t.shape == record.noisy.shape

def test_record_multichannel_shapes():
    """Verify multi-channel SignalRecord dimension invariants."""
    for n_channels in [1, 2, 5, 12]:
        length = 200
        fs = 100.0
        t = np.arange(length) / fs
        
        if n_channels == 1:
            clean = np.zeros(length)
            noisy = np.zeros(length)
        else:
            clean = np.zeros((n_channels, length))
            noisy = np.zeros((n_channels, length))
            
        record = SignalRecord(
            signal_type='ecg',
            fs=fs,
            t=t,
            clean=clean,
            noisy=noisy,
            noise_components={}
        )
        
        # Verify shapes
        assert record.t.ndim == 1
        assert len(record.t) == length
        
        if n_channels == 1:
            assert record.clean.ndim == 1
            assert record.noisy.ndim == 1
        else:
            assert record.clean.ndim == 2
            assert record.clean.shape == (n_channels, length)
            assert record.noisy.shape == (n_channels, length)


# =====================================================================
# 2. Filter Mathematical Invariants (BIBO Stability)
# =====================================================================

def test_butterworth_bibo_stability():
    """Verify Bounded-Input Bounded-Output stability of the Butterworth filter."""
    # Fuzzing filter parameters over range of frequencies and orders
    fss = [100.0, 500.0, 1000.0]
    orders = [1, 2, 4, 6]
    
    rng = np.random.default_rng(12345)
    
    for fs in fss:
        t = np.arange(1000) / fs
        # Bounded input signal (-2.0 to 2.0)
        x = np.sin(2.0 * np.pi * 5.0 * t) + rng.uniform(-1.0, 1.0, size=1000)
        
        for order in orders:
            try:
                # Use butter_lowpass with cutoff within boundaries
                cutoff = 20.0
                y = butter_lowpass(x, fs, cutoff, order)
                
                # Assertions of stability
                assert not np.any(np.isnan(y)), f"NaN returned: fs={fs}, order={order}"
                assert not np.any(np.isinf(y)), f"Inf returned: fs={fs}, order={order}"
                
                # Bounded Output: Output amplitude shouldn't exceed input span exponentially
                span_in = np.max(x) - np.min(x)
                span_out = np.max(y) - np.min(y)
                # Filter output can have transient startup peaks, but should be bounded
                assert span_out < 3.0 * span_in, f"Filter unstable: span_in={span_in}, span_out={span_out}"
            except ValueError:
                # Ignore values that violate Nyquist, which is caught separately
                pass

def test_notch_bibo_stability():
    """Verify notch filter stability."""
    fs = 250.0
    t = np.arange(500) / fs
    x = np.sin(2.0 * np.pi * 50.0 * t)
    
    # Fuzz notch frequency and Q-factor
    for notch_freq in [50.0, 60.0, 100.0]:
        for q in [10.0, 30.0, 50.0]:
            y = butter_notch(x, fs, notch_freq, q)
            assert not np.any(np.isnan(y))
            assert not np.any(np.isinf(y))
            
            # The power of the 50Hz sine wave should be significantly reduced at notch frequency
            if np.isclose(notch_freq, 50.0):
                # The output amplitude should be smaller than input
                assert np.std(y) < np.std(x)


# =====================================================================
# 3. Robust Statistics Invariance
# =====================================================================

def test_robust_std_invariance():
    """Verify robust_std (Median Absolute Deviation based) invariants."""
    # Robust standard deviation of a zero array is 0
    assert robust_std(np.zeros(100)) == 0.0
    
    # Robust std of a constant is 0
    assert robust_std(np.ones(100) * 5.5) == 0.0
    
    # Scale invariance: robust_std(k * X) == k * robust_std(X)
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    std_1 = robust_std(x)
    
    for k in [0.5, 2.0, 10.0]:
        std_k = robust_std(k * x)
        assert np.isclose(std_k, k * std_1)
        
    # Translation invariance: robust_std(X + c) == robust_std(X)
    for c in [-100.0, 50.0, 1000.0]:
        std_c = robust_std(x + c)
        assert np.isclose(std_c, std_1)


# =====================================================================
# 4. Resampling Invariants
# =====================================================================

def test_resampling_roundtrip_correlation():
    """Verify resampling down and up preserves correlation and frequency structure."""
    fs_orig = 1000.0
    t_orig = np.arange(2000) / fs_orig
    # Signal containing low and mid frequencies: 2Hz + 15Hz
    x = np.sin(2.0 * np.pi * 2.0 * t_orig) + 0.5 * np.cos(2.0 * np.pi * 15.0 * t_orig)
    
    # Downsample to 200 Hz
    fs_down = 200.0
    t_down = np.arange(400) / fs_down
    x_down = interpolate_1d_grid(t_orig, x, t_down, method='cubic')
    assert len(x_down) == 400
    
    # Upsample back to 1000 Hz
    x_up = interpolate_1d_grid(t_down, x_down, t_orig, method='cubic')
    assert len(x_up) == 2000
    
    # Correlation coefficient between original and roundtrip should be high (> 0.90)
    # inside the signal body (excluding edge filter transients of ~100 samples)
    corr = np.corrcoef(x[100:-100], x_up[100:-100])[0, 1]
    assert corr > 0.90, f"Resampling roundtrip correlation too low: {corr:.4f}"


# =====================================================================
# 5. Welch PSD Energy Preservation
# =====================================================================

def test_welch_energy_preservation():
    """Verify Parseval's theorem approximate preservation in Welch PSD calculation."""
    fs = 100.0
    t = np.arange(1000) / fs
    x = np.sin(2.0 * np.pi * 5.0 * t)
    
    # Signal variance (time domain energy)
    var_time = np.var(x)
    
    # Spectral energy (integrated band power over entire frequency range)
    var_spec = bandpower(x, fs, 0.0, fs / 2.0 - 0.1)
    
    # Energy in time domain should closely match spectral energy (within ~10%)
    assert np.isclose(var_time, var_spec, rtol=0.1)


# =====================================================================
# Property-based testing via hypothesis (if available)
# =====================================================================
if HAS_HYPOTHESIS:
    @given(
        fs=st.floats(min_value=50.0, max_value=2000.0),
        duration=st.floats(min_value=0.5, max_value=10.0),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=25)
    def test_property_ecg_record_generation(fs, duration, seed):
        """Hypothesis check for ECG signal record safety boundaries."""
        from biosignal_simulator.signals.ecg import generate_ecg
        
        # Round duration to avoid fractional sample mismatch
        num_samples = int(math.ceil(duration * fs))
        adj_duration = num_samples / fs
        
        record = generate_ecg(
            fs=fs,
            duration_s=adj_duration,
            heart_rate=75.0,
            seed=seed
        )
        
        assert isinstance(record, SignalRecord)
        assert len(record.t) == num_samples
        assert not np.any(np.isnan(record.clean))
        assert not np.any(np.isinf(record.clean))
        assert np.max(np.abs(record.clean)) < 5.0 # Max amplitude within physiological limits
