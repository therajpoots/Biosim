"""
BioSignal Simulator Visualization Engine.

This module provides standard static plotting tools using matplotlib, and interactive
HTML dashboard exporters for analyzing physiological waveforms and noise components.

Features:
1. 12-Lead ECG Plotting:
   Arranges ECG channels into clinical 3x4 + 1 rhythm layout with standardized grid markings.
   - Standard paper speed: 25 mm/s (0.04 s per 1 mm small grid, 0.2 s per 5 mm large grid).
   - Standard gain: 10 mm/mV (0.1 mV per 1 mm small grid, 0.5 mV per 5 mm large grid).

2. EEG Time-Frequency Analysis:
   Renders spectrogram panels side-by-side with relative power bands.

3. EMG Fatigue Trend Monitoring:
   Plots rolling mean and median spectral frequency shifts indicating muscle fatigue.

4. Interactive HTML Dashboard:
   Saves signal records as self-contained HTML files containing vector canvas rendering,
   allowing users to drag-to-zoom, pan, toggle signals, and view tooltips offline.
"""

import os
import json
import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

from biosignal_simulator.core.record import SignalRecord
from biosignal_simulator.metrics.spectral import compute_psd
from scipy import signal as sp_signal


def _check_matplotlib():
    """Verify that matplotlib is installed before plotting."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization utilities. "
            "Please install it using `pip install matplotlib`."
        )
    return plt


def plot_record(record: SignalRecord, show_components: bool = True, fig_kwargs: Optional[Dict[str, Any]] = None) -> object:
    """
    Plot the clean signal, noisy signal, and optionally the individual noise components.
    
    Parameters
    ----------
    record : SignalRecord
        The SignalRecord to plot.
    show_components : bool
        If True, plots each individual noise component in a separate panel.
    fig_kwargs : Optional[Dict[str, Any]]
        Optional arguments passed to matplotlib subplots.
        
    Returns
    -------
    matplotlib.figure.Figure
        The constructed figure object.
    """
    plt = _check_matplotlib()
    
    # Check dimensions
    clean_data = record.clean
    noisy_data = record.noisy
    
    # Handle multi-channel: take first channel for standard plot
    if clean_data.ndim > 1:
        clean_data = clean_data[0]
        noisy_data = noisy_data[0]
        noise_keys = list(record.noise_components.keys())
        noise_comps = {k: (v[0] if v.ndim > 1 else v) for k, v in record.noise_components.items()}
    else:
        noise_keys = list(record.noise_components.keys())
        noise_comps = record.noise_components
        
    n_panels = 2
    if show_components:
        n_panels += len(noise_keys)
        
    fk = fig_kwargs or {}
    if 'figsize' not in fk:
        fk['figsize'] = (12, 2.2 * n_panels)
        
    fig, axes = plt.subplots(n_panels, 1, sharex=True, **fk)
    if n_panels == 1:
        axes = [axes]
        
    # Panel 1: Clean Signal
    axes[0].plot(record.t, clean_data, color='#10b981', lw=1.5, label='Clean')
    axes[0].set_title(f"Clean {record.signal_type.upper()} Signal (fs={record.fs:.1f} Hz)", color='#e2e8f0', fontsize=12)
    axes[0].set_ylabel("Amplitude (mV)", color='#94a3b8')
    axes[0].grid(True, alpha=0.15, color='#e2e8f0')
    axes[0].tick_params(colors='#94a3b8')
    axes[0].set_facecolor('#111827')
    
    # Panel 2: Noisy Signal
    snr_str = f" [SNR: {record.snr_db:.1f} dB]" if record.snr_db is not None else ""
    axes[1].plot(record.t, noisy_data, color='#f59e0b', lw=1.5, label='Noisy')
    axes[1].set_title(f"Noisy Signal{snr_str}", color='#e2e8f0', fontsize=12)
    axes[1].set_ylabel("Amplitude (mV)", color='#94a3b8')
    axes[1].grid(True, alpha=0.15, color='#e2e8f0')
    axes[1].tick_params(colors='#94a3b8')
    axes[1].set_facecolor('#111827')
    
    # Panel 3..N: Noise Components
    if show_components:
        for idx, key in enumerate(noise_keys):
            ax = axes[2 + idx]
            ax.plot(record.t, noise_comps[key], color='#ef4444', lw=1.0)
            ax.set_title(f"Noise Component: {key}", color='#e2e8f0', fontsize=11)
            ax.set_ylabel("Amplitude", color='#94a3b8')
            ax.grid(True, alpha=0.15, color='#e2e8f0')
            ax.tick_params(colors='#94a3b8')
            ax.set_facecolor('#111827')
            
    axes[-1].set_xlabel("Time (s)", color='#94a3b8')
    fig.patch.set_facecolor('#0f172a')
    plt.tight_layout()
    return fig


def plot_psd_comparison(signals_dict: Dict[str, np.ndarray], fs: float, fmin: float = 0.1, fmax: float = 100.0) -> object:
    """
    Overlay PSD of multiple signals on a single plot for comparison.
    """
    plt = _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#111827')
    
    for label, sig in signals_dict.items():
        # Handle multi-channel
        if sig.ndim > 1:
            sig = sig[0]
        f, psd = sp_signal.welch(sig - np.mean(sig), fs=fs, nperseg=min(1024, len(sig)))
        mask = (f >= fmin) & (f <= fmax)
        ax.semilogy(f[mask], psd[mask], label=label, lw=2, alpha=0.85)
        
    ax.set_title("Power Spectral Density (PSD) Comparison", color='#e2e8f0', fontsize=13)
    ax.set_xlabel("Frequency (Hz)", color='#94a3b8')
    ax.set_ylabel("Power/Frequency (dB/Hz)", color='#94a3b8')
    ax.grid(True, which='both', alpha=0.15, color='#e2e8f0')
    ax.tick_params(colors='#94a3b8')
    ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
    plt.tight_layout()
    return fig


def plot_noise_characterization(noise_model: Any, n_samples: int, fs: float) -> object:
    """
    Generate a 4-panel analysis figure for verifying a noise model's mathematical properties.
    """
    plt = _check_matplotlib()
    
    # Generate noise
    if hasattr(noise_model, 'generate'):
        noise = noise_model.generate(n_samples, fs)
    else:
        raise ValueError("Provided noise_model does not have a generate method.")
        
    # Take first channel if multi-channel
    if noise.ndim > 1:
        noise = noise[0]
        
    t = np.arange(n_samples) / fs
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes.flat:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.15, color='#e2e8f0')
        
    # Panel 1: Time Series
    axes[0, 0].plot(t, noise, color='#ef4444', lw=0.8)
    axes[0, 0].set_title("Time Series", color='#e2e8f0')
    axes[0, 0].set_xlabel("Time (s)", color='#94a3b8')
    axes[0, 0].set_ylabel("Amplitude", color='#94a3b8')
    
    # Panel 2: Histogram (Amplitude Distribution)
    axes[0, 1].hist(noise, bins=50, color='#8b5cf6', edgecolor='none', alpha=0.85)
    axes[0, 1].set_title("Amplitude Histogram", color='#e2e8f0')
    axes[0, 1].set_xlabel("Value", color='#94a3b8')
    axes[0, 1].set_ylabel("Count", color='#94a3b8')
    
    # Panel 3: Power Spectral Density
    f, psd = sp_signal.welch(noise - np.mean(noise), fs=fs, nperseg=min(1024, len(noise)))
    axes[1, 0].loglog(f[f > 0], psd[f > 0], color='#10b981', lw=1.5)
    axes[1, 0].set_title("Power Spectral Density (Log-Log)", color='#e2e8f0')
    axes[1, 0].set_xlabel("Frequency (Hz)", color='#94a3b8')
    axes[1, 0].set_ylabel("Power/Frequency", color='#94a3b8')
    
    # Panel 4: Autocorrelation Function
    max_lag = min(500, n_samples - 1)
    n_centered = noise - np.mean(noise)
    var = np.var(noise)
    if var > 1e-12:
        autocorr = np.correlate(n_centered, n_centered, mode='full')
        mid = len(autocorr) // 2
        autocorr = autocorr[mid:mid + max_lag + 1] / (var * n_samples)
    else:
        autocorr = np.zeros(max_lag + 1)
        autocorr[0] = 1.0
        
    lags = np.arange(0, max_lag + 1) / fs
    axes[1, 1].plot(lags, autocorr, color='#f59e0b', lw=1.8)
    axes[1, 1].set_title("Autocorrelation", color='#e2e8f0')
    axes[1, 1].set_xlabel("Lag (s)", color='#94a3b8')
    axes[1, 1].set_ylabel("Correlation Coefficient", color='#94a3b8')
    
    plt.suptitle(f"Characterization of {noise_model.__class__.__name__}", fontsize=14, color='#ffffff')
    plt.tight_layout()
    return fig


def plot_snr_sweep(results: Any) -> object:
    """
    Plot sweep results (RMSE, SNR Out, Correlation) vs Target SNR.
    """
    plt = _check_matplotlib()
    
    if isinstance(results, dict):
        try:
            import pandas as pd
            df = pd.DataFrame(results)
        except ImportError:
            df = results
    else:
        df = results
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.15, color='#e2e8f0')
        
    x_col = None
    for col in ['snr_target_db', 'sig_std', 'noise_std', 'snr_in']:
        if col in df:
            x_col = col
            break
    if x_col is None:
        x = np.arange(len(df))
        xlabel = "Experiment Index"
    else:
        x = df[x_col]
        xlabel = x_col
        
    # Plot SNR Out
    if 'snr_out' in df:
        axes[0].plot(x, df['snr_out'], 'o-', color='#3b82f6', lw=2.5, ms=6)
        if 'snr_in' in df:
            axes[0].plot(x, df['snr_in'], '--', color='#64748b', label='Input SNR')
            axes[0].legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
        axes[0].set_title("Output SNR vs Input", color='#e2e8f0')
        axes[0].set_ylabel("SNR (dB)", color='#94a3b8')
    else:
        axes[0].text(0.5, 0.5, "snr_out missing", ha='center', va='center', color='#64748b')
    axes[0].set_xlabel(xlabel, color='#94a3b8')
    
    # Plot RMSE
    rmse_col = 'rmse_after' if 'rmse_after' in df else ('rmse' if 'rmse' in df else None)
    if rmse_col:
        axes[1].plot(x, df[rmse_col], 's-', color='#ef4444', lw=2.5, ms=6)
        axes[1].set_title("RMSE (Lower is Better)", color='#e2e8f0')
        axes[1].set_ylabel("RMSE", color='#94a3b8')
    else:
        axes[1].text(0.5, 0.5, "rmse missing", ha='center', va='center', color='#64748b')
    axes[1].set_xlabel(xlabel, color='#94a3b8')
    
    # Plot Correlation
    corr_col = 'correlation_after' if 'correlation_after' in df else ('correlation' if 'correlation' in df else None)
    if corr_col:
        axes[2].plot(x, df[corr_col], '^-', color='#10b981', lw=2.5, ms=6)
        axes[2].set_title("Correlation (Higher is Better)", color='#e2e8f0')
        axes[2].set_ylabel("Pearson Correlation", color='#94a3b8')
    else:
        axes[2].text(0.5, 0.5, "correlation missing", ha='center', va='center', color='#64748b')
    axes[2].set_xlabel(xlabel, color='#94a3b8')
    
    plt.tight_layout()
    return fig


def plot_filter_response(b: np.ndarray, a: np.ndarray, fs: float, fmax: Optional[float] = None, title: str = "Filter Response") -> object:
    """Plot magnitude and phase response of a digital filter."""
    plt = _check_matplotlib()
    from scipy.signal import freqz
    
    w, h = freqz(b, a, worN=2000)
    freq = w * fs / (2.0 * np.pi)
    
    if fmax is not None:
        mask = freq <= fmax
        freq = freq[mask]
        h = h[mask]
        
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#111827')
    ax1.tick_params(colors='#94a3b8')
    
    # Magnitude
    color = '#3b82f6'
    ax1.set_title(title, color='#e2e8f0', fontsize=13)
    ax1.set_xlabel('Frequency (Hz)', color='#94a3b8')
    ax1.set_ylabel('Gain (dB)', color=color)
    gain_db = 20 * np.log10(np.clip(np.abs(h), 1e-15, None))
    ax1.plot(freq, gain_db, color=color, lw=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.15, color='#e2e8f0')
    
    # Phase
    ax2 = ax1.twinx()  
    color = '#ef4444'
    ax2.set_ylabel('Phase (radians)', color=color)
    phase = np.unwrap(np.angle(h))
    ax2.plot(freq, phase, color=color, lw=1.8, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.tight_layout()
    return fig


def plot_multi_lead_ecg(record: SignalRecord, lead_format: str = "12-lead") -> object:
    """
    Renders the record clean and noisy data in a standardized clinical ECG layout.
    
    Layout maps:
    - 3x4 + 1 layout:
      Column 1: Lead I, Lead II, Lead III
      Column 2: aVR, aVL, aVF
      Column 3: V1, V2, V3
      Column 4: V4, V5, V6
      Bottom row (spanning full width): Lead II rhythm lead
      
    Background is rendered as a reddish-brown grid representing clinical paper:
    - Small squares: 1 mm x 1 mm (0.04 s x 0.1 mV)
    - Large squares: 5 mm x 5 mm (0.2 s x 0.5 mV)
    """
    plt = _check_matplotlib()
    
    clean = record.clean
    noisy = record.noisy
    fs = record.fs
    t = record.t
    
    # 12-lead ECG requires 12 signals
    # If the simulation record does not contain 12 signals, we will mock them by applying
    # different phase shifts and scaling to simulate leads, or plot channels that are available.
    n_ch = 1 if clean.ndim == 1 else clean.shape[0]
    
    leads_labels = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # Generate mock 12 lead signals if single-channel or 12-channel is not available
    clean_leads = []
    noisy_leads = []
    
    if n_ch >= 12:
        for idx in range(12):
            clean_leads.append(clean[idx])
            noisy_leads.append(noisy[idx])
    else:
        # Mock projection from Lead I (Lead 0) using projection constants
        base_clean = clean[0] if clean.ndim > 1 else clean
        base_noisy = noisy[0] if noisy.ndim > 1 else noisy
        
        # Simple projection matrix for testing visual output
        for idx in range(12):
            # Phase shifts to create visual variance
            shift = int(np.round(idx * 0.05 * fs))
            rolled_clean = np.roll(base_clean, shift)
            rolled_noisy = np.roll(base_noisy, shift)
            
            scale = 1.0 - 0.05 * (idx % 4)
            if idx in [3, 4]:  # Augmented leads inverted
                scale = -scale
                
            clean_leads.append(rolled_clean * scale)
            noisy_leads.append(rolled_noisy * scale)
            
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#ffffff') # standard paper is white
    
    # 3x4 + 1 grid: 4 rows total. 
    # Row 0, 1, 2 have 4 columns
    # Row 3 contains the rhythm strip spanning all 4 columns
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1.2])
    
    # Setup subplots
    axes = []
    for r in range(3):
        for c in range(4):
            axes.append(fig.add_subplot(gs[r, c]))
    # Add rhythm strip
    rhythm_ax = fig.add_subplot(gs[3, :])
    axes.append(rhythm_ax)
    
    # ECG Grid Drawer Helper
    def draw_ecg_grid(ax, t_max, v_min, v_max):
        # Grid settings
        # Paper speed 25 mm/s -> 1 s = 25 mm. Grid spacing = 1 mm -> 0.04 s.
        # Amplitude 10 mm/mV -> 1 mV = 10 mm. Grid spacing = 1 mm -> 0.1 mV.
        # Major grid: 5 mm -> 0.2 s, 0.5 mV.
        
        ax.set_facecolor('#ffebee') # Light peach/pink ECG paper
        
        t_minor = np.arange(0, t_max, 0.04)
        t_major = np.arange(0, t_max, 0.2)
        
        v_minor = np.arange(v_min, v_max, 0.1)
        v_major = np.arange(v_min, v_max, 0.5)
        
        # Draw minor lines
        for x in t_minor:
            ax.axvline(x, color='#ffcdd2', lw=0.4, zorder=0)
        for y in v_minor:
            ax.axhline(y, color='#ffcdd2', lw=0.4, zorder=0)
            
        # Draw major lines
        for x in t_major:
            ax.axvline(x, color='#ef9a9a', lw=0.9, zorder=0)
        for y in v_major:
            ax.axhline(y, color='#ef9a9a', lw=0.9, zorder=0)
            
        ax.set_xlim(0, t_max)
        ax.set_ylim(v_min, v_max)
        ax.tick_params(colors='#d32f2f', labelsize=8)
        
    t_span = min(2.5, t[-1])  # Standard clinical strip segment is 2.5 seconds
    n_pts_span = int(np.round(t_span * fs))
    
    v_min, v_max = -2.0, 2.0
    
    # Plot first 12 leads
    for idx in range(12):
        ax = axes[idx]
        draw_ecg_grid(ax, t_span, v_min, v_max)
        
        # Plot noisy in orange/brown, clean in black/dark-red
        ax.plot(t[:n_pts_span], noisy_leads[idx][:n_pts_span], color='#795548', lw=1.0, alpha=0.75, zorder=1)
        ax.plot(t[:n_pts_span], clean_leads[idx][:n_pts_span], color='#b71c1c', lw=1.2, zorder=2)
        
        ax.text(0.05, 1.6, leads_labels[idx], color='#b71c1c', weight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Standard calibration mark on column 0, row 0 (I)
        if idx == 0:
            # Mark: 0.2 s wide, 1.0 mV high
            cal_t = np.array([0.05, 0.05, 0.15, 0.15, 0.25, 0.25])
            cal_v = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
            ax.plot(cal_t, cal_v - 1.0, color='#b71c1c', lw=2.0, zorder=3)
            
    # Plot rhythm lead (Lead II) over longer duration
    rhythm_t_span = min(10.0, t[-1])
    n_rhythm_pts = int(np.round(rhythm_t_span * fs))
    
    draw_ecg_grid(rhythm_ax, rhythm_t_span, v_min, v_max)
    rhythm_ax.plot(t[:n_rhythm_pts], noisy_leads[1][:n_rhythm_pts], color='#795548', lw=1.0, alpha=0.75, zorder=1)
    rhythm_ax.plot(t[:n_rhythm_pts], clean_leads[1][:n_rhythm_pts], color='#b71c1c', lw=1.3, zorder=2)
    rhythm_ax.text(0.05, 1.5, "Rhythm Strip: II (Continuous)", color='#b71c1c', weight='bold', fontsize=11, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.suptitle("Standard Clinical 12-Lead ECG Report (25 mm/s, 10 mm/mV)", fontsize=16, color='#b71c1c', weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def plot_eeg_spectrogram_and_bands(record: SignalRecord) -> object:
    """
    Renders a joint panel showing raw EEG time-series, spectrogram density, and relative band power trends.
    """
    plt = _check_matplotlib()
    
    eeg = record.clean[0] if record.clean.ndim > 1 else record.clean
    noisy = record.noisy[0] if record.noisy.ndim > 1 else record.noisy
    fs = record.fs
    t = record.t
    
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#0f172a')
    
    # Layout:
    # Top panel: raw time series (clean vs noisy)
    # Bottom left: spectrogram
    # Bottom right: relative spectral band energy distribution
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2], width_ratios=[1.2, 0.8])
    
    ax_time = fig.add_subplot(gs[0, :])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_bands = fig.add_subplot(gs[1, 1])
    
    # Time domain
    ax_time.set_facecolor('#111827')
    ax_time.plot(t, noisy, color='#f59e0b', lw=1.0, alpha=0.6, label='Noisy EEG')
    ax_time.plot(t, eeg, color='#3b82f6', lw=1.2, label='Clean EEG')
    ax_time.set_title("EEG Time Series (Channel 1)", color='#e2e8f0', fontsize=12)
    ax_time.set_xlabel("Time (s)", color='#94a3b8')
    ax_time.set_ylabel("Amplitude (uV)", color='#94a3b8')
    ax_time.tick_params(colors='#94a3b8')
    ax_time.grid(True, alpha=0.1, color='#e2e8f0')
    ax_time.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
    
    # Spectrogram
    # Using scipy spectrogram
    nperseg = min(256, len(eeg))
    noverlap = nperseg // 2
    f, ts, Sxx = sp_signal.spectrogram(eeg - np.mean(eeg), fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Focus on standard EEG bands: 0.5 - 40 Hz
    mask = f <= 40.0
    f_masked = f[mask]
    Sxx_masked = Sxx[mask, :]
    
    # Convert to log scale dB
    Sxx_db = 10 * np.log10(np.clip(Sxx_masked, 1e-15, None))
    
    ax_spec.set_facecolor('#111827')
    im = ax_spec.pcolormesh(ts, f_masked, Sxx_db, shading='gouraud', cmap='viridis')
    ax_spec.set_title("Spectrogram Density (0.5 - 40 Hz)", color='#e2e8f0', fontsize=12)
    ax_spec.set_xlabel("Time (s)", color='#94a3b8')
    ax_spec.set_ylabel("Frequency (Hz)", color='#94a3b8')
    ax_spec.tick_params(colors='#94a3b8')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_spec, orientation='horizontal', pad=0.15)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=8)
    cbar.set_label('Power Spectral Density (dB)', color='#94a3b8')
    
    # Bands relative distribution
    bands = {
        'Delta\n(0.5-4Hz)': (0.5, 4.0),
        'Theta\n(4-8Hz)': (4.0, 8.0),
        'Alpha\n(8-12Hz)': (8.0, 12.0),
        'Beta\n(12-30Hz)': (12.0, 30.0),
        'Gamma\n(30-50Hz)': (30.0, 50.0)
    }
    
    # Extract total PSD for band power bar plot
    f_psd, psd = sp_signal.welch(eeg - np.mean(eeg), fs=fs, nperseg=min(512, len(eeg)))
    total_power = 1e-15
    powers = []
    
    for label, (fmin, fmax) in bands.items():
        band_mask = (f_psd >= fmin) & (f_psd <= fmax)
        if np.any(band_mask):
            p = float(_trapz(psd[band_mask], f_psd[band_mask]))
        else:
            p = 0.0
        powers.append(p)
        total_power += p
        
    rel_powers = [p / total_power for p in powers]
    
    ax_bands.set_facecolor('#111827')
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    bars = ax_bands.bar(bands.keys(), rel_powers, color=colors, edgecolor='none', alpha=0.8)
    ax_bands.set_title("Relative Spectral Band Energy", color='#e2e8f0', fontsize=12)
    ax_bands.set_ylabel("Relative Power Ratio", color='#94a3b8')
    ax_bands.tick_params(colors='#94a3b8', labelsize=9)
    ax_bands.grid(True, axis='y', alpha=0.1, color='#e2e8f0')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_bands.text(
            bar.get_x() + bar.get_width()/2.0, height + 0.02,
            f"{height*100:.1f}%", ha='center', va='bottom', color='#e2e8f0', fontsize=9
        )
        
    plt.suptitle("EEG Multi-Domain Analysis Report", fontsize=15, color='#ffffff', weight='bold')
    plt.tight_layout()
    return fig


def plot_emg_fatigue_indicators(record: SignalRecord, window_size_s: float = 1.0) -> object:
    """
    Plots the EMG activity profile and tracking indices showing shifts in Mean/Median frequencies
    as indicators of localized muscle fatigue.
    """
    plt = _check_matplotlib()
    
    emg = record.clean[0] if record.clean.ndim > 1 else record.clean
    fs = record.fs
    t = record.t
    
    # Calculate rolling metrics
    win_samples = int(np.round(window_size_s * fs))
    step_samples = win_samples // 4  # 75% overlap
    
    n_samples = len(emg)
    times = []
    mnf_profile = []
    mdf_profile = []
    rms_profile = []
    
    start = 0
    while (start + win_samples) <= n_samples:
        segment = emg[start:start+win_samples]
        segment_centered = segment - np.mean(segment)
        
        # Time of segment center
        times.append((start + win_samples//2) / fs)
        
        # RMS power
        rms_profile.append(np.sqrt(np.mean(segment ** 2)))
        
        # PSD
        f, psd = sp_signal.welch(segment_centered, fs=fs, nperseg=min(256, len(segment_centered)))
        mask = (f >= 10.0) & (f <= 250.0)
        f_m = f[mask]
        psd_m = psd[mask]
        
        if len(psd_m) > 0:
            mnf = np.sum(f_m * psd_m) / np.sum(psd_m)
            
            cum = np.cumsum(psd_m)
            half = cum[-1] / 2.0
            idx = np.where(cum >= half)[0][0]
            mdf = f_m[idx]
            
            mnf_profile.append(mnf)
            mdf_profile.append(mdf)
        else:
            mnf_profile.append(0.0)
            mdf_profile.append(0.0)
            
        start += step_samples
        
    times_arr = np.array(times)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.15, color='#e2e8f0')
        
    # Panel 1: Raw EMG Envelope
    axes[0].plot(t, emg, color='#94a3b8', lw=0.6, alpha=0.5, label='Raw EMG')
    # Rolling RMS overlay
    axes[0].plot(times_arr, rms_profile, color='#ef4444', lw=2.0, label='Rolling RMS Envelope')
    axes[0].set_title("EMG Amplitude Profile & Envelope", color='#e2e8f0', fontsize=12)
    axes[0].set_ylabel("Amplitude (mV)", color='#94a3b8')
    axes[0].legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
    
    # Panel 2: Mean Frequency Trend (MNF)
    axes[1].plot(times_arr, mnf_profile, 'o-', color='#10b981', lw=1.8, ms=4, label='Mean Frequency')
    # Linear fit to show trend slope
    if len(times_arr) > 1:
        slope, intercept = np.polyfit(times_arr, mnf_profile, 1)
        axes[1].plot(times_arr, slope * times_arr + intercept, '--', color='#a7f3d0', lw=1.5, label=f"Trend (Slope: {slope:.3f} Hz/s)")
    axes[1].set_title("Mean Frequency (MNF) Trend", color='#e2e8f0', fontsize=12)
    axes[1].set_ylabel("MNF (Hz)", color='#94a3b8')
    axes[1].legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
    
    # Panel 3: Median Frequency Trend (MDF)
    axes[2].plot(times_arr, mdf_profile, 's-', color='#8b5cf6', lw=1.8, ms=4, label='Median Frequency')
    if len(times_arr) > 1:
        slope, intercept = np.polyfit(times_arr, mdf_profile, 1)
        axes[2].plot(times_arr, slope * times_arr + intercept, '--', color='#ddd6fe', lw=1.5, label=f"Trend (Slope: {slope:.3f} Hz/s)")
    axes[2].set_title("Median Frequency (MDF) Fatigue Tracking", color='#e2e8f0', fontsize=12)
    axes[2].set_ylabel("MDF (Hz)", color='#94a3b8')
    axes[2].set_xlabel("Time (s)", color='#94a3b8')
    axes[2].legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
    
    # Analysis summary
    fatigue_text = "No Fatigue Indication"
    if len(times_arr) > 1:
        if slope < -0.2:
            fatigue_text = "CRITICAL FATIGUE DETECTED (Significant downward shift)"
        elif slope < 0.0:
            fatigue_text = "MODERATE FATIGUE (Stable downward trend)"
            
    plt.suptitle(f"EMG Muscle Fatigue Characterization ({fatigue_text})", fontsize=14, color='#ffffff', weight='bold')
    plt.tight_layout()
    return fig


def plot_eda_decomposition(record: SignalRecord) -> object:
    """
    Renders raw EDA signal side-by-side with separated tonic (Skin Conductance Level)
    and phasic (Skin Conductance Response) components.
    """
    plt = _check_matplotlib()
    
    eda = record.clean[0] if record.clean.ndim > 1 else record.clean
    t = record.t
    
    # Reconstruct components from metadata if available, otherwise apply basic filters
    # Tonic (low pass filter below 0.05 Hz)
    # Phasic (high pass filter or subtracted tonic)
    fs = record.fs
    nyq = 0.5 * fs
    
    b_tonic, a_tonic = sp_signal.butter(2, 0.05 / nyq, btype='lowpass')
    tonic = sp_signal.filtfilt(b_tonic, a_tonic, eda)
    phasic = eda - tonic
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.15, color='#e2e8f0')
        
    # Panel 1: Raw EDA
    axes[0].plot(t, eda, color='#8b5cf6', lw=1.8, label='Raw EDA')
    axes[0].plot(t, tonic, '--', color='#10b981', lw=1.2, label='Decomposed Tonic Baseline')
    axes[0].set_title("Electrodermal Activity (EDA)", color='#e2e8f0', fontsize=12)
    axes[0].set_ylabel("Conductance (uS)", color='#94a3b8')
    axes[0].legend(facecolor='#1e293b', edgecolor='none', labelcolor='#e2e8f0')
    
    # Panel 2: Decomposed Tonic Component (SCL)
    axes[1].plot(t, tonic, color='#10b981', lw=2.0)
    axes[1].set_title("Skin Conductance Level (Tonic SCL)", color='#e2e8f0', fontsize=12)
    axes[1].set_ylabel("Tonic (uS)", color='#94a3b8')
    
    # Panel 3: Decomposed Phasic Component (SCR)
    axes[2].plot(t, phasic, color='#f59e0b', lw=1.5)
    axes[2].set_title("Skin Conductance Response (Phasic SCR Peaks)", color='#e2e8f0', fontsize=12)
    axes[2].set_ylabel("Phasic (uS)", color='#94a3b8')
    axes[2].set_xlabel("Time (s)", color='#94a3b8')
    
    plt.suptitle("EDA Signal Decomposition Report", fontsize=14, color='#ffffff', weight='bold')
    plt.tight_layout()
    return fig


def generate_interactive_html_dashboard(record: SignalRecord, path: str) -> None:
    """
    Generates a premium standalone interactive HTML dashboard.
    
    Embeds the entire signal data series and renders interactive zoomable vector
    charts using HTML5 canvas/SVG and native JavaScript with drag controls.
    """
    clean_data = record.clean
    noisy_data = record.noisy
    
    n_ch = 1 if clean_data.ndim == 1 else clean_data.shape[0]
    
    # Serialize arrays to lists to embed in JS
    js_data = {
        't': record.t.tolist(),
        'clean': clean_data.tolist() if clean_data.ndim == 1 else [clean_data[c].tolist() for c in range(n_ch)],
        'noisy': noisy_data.tolist() if noisy_data.ndim == 1 else [noisy_data[c].tolist() for c in range(n_ch)],
        'noise_components': {k: (v.tolist() if v.ndim == 1 else [v[c].tolist() for c in range(n_ch)]) for k, v in record.noise_components.items()}
    }
    
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BioSignal Simulator Interactive Dashboard</title>
    <style>
        body {{
            font-family: 'Outfit', 'Inter', -apple-system, sans-serif;
            background-color: #0b0f19;
            color: #e2e8f0;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
            color: #ffffff;
        }}
        .info-panel {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .info-card {{
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
            backdrop-filter: blur(8px);
        }}
        .info-card span {{
            display: block;
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .info-card strong {{
            font-size: 18px;
            color: #cbd5e1;
        }}
        .main-layout {{
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 20px;
        }}
        .chart-container {{
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}
        .chart-controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }}
        button {{
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s;
        }}
        button:hover {{
            background-color: #2563eb;
        }}
        button.secondary {{
            background-color: rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
        }}
        button.secondary:hover {{
            background-color: rgba(255, 255, 255, 0.15);
        }}
        canvas {{
            background-color: #070a13;
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            width: 100%;
            height: 450px;
            cursor: crosshair;
        }}
        .sidebar {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .control-box {{
            background: rgba(30, 41, 59, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
        }}
        .control-box h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 15px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
            color: #818cf8;
        }}
        .checkbox-group {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .checkbox-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
        }}
        .checkbox-item input {{
            cursor: pointer;
        }}
        .color-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}
        .instructions {{
            font-size: 12px;
            color: #64748b;
            line-height: 1.5;
        }}
        .instructions ul {{
            padding-left: 15px;
            margin: 5px 0 0 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>BioSignal Waveform Simulator Dashboard</h1>
            <p style="margin: 5px 0 0 0; font-size: 13px; color: #64748b;">Interactive High-Fidelity Signal Diagnostic System</p>
        </div>
        <div>
            <span style="font-size: 12px; color: #475569;">Generated on: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
        </div>
    </div>
    
    <div class="info-panel">
        <div class="info-card">
            <span>Signal Type</span>
            <strong>{record.signal_type.upper()}</strong>
        </div>
        <div class="info-card">
            <span>Sampling Frequency</span>
            <strong>{record.fs:.1f} Hz</strong>
        </div>
        <div class="info-card">
            <span>Duration</span>
            <strong>{record.t[-1]:.2f} seconds</strong>
        </div>
        <div class="info-card">
            <span>SNR Level</span>
            <strong>{f"{record.snr_db:.1f} dB" if record.snr_db is not None else "No Noise Added"}</strong>
        </div>
    </div>
    
    <div class="main-layout">
        <div class="chart-container">
            <div class="chart-controls">
                <button id="btn-zoom-fit">Fit Screen</button>
                <button id="btn-reset" class="secondary">Reset View</button>
                <span id="coordinates" style="margin-left: auto; align-self: center; font-size: 12px; color: #64748b;">Coords: --</span>
            </div>
            
            <canvas id="chart-canvas"></canvas>
        </div>
        
        <div class="sidebar">
            <div class="control-box">
                <h3>Visible Channels</h3>
                <div class="checkbox-group" id="channels-toggle">
                    <label class="checkbox-item">
                        <input type="checkbox" id="chk-clean" checked>
                        <span class="color-dot" style="background-color: #10b981;"></span>
                        Clean Signal
                    </label>
                    <label class="checkbox-item">
                        <input type="checkbox" id="chk-noisy" checked>
                        <span class="color-dot" style="background-color: #f59e0b;"></span>
                        Noisy Signal
                    </label>
                    <!-- Noise components will be injected dynamically in JavaScript -->
                </div>
            </div>
            
            <div class="control-box">
                <h3>Interactivity Guides</h3>
                <div class="instructions">
                    <ul>
                        <li><strong>Zoom</strong>: Click and drag horizontally/vertically over the canvas to zoom in on a region.</li>
                        <li><strong>Pan</strong>: Hold the <code>Shift</code> key while dragging to pan horizontally.</li>
                        <li><strong>Reset</strong>: Double click anywhere on the chart, or click <em>Fit Screen</em> above to restore original zoom levels.</li>
                        <li><strong>Toggle</strong>: Use the sidebar checkboxes to show or hide signal waveforms and subcomponents.</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Inject data
        const signalData = {json.dumps(js_data)};
        const fs = {record.fs};
        const nChannels = {n_ch};
        
        // Setup Canvas variables
        const canvas = document.getElementById('chart-canvas');
        const ctx = canvas.getContext('2d');
        
        let visibleComponents = {{
            clean: true,
            noisy: true
        }};
        
        // Setup dynamic toggles for noise components
        const toggleContainer = document.getElementById('channels-toggle');
        const colors = ['#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#f43f5e', '#10b981'];
        let colorIdx = 0;
        
        const componentColors = {{
            clean: '#10b981',
            noisy: '#f59e0b'
        }};
        
        for (const name in signalData.noise_components) {{
            visibleComponents[name] = false; // default hidden to avoid clutter
            const col = colors[colorIdx % colors.length];
            colorIdx++;
            componentColors[name] = col;
            
            const label = document.createElement('label');
            label.className = 'checkbox-item';
            label.innerHTML = `
                <input type="checkbox" id="chk-${{name}}">
                <span class="color-dot" style="background-color: ${{col}};"></span>
                Noise: ${{name}}
            `;
            toggleContainer.appendChild(label);
            
            document.getElementById(`chk-${{name}}`).addEventListener('change', (e) => {{
                visibleComponents[name] = e.target.checked;
                draw();
            }});
        }}
        
        document.getElementById('chk-clean').addEventListener('change', (e) => {{
            visibleComponents.clean = e.target.checked;
            draw();
        }});
        document.getElementById('chk-noisy').addEventListener('change', (e) => {{
            visibleComponents.noisy = e.target.checked;
            draw();
        }});
        
        // Chart limits
        const totalSamples = signalData.t.length;
        let xMin = signalData.t[0];
        let xMax = signalData.t[totalSamples - 1];
        
        let yMin = -1.5;
        let yMax = 1.5;
        
        // Reset limits to automatically encompass signals
        function fitLimits() {{
            xMin = signalData.t[0];
            xMax = signalData.t[totalSamples - 1];
            
            let minVal = Infinity;
            let maxVal = -Infinity;
            
            for (let i = 0; i < totalSamples; i++) {{
                if (visibleComponents.clean) {{
                    const val = nChannels > 1 ? signalData.clean[0][i] : signalData.clean[i];
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }}
                if (visibleComponents.noisy) {{
                    const val = nChannels > 1 ? signalData.noisy[0][i] : signalData.noisy[i];
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                }}
                for (const name in signalData.noise_components) {{
                    if (visibleComponents[name]) {{
                        const val = nChannels > 1 ? signalData.noise_components[name][0][i] : signalData.noise_components[name][i];
                        if (val < minVal) minVal = val;
                        if (val > maxVal) maxVal = val;
                    }}
                }}
            }}
            
            if (minVal === Infinity) {{
                yMin = -1.0;
                yMax = 1.0;
            }} else {{
                const pad = (maxVal - minVal) * 0.1 || 0.5;
                yMin = minVal - pad;
                yMax = maxVal + pad;
            }}
        }}
        
        // Coordinate conversion
        function toScreenX(x) {{
            const margin = 60;
            const w = canvas.width - margin - 20;
            return margin + ((x - xMin) / (xMax - xMin)) * w;
        }}
        
        function toScreenY(y) {{
            const margin = 40;
            const h = canvas.height - margin - 40;
            return margin + (1.0 - (y - yMin) / (yMax - yMin)) * h;
        }}
        
        function toDataX(sx) {{
            const margin = 60;
            const w = canvas.width - margin - 20;
            return xMin + ((sx - margin) / w) * (xMax - xMin);
        }}
        
        function toDataY(sy) {{
            const margin = 40;
            const h = canvas.height - margin - 40;
            return yMin + (1.0 - (sy - margin) / h) * (yMax - yMin);
        }}
        
        // Handle resizing
        function resizeCanvas() {{
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            // Redraw
            draw();
        }}
        
        window.addEventListener('resize', resizeCanvas);
        
        // Draw chart
        function draw() {{
            // Clear
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const w = canvas.width / (window.devicePixelRatio || 1);
            const h = canvas.height / (window.devicePixelRatio || 1);
            
            const chartLeft = 60;
            const chartRight = w - 20;
            const chartTop = 40;
            const chartBottom = h - 40;
            
            // Draw plot background grid
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
            ctx.lineWidth = 1;
            
            // Vertical grid lines
            const xStep = (xMax - xMin) / 10;
            for (let i = 0; i <= 10; i++) {{
                const xVal = xMin + i * xStep;
                const sx = toScreenX(xVal);
                if (sx >= chartLeft && sx <= chartRight) {{
                    ctx.beginPath();
                    ctx.moveTo(sx, chartTop);
                    ctx.lineTo(sx, chartBottom);
                    ctx.stroke();
                    
                    // Draw axis text
                    ctx.fillStyle = '#64748b';
                    ctx.font = '10px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(xVal.toFixed(2) + 's', sx, chartBottom + 18);
                }}
            }}
            
            // Horizontal grid lines
            const yStep = (yMax - yMin) / 8;
            for (let i = 0; i <= 8; i++) {{
                const yVal = yMin + i * yStep;
                const sy = toScreenY(yVal);
                if (sy >= chartTop && sy <= chartBottom) {{
                    ctx.beginPath();
                    ctx.moveTo(chartLeft, sy);
                    ctx.lineTo(chartRight, sy);
                    ctx.stroke();
                    
                    // Axis label
                    ctx.fillStyle = '#64748b';
                    ctx.font = '10px sans-serif';
                    ctx.textAlign = 'right';
                    ctx.fillText(yVal.toFixed(2), chartLeft - 8, sy + 3);
                }}
            }}
            
            // Draw signal traces
            function drawTrace(points, color, width) {{
                ctx.strokeStyle = color;
                ctx.lineWidth = width;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.beginPath();
                
                let active = false;
                
                // Optimized stride plotting for long files
                const pixelWidth = xMax - xMin;
                const stride = Math.max(1, Math.floor(totalSamples / (w * 2)));
                
                for (let i = 0; i < totalSamples; i += stride) {{
                    const tx = signalData.t[i];
                    if (tx < xMin || tx > xMax) continue;
                    
                    const val = points[i];
                    const sx = toScreenX(tx);
                    const sy = toScreenY(val);
                    
                    if (!active) {{
                        ctx.moveTo(sx, sy);
                        active = true;
                    }} else {{
                        ctx.lineTo(sx, sy);
                    }}
                }}
                ctx.stroke();
            }}
            
            if (visibleComponents.noisy) {{
                const raw = nChannels > 1 ? signalData.noisy[0] : signalData.noisy;
                drawTrace(raw, componentColors.noisy, 1.5);
            }}
            if (visibleComponents.clean) {{
                const raw = nChannels > 1 ? signalData.clean[0] : signalData.clean;
                drawTrace(raw, componentColors.clean, 1.8);
            }}
            for (const name in signalData.noise_components) {{
                if (visibleComponents[name]) {{
                    const raw = nChannels > 1 ? signalData.noise_components[name][0] : signalData.noise_components[name];
                    drawTrace(raw, componentColors[name], 1.0);
                }}
            }}
            
            // Border box
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(chartLeft, chartTop, chartRight - chartLeft, chartBottom - chartTop);
            
            // Draw zoom selection box if dragging
            if (isDragging && !isPanning && dragStartScreenX !== null && dragCurrentScreenX !== null) {{
                ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
                ctx.strokeStyle = '#3b82f6';
                ctx.lineWidth = 1;
                const x = Math.min(dragStartScreenX, dragCurrentScreenX);
                const width = Math.abs(dragStartScreenX - dragCurrentScreenX);
                const y = Math.min(dragStartScreenY, dragCurrentScreenY);
                const height = Math.abs(dragStartScreenY - dragCurrentScreenY);
                ctx.fillRect(x, y, width, height);
                ctx.strokeRect(x, y, width, height);
            }}
        }}
        
        // Mouse/Touch Interaction Controls
        let isDragging = false;
        let isPanning = false;
        let dragStartScreenX = null;
        let dragStartScreenY = null;
        let dragCurrentScreenX = null;
        let dragCurrentScreenY = null;
        
        let dragStartDataX = null;
        let dragStartDataY = null;
        
        canvas.addEventListener('mousedown', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            
            const chartLeft = 60;
            const w = canvas.width / (window.devicePixelRatio || 1);
            const chartRight = w - 20;
            
            if (sx < chartLeft || sx > chartRight) return;
            
            isDragging = true;
            isPanning = e.shiftKey;
            
            dragStartScreenX = sx;
            dragStartScreenY = sy;
            dragCurrentScreenX = sx;
            dragCurrentScreenY = sy;
            
            dragStartDataX = toDataX(sx);
            dragStartDataY = toDataY(sy);
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            
            // Show coordinate label
            const valX = toDataX(sx);
            const valY = toDataY(sy);
            document.getElementById('coordinates').innerText = `Time: ${{valX.toFixed(2)}}s, Val: ${{valY.toFixed(3)}}`;
            
            if (!isDragging) return;
            
            dragCurrentScreenX = sx;
            dragCurrentScreenY = sy;
            
            if (isPanning) {{
                const dx = valX - dragStartDataX;
                const dy = valY - dragStartDataY;
                xMin -= dx;
                xMax -= dx;
                yMin -= dy;
                yMax -= dy;
                dragStartDataX = toDataX(sx);
                dragStartDataY = toDataY(sy);
            }}
            
            draw();
        }});
        
        window.addEventListener('mouseup', () => {{
            if (!isDragging) return;
            isDragging = false;
            
            if (!isPanning && dragStartScreenX !== null && dragCurrentScreenX !== null) {{
                const dX = Math.abs(dragStartScreenX - dragCurrentScreenX);
                const dY = Math.abs(dragStartScreenY - dragCurrentScreenY);
                
                // Threshold to avoid zoom on simple click
                if (dX > 10 || dY > 10) {{
                    const x0 = toDataX(Math.min(dragStartScreenX, dragCurrentScreenX));
                    const x1 = toDataX(Math.max(dragStartScreenX, dragCurrentScreenX));
                    const y0 = toDataY(Math.max(dragStartScreenY, dragCurrentScreenY)); // y is inverted on screen
                    const y1 = toDataY(Math.min(dragStartScreenY, dragCurrentScreenY));
                    
                    xMin = x0;
                    xMax = x1;
                    yMin = y0;
                    yMax = y1;
                }}
            }}
            
            dragStartScreenX = null;
            dragCurrentScreenX = null;
            draw();
        }});
        
        canvas.addEventListener('dblclick', () => {{
            fitLimits();
            draw();
        }});
        
        document.getElementById('btn-zoom-fit').addEventListener('click', () => {{
            fitLimits();
            draw();
        }});
        
        document.getElementById('btn-reset').addEventListener('click', () => {{
            fitLimits();
            draw();
        }});
        
        // Initialize
        fitLimits();
        setTimeout(resizeCanvas, 100);
        
    </script>
</body>
</html>
"""
    
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_template)
