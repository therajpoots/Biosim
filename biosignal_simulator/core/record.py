"""
Metadata and storage containers for generated physiological signal records.

This module provides:
1. SignalRecordMetadata: Nested metadata container with UUIDs, timestamps, and annotations.
2. SignalRecord: Core record container holding clean, noisy, and raw noise arrays,
   along with calibration details, achieved SNR, and automatic signal quality checks.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple, List
import json
import numpy as np
from datetime import datetime, timezone
import uuid as uuid_tool

# Import mathematical and statistical utilities
from biosignal_simulator.core.math_utils import (
    compute_rms,
    compute_skewness,
    compute_kurtosis,
    compute_zcr,
    compute_shannon_entropy,
    crest_factor,
    peak_to_average_power_ratio,
    compute_rmse,
    compute_ssim_1d,
    compute_snr_components
)

@dataclass
class SignalRecordMetadata:
    """
    Metadata information for a physiological simulation record.
    
    Attributes
    ----------
    signal_type : str
        Type of signal (e.g. 'ecg', 'eeg', 'emg', 'ppg', 'eda', 'resp').
    fs : float
        Sampling frequency in Hz.
    signal_duration_s : float
        Signal duration in seconds.
    n_channels : int
        Number of channels represented in the signal.
    n_samples : int
        Total number of digitized samples per channel.
    timestamp : str
        Creation timestamp in ISO-8601 format.
    uuid : str
        Globally unique identifier for the record.
    user_notes : str
        Optional custom annotation string.
    """
    signal_type: str
    fs: float
    signal_duration_s: float
    n_channels: int
    n_samples: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    uuid: str = field(default_factory=lambda: str(uuid_tool.uuid4()))
    user_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata fields to a standard dictionary."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalRecordMetadata':
        """Reconstruct metadata container from a dictionary."""
        return cls(
            signal_type=data["signal_type"],
            fs=data["fs"],
            signal_duration_s=data["signal_duration_s"],
            n_channels=data["n_channels"],
            n_samples=data["n_samples"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            uuid=data.get("uuid", str(uuid_tool.uuid4())),
            user_notes=data.get("user_notes", "")
        )


@dataclass
class SignalRecord:
    """
    Container representing a digitized physiological signal record.
    
    Holds the original clean reference signal, the synthesized noisy output,
    individual noise component realizations, simulation configurations, 
    and achieved quality indicators.
    
    Attributes
    ----------
    signal_type : str
        Physiological signal type (e.g. 'ecg', 'eeg', 'emg', 'ppg', 'eda', 'resp').
    fs : float
        Sampling frequency in Hz.
    t : np.ndarray
        Time vector of shape (n_samples,).
    clean : np.ndarray
        Clean source signal of shape (n_samples,) or (n_channels, n_samples).
    noisy : np.ndarray
        Polluted signal array of shape matching `clean`.
    noise_components : Dict[str, np.ndarray]
        Realized noise channels indexable by noise model name.
    signal_params : Dict[str, Any]
        Configuration parameters used to generate the clean signal.
    noise_params : Dict[str, Any]
        Configuration parameters used for each applied noise model.
    snr_db : Optional[float]
        Post-hoc calculated SNR in dB.
    metadata : Dict[str, Any]
        Additional custom user-provided annotations or attributes.
    quality_flags : Dict[str, bool]
        Signal quality indicators computed during initialization.
    quality_metrics : Dict[str, Any]
        Statistical signal quality metrics computed during initialization.
    """
    signal_type: str
    fs: float
    t: np.ndarray
    clean: np.ndarray
    noisy: np.ndarray
    noise_components: Dict[str, np.ndarray] = field(default_factory=dict)
    signal_params: Dict[str, Any] = field(default_factory=dict)
    noise_params: Dict[str, Any] = field(default_factory=dict)
    snr_db: Optional[float] = None
    target_snr_db: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validate record dimensions and compute signal quality flags and metrics.
        
        Raises
        ------
        ValueError
            If array shapes do not match or are inconsistent with fs.
        """
        # Type and dimensionality checks
        if not isinstance(self.clean, np.ndarray):
            self.clean = np.array(self.clean, dtype=np.float64)
        if not isinstance(self.noisy, np.ndarray):
            self.noisy = np.array(self.noisy, dtype=np.float64)
            
        # Squeeze (1, N) arrays to 1D (N,) to standardize single-channel representation
        if self.clean.ndim == 2 and self.clean.shape[0] == 1:
            self.clean = self.clean.squeeze(0)
        if self.noisy.ndim == 2 and self.noisy.shape[0] == 1:
            self.noisy = self.noisy.squeeze(0)
            
        n_samples = len(self.t)
        
        # Check alignment of clean and noisy signals
        if self.clean.shape != self.noisy.shape:
            raise ValueError(
                f"Shape mismatch: clean signal is {self.clean.shape}, "
                f"but noisy signal is {self.noisy.shape}."
            )
            
        # Check matching length against time vector
        sig_len = self.clean.shape[-1]
        if sig_len != n_samples:
            raise ValueError(
                f"Length mismatch: Time vector has {n_samples} points, "
                f"but signal arrays have {sig_len} points."
            )
            
        # Check noise components
        for name, noise_arr in list(self.noise_components.items()):
            if not isinstance(noise_arr, np.ndarray):
                self.noise_components[name] = np.array(noise_arr, dtype=np.float64)
            if self.noise_components[name].ndim == 2 and self.noise_components[name].shape[0] == 1:
                self.noise_components[name] = self.noise_components[name].squeeze(0)
            if self.noise_components[name].shape != self.clean.shape:
                raise ValueError(
                    f"Shape mismatch in noise component '{name}': "
                    f"expected {self.clean.shape}, got {self.noise_components[name].shape}."
                )
                
        # Calculate channels
        self._n_channels = 1 if self.clean.ndim == 1 else self.clean.shape[0]
        
        # Instantiate metadata object
        self._metadata_obj = SignalRecordMetadata(
            signal_type=self.signal_type,
            fs=self.fs,
            # B-15 FIX: use n_samples/fs (true duration), not t[-1] (last sample time)
            signal_duration_s=float(n_samples) / self.fs if n_samples > 0 else 0.0,
            n_channels=self._n_channels,
            n_samples=n_samples,
            user_notes=str(self.metadata.get("notes", ""))
        )
        
        # Calculate detailed quality metrics
        self._calculate_quality_metrics()
        
        # Automatically scan signal quality
        self.quality_flags = self._compute_quality_flags()
        
    def _calculate_quality_metrics(self) -> None:
        """
        Compute descriptive statistical indices of clean, noisy, and noise components.
        
        Indices calculated:
        - Root Mean Square (RMS)
        - Skewness (biased and unbiased)
        - Kurtosis (excess)
        - Zero-Crossing Rate (ZCR)
        - Shannon Entropy
        - Crest Factor
        - Peak-to-Average Power Ratio (PAPR)
        - Root Mean Square Error (RMSE) between clean and noisy
        - 1-D Structural Similarity Index (SSIM)
        - SNR breakdown per noise component
        """
        self.quality_metrics = {}
        
        # Helper to compute standard statistics for a signal array
        def compute_stats(arr: np.ndarray, suffix: str):
            if arr.ndim == 1:
                self.quality_metrics[f"{suffix}_rms"] = float(compute_rms(arr))
                self.quality_metrics[f"{suffix}_skewness"] = float(compute_skewness(arr))
                self.quality_metrics[f"{suffix}_kurtosis"] = float(compute_kurtosis(arr))
                self.quality_metrics[f"{suffix}_zcr"] = float(compute_zcr(arr))
                self.quality_metrics[f"{suffix}_entropy"] = float(compute_shannon_entropy(arr))
                self.quality_metrics[f"{suffix}_crest_factor"] = float(crest_factor(arr))
                self.quality_metrics[f"{suffix}_papr"] = float(peak_to_average_power_ratio(arr))
            else:
                rms_vals, skew_vals, kurt_vals, zcr_vals, ent_vals, cf_vals, papr_vals = [], [], [], [], [], [], []
                for idx, ch_arr in enumerate(arr):
                    r = float(compute_rms(ch_arr))
                    sk = float(compute_skewness(ch_arr))
                    kt = float(compute_kurtosis(ch_arr))
                    z = float(compute_zcr(ch_arr))
                    e = float(compute_shannon_entropy(ch_arr))
                    cf = float(crest_factor(ch_arr))
                    pa = float(peak_to_average_power_ratio(ch_arr))
                    
                    rms_vals.append(r)
                    skew_vals.append(sk)
                    kurt_vals.append(kt)
                    zcr_vals.append(z)
                    ent_vals.append(e)
                    cf_vals.append(cf)
                    papr_vals.append(pa)
                    
                    self.quality_metrics[f"ch{idx}_{suffix}_rms"] = r
                    self.quality_metrics[f"ch{idx}_{suffix}_skewness"] = sk
                    self.quality_metrics[f"ch{idx}_{suffix}_kurtosis"] = kt
                    self.quality_metrics[f"ch{idx}_{suffix}_zcr"] = z
                    self.quality_metrics[f"ch{idx}_{suffix}_entropy"] = e
                    self.quality_metrics[f"ch{idx}_{suffix}_crest_factor"] = cf
                    self.quality_metrics[f"ch{idx}_{suffix}_papr"] = pa
                    
                self.quality_metrics[f"{suffix}_rms"] = float(np.mean(rms_vals))
                self.quality_metrics[f"{suffix}_skewness"] = float(np.mean(skew_vals))
                self.quality_metrics[f"{suffix}_kurtosis"] = float(np.mean(kurt_vals))
                self.quality_metrics[f"{suffix}_zcr"] = float(np.mean(zcr_vals))
                self.quality_metrics[f"{suffix}_entropy"] = float(np.mean(ent_vals))
                self.quality_metrics[f"{suffix}_crest_factor"] = float(np.mean(cf_vals))
                self.quality_metrics[f"{suffix}_papr"] = float(np.mean(papr_vals))
                
        # Calculate stats for clean and noisy signals
        compute_stats(self.clean, "clean")
        compute_stats(self.noisy, "noisy")
        
        # Calculate similarity and distortion metrics
        if self.clean.ndim == 1:
            self.quality_metrics["rmse"] = float(compute_rmse(self.clean, self.noisy))
            self.quality_metrics["ssim_index"] = float(compute_ssim_1d(self.clean, self.noisy))
        else:
            rmse_vals, ssim_vals = [], []
            for ch in range(self._n_channels):
                r_val = float(compute_rmse(self.clean[ch], self.noisy[ch]))
                s_val = float(compute_ssim_1d(self.clean[ch], self.noisy[ch]))
                rmse_vals.append(r_val)
                ssim_vals.append(s_val)
                self.quality_metrics[f"ch{ch}_rmse"] = r_val
                self.quality_metrics[f"ch{ch}_ssim_index"] = s_val
            self.quality_metrics["rmse"] = float(np.mean(rmse_vals))
            self.quality_metrics["ssim_index"] = float(np.mean(ssim_vals))
            
        # Calculate individual noise component contribution SNRs
        if self.noise_components:
            self.quality_metrics["noise_snr_breakdown"] = compute_snr_components(self.clean, self.noise_components)
            
    def _compute_quality_flags(self) -> Dict[str, bool]:
        """
        Check signal arrays for saturation, flatlines, gaps, and anomalous drift.
        
        Evaluated indicators:
        - `has_nan`: True if any value is NaN.
        - `has_inf`: True if any value is Infinite.
        - `is_clipped`: True if signal hits extreme saturation limits.
        - `has_dc_offset`: True if average amplitude exceeds 15% of standard deviation.
        - `too_noisy`: True if post-hoc SNR drops below -5.0 dB.
        - `is_flatline`: True if signal variance is near-zero (sensor detached or dead).
        - `has_high_kurtosis`: True if excess kurtosis exceeds 8.0, indicating spikes.
        
        Returns
        -------
        Dict[str, bool]
            A dictionary maps quality metrics to boolean flags.
        """
        flags = {
            "has_nan": bool(np.isnan(self.noisy).any() or np.isnan(self.clean).any()),
            "has_inf": bool(np.isinf(self.noisy).any() or np.isinf(self.clean).any()),
            "is_clipped": False,
            "has_dc_offset": False,
            "too_noisy": False,
            "is_flatline": False,
            "has_high_kurtosis": False
        }
        
        if flags["has_nan"] or flags["has_inf"]:
            return flags
            
        # Clipping (saturation) detection
        # B-09 FIX: Use consecutive-sample flatness (>=3 samples in a row at extrema)
        # instead of raw proximity to max/min, to avoid false positives on periodic signals.
        # Flatten to 1D for the diff/consecutive checks (works for 1D and 2D signals alike).
        noisy_flat = self.noisy.ravel()
        max_val = float(np.max(noisy_flat))
        min_val = float(np.min(noisy_flat))
        
        if max_val - min_val > 1e-3:
            tol = 1e-4 * (max_val - min_val)
            diffs = np.abs(np.diff(noisy_flat))
            is_flat = diffs < 1e-10
            # Consecutive flatness: at least 3 flat steps in a row
            has_consecutive_flat = is_flat[:-1] & is_flat[1:]
            flat_runs = np.zeros(len(noisy_flat), dtype=bool)
            flat_runs[:-2] |= has_consecutive_flat
            flat_runs[1:-1] |= has_consecutive_flat
            flat_runs[2:] |= has_consecutive_flat
            
            top_clip = np.sum((noisy_flat >= max_val - tol) & flat_runs)
            bot_clip = np.sum((noisy_flat <= min_val + tol) & flat_runs)
            total_pts = noisy_flat.size
            if top_clip > 0.001 * total_pts or bot_clip > 0.001 * total_pts:
                flags["is_clipped"] = True
            
        # DC offset detection
        mean_val = np.mean(self.noisy)
        std_val = np.std(self.noisy)
        if std_val > 1e-6 and abs(mean_val) > 0.15 * std_val:
            flags["has_dc_offset"] = True
            
        # Flatline detection
        if std_val <= 1e-6:
            flags["is_flatline"] = True
            
        # Kurtosis checking (excess artifact detection)
        if self.quality_metrics.get("noisy_kurtosis", 0.0) > 8.0:
            flags["has_high_kurtosis"] = True
            
        # Low SNR flag
        if self.snr_db is not None and self.snr_db < -5.0:
            flags["too_noisy"] = True
            
        return flags

    @property
    def duration_s(self) -> float:
        """Get the duration of the signal in seconds."""
        return self._metadata_obj.signal_duration_s

    @property
    def n_channels(self) -> int:
        """Get the number of signal channels."""
        return self._n_channels

    @property
    def n_samples(self) -> int:
        """Get the total number of digitized samples per channel."""
        return self._metadata_obj.n_samples

    @property
    def is_valid(self) -> bool:
        """
        Check if the record passes all automated quality criteria.
        
        Returns
        -------
        bool
            True if all quality_flags are False (i.e. no nan, inf, flatline, etc.).
        """
        return not any(self.quality_flags.values())

    @property
    def dynamic_range(self) -> float:
        """Get the dynamic range of the noisy signal (max - min)."""
        return float(np.max(self.noisy) - np.min(self.noisy))

    @property
    def peak_to_peak(self) -> float:
        """Get the peak-to-peak amplitude of the noisy signal."""
        return self.dynamic_range

    @property
    def sampling_rate(self) -> float:
        """Alias for the sampling frequency in Hz (fs)."""
        return self.fs

    @property
    def has_artifacts(self) -> bool:
        """
        Check if any environmental or hardware artifact quality flags are active.
        
        Returns
        -------
        bool
            True if clipped, dc offset, flatline, or high kurtosis flags are active.
        """
        return (
            self.quality_flags.get("is_clipped", False) or
            self.quality_flags.get("has_dc_offset", False) or
            self.quality_flags.get("is_flatline", False) or
            self.quality_flags.get("has_high_kurtosis", False)
        )

    @property
    def dynamic_range_clean(self) -> float:
        """Get the dynamic range of the clean signal (max - min)."""
        return float(np.max(self.clean) - np.min(self.clean))

    @property
    def peak_to_peak_clean(self) -> float:
        """Get the peak-to-peak amplitude of the clean signal."""
        return self.dynamic_range_clean

    @property
    def snr_linear(self) -> float:
        """
        Convert the post-hoc SNR from decibels to a linear power ratio.
        
        Formula:
            SNR_linear = 10^(SNR_dB / 10)
            
        Returns
        -------
        float
            Linear power ratio of signal to noise. Returns float('inf') if SNR is None.
        """
        if self.snr_db is None:
            return float('inf')
        return 10.0 ** (self.snr_db / 10.0)

    def plot(self, show: bool = True, filepath: Optional[str] = None) -> Any:
        """
        Plot the clean and noisy signals along with individual noise components.
        
        Uses matplotlib to render a multi-panel visualizer.
        
        Parameters
        ----------
        show : bool
            If True, displays the plot interactively (default: True).
        filepath : Optional[str]
            If provided, saves the plot to the specified path (default: None).
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Run `pip install matplotlib`.")
            
        # Determine number of rows based on number of channels and noise components
        n_plots = 1 + (1 if self.noise_components else 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
            
        # Plot Clean vs Noisy on the first panel
        ax0 = axes[0]
        if self._n_channels == 1:
            ax0.plot(self.t, self.clean, label="Clean Reference", color="#008080", alpha=0.9, linewidth=1.5)
            ax0.plot(self.t, self.noisy, label="Noisy Signal", color="#D2691E", alpha=0.7, linewidth=1.0)
        else:
            for ch in range(min(self._n_channels, 3)):  # Plot up to first 3 channels
                ax0.plot(self.t, self.clean[ch], label=f"Ch {ch} Clean", alpha=0.9, linewidth=1.5)
                ax0.plot(self.t, self.noisy[ch], label=f"Ch {ch} Noisy", alpha=0.6, linewidth=1.0)
                
        ax0.set_title(f"BioSignal Simulation: {self.signal_type.upper()} Clean vs Noisy (Actual SNR: {f'{self.snr_db:.2f} dB' if self.snr_db is not None else 'N/A'})")
        ax0.set_ylabel("Amplitude")
        ax0.legend(loc="upper right")
        ax0.grid(True, linestyle="--", alpha=0.5)
        
        # Plot individual noise components if present on the second panel
        if self.noise_components and len(axes) > 1:
            ax1 = axes[1]
            for name, noise in self.noise_components.items():
                if self._n_channels == 1:
                    ax1.plot(self.t, noise, label=name, alpha=0.8, linewidth=1.0)
                else:
                    ax1.plot(self.t, noise[0], label=f"{name} (Ch 0)", alpha=0.8, linewidth=1.0)
            ax1.set_title("Realized Noise Components")
            ax1.set_ylabel("Amplitude")
            ax1.set_xlabel("Time (seconds)")
            ax1.legend(loc="upper right")
            ax1.grid(True, linestyle="--", alpha=0.5)
        else:
            axes[0].set_xlabel("Time (seconds)")
            
        plt.tight_layout()
        
        if filepath:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            
        if show:
            plt.show()
            
        return fig

    def export_quality_report(self, filepath: str) -> None:
        """
        Generate a detailed Markdown diagnostic report of the record's signal quality.
        
        The report contains:
        - Metadata (UUID, timestamp, duration, sample rate, channels, samples).
        - Quality Assessment Flags (flatline, clipping, nan/inf, dc offset, high noise).
        - Detailed statistics (RMS, skewness, kurtosis, entropy, crest factor, PAPR, ZCR).
        - Discrepancy analysis (RMSE, SSIM between clean and noisy).
        - Noise components breakdown (RMS and SNR contributions).
        
        Parameters
        ----------
        filepath : str
            Target file path where the markdown report will be written.
        """
        lines = [
            f"# Physiological Signal Simulation Quality Report",
            f"",
            f"## 1. General Metadata",
            f"- **UUID**: {self._metadata_obj.uuid}",
            f"- **Timestamp**: {self._metadata_obj.timestamp}",
            f"- **Signal Type**: `{self.signal_type.upper()}`",
            f"- **Sampling Frequency**: {self.fs} Hz",
            f"- **Duration**: {self._metadata_obj.signal_duration_s:.3f} seconds",
            f"- **Number of Channels**: {self._n_channels}",
            f"- **Samples per Channel**: {self._metadata_obj.n_samples}",
            f"",
            f"## 2. Quality Flag Assessment",
            f"| Quality Check | Status | Description |",
            f"| :--- | :---: | :--- |",
        ]
        
        descriptions = {
            "has_nan": "Contains NaN (Not a Number) values",
            "has_inf": "Contains infinite values",
            "is_clipped": "Signal amplitude reaches extreme saturation boundaries (>0.1% points)",
            "has_dc_offset": "Significant DC baseline offset present (>15% standard deviation)",
            "too_noisy": "Achieved SNR is unacceptably low (below -5 dB)",
            "is_flatline": "Near-zero variance detected (sensor detached/dead signal)",
            "has_high_kurtosis": "Excess kurtosis is unusually high (>8.0), indicating large artifact spikes"
        }
        
        for flag, failed in self.quality_flags.items():
            status = "❌ FAILED" if failed else "✅ PASSED"
            desc = descriptions.get(flag, "Custom quality parameter check")
            lines.append(f"| {flag} | {status} | {desc} |")
            
        lines.extend([
            f"",
            f"## 3. Signal Statistical Metrics",
            f"| Metric / Statistical Moment | Clean Reference | Noisy Signal | Unit / Interpretation |",
            f"| :--- | :---: | :---: | :--- |",
            f"| **Root Mean Square (RMS)** | {self.quality_metrics.get('clean_rms', 0.0):.6f} | {self.quality_metrics.get('noisy_rms', 0.0):.6f} | Effective amplitude |",
            f"| **Skewness (Asymmetry)** | {self.quality_metrics.get('clean_skewness', 0.0):.6f} | {self.quality_metrics.get('noisy_skewness', 0.0):.6f} | Symmetry around mean (Normal = 0) |",
            f"| **Kurtosis (Tailedness)** | {self.quality_metrics.get('clean_kurtosis', 0.0):.6f} | {self.quality_metrics.get('noisy_kurtosis', 0.0):.6f} | Outlier severity (Excess, Normal = 0) |",
            f"| **Zero-Crossing Rate (ZCR)** | {self.quality_metrics.get('clean_zcr', 0.0):.6f} | {self.quality_metrics.get('noisy_zcr', 0.0):.6f} | Frequency threshold crossings |",
            f"| **Shannon Entropy** | {self.quality_metrics.get('clean_entropy', 0.0):.4f} | {self.quality_metrics.get('noisy_entropy', 0.0):.4f} | Information content (bits) |",
            f"| **Crest Factor** | {self.quality_metrics.get('clean_crest_factor', 0.0):.4f} | {self.quality_metrics.get('noisy_crest_factor', 0.0):.4f} | Peak amplitude to RMS ratio |",
            f"| **PAPR** | {self.quality_metrics.get('clean_papr', 0.0):.4f} | {self.quality_metrics.get('noisy_papr', 0.0):.4f} | Peak-to-Average Power Ratio (dB) |",
            f"",
            f"## 4. Discrepancy & Distortion Analysis",
            f"- **Root Mean Square Error (RMSE)**: {self.quality_metrics.get('rmse', 0.0):.6f}",
            f"- **1-D Structural Similarity Index (SSIM)**: {self.quality_metrics.get('ssim_index', 0.0):.6f} *(1.0 indicates perfect structural identity)*",
            f"- **Target SNR**: {self.signal_params.get('target_snr_db', 'N/A')} dB",
            f"- **Actual Signal-to-Noise Ratio (SNR)**: {f'{self.snr_db:.4f} dB' if self.snr_db is not None else 'N/A'}",
            f"",
            f"## 5. Noise Components Contribution Breakdown",
        ])
        
        if "noise_snr_breakdown" in self.quality_metrics:
            lines.append("| Noise Component | RMS Amplitude | Individual Contribution SNR |")
            lines.append("| :--- | :---: | :---: |")
            for name, snr_val in self.quality_metrics["noise_snr_breakdown"].items():
                noise_arr = self.noise_components[name]
                rms_noise = float(np.sqrt(np.mean(np.square(noise_arr))))
                lines.append(f"| {name} | {rms_noise:.6f} | {snr_val:.2f} dB |")
        else:
            lines.append("*No individual noise components recorded or processed.*")
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            
    def summary(self) -> str:

        """
        Create a human-readable clinical summary string of the record.
        
        Returns
        -------
        str
            A detailed, multi-line format report.
        """
        duration = self.t[-1] if len(self.t) > 0 else 0.0
        n_samples = len(self.t)
        
        lines = [
            "==================================================",
            "             BIOSIGNAL RECORD SUMMARY             ",
            "==================================================",
            f"UUID:         {self._metadata_obj.uuid}",
            f"Timestamp:    {self._metadata_obj.timestamp}",
            f"Signal Type:  {self.signal_type.upper()}",
            f"Sampling:     {self.fs} Hz",
            f"Duration:     {duration:.3f} seconds",
            f"Channels:     {self._n_channels}",
            f"Samples/Ch:   {n_samples}",
            "--------------------------------------------------",
            "Signal Amplitude & Statistics:",
            f"  Clean RMS:  {self.quality_metrics.get('clean_rms', 0.0):.4f}",
            f"  Noisy RMS:  {self.quality_metrics.get('noisy_rms', 0.0):.4f}",
            f"  RMSE Error: {self.quality_metrics.get('rmse', 0.0):.4f}",
            f"  1-D SSIM:   {self.quality_metrics.get('ssim_index', 0.0):.4f}",
            f"  Target SNR: {self.signal_params.get('target_snr_db', 'N/A')} dB",
            f"  Actual SNR: {f'{self.snr_db:.2f} dB' if self.snr_db is not None else 'N/A'}",
            "--------------------------------------------------",
            "Physiological Shape Properties:",
            f"  Clean Skewness: {self.quality_metrics.get('clean_skewness', 0.0):.4f}",
            f"  Noisy Skewness: {self.quality_metrics.get('noisy_skewness', 0.0):.4f}",
            f"  Clean Kurtosis: {self.quality_metrics.get('clean_kurtosis', 0.0):.4f}",
            f"  Noisy Kurtosis: {self.quality_metrics.get('noisy_kurtosis', 0.0):.4f}",
            f"  Clean ZCR:      {self.quality_metrics.get('clean_zcr', 0.0):.4f}",
            f"  Noisy ZCR:      {self.quality_metrics.get('noisy_zcr', 0.0):.4f}",
            f"  Clean Entropy:  {self.quality_metrics.get('clean_entropy', 0.0):.2f} bits",
            f"  Noisy Entropy:  {self.quality_metrics.get('noisy_entropy', 0.0):.2f} bits",
            f"  Clean Crest F:  {self.quality_metrics.get('clean_crest_factor', 0.0):.4f}",
            f"  Noisy Crest F:  {self.quality_metrics.get('noisy_crest_factor', 0.0):.4f}",
            "--------------------------------------------------",
            f"Noise Components Count: {len(self.noise_components)}",
        ]
        
        # Show SNRs of components if available
        if "noise_snr_breakdown" in self.quality_metrics:
            for name, snr_val in self.quality_metrics["noise_snr_breakdown"].items():
                lines.append(f"  - {name:<12}: SNR = {snr_val:.2f} dB")
        else:
            for name, noise in self.noise_components.items():
                rms_noise = float(np.sqrt(np.mean(np.square(noise))))
                lines.append(f"  - {name:<12}: RMS = {rms_noise:.4f}")
            
        lines.append("--------------------------------------------------")
        lines.append("Quality Indicators:")
        for k, v in self.quality_flags.items():
            status = "FAILED" if v else "PASS"
            lines.append(f"  {k:<18}: {status}")
        lines.append("==================================================")
        
        return "\n".join(lines)
        
    def to_dataframe(self) -> object:
        """
        Convert this record's signals to a pandas DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by time vector.
        """
        from biosignal_simulator.io.exporters import BiosignalExporter
        return BiosignalExporter.export_dataframe(self)

    def to_csv(self, filepath: str) -> None:
        """
        Export this record to a CSV file with metadata.
        
        Parameters
        ----------
        filepath : str
            Target file path.
        """
        from biosignal_simulator.io.exporters import BiosignalExporter
        BiosignalExporter.export_csv(self, filepath)

    def to_hdf5(self, filepath: str) -> None:
        """
        Export this record to an HDF5 file.
        
        Parameters
        ----------
        filepath : str
            Target file path.
        """
        from biosignal_simulator.io.exporters import BiosignalExporter
        BiosignalExporter.export_hdf5(self, filepath)

    def to_edf_lite(self, filepath: str, use_clean: bool = False) -> None:
        """
        Export this record to European Data Format (EDF-lite) binary format.
        
        Parameters
        ----------
        filepath : str
            Target file path.
        use_clean : bool
            If True, exports the clean signal. Otherwise, exports the noisy signal (default: False).
        """
        from biosignal_simulator.io.exporters import BiosignalExporter
        BiosignalExporter.export_edf_lite(self, filepath, use_clean)

    def to_numpy(self, filepath: str) -> None:
        """
        Export this record to a compressed NumPy NPZ archive.
        
        Parameters
        ----------
        filepath : str
            Target file path.
        """
        from biosignal_simulator.io.exporters import BiosignalExporter
        BiosignalExporter.export_numpy(self, filepath)
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize all fields to a dictionary.
        
        Converts all NumPy arrays into lists for serialization compatibility.
        
        Returns
        -------
        Dict[str, Any]
            The serialized record dictionary.
        """
        return {
            "signal_type": self.signal_type,
            "fs": float(self.fs),
            "t": self.t.tolist(),
            "clean": self.clean.tolist(),
            "noisy": self.noisy.tolist(),
            "noise_components": {k: v.tolist() for k, v in self.noise_components.items()},
            "signal_params": self.signal_params,
            "noise_params": self.noise_params,
            "snr_db": self.snr_db,
            "metadata": self.metadata,
            "quality_flags": self.quality_flags,
            "quality_metrics": self.quality_metrics,
            "uuid": self._metadata_obj.uuid,
            "timestamp": self._metadata_obj.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalRecord':
        """
        Reconstruct a SignalRecord from a dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            The dictionary representing the record.
            
        Returns
        -------
        SignalRecord
            An initialized SignalRecord.
        """
        t = np.array(data["t"], dtype=np.float64)
        clean = np.array(data["clean"], dtype=np.float64)
        noisy = np.array(data["noisy"], dtype=np.float64)
        noise_components = {k: np.array(v, dtype=np.float64) for k, v in data["noise_components"].items()}
        
        # Reconstruct standard fields
        record = cls(
            signal_type=data["signal_type"],
            fs=data["fs"],
            t=t,
            clean=clean,
            noisy=noisy,
            noise_components=noise_components,
            signal_params=data.get("signal_params", {}),
            noise_params=data.get("noise_params", {}),
            snr_db=data.get("snr_db"),
            metadata=data.get("metadata", {}),
            quality_flags=data.get("quality_flags", {}),
            quality_metrics=data.get("quality_metrics", {})
        )
        
        # Override generated UUID and timestamp with loaded ones if available
        if "uuid" in data:
            record._metadata_obj.uuid = data["uuid"]
        if "timestamp" in data:
            record._metadata_obj.timestamp = data["timestamp"]
            
        return record
        
    def to_json(self, filepath: str) -> None:
        """
        Write the record to a JSON file.
        
        Parameters
        ----------
        filepath : str
            Target file destination.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    @classmethod
    def from_json(cls, filepath: str) -> 'SignalRecord':
        """
        Load a SignalRecord from a JSON file.
        
        Parameters
        ----------
        filepath : str
            Source file path.
            
        Returns
        -------
        SignalRecord
            Reconstructed SignalRecord object.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
        
    def __repr__(self) -> str:
        return (
            f"SignalRecord(type={self.signal_type}, fs={self.fs}, "
            f"samples={len(self.t)}, channels={self._n_channels}, snr={self.snr_db})"
        )

