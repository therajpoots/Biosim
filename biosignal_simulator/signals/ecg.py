"""
High-fidelity Electrocardiogram (ECG) and Vectorcardiogram (VCG) simulator.

This module provides a comprehensive, physiologically accurate simulation of cardiac
electrical activity, based on the Vectorcardiogram (VCG) dipole model and its projection
to standard clinical 12-lead ECG using the Dower Transformation Matrix.

Physiological Basis
-------------------
The ECG is a surface recording of the electrical dipole vector d(t) = [x(t), y(t), z(t)]^T
that represents the collective action potential propagation across the myocardium.
This 3-dimensional trajectory is called the Vectorcardiogram (VCG). It is decomposed into
three major loops:

  - P-loop   : Atrial depolarization (60-100 ms duration)
  - QRS-loop : Ventricular depolarization (60-120 ms duration, large amplitude)
  - T-loop   : Ventricular repolarization (160-300 ms duration)

Each loop is modeled as an amplitude-modulated Gaussian pulse in 3D space, summed to form
the complete VCG signal.

12-Lead Projection
------------------
The Frank lead system VCG (X, Y, Z orthogonal leads) is projected to 12 standard clinical
ECG leads (I, II, III, aVR, aVL, aVF, V1-V6) using the inverse Dower matrix:

    V_12(t) = P_Dower @ d(t)

where P_Dower is the standard 12x3 Dower transformation matrix.

Arrhythmias and Pathologies Supported
--------------------------------------
- Normal Sinus Rhythm (NSR)
- Sinus Bradycardia (HR < 60 bpm)
- Sinus Tachycardia (HR > 100 bpm)
- Atrial Fibrillation (AFib): no P-waves, irregularly irregular RR, f-waves (4-8 Hz)
- Atrial Flutter: sawtooth F-waves at 250-350 bpm with 2:1 or 4:1 AV conduction
- Premature Ventricular Contractions (PVC): wide ectopic QRS, compensatory pause
- Premature Atrial Contractions (PAC): early narrow QRS, non-compensatory pause
- Ventricular Tachycardia (VTach): 3+ consecutive PVCs at rate > 100 bpm
- Ventricular Fibrillation (VFib): chaotic low-amplitude activity, no organized QRS
- Second-Degree AV Block Mobitz I (Wenckebach): progressive PR lengthening → dropped QRS
- Second-Degree AV Block Mobitz II: fixed PR, sudden dropped QRS (every Nth beat)
- Third-Degree Complete AV Block: complete P/QRS dissociation, ventricular escape rhythm
- Right Bundle Branch Block (RBBB): late R' in V1, wide S in I and V6
- Left Bundle Branch Block (LBBB): wide QRS, notched R in V6, deep S in V1
- Wolff-Parkinson-White (WPW): short PR, delta wave, widened QRS
- Long QT Syndrome: prolonged QT interval, flattened T-wave
- ST-Elevation Myocardial Infarction (STEMI): ST elevation in territory leads
- ST Depression (Ischemia): horizontal or down-sloping ST depression

References
----------
- Macfarlane, P.W. et al. (2010). Comprehensive Electrocardiology.
- Dower, G.E. et al. (1988). Derivation of the 12-Lead Electrocardiogram from Four
  (EASI) Electrodes. J Electrocardiol.
- Clifford, G.D. et al. (2006). Advanced Methods for ECG Analysis.
- Pan, J. & Tompkins, W.J. (1985). A Real-Time QRS Detection Algorithm.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

from biosignal_simulator.core.base import BaseSignal
from biosignal_simulator.core.config import ECGConfig


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations for Rhythm and Pathology Types
# ──────────────────────────────────────────────────────────────────────────────

class ECGRhythm(str, Enum):
    """Enumeration of supported ECG rhythm types."""
    NORMAL = "normal"
    BRADYCARDIA = "bradycardia"
    TACHYCARDIA = "tachycardia"
    AFIB = "afib"
    AFLUTTER = "aflutter"
    PVC = "pvc"
    PAC = "pac"
    VTACH = "vtach"
    VFIB = "vfib"
    AV_BLOCK = "av_block"
    WENCKEBACH = "wenckebach"
    COMPLETE_AV_BLOCK = "complete_av_block"
    RBBB = "rbbb"
    LBBB = "lbbb"
    WPW = "wpw"
    LONG_QT = "long_qt"
    STEMI = "stemi"
    ISCHEMIA = "ischemia"


class ECGLeadType(str, Enum):
    """Supported ECG lead output configurations."""
    SINGLE = "single"
    VCG = "vcg"
    TWELVE_LEAD = "12lead"


# ──────────────────────────────────────────────────────────────────────────────
# Internal data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BeatEvent:
    """
    Container for a single cardiac beat event and its associated metadata.

    Attributes
    ----------
    r_time : float
        Timestamp of the R-peak (or equivalent ventricular activation) in seconds.
    beat_type : str
        Type of beat: 'normal', 'pvc', 'pac', 'vtach', 'escape', 'dropped'.
    pr_interval : float
        PR interval in seconds (time from P-wave onset to QRS onset).
    qrs_width : float
        QRS complex width in seconds.
    qt_interval : float
        QT interval in seconds.
    st_level : float
        ST segment elevation/depression in mV relative to baseline.
    p_amplitude_scale : float
        Multiplicative scaling factor for P-wave amplitude.
    has_p_wave : bool
        Whether a preceding P-wave is present.
    has_delta_wave : bool
        Whether a delta wave (WPW pre-excitation) is present.
    """
    r_time: float = 0.0
    beat_type: str = "normal"
    pr_interval: float = 0.16
    qrs_width: float = 0.08
    qt_interval: float = 0.40
    st_level: float = 0.0
    p_amplitude_scale: float = 1.0
    has_p_wave: bool = True
    has_delta_wave: bool = False


@dataclass
class VCGWaveComponents:
    """
    Container for 3D VCG direction vectors and amplitudes for each cardiac wave.

    Attributes
    ----------
    v_P, v_Q, v_R, v_S, v_T : np.ndarray
        3D direction vectors (Frank XYZ coordinates) for each waveform component.
    amp_P, amp_Q, amp_R, amp_S, amp_T : float
        Amplitude scalars for each component.
    sigma_P, sigma_Q, sigma_R, sigma_S, sigma_T : float
        Gaussian width (standard deviation in seconds) for each component.
    dt_Q, dt_S, dt_T : float
        Timing offsets relative to R-peak for Q, S, and T waves.
    """
    # Direction vectors (Frank XYZ: X = left, Y = down, Z = back)
    v_P: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.40, 0.10]))
    v_Q: np.ndarray = field(default_factory=lambda: np.array([-0.15, 0.05, 0.15]))
    v_R: np.ndarray = field(default_factory=lambda: np.array([0.70, 0.60, -0.15]))
    v_S: np.ndarray = field(default_factory=lambda: np.array([-0.25, -0.30, -0.20]))
    v_T: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.45, 0.05]))
    v_ST: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.45, 0.05]))

    # Base amplitudes
    amp_P: float = 0.225
    amp_Q: float = 0.180
    amp_R: float = 1.000
    amp_S: float = 0.350
    amp_T: float = 0.300

    # Gaussian widths (seconds)
    sigma_P: float = 0.018
    sigma_Q: float = 0.004
    sigma_R: float = 0.009
    sigma_S: float = 0.007
    sigma_T: float = 0.038

    # Timing offsets relative to R-peak (seconds)
    dt_P: float = -0.150  # P-wave center offset (before R)
    dt_Q: float = -0.015  # Q-wave center offset
    dt_S: float = +0.018  # S-wave center offset
    dt_T: float = +0.220  # T-wave center offset


# ──────────────────────────────────────────────────────────────────────────────
# Dower Transformation Matrix (VCG → 12-Lead ECG)
# ──────────────────────────────────────────────────────────────────────────────

# Standard 12x3 Dower matrix for projecting Frank VCG (X, Y, Z) to 12 ECG leads.
# Row order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
# Reference: Dower et al. (1988), J Electrocardiol
DOWER_MATRIX = np.array([
    [ 0.632, -0.235,  0.059],  # Lead I
    [ 0.235,  1.066, -0.132],  # Lead II
    [-0.397,  0.831, -0.191],  # Lead III
    [-0.433, -0.415,  0.037],  # aVR
    [ 0.515, -0.533,  0.125],  # aVL
    [-0.081,  0.948, -0.162],  # aVF
    [ 0.229,  0.191, -0.735],  # V1
    [ 0.266,  0.554, -1.045],  # V2
    [ 0.339,  0.697, -0.951],  # V3
    [ 0.398,  0.939, -0.771],  # V4
    [ 0.711,  1.221, -0.410],  # V5
    [ 0.766,  0.899, -0.220],  # V6
], dtype=np.float64)

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _gaussian_pulse(t: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    """
    Compute a Gaussian pulse centered at ``center`` with standard deviation ``sigma``.

    Parameters
    ----------
    t : np.ndarray
        Time array in seconds.
    center : float
        Center time of the Gaussian (seconds).
    sigma : float
        Standard deviation of the Gaussian (seconds). Must be > 0.
    amplitude : float
        Peak amplitude of the Gaussian.

    Returns
    -------
    np.ndarray
        Gaussian pulse values, same shape as ``t``.

    Examples
    --------
    >>> t = np.linspace(0, 1, 500)
    >>> pulse = _gaussian_pulse(t, center=0.5, sigma=0.05, amplitude=1.0)
    >>> float(np.max(pulse))
    1.0
    """
    if sigma <= 0.0:
        sigma = 1e-9
    return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _add_vcg_wave(
    vcg: np.ndarray,
    t: np.ndarray,
    center: float,
    sigma: float,
    amplitude: float,
    direction: np.ndarray,
) -> None:
    """
    Add a single Gaussian wave contribution to the 3D VCG signal (in-place).

    Parameters
    ----------
    vcg : np.ndarray
        Shape (3, n_samples) VCG array to update in-place.
    t : np.ndarray
        Time axis array.
    center : float
        Center time of the Gaussian pulse.
    sigma : float
        Standard deviation of the Gaussian pulse.
    amplitude : float
        Scalar amplitude of the wave.
    direction : np.ndarray
        3-element unit direction vector in Frank XYZ space.
    """
    env = _gaussian_pulse(t, center, sigma, amplitude)
    vcg[0] += env * direction[0]
    vcg[1] += env * direction[1]
    vcg[2] += env * direction[2]


def _apply_iir_filter(
    signal_arr: np.ndarray,
    fs: float,
    low_hz: Optional[float] = None,
    high_hz: Optional[float] = None,
    order: int = 4,
    filter_type: str = "butter",
) -> np.ndarray:
    """
    Apply a zero-phase IIR band-pass filter to a 1-D signal.

    Parameters
    ----------
    signal_arr : np.ndarray
        Input 1-D signal array.
    fs : float
        Sampling frequency in Hz.
    low_hz : float, optional
        Low-cut frequency in Hz. If None, a low-pass filter is applied.
    high_hz : float, optional
        High-cut frequency in Hz. If None, a high-pass filter is applied.
    order : int
        Filter order.
    filter_type : str
        Filter design: 'butter', 'cheby1', 'cheby2', 'ellip', 'bessel'.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyq = 0.5 * fs

    if low_hz is not None and high_hz is not None:
        wn = [np.clip(low_hz / nyq, 1e-4, 0.999), np.clip(high_hz / nyq, 1e-4, 0.999)]
        if wn[0] >= wn[1]:
            return signal_arr
        btype = "bandpass"
    elif low_hz is not None:
        wn = np.clip(low_hz / nyq, 1e-4, 0.999)
        btype = "highpass"
    elif high_hz is not None:
        wn = np.clip(high_hz / nyq, 1e-4, 0.999)
        btype = "lowpass"
    else:
        return signal_arr

    if filter_type == "butter":
        b, a = sp_signal.butter(order, wn, btype=btype)
    elif filter_type == "cheby1":
        b, a = sp_signal.cheby1(order, 0.5, wn, btype=btype)
    elif filter_type == "bessel":
        b, a = sp_signal.bessel(order, wn, btype=btype, norm="phase")
    else:
        b, a = sp_signal.butter(order, wn, btype=btype)

    return sp_signal.filtfilt(b, a, signal_arr)


def _compute_qtc(qt: float, rr: float, method: str = "bazett") -> float:
    """
    Compute the heart-rate-corrected QT interval (QTc).

    Parameters
    ----------
    qt : float
        QT interval in seconds.
    rr : float
        RR interval in seconds.
    method : str
        Correction formula: 'bazett', 'fridericia', 'hodges', 'framingham'.

    Returns
    -------
    float
        QTc in seconds.

    Notes
    -----
    Bazett formula:    QTc = QT / sqrt(RR)
    Fridericia formula: QTc = QT / RR^(1/3)
    Hodges formula:    QTc = QT + 1.75 * (HR - 60)   [HR in bpm]
    Framingham formula: QTc = QT + 0.154 * (1 - RR)
    """
    if rr <= 0:
        rr = 0.833  # Default 72 bpm
    method = method.lower()
    if method == "bazett":
        return qt / np.sqrt(rr)
    elif method == "fridericia":
        return qt / (rr ** (1.0 / 3.0))
    elif method == "hodges":
        hr = 60.0 / rr
        return qt + 0.00175 * (hr - 60.0)
    elif method == "framingham":
        return qt + 0.154 * (1.0 - rr)
    else:
        return qt / np.sqrt(rr)


def _lognormal_rr(rng: np.random.Generator, mean_rr: float, hrv_std: float) -> float:
    """
    Sample a single RR interval from a lognormal distribution parameterized by
    mean and coefficient of variation (CV = hrv_std).

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random generator.
    mean_rr : float
        Mean RR interval in seconds.
    hrv_std : float
        Standard deviation of RR interval in seconds (HRV).

    Returns
    -------
    float
        Sampled RR interval in seconds. Always > 0.
    """
    if hrv_std <= 0.0:
        return mean_rr
    # Parameterize the lognormal distribution in terms of mean and std of RR
    sigma_ln = np.sqrt(np.log(1.0 + (hrv_std / mean_rr) ** 2))
    mu_ln = np.log(mean_rr) - 0.5 * sigma_ln ** 2
    return float(rng.lognormal(mean=mu_ln, sigma=sigma_ln))


def _afib_rr_interval(rng: np.random.Generator, mean_rr: float) -> float:
    """
    Sample an RR interval for Atrial Fibrillation — extreme irregularity.

    AFib is characterized by very high RR variability. We model this using a
    Gamma distribution (as clinically validated: Meriggi et al. 2013).

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    mean_rr : float
        Mean RR interval in seconds.

    Returns
    -------
    float
        Sampled RR interval in seconds.
    """
    # Gamma shape k ~ 2 gives moderate variability (CV ≈ 0.45-0.6 for AFib)
    k = 2.0
    theta = mean_rr / k
    return float(rng.gamma(shape=k, scale=theta))


def _flutter_rr_interval(
    rng: np.random.Generator, flutter_rate: float, conduction_ratio: float
) -> float:
    """
    Sample an RR interval for Atrial Flutter.

    AFL is characterized by a regular atrial rate (~300 bpm) and a fixed AV
    conduction ratio (e.g., 2:1, 3:1, 4:1), producing a relatively regular
    ventricular rate with slight variation.

    Parameters
    ----------
    rng : np.random.Generator
        Random generator.
    flutter_rate : float
        Atrial flutter rate in bpm (usually 250-350).
    conduction_ratio : float
        AV conduction ratio (e.g., 2.0 for 2:1 block).

    Returns
    -------
    float
        RR interval in seconds.
    """
    atrial_rr = 60.0 / flutter_rate
    ventricular_rr = atrial_rr * conduction_ratio
    # Small jitter (±3%)
    jitter = rng.uniform(-0.03, 0.03) * ventricular_rr
    return float(ventricular_rr + jitter)


def calculate_axis_from_vcg(vcg: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    """
    Compute the cardiac electrical axis from the VCG signal.

    Calculates mean QRS vector in the frontal plane (Lead I vs. Lead aVF)
    and the transverse plane (V1 vs. V5).

    Parameters
    ----------
    vcg : np.ndarray
        Shape (3, n_samples) VCG array.
    t : np.ndarray
        Time axis in seconds.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:
        - 'frontal_axis_deg'  : Frontal plane mean QRS axis in degrees
        - 'horizontal_axis_deg': Transverse plane axis in degrees
        - 'mean_X', 'mean_Y', 'mean_Z': Mean VCG deflection amplitudes
    """
    ecg_12 = DOWER_MATRIX @ vcg
    lead_I = ecg_12[0]
    lead_avf = ecg_12[5]

    # Frontal axis from Lead I and aVF net deflection
    net_I = np.max(lead_I) - np.abs(np.min(lead_I))
    net_avf = np.max(lead_avf) - np.abs(np.min(lead_avf))
    frontal_axis_deg = float(np.degrees(np.arctan2(net_avf, net_I)))

    # Transverse axis from V1 and V5
    lead_V1 = ecg_12[6]
    lead_V5 = ecg_12[10]
    net_V1 = np.max(lead_V1) - np.abs(np.min(lead_V1))
    net_V5 = np.max(lead_V5) - np.abs(np.min(lead_V5))
    horizontal_axis_deg = float(np.degrees(np.arctan2(net_V1, net_V5)))

    return {
        "frontal_axis_deg": frontal_axis_deg,
        "horizontal_axis_deg": horizontal_axis_deg,
        "mean_X": float(np.mean(np.abs(vcg[0]))),
        "mean_Y": float(np.mean(np.abs(vcg[1]))),
        "mean_Z": float(np.mean(np.abs(vcg[2]))),
    }


def detect_r_peaks(ecg_lead: np.ndarray, fs: float, min_rr_s: float = 0.3) -> np.ndarray:
    """
    Simple Pan-Tompkins inspired R-peak detector for a single ECG lead.

    This implements a lightweight version suitable for synthetic ECG evaluation:
    1. Bandpass filter (5-15 Hz)
    2. Differentiate
    3. Square
    4. Moving average integration
    5. Adaptive threshold with refractory period

    Parameters
    ----------
    ecg_lead : np.ndarray
        Single-channel ECG signal.
    fs : float
        Sampling frequency in Hz.
    min_rr_s : float
        Minimum RR interval in seconds (refractory period). Default: 0.3 s.

    Returns
    -------
    np.ndarray
        Array of R-peak sample indices.

    References
    ----------
    - Pan, J. & Tompkins, W.J. (1985). IEEE Trans. Biomed. Eng., 32(3), 230-236.
    """
    # Step 1: Bandpass filter (5-15 Hz)
    filtered = _apply_iir_filter(ecg_lead, fs, low_hz=5.0, high_hz=15.0, order=2)

    # Step 2: Differentiate and square
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2

    # Step 3: Moving average integration (window ≈ 150 ms)
    win = max(1, int(0.15 * fs))
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")

    # Step 4: Adaptive threshold and peak detection
    threshold = 0.5 * np.max(integrated)
    min_samples = int(min_rr_s * fs)

    peaks = []
    i = 0
    while i < len(integrated):
        if integrated[i] > threshold:
            # Find local max in next min_samples window
            end = min(i + min_samples, len(integrated))
            local_idx = np.argmax(ecg_lead[i:end]) + i
            peaks.append(local_idx)
            i = end
        else:
            i += 1

    return np.array(peaks, dtype=int)


def compute_hrv_metrics(r_peaks: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Compute standard HRV time-domain metrics from detected R-peak indices.

    Parameters
    ----------
    r_peaks : np.ndarray
        Array of R-peak sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict[str, float]
        Dictionary containing HRV metrics:
        - 'mean_rr_ms'   : Mean RR interval (ms)
        - 'sdnn_ms'      : SD of NN intervals (ms) — overall HRV
        - 'rmssd_ms'     : Root Mean Square of Successive Differences (ms) — short-term HRV
        - 'pnn50'        : % NN intervals > 50 ms apart
        - 'mean_hr_bpm'  : Mean heart rate (bpm)
        - 'min_rr_ms'    : Minimum RR interval (ms)
        - 'max_rr_ms'    : Maximum RR interval (ms)
    """
    if len(r_peaks) < 2:
        return {
            "mean_rr_ms": 0.0, "sdnn_ms": 0.0, "rmssd_ms": 0.0,
            "pnn50": 0.0, "mean_hr_bpm": 0.0, "min_rr_ms": 0.0, "max_rr_ms": 0.0,
        }

    rr_samples = np.diff(r_peaks)
    rr_ms = rr_samples * 1000.0 / fs
    successive_diff_ms = np.abs(np.diff(rr_ms))

    mean_rr = float(np.mean(rr_ms))
    sdnn = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(successive_diff_ms ** 2))) if len(successive_diff_ms) > 0 else 0.0
    pnn50 = float(100.0 * np.sum(successive_diff_ms > 50.0) / len(successive_diff_ms)) if len(successive_diff_ms) > 0 else 0.0

    return {
        "mean_rr_ms": mean_rr,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50": pnn50,
        "mean_hr_bpm": float(60000.0 / mean_rr) if mean_rr > 0 else 0.0,
        "min_rr_ms": float(np.min(rr_ms)),
        "max_rr_ms": float(np.max(rr_ms)),
    }


def compute_qrs_morphology(
    ecg_lead: np.ndarray, r_peaks: np.ndarray, fs: float
) -> Dict[str, float]:
    """
    Extract basic QRS morphology statistics from a single ECG lead.

    Parameters
    ----------
    ecg_lead : np.ndarray
        Single-channel ECG signal.
    r_peaks : np.ndarray
        R-peak indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'mean_r_amplitude_mv'  : Mean R-wave amplitude
        - 'std_r_amplitude_mv'   : Std of R-wave amplitudes
        - 'mean_qrs_duration_ms' : Mean QRS complex duration (ms)
    """
    if len(r_peaks) == 0:
        return {"mean_r_amplitude_mv": 0.0, "std_r_amplitude_mv": 0.0,
                "mean_qrs_duration_ms": 0.0}

    r_amplitudes = []
    qrs_durations = []
    qrs_half = int(0.06 * fs)  # ±60 ms window around R-peak

    for rp in r_peaks:
        start = max(0, rp - qrs_half)
        end = min(len(ecg_lead), rp + qrs_half)
        segment = ecg_lead[start:end]

        if len(segment) == 0:
            continue

        r_amplitudes.append(float(ecg_lead[rp]))
        # QRS duration: width at 50% of R-peak amplitude
        half_amp = 0.5 * ecg_lead[rp]
        above_half = np.where(segment > half_amp)[0]
        if len(above_half) > 1:
            qrs_dur_ms = (above_half[-1] - above_half[0]) * 1000.0 / fs
            qrs_durations.append(qrs_dur_ms)

    return {
        "mean_r_amplitude_mv": float(np.mean(r_amplitudes)) if r_amplitudes else 0.0,
        "std_r_amplitude_mv": float(np.std(r_amplitudes)) if r_amplitudes else 0.0,
        "mean_qrs_duration_ms": float(np.mean(qrs_durations)) if qrs_durations else 0.0,
    }


def vcg_loop_area(vcg: np.ndarray, plane: str = "frontal") -> float:
    """
    Compute the area enclosed by the VCG loop projected onto a 2D plane.

    The loop area is computed using the Shoelace (Gauss) formula.

    Parameters
    ----------
    vcg : np.ndarray
        Shape (3, n_samples) VCG array.
    plane : str
        Plane to project onto: 'frontal' (X-Y), 'sagittal' (Y-Z), 'transverse' (X-Z).

    Returns
    -------
    float
        Signed loop area (positive = counter-clockwise, negative = clockwise).
    """
    if plane == "frontal":
        x, y = vcg[0], vcg[1]   # X (left-right), Y (up-down)
    elif plane == "sagittal":
        x, y = vcg[1], vcg[2]   # Y (up-down), Z (front-back)
    elif plane == "transverse":
        x, y = vcg[0], vcg[2]   # X (left-right), Z (front-back)
    else:
        raise ValueError(f"Unknown VCG plane: {plane!r}. Use 'frontal', 'sagittal', 'transverse'.")

    # Shoelace formula for polygon area
    n = len(x)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


# ──────────────────────────────────────────────────────────────────────────────
# Beat Event Generator
# ──────────────────────────────────────────────────────────────────────────────

class BeatEventGenerator:
    """
    Generate sequences of cardiac beat events for a given rhythm type and duration.

    This class handles all timing logic: RR interval generation, PR interval
    adaptation, QRS morphology parameters, beat type flags (PVC, PAC, dropped),
    and rhythm-specific behaviors (Wenckebach PR lengthening, AFib irregularity, etc.).

    Parameters
    ----------
    rhythm : str
        ECG rhythm type (see ``ECGRhythm`` enum values).
    base_hr_bpm : float
        Nominal heart rate in beats per minute.
    hrv_std_s : float
        HRV standard deviation in seconds. Controls beat-to-beat variability.
    duration_s : float
        Total signal duration in seconds.
    rng : np.random.Generator
        Random number generator for reproducibility.
    config : ECGConfig
        Full ECG configuration (for additional rhythm parameters).
    """

    def __init__(
        self,
        rhythm: str,
        base_hr_bpm: float,
        hrv_std_s: float,
        duration_s: float,
        rng: np.random.Generator,
        config: "ECGConfig",
    ) -> None:
        self.rhythm = rhythm.lower()
        self.base_hr_bpm = base_hr_bpm
        self.hrv_std_s = hrv_std_s
        self.duration_s = duration_s
        self.rng = rng
        self.config = config

        # Effective mean RR interval
        self._mean_rr = 60.0 / base_hr_bpm

    def generate(self) -> List[BeatEvent]:
        """
        Generate the full list of beat events for the configured rhythm.

        Returns
        -------
        List[BeatEvent]
            Chronologically ordered list of beat events.
        """
        if self.rhythm == "afib":
            return self._generate_afib()
        elif self.rhythm == "aflutter":
            return self._generate_aflutter()
        elif self.rhythm == "vtach":
            return self._generate_vtach()
        elif self.rhythm == "vfib":
            return self._generate_vfib()
        elif self.rhythm == "complete_av_block":
            return self._generate_complete_av_block()
        elif self.rhythm == "wenckebach":
            return self._generate_wenckebach()
        elif self.rhythm in {"rbbb", "lbbb", "wpw", "long_qt", "stemi", "ischemia",
                             "normal", "bradycardia", "tachycardia"}:
            return self._generate_normal_variant()
        elif self.rhythm == "av_block":
            return self._generate_mobitz_ii()
        elif self.rhythm == "pvc":
            return self._generate_with_pvcs()
        elif self.rhythm == "pac":
            return self._generate_with_pacs()
        else:
            return self._generate_normal_variant()

    def _next_normal_rr(self, prev_rr: Optional[float] = None) -> float:
        """Sample the next normal RR interval with lognormal HRV."""
        return _lognormal_rr(self.rng, self._mean_rr, self.hrv_std_s)

    def _generate_normal_variant(self) -> List[BeatEvent]:
        """Generate normal sinus rhythm with pathological QRS morphology overlays."""
        events: List[BeatEvent] = []
        curr_t = self._mean_rr * 0.5

        # Rhythm-specific QRS parameters
        qrs_width = self.config.qrs_width
        qt_interval = 0.40
        pr_interval = self.config.pr_interval

        if self.rhythm == "rbbb":
            qrs_width = 0.13  # >120 ms
        elif self.rhythm == "lbbb":
            qrs_width = 0.14  # >120 ms
        elif self.rhythm == "wpw":
            pr_interval = 0.09   # Short PR (delta wave present)
            qrs_width = 0.11    # Widened QRS
        elif self.rhythm == "long_qt":
            qt_interval = 0.55  # QTc >500 ms at 72 bpm
        elif self.rhythm == "stemi":
            pass  # ST handled separately
        elif self.rhythm == "ischemia":
            pass  # ST handled separately
        elif self.rhythm == "bradycardia":
            self._mean_rr = max(self._mean_rr, 60.0 / 50.0)
        elif self.rhythm == "tachycardia":
            self._mean_rr = min(self._mean_rr, 60.0 / 110.0)

        while curr_t < self.duration_s + 1.0:
            rr = self._next_normal_rr()
            st_level = 0.0
            has_delta = False

            if self.rhythm == "stemi":
                st_level = self.config.st_elevation
            elif self.rhythm == "ischemia":
                st_level = -abs(self.config.st_elevation) * 0.6
            elif self.rhythm == "wpw":
                has_delta = True

            events.append(BeatEvent(
                r_time=curr_t,
                beat_type="normal",
                pr_interval=pr_interval,
                qrs_width=qrs_width,
                qt_interval=qt_interval,
                st_level=st_level,
                p_amplitude_scale=1.0,
                has_p_wave=True,
                has_delta_wave=has_delta,
            ))
            curr_t += rr

        return events

    def _generate_with_pvcs(self) -> List[BeatEvent]:
        """Normal rhythm with randomly inserted PVC beats (≈15% probability)."""
        events: List[BeatEvent] = []
        curr_t = self._mean_rr * 0.5
        last_was_pvc = False

        while curr_t < self.duration_s + 1.0:
            is_pvc = False

            if not last_was_pvc and self.rng.random() < 0.15:
                # PVC: arrives early, wide QRS, no P-wave, compensatory pause after
                pvc_rr = self._mean_rr * self.rng.uniform(0.55, 0.72)
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="pvc",
                    pr_interval=0.0,
                    qrs_width=self.rng.uniform(0.14, 0.18),
                    qt_interval=self.rng.uniform(0.36, 0.42),
                    st_level=0.0,
                    p_amplitude_scale=0.0,
                    has_p_wave=False,
                ))
                # Compensatory pause
                curr_t += pvc_rr + self._mean_rr * self.rng.uniform(1.3, 1.55)
                is_pvc = True
            else:
                rr = self._next_normal_rr()
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="normal",
                    pr_interval=self.config.pr_interval,
                    qrs_width=self.config.qrs_width,
                    qt_interval=0.40,
                    st_level=0.0,
                    p_amplitude_scale=1.0,
                    has_p_wave=True,
                ))
                curr_t += rr

            last_was_pvc = is_pvc

        return events

    def _generate_with_pacs(self) -> List[BeatEvent]:
        """Normal rhythm with randomly inserted PAC beats (≈12% probability)."""
        events: List[BeatEvent] = []
        curr_t = self._mean_rr * 0.5

        while curr_t < self.duration_s + 1.0:
            if self.rng.random() < 0.12:
                # PAC: slightly early, narrow QRS, abnormal P-wave morphology
                pac_rr = self._mean_rr * self.rng.uniform(0.65, 0.82)
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="pac",
                    pr_interval=self.rng.uniform(0.10, 0.14),  # Often shorter
                    qrs_width=self.config.qrs_width,  # Narrow QRS (supraventricular)
                    qt_interval=0.40,
                    st_level=0.0,
                    p_amplitude_scale=self.rng.uniform(0.4, 0.8),  # Abnormal P-wave
                    has_p_wave=True,
                ))
                curr_t += pac_rr + self._next_normal_rr() * self.rng.uniform(0.9, 1.1)
            else:
                rr = self._next_normal_rr()
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="normal",
                    pr_interval=self.config.pr_interval,
                    qrs_width=self.config.qrs_width,
                    qt_interval=0.40,
                    st_level=0.0,
                    p_amplitude_scale=1.0,
                    has_p_wave=True,
                ))
                curr_t += rr

        return events

    def _generate_afib(self) -> List[BeatEvent]:
        """
        Atrial Fibrillation: no P-waves, highly irregular RR intervals.

        RR intervals follow a Gamma distribution with high CV (≈ 0.45-0.60).
        F-waves (fibrillatory baseline) are generated separately in the main
        generator and superimposed on the VCG.
        """
        events: List[BeatEvent] = []
        curr_t = self._mean_rr * 0.3

        while curr_t < self.duration_s + 1.0:
            rr = _afib_rr_interval(self.rng, self._mean_rr)
            events.append(BeatEvent(
                r_time=curr_t,
                beat_type="normal",
                pr_interval=0.0,
                qrs_width=self.config.qrs_width,
                qt_interval=0.38,
                st_level=0.0,
                p_amplitude_scale=0.0,
                has_p_wave=False,
            ))
            curr_t += max(0.30, rr)

        return events

    def _generate_aflutter(self) -> List[BeatEvent]:
        """
        Atrial Flutter: regular sawtooth F-waves at 250-350 bpm.

        Ventricular response depends on AV conduction ratio (2:1, 3:1, 4:1).
        """
        events: List[BeatEvent] = []
        flutter_rate = self.rng.uniform(250.0, 320.0)  # Atrial flutter rate in bpm
        conduction_ratio = float(self.rng.choice([2.0, 3.0, 4.0], p=[0.6, 0.25, 0.15]))
        curr_t = self._mean_rr * 0.3

        while curr_t < self.duration_s + 1.0:
            rr = _flutter_rr_interval(self.rng, flutter_rate, conduction_ratio)
            events.append(BeatEvent(
                r_time=curr_t,
                beat_type="normal",
                pr_interval=0.14,
                qrs_width=self.config.qrs_width,
                qt_interval=0.38,
                st_level=0.0,
                p_amplitude_scale=0.0,  # Flutter waves, not P-waves
                has_p_wave=False,
            ))
            curr_t += max(0.25, rr)

        # Store flutter parameters for use in F-wave generation
        self._flutter_rate_bpm = flutter_rate
        self._flutter_conduction_ratio = conduction_ratio

        return events

    def _generate_vtach(self) -> List[BeatEvent]:
        """
        Ventricular Tachycardia: rapid, regular, wide QRS complexes.

        Rate: 120-250 bpm. No visible P-waves. Wide, monomorphic QRS.
        """
        events: List[BeatEvent] = []
        vtach_hr = self.rng.uniform(150.0, 200.0)
        vtach_rr = 60.0 / vtach_hr
        curr_t = vtach_rr * 0.5

        while curr_t < self.duration_s + 1.0:
            jitter = self.rng.uniform(-0.005, 0.005)  # Minimal jitter
            events.append(BeatEvent(
                r_time=curr_t + jitter,
                beat_type="vtach",
                pr_interval=0.0,
                qrs_width=self.rng.uniform(0.14, 0.20),  # Very wide QRS
                qt_interval=0.35,
                st_level=0.0,
                p_amplitude_scale=0.0,
                has_p_wave=False,
            ))
            curr_t += vtach_rr

        return events

    def _generate_vfib(self) -> List[BeatEvent]:
        """
        Ventricular Fibrillation: chaotic activity with no organized QRS.

        VFib is generated as pure noise with no beat events.
        """
        # No structured beats — VFib is pure chaotic baseline
        return []

    def _generate_mobitz_ii(self) -> List[BeatEvent]:
        """
        Second-Degree AV Block Mobitz II: every Nth QRS is dropped.

        Default: every 3rd beat is dropped (3:2 conduction).
        Fixed PR interval in all conducted beats.
        """
        events: List[BeatEvent] = []
        curr_t = self._mean_rr * 0.5
        beat_count = 0

        while curr_t < self.duration_s + 1.0:
            rr = self._next_normal_rr()
            beat_count += 1

            if beat_count % 3 == 0:
                # Dropped beat: P-wave occurs but no QRS
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="dropped",
                    pr_interval=self.config.pr_interval,
                    qrs_width=0.0,
                    qt_interval=0.0,
                    st_level=0.0,
                    p_amplitude_scale=1.0,
                    has_p_wave=True,
                ))
            else:
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="normal",
                    pr_interval=self.config.pr_interval,  # Fixed PR in Mobitz II
                    qrs_width=self.config.qrs_width,
                    qt_interval=0.40,
                    st_level=0.0,
                    p_amplitude_scale=1.0,
                    has_p_wave=True,
                ))
            curr_t += rr

        return events

    def _generate_wenckebach(self) -> List[BeatEvent]:
        """
        Second-Degree AV Block Mobitz I (Wenckebach): progressive PR lengthening,
        followed by a dropped beat, then reset.

        Typical Wenckebach cycle length: 3:2 to 5:4 (P-waves per QRS).
        PR increments by ≈20-40 ms per beat.
        """
        events: List[BeatEvent] = []
        curr_t = self._mean_rr * 0.5

        # Wenckebach parameters
        cycle_length = int(self.rng.choice([3, 4, 5]))  # Beats per cycle (last is dropped)
        pr_base = self.config.pr_interval
        pr_increment = self.rng.uniform(0.020, 0.040)  # ms per beat

        beat_in_cycle = 0

        while curr_t < self.duration_s + 1.0:
            rr = self._next_normal_rr()
            beat_in_cycle += 1

            if beat_in_cycle == cycle_length:
                # Dropped beat at the end of Wenckebach cycle
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="dropped",
                    pr_interval=pr_base + (beat_in_cycle - 1) * pr_increment,
                    qrs_width=0.0,
                    qt_interval=0.0,
                    st_level=0.0,
                    p_amplitude_scale=1.0,
                    has_p_wave=True,
                ))
                beat_in_cycle = 0
            else:
                events.append(BeatEvent(
                    r_time=curr_t,
                    beat_type="normal",
                    pr_interval=pr_base + (beat_in_cycle - 1) * pr_increment,
                    qrs_width=self.config.qrs_width,
                    qt_interval=0.40,
                    st_level=0.0,
                    p_amplitude_scale=1.0,
                    has_p_wave=True,
                ))
            curr_t += rr

        return events

    def _generate_complete_av_block(self) -> List[BeatEvent]:
        """
        Third-Degree (Complete) AV Block: complete dissociation between atria and ventricles.

        - P-waves fire at atrial rate (60-100 bpm)
        - QRS complexes fire at escape (ventricular) rate (20-40 bpm)
        - P-waves and QRS are completely independent
        """
        events: List[BeatEvent] = []

        # Atrial P-wave events (independent of QRS)
        atrial_rr = 60.0 / self.rng.uniform(65.0, 90.0)
        p_curr_t = atrial_rr * 0.3
        p_times = []
        while p_curr_t < self.duration_s + 1.0:
            p_times.append(p_curr_t)
            p_curr_t += _lognormal_rr(self.rng, atrial_rr, 0.01)

        # Ventricular escape events (idioventricular)
        escape_rr = 60.0 / self.rng.uniform(25.0, 40.0)
        v_curr_t = escape_rr * 0.5
        v_times = []
        while v_curr_t < self.duration_s + 1.0:
            v_times.append(v_curr_t)
            v_curr_t += _lognormal_rr(self.rng, escape_rr, 0.02)

        # Create P-only events
        for pt in p_times:
            events.append(BeatEvent(
                r_time=pt,
                beat_type="p_only",
                pr_interval=0.0,
                qrs_width=0.0,
                qt_interval=0.0,
                st_level=0.0,
                p_amplitude_scale=1.0,
                has_p_wave=True,
            ))

        # Create QRS-only events (wide escape complexes)
        for vt in v_times:
            events.append(BeatEvent(
                r_time=vt,
                beat_type="escape",
                pr_interval=0.0,
                qrs_width=self.rng.uniform(0.12, 0.16),  # Wide escape
                qt_interval=self.rng.uniform(0.40, 0.48),
                st_level=0.0,
                p_amplitude_scale=0.0,
                has_p_wave=False,
            ))

        # Sort events chronologically
        events.sort(key=lambda e: e.r_time)

        return events


# ──────────────────────────────────────────────────────────────────────────────
# Main ECG Generator Class
# ──────────────────────────────────────────────────────────────────────────────

class ECGGenerator(BaseSignal):
    """
    High-fidelity 3D VCG-projected 12-Lead Electrocardiogram generator.

    This generator models cardiac electrical activity as a 3D dipole vector
    trajectory (Vectorcardiogram) and projects it to any combination of the
    12 standard clinical ECG leads using the Dower transformation matrix.

    It supports a comprehensive range of cardiac rhythms, arrhythmias, and
    conduction abnormalities with physiologically accurate morphology.

    Parameters
    ----------
    config : ECGConfig
        ECG simulation configuration.

    Attributes
    ----------
    config : ECGConfig
        The configuration used to generate the signal.
    lead_names : list of str
        Names of the 12 standard ECG leads.
    vcg_components : VCGWaveComponents
        VCG direction vectors and amplitudes.

    Examples
    --------
    >>> from biosignal_simulator.core.config import ECGConfig
    >>> config = ECGConfig(fs=500.0, duration_s=10.0, heart_rate=72.0,
    ...                     rhythm_type='normal', lead_type='12lead')
    >>> gen = ECGGenerator(config)
    >>> ecg_12lead = gen.generate()
    >>> ecg_12lead.shape
    (12, 5000)

    >>> # Generate a single-lead ECG with PVC arrhythmia
    >>> config_pvc = ECGConfig(fs=500.0, duration_s=10.0, heart_rate=72.0,
    ...                         rhythm_type='pvc', lead_type='single',
    ...                         lead_name='II')
    >>> gen_pvc = ECGGenerator(config_pvc)
    >>> lead_II = gen_pvc.generate()
    >>> lead_II.shape
    (5000,)
    """

    def __init__(self, config: ECGConfig) -> None:
        config.__post_init__()
        lead_type = config.lead_type.lower()
        if lead_type == "12lead":
            n_ch = 12
            multi = True
        elif lead_type == "vcg":
            n_ch = 3
            multi = True
        else:
            n_ch = 1
            multi = False

        super().__init__(
            fs=config.fs,
            duration_s=config.duration_s,
            seed=config.seed,
            multichannel=multi,
            n_channels=n_ch,
        )
        self.config = config
        self.lead_names: List[str] = LEAD_NAMES

        # Build default VCG wave components from config
        self.vcg_components = self._build_vcg_components()

    @property
    def dower_matrix(self) -> np.ndarray:
        """Return the 12x3 Dower transformation matrix."""
        return DOWER_MATRIX.copy()

    def _build_vcg_components(self) -> VCGWaveComponents:
        """
        Build the VCG wave component set from the current configuration.

        Returns
        -------
        VCGWaveComponents
            Populated VCG wave components with amplitude and direction vectors.
        """
        comp = VCGWaveComponents()

        # Scale amplitudes from config
        comp.amp_P = self.config.p_amplitude / 1.875
        comp.amp_R = self.config.qrs_amplitude
        comp.amp_T = self.config.t_amplitude / 1.167
        comp.sigma_R = (self.config.qrs_width / 0.08) * 0.009

        return comp

    def validate_parameters(self) -> Tuple[bool, str]:
        """
        Validate the ECG configuration parameters.

        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message). error_message is empty if valid.
        """
        try:
            self.config.__post_init__()
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def generate(self) -> np.ndarray:
        """
        Generate the ECG or VCG signal according to the configuration.

        Returns
        -------
        np.ndarray
            - Shape (12, n_samples) for 12-lead ECG
            - Shape (3, n_samples) for VCG (Frank X, Y, Z)
            - Shape (n_samples,) for a single-lead ECG

        Raises
        ------
        ValueError
            If an unknown lead name is requested.
        """
        t = self.t
        fs = self.fs
        n_samples = self.n_samples
        rhythm = self.config.rhythm_type.lower()

        # Determine effective heart rate
        target_hr = self._effective_heart_rate(rhythm)
        mean_rr = 60.0 / target_hr

        # Generate beat events
        beat_gen = BeatEventGenerator(
            rhythm=rhythm,
            base_hr_bpm=target_hr,
            hrv_std_s=self.config.hr_variability_std,
            duration_s=self.config.duration_s,
            rng=self.rng,
            config=self.config,
        )
        beat_events = beat_gen.generate()

        # Handle VFib separately (pure chaotic signal)
        if rhythm == "vfib":
            return self._generate_vfib_signal()

        # Build 3D VCG from beat events
        vcg = self._build_vcg_from_events(beat_events, t)

        # Add rhythm-specific baseline signals
        vcg = self._add_rhythm_baseline(vcg, t, rhythm, beat_gen)

        # Apply bundle branch block morphology modifications
        vcg = self._apply_bundle_branch_block(vcg, t, rhythm, beat_events)

        # Zero-mean centering
        for i in range(3):
            vcg[i] -= np.mean(vcg[i])

        # Project VCG to 12-lead ECG
        ecg_12 = DOWER_MATRIX @ vcg

        # Post-process for RBBB/LBBB precordial morphology
        ecg_12 = self._apply_bundle_branch_precordial(ecg_12, rhythm)

        # Zero-mean all leads
        for i in range(12):
            ecg_12[i] -= np.mean(ecg_12[i])

        # Output selection
        return self._select_output(vcg, ecg_12)

    def _effective_heart_rate(self, rhythm: str) -> float:
        """
        Compute the effective heart rate for the given rhythm.

        Parameters
        ----------
        rhythm : str
            Rhythm type string.

        Returns
        -------
        float
            Effective heart rate in bpm.
        """
        hr = self.config.heart_rate
        if rhythm == "bradycardia":
            return min(hr, 50.0)
        elif rhythm in {"tachycardia", "aflutter"}:
            return max(hr, 110.0)
        elif rhythm == "vtach":
            return self.rng.uniform(150.0, 200.0)
        elif rhythm == "complete_av_block":
            return 70.0  # Atrial rate; ventricular escape is separate
        else:
            return hr

    def _build_vcg_from_events(
        self, beat_events: List[BeatEvent], t: np.ndarray
    ) -> np.ndarray:
        """
        Construct the 3D VCG signal by superposing all beat events.

        Each beat event contributes P, QRS (Q, R, S), and T wave Gaussian
        pulses in the Frank XYZ space according to the VCG wave components.

        Parameters
        ----------
        beat_events : List[BeatEvent]
            Chronologically ordered list of beat events.
        t : np.ndarray
            Time axis in seconds.

        Returns
        -------
        np.ndarray
            Shape (3, n_samples) VCG array.
        """
        vcg = np.zeros((3, len(t)))
        comp = self.vcg_components
        rhythm = self.config.rhythm_type.lower()

        for event in beat_events:
            r_t = event.r_time

            # Skip events outside time range
            if r_t < -0.5 or r_t > self.duration_s + 0.5:
                continue

            beat_type = event.beat_type

            # ── P-wave ────────────────────────────────────────────────────────
            if event.has_p_wave and beat_type not in {"vtach", "escape"}:
                p_center = r_t + comp.dt_P - (event.pr_interval - 0.16)
                p_amp = comp.amp_P * event.p_amplitude_scale
                if abs(p_amp) > 1e-9:
                    _add_vcg_wave(vcg, t, p_center, comp.sigma_P, p_amp, comp.v_P)

            # ── Dropped beat: only P-wave, no QRS ────────────────────────────
            if beat_type in {"dropped", "p_only"}:
                continue

            # ── Delta wave (WPW pre-excitation) ──────────────────────────────
            if event.has_delta_wave:
                delta_center = r_t - event.qrs_width * 0.5
                delta_amp = comp.amp_R * 0.18
                delta_sigma = event.qrs_width * 0.15
                _add_vcg_wave(vcg, t, delta_center, delta_sigma, delta_amp, comp.v_R)

            # ── QRS Complex ───────────────────────────────────────────────────
            self._add_qrs_to_vcg(vcg, t, r_t, event, comp, beat_type, rhythm)

            # ── T-wave ────────────────────────────────────────────────────────
            self._add_t_wave_to_vcg(vcg, t, r_t, event, comp, beat_type, rhythm)

            # ── ST Segment ───────────────────────────────────────────────────
            if abs(event.st_level) > 1e-9:
                self._add_st_segment(vcg, t, r_t, event, comp)

        return vcg

    def _add_qrs_to_vcg(
        self,
        vcg: np.ndarray,
        t: np.ndarray,
        r_t: float,
        event: BeatEvent,
        comp: VCGWaveComponents,
        beat_type: str,
        rhythm: str,
    ) -> None:
        """
        Add Q, R, S wave components to the VCG for a single beat.

        Beat-type specific QRS morphology:
        - PVC/VTach: inverted, very wide, discordant QRS
        - Escape: wide upright QRS (bundle of His escape)
        - Normal/PAC/LBBB/RBBB: standard QRS with width scaling

        Parameters
        ----------
        vcg : np.ndarray
            In-place (3, n_samples) VCG array.
        t : np.ndarray
            Time axis.
        r_t : float
            R-peak timestamp.
        event : BeatEvent
            The beat event descriptor.
        comp : VCGWaveComponents
            VCG wave components.
        beat_type : str
            Type of beat: 'normal', 'pvc', 'vtach', 'escape', etc.
        rhythm : str
            Current rhythm string.
        """
        width_scale = event.qrs_width / 0.08  # Normalize relative to 80 ms

        if beat_type in {"pvc", "vtach"}:
            # PVC/VTach: inverted wide QRS, no Q-wave, slurred S
            r_amp = -1.6 * comp.amp_R
            s_amp = -1.2 * comp.amp_S
            r_sigma = comp.sigma_R * width_scale * 2.2
            s_sigma = comp.sigma_S * width_scale * 2.0

            _add_vcg_wave(vcg, t, r_t, r_sigma, r_amp, comp.v_R)
            _add_vcg_wave(vcg, t, r_t + comp.dt_S * width_scale, s_sigma, s_amp, comp.v_S)

        elif beat_type == "escape":
            # Idioventricular escape: wide, upright but aberrant
            r_amp = comp.amp_R * 0.75
            r_sigma = comp.sigma_R * width_scale * 1.7
            s_amp = comp.amp_S * 0.5
            s_sigma = comp.sigma_S * width_scale * 1.5

            _add_vcg_wave(vcg, t, r_t, r_sigma, r_amp, comp.v_R)
            _add_vcg_wave(vcg, t, r_t + comp.dt_S, s_sigma, s_amp, comp.v_S)

        else:
            # Normal/PAC/other: standard Q, R, S with width scaling
            q_sigma = comp.sigma_Q * width_scale
            r_sigma = comp.sigma_R * width_scale
            s_sigma = comp.sigma_S * width_scale

            _add_vcg_wave(vcg, t, r_t + comp.dt_Q, q_sigma, comp.amp_Q, comp.v_Q)
            _add_vcg_wave(vcg, t, r_t, r_sigma, comp.amp_R, comp.v_R)
            _add_vcg_wave(vcg, t, r_t + comp.dt_S * width_scale, s_sigma, comp.amp_S, comp.v_S)

    def _add_t_wave_to_vcg(
        self,
        vcg: np.ndarray,
        t: np.ndarray,
        r_t: float,
        event: BeatEvent,
        comp: VCGWaveComponents,
        beat_type: str,
        rhythm: str,
    ) -> None:
        """
        Add T-wave component to the VCG for a single beat.

        T-wave timing is linked to the QT interval. For PVC/VTach beats,
        the T-wave is discordant (opposite polarity to QRS).

        Parameters
        ----------
        vcg : np.ndarray
            In-place (3, n_samples) VCG array.
        t : np.ndarray
            Time axis.
        r_t : float
            R-peak timestamp.
        event : BeatEvent
            Beat event descriptor.
        comp : VCGWaveComponents
            VCG wave components.
        beat_type : str
            Type of beat.
        rhythm : str
            Current rhythm type.
        """
        # T-wave center relative to R-peak (proportional to QT interval)
        t_center = r_t + event.qt_interval * 0.75

        if beat_type in {"pvc", "vtach"}:
            # Discordant (inverted) T-wave, wider
            t_amp = -1.5 * comp.amp_T
            t_sigma = comp.sigma_T * 1.8
        elif beat_type == "escape":
            # Idioventricular: discordant T
            t_amp = -comp.amp_T * 0.8
            t_sigma = comp.sigma_T * 1.4
        elif rhythm == "long_qt":
            # Flattened T-wave, very prolonged
            t_amp = comp.amp_T * 0.45
            t_sigma = comp.sigma_T * 2.5
        else:
            t_amp = comp.amp_T
            t_sigma = comp.sigma_T

        _add_vcg_wave(vcg, t, t_center, t_sigma, t_amp, comp.v_T)

    def _add_st_segment(
        self,
        vcg: np.ndarray,
        t: np.ndarray,
        r_t: float,
        event: BeatEvent,
        comp: VCGWaveComponents,
    ) -> None:
        """
        Add ST segment elevation or depression to the VCG.

        The ST segment is modeled as a sinusoidal arch between the S-wave
        and T-wave peak, scaled by the configured elevation level.

        Parameters
        ----------
        vcg : np.ndarray
            In-place (3, n_samples) VCG array.
        t : np.ndarray
            Time axis.
        r_t : float
            R-peak timestamp.
        event : BeatEvent
            Beat event descriptor.
        comp : VCGWaveComponents
            VCG wave components.
        """
        st_start = r_t + comp.dt_S + 0.02  # J-point
        st_end = r_t + event.qt_interval * 0.75  # T-wave peak

        mask = (t >= st_start) & (t <= st_end)
        if not np.any(mask):
            return

        t_sub = t[mask]
        duration = max(st_end - st_start, 1e-6)
        # Sinusoidal ST arch
        env_st = event.st_level * np.sin(np.pi * (t_sub - st_start) / duration)

        for coord in range(3):
            vcg[coord, mask] += env_st * comp.v_ST[coord]

    def _add_rhythm_baseline(
        self,
        vcg: np.ndarray,
        t: np.ndarray,
        rhythm: str,
        beat_gen: BeatEventGenerator,
    ) -> np.ndarray:
        """
        Add rhythm-specific baseline signals to the VCG.

        - AFib: fibrillatory f-waves (4-8 Hz irregular oscillations)
        - AFL: sawtooth F-waves (250-350 bpm atrial activity)

        Parameters
        ----------
        vcg : np.ndarray
            In-place (3, n_samples) VCG array.
        t : np.ndarray
            Time axis.
        rhythm : str
            Rhythm type.
        beat_gen : BeatEventGenerator
            Beat generator (may carry rhythm-specific state).

        Returns
        -------
        np.ndarray
            Updated VCG array.
        """
        n_samples = len(t)
        fs = self.fs

        if rhythm == "afib":
            # AFib fibrillatory baseline: continuous 4-8 Hz irregular oscillations
            # Frequency-modulated sine wave with random amplitude fluctuations
            f_wave_freqs = self.rng.uniform(4.5, 7.5, size=n_samples)
            phase = 2.0 * np.pi * np.cumsum(f_wave_freqs) / fs
            amp_mod = 0.045 * (1.0 + 0.35 * np.sin(2.0 * np.pi * 0.22 * t))
            # Add slight noise to f-waves
            f_waves = amp_mod * np.sin(phase) + 0.012 * self.rng.normal(size=n_samples)
            # Inject into atrial-dominant X and Y components
            vcg[0] += f_waves
            vcg[1] += 0.65 * f_waves

        elif rhythm == "aflutter" and hasattr(beat_gen, "_flutter_rate_bpm"):
            # Atrial Flutter: regular sawtooth F-waves at flutter rate
            f_flutter_hz = beat_gen._flutter_rate_bpm / 60.0
            # Sawtooth wave at atrial flutter rate
            sawtooth = 0.06 * sp_signal.sawtooth(2.0 * np.pi * f_flutter_hz * t, width=0.5)
            # Apply to atrial components (X, Y)
            vcg[0] += sawtooth * 0.8
            vcg[1] += sawtooth

        return vcg

    def _apply_bundle_branch_block(
        self,
        vcg: np.ndarray,
        t: np.ndarray,
        rhythm: str,
        beat_events: List[BeatEvent],
    ) -> np.ndarray:
        """
        Apply bundle branch block VCG morphology modifications.

        RBBB and LBBB alter the sequence of ventricular activation, which
        changes the direction of the terminal QRS forces in the VCG:

        - RBBB: late rightward and anterior forces (v_R' component in V1)
        - LBBB: notched R in lateral leads, deep S in V1

        Parameters
        ----------
        vcg : np.ndarray
            In-place (3, n_samples) VCG array.
        t : np.ndarray
            Time axis.
        rhythm : str
            Rhythm type.
        beat_events : List[BeatEvent]
            Beat events for timing reference.

        Returns
        -------
        np.ndarray
            Modified VCG array.
        """
        if rhythm not in {"rbbb", "lbbb"}:
            return vcg

        comp = self.vcg_components

        for event in beat_events:
            if event.beat_type in {"dropped", "p_only"}:
                continue

            r_t = event.r_time
            width_scale = event.qrs_width / 0.08

            if rhythm == "rbbb":
                # RBBB: Add terminal late R' wave (rightward and anterior)
                # Late activation of right ventricle via aberrant conduction
                v_r_prime = np.array([-0.4, 0.2, 0.6])  # Rightward, anterior
                r_prime_center = r_t + 0.055 * width_scale  # Late terminal force
                r_prime_sigma = comp.sigma_R * 1.2
                r_prime_amp = comp.amp_R * 0.55
                _add_vcg_wave(vcg, t, r_prime_center, r_prime_sigma, r_prime_amp, v_r_prime)

            elif rhythm == "lbbb":
                # LBBB: Notch the R wave in lateral direction + slow initial forces
                # The normal early septal Q is absent; all forces go left
                v_notch = np.array([0.65, 0.35, -0.05])  # Lateral-leftward
                notch_center = r_t + 0.04 * width_scale
                notch_sigma = comp.sigma_R * 0.8
                notch_amp = comp.amp_R * 0.40
                _add_vcg_wave(vcg, t, notch_center, notch_sigma, notch_amp, v_notch)

        return vcg

    def _apply_bundle_branch_precordial(
        self, ecg_12: np.ndarray, rhythm: str
    ) -> np.ndarray:
        """
        Apply precordial lead-specific post-processing for bundle branch blocks.

        After VCG projection, RBBB shows RSR' pattern in V1 and wide S in V6.
        LBBB shows broad monophasic R in V6 and QS pattern in V1.

        Parameters
        ----------
        ecg_12 : np.ndarray
            Shape (12, n_samples) 12-lead ECG.
        rhythm : str
            Rhythm type.

        Returns
        -------
        np.ndarray
            Modified 12-lead ECG.
        """
        if rhythm == "rbbb":
            # Enhance RSR' pattern in V1 (lead index 6)
            ecg_12[6] *= 1.15
            # Add slight slurring to wide S in V6 (index 11) and Lead I (index 0)
            ecg_12[11] *= 0.92
            ecg_12[0] *= 0.88

        elif rhythm == "lbbb":
            # V1: Deep QS (invert and amplify)
            ecg_12[6] *= -0.85
            # V6, V5: Broad upright R (enhance)
            ecg_12[10] *= 1.18
            ecg_12[11] *= 1.20
            # Lead I: Broad R
            ecg_12[0] *= 1.12

        return ecg_12

    def _generate_vfib_signal(self) -> np.ndarray:
        """
        Generate Ventricular Fibrillation signal.

        VFib is characterized by chaotic, disorganized electrical activity with
        no recognizable P waves or QRS complexes. It is modeled as:
        1. Band-limited (3-15 Hz) colored noise
        2. Amplitude-modulated with a slow irregular envelope
        3. Superposition of multiple chaotic frequency components

        Returns
        -------
        np.ndarray
            VFib signal in the same shape as the configured output type.
        """
        n_samples = self.n_samples
        fs = self.fs
        t = self.t

        # Generate chaotic base noise
        raw_noise = self.rng.normal(0.0, 1.0, size=n_samples)

        # Band-limit to VFib frequency range (3-15 Hz)
        vfib_signal = _apply_iir_filter(raw_noise, fs, low_hz=3.0, high_hz=15.0, order=3)

        # Slow amplitude modulation (0.5-2 Hz) — coarseness varies
        mod_freq = self.rng.uniform(0.8, 1.5)
        amplitude_env = (0.6 + 0.4 * np.abs(np.sin(2.0 * np.pi * mod_freq * t)))
        vfib_signal *= amplitude_env

        # Scale to realistic VFib amplitude (small, 0.1-0.3 mV)
        vfib_signal = vfib_signal / (np.std(vfib_signal) + 1e-12) * 0.18

        # Zero-mean
        vfib_signal -= np.mean(vfib_signal)

        # Project to requested output format
        if self.config.lead_type.lower() == "12lead":
            # Spread VFib across all 12 leads with slight variation
            vcg = np.zeros((3, n_samples))
            vcg[0] = vfib_signal
            vcg[1] = vfib_signal * 0.85 + self.rng.normal(0, 0.02, n_samples)
            vcg[2] = vfib_signal * 0.70 + self.rng.normal(0, 0.02, n_samples)
            return DOWER_MATRIX @ vcg
        elif self.config.lead_type.lower() == "vcg":
            vcg = np.zeros((3, n_samples))
            vcg[0] = vfib_signal
            vcg[1] = vfib_signal * 0.85 + self.rng.normal(0, 0.02, n_samples)
            vcg[2] = vfib_signal * 0.70 + self.rng.normal(0, 0.02, n_samples)
            return vcg
        else:
            return vfib_signal

    def _select_output(self, vcg: np.ndarray, ecg_12: np.ndarray) -> np.ndarray:
        """
        Select the output format based on the configured lead type.

        Parameters
        ----------
        vcg : np.ndarray
            Shape (3, n_samples) VCG array.
        ecg_12 : np.ndarray
            Shape (12, n_samples) 12-lead ECG.

        Returns
        -------
        np.ndarray
            The requested output array.
        """
        l_type = self.config.lead_type.lower()
        if l_type == "vcg":
            return vcg
        elif l_type == "12lead":
            return ecg_12
        else:
            lead_name = self.config.lead_name
            if lead_name not in LEAD_NAMES:
                raise ValueError(
                    f"Unknown lead name '{lead_name}'. "
                    f"Valid options: {LEAD_NAMES}"
                )
            idx = LEAD_NAMES.index(lead_name)
            return ecg_12[idx]

    def get_12lead_labels(self) -> List[str]:
        """
        Return the ordered list of 12-lead ECG label names.

        Returns
        -------
        List[str]
            ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', ..., 'V6']
        """
        return list(LEAD_NAMES)

    def get_vcg_axis(self) -> Dict[str, float]:
        """
        Compute and return the cardiac electrical axis from the generated VCG.

        Returns
        -------
        Dict[str, float]
            Dictionary with 'frontal_axis_deg', 'horizontal_axis_deg',
            'mean_X', 'mean_Y', 'mean_Z'.
        """
        # Generate VCG component only for axis calculation
        vcg_config = ECGConfig(
            fs=self.config.fs,
            duration_s=self.config.duration_s,
            heart_rate=self.config.heart_rate,
            rhythm_type=self.config.rhythm_type,
            lead_type="vcg",
            seed=self.config.seed,
        )
        vcg_gen = ECGGenerator(vcg_config)
        vcg = vcg_gen.generate()
        return calculate_axis_from_vcg(vcg, self.t)

    def detect_peaks(self, lead_name: str = "II") -> np.ndarray:
        """
        Detect R-peaks on a specified ECG lead using Pan-Tompkins algorithm.

        Parameters
        ----------
        lead_name : str
            Lead to analyze. Default: 'II'.

        Returns
        -------
        np.ndarray
            Array of R-peak sample indices.
        """
        ecg = self.generate()
        if self.config.lead_type.lower() == "12lead":
            lead_idx = LEAD_NAMES.index(lead_name) if lead_name in LEAD_NAMES else 1
            ecg_lead = ecg[lead_idx]
        elif self.config.lead_type.lower() == "vcg":
            ecg_lead = ecg[1]  # Y component (similar to Lead II)
        else:
            ecg_lead = ecg

        return detect_r_peaks(ecg_lead, self.fs)

    def compute_hrv(self, lead_name: str = "II") -> Dict[str, float]:
        """
        Detect R-peaks and compute HRV metrics.

        Parameters
        ----------
        lead_name : str
            Lead to use for R-peak detection.

        Returns
        -------
        Dict[str, float]
            HRV metric dictionary (see ``compute_hrv_metrics``).
        """
        peaks = self.detect_peaks(lead_name)
        return compute_hrv_metrics(peaks, self.fs)

    def get_lead(self, lead_name: str) -> np.ndarray:
        """
        Generate and extract a single named ECG lead.

        Parameters
        ----------
        lead_name : str
            Lead name from ['I', 'II', ..., 'V6'].

        Returns
        -------
        np.ndarray
            Single-lead ECG signal of shape (n_samples,).

        Raises
        ------
        ValueError
            If lead_name is not a valid ECG lead name.
        """
        if lead_name not in LEAD_NAMES:
            raise ValueError(f"'{lead_name}' is not a valid lead name. Use: {LEAD_NAMES}")

        # Generate 12-lead and extract
        old_type = self.config.lead_type
        self.config.lead_type = "12lead"
        ecg_12 = self.generate()
        self.config.lead_type = old_type

        idx = LEAD_NAMES.index(lead_name)
        return ecg_12[idx]

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Generate ECG and return as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time as index and lead names as columns.
            For 12-lead output: 12 columns.
            For VCG output: 3 columns (X, Y, Z).
            For single lead: 1 column.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with: pip install pandas")

        data = self.generate()
        t = self.t

        if self.config.lead_type.lower() == "12lead":
            df = pd.DataFrame(data.T, index=t, columns=LEAD_NAMES)
        elif self.config.lead_type.lower() == "vcg":
            df = pd.DataFrame(data.T, index=t, columns=["X", "Y", "Z"])
        else:
            df = pd.DataFrame(data, index=t, columns=[self.config.lead_name])

        df.index.name = "time_s"
        return df

    def summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive summary of the ECG generator configuration.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary including rhythm type, heart rate, HRV,
            QRS width, lead type, and sample count.
        """
        return {
            "rhythm_type": self.config.rhythm_type,
            "heart_rate_bpm": self.config.heart_rate,
            "hrv_std_s": self.config.hr_variability_std,
            "qrs_width_s": self.config.qrs_width,
            "p_amplitude": self.config.p_amplitude,
            "qrs_amplitude": self.config.qrs_amplitude,
            "t_amplitude": self.config.t_amplitude,
            "st_elevation": self.config.st_elevation,
            "lead_type": self.config.lead_type,
            "lead_name": self.config.lead_name,
            "fs_hz": self.fs,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
            "seed": self.config.seed,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience Factory Functions
# ──────────────────────────────────────────────────────────────────────────────

def make_normal_sinus(
    duration_s: float = 10.0,
    heart_rate: float = 72.0,
    fs: float = 500.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate a normal sinus rhythm ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    heart_rate : float
        Heart rate in bpm.
    fs : float
        Sampling frequency in Hz.
    lead_type : str
        Output format: '12lead', 'vcg', or 'single'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        ECG signal array.

    Examples
    --------
    >>> ecg = make_normal_sinus(duration_s=5.0, heart_rate=72.0, fs=500.0)
    >>> ecg.shape
    (12, 2500)
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=heart_rate,
        rhythm_type="normal",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_afib(
    duration_s: float = 10.0,
    mean_heart_rate: float = 85.0,
    fs: float = 500.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate an Atrial Fibrillation ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    mean_heart_rate : float
        Mean ventricular rate in bpm (highly irregular in AFib).
    fs : float
        Sampling frequency in Hz.
    lead_type : str
        Output format: '12lead', 'vcg', or 'single'.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        AFib ECG signal.

    Examples
    --------
    >>> ecg_afib = make_afib(duration_s=10.0, mean_heart_rate=85.0)
    >>> ecg_afib.shape[0]
    12
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=mean_heart_rate,
        rhythm_type="afib",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_vtach(
    duration_s: float = 10.0,
    fs: float = 500.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate a Ventricular Tachycardia ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    lead_type : str
        Output format.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        VTach ECG signal.
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=180.0,
        rhythm_type="vtach",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_vfib(
    duration_s: float = 10.0,
    fs: float = 500.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate a Ventricular Fibrillation ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    lead_type : str
        Output format.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        VFib ECG signal.
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=250.0,
        rhythm_type="vfib",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_stemi(
    duration_s: float = 10.0,
    fs: float = 500.0,
    st_elevation: float = 0.25,
    heart_rate: float = 72.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate an ST-Elevation MI (STEMI) ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    st_elevation : float
        ST segment elevation in mV. Positive for STEMI.
    heart_rate : float
        Heart rate in bpm.
    lead_type : str
        Output format.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        STEMI ECG signal.
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=heart_rate,
        rhythm_type="stemi",
        st_elevation=st_elevation,
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_complete_av_block(
    duration_s: float = 15.0,
    fs: float = 500.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate a Third-Degree (Complete) AV Block ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    lead_type : str
        Output format.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Complete AV block ECG signal.
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=70.0,
        rhythm_type="complete_av_block",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_wenckebach(
    duration_s: float = 15.0,
    fs: float = 500.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate a Wenckebach (Mobitz I) AV Block ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    lead_type : str
        Output format.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Wenckebach ECG signal.
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=72.0,
        rhythm_type="wenckebach",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_lbbb(
    duration_s: float = 10.0,
    fs: float = 500.0,
    heart_rate: float = 72.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate a Left Bundle Branch Block (LBBB) ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    heart_rate : float
        Heart rate in bpm.
    lead_type : str
        Output format.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        LBBB ECG signal.
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=heart_rate,
        rhythm_type="lbbb",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def make_rbbb(
    duration_s: float = 10.0,
    fs: float = 500.0,
    heart_rate: float = 72.0,
    lead_type: str = "12lead",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Convenience function: generate a Right Bundle Branch Block (RBBB) ECG.

    Parameters
    ----------
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    heart_rate : float
        Heart rate in bpm.
    lead_type : str
        Output format.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        RBBB ECG signal.
    """
    config = ECGConfig(
        fs=fs,
        duration_s=duration_s,
        heart_rate=heart_rate,
        rhythm_type="rbbb",
        lead_type=lead_type,
        seed=seed,
    )
    return ECGGenerator(config).generate()


def batch_generate_rhythms(
    rhythms: List[str],
    duration_s: float = 10.0,
    fs: float = 500.0,
    lead_type: str = "single",
    lead_name: str = "II",
    heart_rate: float = 72.0,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate ECG signals for a list of rhythm types and return as a dictionary.

    Parameters
    ----------
    rhythms : List[str]
        List of rhythm type strings (see ECGRhythm enum).
    duration_s : float
        Signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    lead_type : str
        Output format.
    lead_name : str
        Lead name for single-lead output.
    heart_rate : float
        Base heart rate in bpm.
    seed : int, optional
        Random seed (same for all rhythms, offset by index for independence).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping rhythm name to generated signal array.

    Examples
    --------
    >>> signals = batch_generate_rhythms(['normal', 'afib', 'pvc'], duration_s=5.0)
    >>> list(signals.keys())
    ['normal', 'afib', 'pvc']
    """
    results: Dict[str, np.ndarray] = {}
    for i, rhythm in enumerate(rhythms):
        rng_seed = (seed + i) if seed is not None else None
        config = ECGConfig(
            fs=fs,
            duration_s=duration_s,
            heart_rate=heart_rate,
            rhythm_type=rhythm,
            lead_type=lead_type,
            lead_name=lead_name,
            seed=rng_seed,
        )
        try:
            results[rhythm] = ECGGenerator(config).generate()
        except Exception as exc:
            warnings.warn(f"Failed to generate rhythm '{rhythm}': {exc}")

    return results


def interpolate_vcg_to_frank(
    ecg_12: np.ndarray, fs: float
) -> np.ndarray:
    """
    Estimate the Frank VCG from 12-lead ECG using the inverse Dower matrix.

    The inverse Dower transformation allows reconstruction of approximate
    Frank lead X, Y, Z signals from standard 12-lead ECG.

    Parameters
    ----------
    ecg_12 : np.ndarray
        Shape (12, n_samples) 12-lead ECG array.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Shape (3, n_samples) estimated Frank VCG (X, Y, Z).

    Notes
    -----
    The reconstruction uses the least-squares inverse: (D^T D)^-1 D^T
    This is the pseudoinverse of the 12x3 Dower matrix.
    """
    # Compute Moore-Penrose pseudoinverse of Dower matrix
    D_pinv = np.linalg.pinv(DOWER_MATRIX)  # Shape (3, 12)
    return D_pinv @ ecg_12  # Shape (3, n_samples)


def resample_ecg(
    ecg: np.ndarray, original_fs: float, target_fs: float
) -> np.ndarray:
    """
    Resample an ECG signal to a new sampling frequency using polyphase filtering.

    Parameters
    ----------
    ecg : np.ndarray
        ECG signal. Shape: (n_samples,) or (n_leads, n_samples).
    original_fs : float
        Original sampling frequency in Hz.
    target_fs : float
        Target sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Resampled ECG signal, same number of dimensions as input.
    """
    ratio = target_fs / original_fs
    if ratio == 1.0:
        return ecg.copy()

    # Compute resampling ratio as ratio of integers
    from fractions import Fraction
    frac = Fraction(target_fs / original_fs).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    if ecg.ndim == 1:
        return sp_signal.resample_poly(ecg, up, down)
    else:
        # Multi-lead: resample each lead independently
        resampled = []
        for lead in ecg:
            resampled.append(sp_signal.resample_poly(lead, up, down))
        return np.array(resampled)


def segment_ecg_beats(
    ecg_lead: np.ndarray,
    fs: float,
    pre_r_ms: float = 200.0,
    post_r_ms: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment an ECG lead into individual beat templates aligned at the R-peak.

    Parameters
    ----------
    ecg_lead : np.ndarray
        Single-lead ECG signal.
    fs : float
        Sampling frequency in Hz.
    pre_r_ms : float
        Samples to include before R-peak (ms). Default: 200 ms.
    post_r_ms : float
        Samples to include after R-peak (ms). Default: 400 ms.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - beats : np.ndarray of shape (n_beats, beat_length) — beat templates
        - r_peaks : np.ndarray — detected R-peak indices

    Notes
    -----
    Beats at the signal boundaries (where the window extends outside the signal)
    are excluded.
    """
    r_peaks = detect_r_peaks(ecg_lead, fs)
    pre_samples = int(pre_r_ms * fs / 1000.0)
    post_samples = int(post_r_ms * fs / 1000.0)
    beat_len = pre_samples + post_samples

    beats = []
    valid_peaks = []

    for rp in r_peaks:
        start = rp - pre_samples
        end = rp + post_samples
        if start >= 0 and end <= len(ecg_lead):
            beats.append(ecg_lead[start:end])
            valid_peaks.append(rp)

    if not beats:
        return np.empty((0, beat_len)), np.array(valid_peaks)

    return np.array(beats), np.array(valid_peaks)


def average_beat_template(
    ecg_lead: np.ndarray,
    fs: float,
    pre_r_ms: float = 200.0,
    post_r_ms: float = 400.0,
) -> np.ndarray:
    """
    Compute the average beat template from an ECG signal.

    Parameters
    ----------
    ecg_lead : np.ndarray
        Single-lead ECG signal.
    fs : float
        Sampling frequency in Hz.
    pre_r_ms : float
        Pre-R-peak window in ms.
    post_r_ms : float
        Post-R-peak window in ms.

    Returns
    -------
    np.ndarray
        Average beat template of shape (beat_length,).
        Returns empty array if no beats detected.
    """
    beats, _ = segment_ecg_beats(ecg_lead, fs, pre_r_ms, post_r_ms)
    if len(beats) == 0:
        return np.array([])
    return np.mean(beats, axis=0)


def classify_rhythm_from_hrv(hrv_metrics: Dict[str, float]) -> str:
    """
    Perform simple rule-based rhythm classification from HRV metrics.

    This is a heuristic classifier, not a clinical-grade diagnostic tool.

    Parameters
    ----------
    hrv_metrics : Dict[str, float]
        Dictionary from ``compute_hrv_metrics()``.

    Returns
    -------
    str
        Suspected rhythm: 'normal', 'bradycardia', 'tachycardia', 'afib',
        'high_hrv', or 'unknown'.

    Notes
    -----
    Clinical rules used:
    - HR < 60 bpm → bradycardia
    - HR > 100 bpm → tachycardia
    - RMSSD > 80 ms AND SDNN > 60 ms → possible AFib (very irregular)
    - pNN50 > 30% → high HRV / young subject
    """
    hr = hrv_metrics.get("mean_hr_bpm", 0.0)
    rmssd = hrv_metrics.get("rmssd_ms", 0.0)
    sdnn = hrv_metrics.get("sdnn_ms", 0.0)
    pnn50 = hrv_metrics.get("pnn50", 0.0)

    if hr < 60.0:
        return "bradycardia"
    elif hr > 100.0:
        return "tachycardia"
    elif rmssd > 80.0 and sdnn > 60.0:
        return "afib"
    elif pnn50 > 30.0:
        return "high_hrv"
    elif 60.0 <= hr <= 100.0:
        return "normal"
    else:
        return "unknown"
