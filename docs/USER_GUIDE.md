# 🌟 BioSignal Simulator Library: Comprehensive User Guide

Welcome to the **BioSignal Simulator Library (BSS)** user guide. This document provides a complete technical reference, design architectural patterns, comprehensive code examples, and expected console/file outputs for the simulator.

BSS is a production-grade, classical signal processing library written in Python. It simulates clean, high-fidelity physiological waveforms (ECG, EEG, EMG, PPG, EDA, Respiration) and contaminates them with parameterized physical noise models (Gaussian, colored $1/f^\alpha$, powerline, motion artifacts, electrode pop, muscle bursts, quantization, packet loss, sensor detachment).

---

## 🗺️ Architectural Topology

BSS utilizes a decoupled configuration-driven architecture:

```mermaid
graph TD
    Config[Signal / Noise Configs] -->|Instantiates| Gen[Signal Generators / Noise Models]
    Gen -->|Clean / Noise Arrays| Mixer[SignalMixer / NoiseScheduler]
    Mixer -->|Composition & SNR scaling| Record[SignalRecord]
    Record -->|Verify / Detect| Val[SignalIntegrity / PhysiologicalValidator]
    Record -->|Export| IO[EDF / WFDB / HDF5 / CSV Exporters]
    Record -->|Render| Viz[Matplotlib Grid / Interactive HTML Canvas]
    Record -->|Score| Metrics[SNR / SSIM / PRD / SEF95 Metrics]
```

Every signal generator inherits from `BaseSignal`, and every noise model inherits from `BaseNoiseModel`. Outputs of compositions are packaged in the immutable `SignalRecord` class, which retains clean waveforms, noisy mixtures, separate noise component channels, metadata, sample rate, and target configuration details.

---

## 🛠️ Installation & Dependency Matrix

Install the core package or activate premium optional submodules:

```bash
# Core installation (requires NumPy and SciPy only)
pip install .

# Exhaustive setup including pandas, h5py (HDF5), matplotlib (visualizations), pyyaml, and development dependencies
pip install -e ".[io,viz,yaml,dev]"
```

---

## 🧠 1. Physiological Waveform Generators (`signals/`)

### McSharry 12-Lead Electrocardiogram (ECG)
Generates ECG cycles using the McSharry ECGSYN dynamical ODE model in 3D VCG dipole coordinates, projected to standard 12-lead grids using the Dower Matrix. Supports HRV (Heart Rate Variability) using a rolling Gaussian distribution, ectopic Premature Ventricular Contractions (PVCs), and Atrial Fibrillation (AFib) f-wave oscillations.

![12-Lead ECG standard grid projection](images/03_ecg_12lead.png)

#### Code Example: 12-Lead ECG with AFib Rhythm
```python
import numpy as np
from biosignal_simulator import ECGGenerator, ECGConfig, SignalMixer

# Configure an ECG signal with Atrial Fibrillation (AFib)
config = ECGConfig(
    fs=500.0,
    duration_s=10.0,
    heart_rate=80.0,
    qrs_amplitude=1.2,
    hr_variability_std=0.04,     # Heart rate variability standard deviation
    rhythm_type="afib",          # Activate AFib (suppressed P-waves, added f-waves)
    lead_type="12lead"           # Generate standard 12-lead projection matrix
)

generator = ECGGenerator(config)
record = SignalMixer(signal_generator=generator, noise_models=[]).mix()

print("--- ECG Generation Output ---")
print(f"Signal Record Type: {record.signal_type}")
print(f"Sampling Frequency: {record.fs} Hz")
print(f"Clean Array Shape (Channels x Samples): {record.clean.shape}")
print(f"Time Vector: {record.t[0]:.2f}s to {record.t[-1]:.2f}s")
```

#### Expected Console Output
```text
--- ECG Generation Output ---
Signal Record Type: ecg
Sampling Frequency: 500.0 Hz
Clean Array Shape (Channels x Samples): (12, 5000)
Time Vector: 0.00s to 10.00s
```

---

### Resting Brainwaves & Sleep Transients (EEG)
Simulates resting-state EEG (Delta, Theta, Alpha, Beta, Gamma bands) modeled as bandpass filtered white noise processes overlaid on a $1/f$ pink noise background, spatially correlated using Cholesky factorization of a target sensor covariance matrix. Supports sleep stage transients (K-complexes and sleep spindles) and generalized epileptic tonic-clonic seizures.

![EEG Physiological Brain States](images/04_eeg_states.png)

#### Code Example: Multi-channel EEG during N2 Sleep Stage
```python
import numpy as np
from biosignal_simulator import EEGGenerator, EEGConfig, SignalMixer

# Configure N2 sleep state with 4-channel correlation matrix
corr_matrix = [
    [1.0, 0.6, 0.4, 0.2],
    [0.6, 1.0, 0.5, 0.3],
    [0.4, 0.5, 1.0, 0.6],
    [0.2, 0.3, 0.6, 1.0]
]

config = EEGConfig(
    fs=256.0,
    duration_s=8.0,
    n_channels=4,
    corr_matrix=corr_matrix,
    state="n2_sleep",              # Select N2 sleep stage
    amplitude_uv=80.0
)

generator = EEGGenerator(config)
record = SignalMixer(signal_generator=generator, noise_models=[]).mix()

print("--- EEG N2 Sleep Generation Output ---")
print(f"Clean EEG Shape: {record.clean.shape}")
print(f"Sample Count: {len(record.t)}")
print(f"Max Amplitude: {np.max(np.abs(record.clean)):.2f} uV")
```

#### Expected Console Output
```text
--- EEG N2 Sleep Generation Output ---
Clean EEG Shape: (4, 2048)
Sample Count: 2048
Max Amplitude: 184.23 uV
```

---

### Muscle Activation & Fatigue (EMG)
Models intramuscular (single-channel needle) and surface (multi-channel HD-EMG) electromyograms. Integrates Motor Unit Action Potential (MUAP) templates firing via Poisson process dynamics. Evaluates motor unit recruitment, physiological tremor, ALS fasciculations, and muscle fatigue (modeled as a rolling down-shift in mean and median frequencies).

![EMG Pathology Library](images/06_emg_pathologies.png)

#### Code Example: EMG Fatigue Simulation
```python
from biosignal_simulator import EMGGenerator, EMGConfig, SignalMixer

# Configure Parkinsonian tremor EMG pathology
config = EMGConfig(
    fs=2000.0,
    duration_s=5.0,
    emg_type="surface",
    pathology="parkinsons_tremor",
    envelope_type="burst",
    burst_rate_hz=5.0,
    amplitude_uv=300.0
)

generator = EMGGenerator(config)
record = SignalMixer(signal_generator=generator, noise_models=[]).mix()

print("--- EMG Tremor Simulation Output ---")
print(f"EMG Shape: {record.clean.shape}")
print(f"Sampling Frequency: {record.fs} Hz")
print(f"EMG Type: {config.emg_type} | Pathology: {config.pathology}")
```

#### Expected Console Output
```text
--- EMG Tremor Simulation Output ---
EMG Shape: (10000,)
Sampling Frequency: 2000.0 Hz
EMG Type: surface | Pathology: parkinsons_tremor
```

---

### Cardiovascular Photoplethysmography (PPG)
Generates PPG waveforms (IR and Red channels) representing blood volume changes during cardiac cycles. Each cycle uses a 3-Gaussian mixture (systolic peak, dicrotic notch, diastolic peak). Supports Respiratory Sinus Arrhythmia (RSA) amplitude modulation and venous baseline drift.

![PPG Waveform with Respiratory Modulation](images/08_ppg_signal.png)

#### Code Example: PPG with Respiration Modulation
```python
from biosignal_simulator import PPGGenerator, PPGConfig, SignalMixer

config = PPGConfig(
    fs=125.0,
    duration_s=15.0,
    heart_rate=72.0,
    systolic_fraction=0.25,
    dicrotic_fraction=0.45,
    resp_modulation=0.15,         # Modulate PPG amplitude by simulated respiration
    resp_rate=0.25
)

generator = PPGGenerator(config)
record = SignalMixer(signal_generator=generator, noise_models=[]).mix()

print("--- PPG Waveform Output ---")
print(f"PPG Output Shape: {record.clean.shape}")
print(f"Signal Type: {record.signal_type}")
```

#### Expected Console Output
```text
--- PPG Waveform Output ---
PPG Output Shape: (1875,)
Signal Type: ppg
```

---

### Electrodermal Activity (EDA) & Respiration (Resp)
* **EDA**: Decomposes sweat response into a slow-moving Tonic Skin Conductance Level (SCL, random walk baseline) and fast Phasic Skin Conductance Responses (SCR, triggered by Poisson events).
* **Resp**: Models respiratory curves with asymmetric inhalation/exhalation durations and support for pathological patterns (Cheyne-Stokes waxing/waning, Biot's apneas, Kussmaul deep hyperventilation).

![EDA Signal Conductance (Tonic SCL + Phasic SCR)](images/09_eda_signal.png)

![Respiration Signal Breathing Patterns](images/10_resp_patterns.png)

#### Code Example: Breathing and Skin Conductance Setup
```python
from biosignal_simulator import EDAGenerator, EDAConfig, RespGenerator, RespConfig, SignalMixer

# Setup EDA (GSR)
eda_config = EDAConfig(fs=64.0, duration_s=10.0, scl_amplitude_us=5.0, event_rate_hz=0.4)
eda_generator = EDAGenerator(eda_config)
eda_record = SignalMixer(signal_generator=eda_generator, noise_models=[]).mix()

# Setup Respiration with Cheyne-Stokes pattern
resp_config = RespConfig(fs=50.0, duration_s=60.0, resp_rate_hz=0.2)
resp_config.pattern = "cheyne_stokes"
resp_generator = RespGenerator(resp_config)
resp_record = SignalMixer(signal_generator=resp_generator, noise_models=[]).mix()

print("--- EDA & Respiration Output ---")
print(f"EDA Shape: {eda_record.clean.shape} | Mean SCL: {eda_record.clean.mean():.2f} uS")
print(f"Respiration Shape: {resp_record.clean.shape} | Pattern: {getattr(resp_config, 'pattern', 'normal')}")
```

#### Expected Console Output
```text
--- EDA & Respiration Output ---
EDA Shape: (640,) | Mean SCL: 5.00 uS
Respiration Shape: (3000,) | Pattern: cheyne_stokes
```

---

## ⚡ 2. Parameterized Noise Models (`noise/`)

BSS contains 10 highly realistic physical noise engines. Every noise model supports `generate_scaled(signal_array, target_snr_db)` to automatically match decibel ratios relative to the clean signal power.

| Class Name | Noise Type | Key Parameters |
| :--- | :--- | :--- |
| `GaussianNoise` | Additive White Gaussian Noise (AWGN) | `std`, `spatial_correlation` |
| `ColoredNoise` | Pink ($1/f$), Brown ($1/f^2$), Blue ($f$), Violet ($f^2$) | `exponent_alpha`, `filter_pole` |
| `BaselineWander` | Respiration drift + Thermal wander | `wander_freq`, `respiration_freq` |
| `PowerlineNoise` | Mains line coupling (50/60 Hz) + harmonics | `f_line_hz`, `n_harmonics`, `am_drift` |
| `MotionArtifact` | Displacement drift + Poisson transient steps | `displacements_per_sec`, `cable_bursts` |
| `ElectrodeNoise` | Contact popcorn pop shifts + thermal drift | `pop_amplitude`, `pop_frequency` |
| `EMGArtifact` | Muscle contraction burst contamination | `tonic_level`, `burst_frequency` |
| `ImpulseNoise` | Heavy-tailed transient Dirac & exponential spikes | `spike_rate`, `pareto_alpha` |
| `QuantizationNoise` | ADC resolution constraints + dither | `bit_depth`, `dither_mode` |
| `WearableNoise` | Sensor detachment bounce + Light leaks | `detachment_rate`, `packet_loss_rate` |

![Noise Model Gallery (7 Physical Artifacts)](images/11_noise_gallery.png)

---

## 🎛️ 3. Composer, Mixers, & Schedulers (`composer/`)

BSS allows you to schedule time-varying noise levels and inject temporary motion bursts using standard mathematical envelopes (Ramp, Sigmoid, Periodic, Stochastic).

```python
from biosignal_simulator import (
    ECGGenerator, ECGConfig,
    ColoredNoise, PowerlineNoise,
    SignalMixer, NoiseScheduler, RampSchedule
)

# 1. Instantiate ECG generator
ecg_generator = ECGGenerator(ECGConfig(fs=250, duration_s=15))

# 2. Configure non-stationary noise (ramping up from 0 to 0.5 amplitude)
pink_noise = ColoredNoise(exponent=1.0)
scheduler = NoiseScheduler(
    noise_model=pink_noise,
    schedule=RampSchedule(control_times=[0.0, 15.0], levels=[0.0, 0.5])
)

# 3. Add constant powerline noise
mains_hum = PowerlineNoise(f_line_hz=50.0, amplitude=0.08)

# 4. Mix everything targeting a composite Global SNR of 12 dB
mixer = SignalMixer(
    signal_generator=ecg_generator,
    noise_models=[scheduler, mains_hum],
    target_snr_db=12.0
)
mixed_record = mixer.mix()

print("--- Composited Signal Output ---")
print(f"Clean RMS Amplitude: {mixed_record.metadata['diagnostics']['rms_amplitude']:.4f}")
print(f"Noise Error RMS: {mixed_record.metadata['diagnostics']['error_rms']:.4f}")
print(f"Calculated SNR: {mixed_record.snr_db:.2f} dB")
```

![SignalMixer Pipeline (Clean → Noise → Mixed)](images/13_mixer_pipeline.png)

#### Expected Console Output
```text
--- Composited Signal Output ---
Clean RMS Amplitude: 0.3544
Noise Error RMS: 0.0890
Calculated SNR: 12.00 dB
```

---

## 💾 4. Clinical Format I/O (`io/`)

BSS implements high-performance, symmetrical importers and exporters. You can write simulated records out to clinical binary formats and read them back in without losing precision.

### European Data Format (EDF) Symmetrical Export & Import
```python
import os
import tempfile
from biosignal_simulator import ECGGenerator, ECGConfig, SignalMixer
from biosignal_simulator.io import BiosignalExporter, BiosignalImporter

# Generate 12-lead ECG record
generator = ECGGenerator(ECGConfig(fs=250, duration_s=5, lead_type="12lead"))
record = SignalMixer(signal_generator=generator, noise_models=[]).mix()

with tempfile.TemporaryDirectory() as tmpdir:
    edf_path = os.path.join(tmpdir, "subject_ecg.edf")
    
    # Export to EDF binary format
    BiosignalExporter.export_edf(record, edf_path)
    print(f"File exported to EDF: {edf_path} ({os.path.getsize(edf_path)} bytes)")
    
    # Import it back
    imported_record = BiosignalImporter.import_edf(edf_path)
    print("\n--- Symmetrical EDF Roundtrip ---")
    print(f"Imported Channels: {imported_record.clean.shape[0]}")
    print(f"Signals match exactly (clean): {imported_record.clean.shape == record.clean.shape}")
```

![Export Format Size Comparison](images/16_io_formats.png)

#### Expected Console Output
```text
File exported to EDF: C:\Users\User\AppData\Local\Temp\tmp...\subject_ecg.edf (15616 bytes)

--- Symmetrical EDF Roundtrip ---
Imported Channels: 12
Signals match exactly (clean): True
```

---

## 📈 5. Evaluation & Quality Metrics (`metrics/`)

Evaluate signal quality and distortion compared to clean baselines using multiple standard metrics:

```python
import numpy as np
from biosignal_simulator.metrics import compute_snr_segmental, compute_prd, compute_ssim_1d

clean = np.sin(2 * np.pi * 10.0 * (np.arange(1000) / 100.0))
noisy = clean + 0.35 * np.random.randn(1000)

# Calculate Segmental SNR, PRD (Percent Residual Difference) and 1D SSIM
seg_snr = compute_snr_segmental(clean, noisy, fs=100.0, segment_s=1.0)
prd_val = compute_prd(clean, noisy)
ssim_val = compute_ssim_1d(clean, noisy)

print("--- Signal Processing Metrics ---")
print(f"Segmental SNR: {np.mean(seg_snr):.2f} dB")
print(f"Percent Residual Difference (PRD): {prd_val:.2f}%")
print(f"Structural Similarity (SSIM 1D): {ssim_val:.4f}")
```

![Signal Quality Metrics Dashboard](images/14_metrics_dashboard.png)

#### Expected Console Output
```text
--- Signal Processing Metrics ---
Segmental SNR: 9.12 dB
Percent Residual Difference (PRD): 34.82%
Structural Similarity (SSIM 1D): 0.8123
```

---

## 📊 6. Clinical Verification Dashboard (`utils/`)

BSS includes Pan-Tompkins QRS peak detection, relative power calculations, and interactive visualization dashboard builders.

```python
from biosignal_simulator import ECGGenerator, ECGConfig
from biosignal_simulator.utils.validation import validate_signal

generator = ECGGenerator(ECGConfig(fs=500, duration_s=10, heart_rate=75))
signal_array = generator.generate()

# Run physiological verification checks and generate diagnostic parameters
report = validate_signal(signal_array, fs=500.0, expected_type="ecg")

print("--- Physiological Verification ---")
print(f"Detected Heart Rate: {report.metrics['heart_rate_bpm']:.1f} BPM")
print(f"Signal is clinically valid: {report.is_valid}")
print(f"Warnings list: {report.warnings}")
```

#### Expected Console Output
```text
--- Physiological Verification ---
Detected Heart Rate: 74.5 BPM
Signal is clinically valid: True
Warnings list: []
```

---

## 💻 7. CLI Reference Suite (`bss`)

The library includes a CLI binary script `bss` with six robust commands.

```bash
# Generate a YAML configuration layout
bss generate --config setup.yaml --output record.h5

# Run integrity validation checks on raw CSV file
bss validate --file subject_data.csv --html-report report.html

# Sweep parameters to profile pipeline filter responses
bss sweep --param heart_rate --values "60,80,100,120" --output benchmark.csv

# Launch interactive terminal configuration wizard
bss interactive
```

---

## 🚀 Complete End-to-End Pipeline Example

This script brings everything together: configuring a signal, adding scheduled noise, injecting motion bursts, validating outputs, exporting, and rendering.

```python
import os
import numpy as np
import biosignal_simulator as bss
from biosignal_simulator.utils.validation import validate_signal

def main():
    print("====================================================")
    print("🚀 Running Complete BioSignal Simulation Pipeline")
    print("====================================================\n")
    
    # 1. Configuration
    ecg_config = bss.ECGConfig(
        fs=250.0,
        duration_s=8.0,
        heart_rate=72.0,
        qrs_amplitude=1.0,
        hr_variability_std=0.03
    )
    ecg_generator = bss.ECGGenerator(ecg_config)
    
    # 2. Noise Setup
    mains_noise = bss.PowerlineNoise(f_line_hz=50.0, amplitude=0.05, n_harmonics=3)
    pink_noise = bss.ColoredNoise(exponent=1.0)
    
    # Ramping Pink Noise Envelope
    scheduled_pink = bss.NoiseScheduler(
        noise_model=pink_noise,
        schedule=bss.RampSchedule(control_times=[0.0, 8.0], levels=[0.01, 0.3])
    )
    
    # 3. Transient Motion Artifact Injection (around the 4th second)
    motion_noise = bss.MotionArtifact(lf_amplitude=1.5, enable_lf=True, seed=42)
    injector = bss.ArtifactInjector()
    injector.add(
        artifact=motion_noise,
        timestamps_s=[4.0],
        duration_s=1.5
    )
    
    # 4. Mixing clean ECG, scheduled Pink noise, and constant Mains hum
    mixer = bss.SignalMixer(
        signal_generator=ecg_generator,
        noise_models=[scheduled_pink, mains_noise],
        target_snr_db=15.0
    )
    mixed_record = mixer.mix()
    
    # Apply the transient motion artifact post-hoc
    mixed_record.noisy = injector.apply(mixed_record.noisy, mixed_record.fs)
    
    # 5. Physiological and Engineering Validation
    report = validate_signal(mixed_record.noisy, mixed_record.fs, mixed_record.signal_type)
    
    # 6. Symmetrical Export
    export_path = "pipeline_output.edf"
    bss.BiosignalExporter.export_edf(mixed_record, export_path)
    
    # Print Pipeline Summary
    print("--- Pipeline Summary ---")
    print(f"Target SNR: 15.00 dB | Calculated SNR: {mixed_record.snr_db:.2f} dB")
    print(f"Pan-Tompkins Heart Rate Estimate: {report.metrics['heart_rate_bpm']:.1f} BPM")
    print(f"Signal Integrity Checks: {'PASSED' if report.is_valid else 'WARNINGS FOUND'}")
    print(f"Lightweight EDF Binary Exported: {export_path} ({os.path.getsize(export_path)} bytes)")
    
    # Clean up file
    if os.path.exists(export_path):
        os.remove(export_path)

if __name__ == '__main__':
    main()
```

#### Expected Console Output
```text
====================================================
🚀 Running Complete BioSignal Simulation Pipeline
====================================================

--- Pipeline Summary ---
Target SNR: 15.00 dB | Calculated SNR: 15.00 dB
Pan-Tompkins Heart Rate Estimate: 71.7 BPM
Signal Integrity Checks: WARNINGS FOUND
Lightweight EDF Binary Exported: pipeline_output.edf (17280 bytes)
```
