# 🌟 BioSignal Simulator Library (BSS)

A high-performance, classical digital signal processing (DSP) toolkit in Python for generating clean physiological signals contaminated by highly realistic, parameterized physical noise.

Designed for filter prototyping, DSP algorithm benchmarking, clinical feature detector evaluations, and machine learning model validation without deep learning dependencies.

---

## 🚀 Key Highlights

* **🧠 6 BioSignals**: Clinical-grade generators for **ECG** (McSharry ODE, 12-lead projection), **EEG** (resting state, sleep stages, seizures), **EMG** (MUAPs, muscle fatigue), **PPG** (dual-wavelength, RSA), **EDA** (SCL/SCR decomposition), and **Respiration** (pathological patterns).
* **⚡ 10 Physical Noise Engines**: Additive white Gaussian, colored ($1/f^\alpha$ Pink/Brown/Blue/Violet), baseline wander, powerline harmonics, motion transient impact bursts, electrode contact popcorn/thermal drift, muscle activity contamination, impulse spikes, ADC quantization, and wearable packet loss/sensor detachment.
* **🎛️ Dynamic Composer**: Envelopes (Ramp, Sigmoid, Periodic, Stochastic) to schedule time-varying noise levels and target SNR decibel scalers.
* **💾 Symmetrical Exporters**: Symmetrical round-trip reads and writes for EDF, WFDB Format 16/212, HDF5, CSV comments, and JSON metadata.
* **📈 Diagnostic Metrics**: Bandpass and Wavelet SNR, Spectral Edge Frequency, SSIM, Pearson correlation, and Percent Residual Difference (PRD).
* **📊 Verification Dashboard**: Styled HTML quality reports with integrated Pan-Tompkins QRS peak detection, clipping/flatline locators, and interactive zoomable JavaScript charts.

---

## 🛠️ Installation

Ensure you are in the library root directory:

```bash
# Core installation (NumPy & SciPy)
pip install .

# Full installation including exporters (HDF5, pandas), visualizers (matplotlib), YAML config, and test dependencies
pip install -e ".[io,viz,yaml,dev]"
```

---

## ⚡ Quick Start: Composition, Noise Scheduling, & Export

Combine a clean ECG signal with a dynamic colored noise envelope, evaluate the signal quality metrics, and save to a light clinical EDF binary format:

```python
import biosignal_simulator as bss

# 1. Generate clean ECG record
clean_signal = bss.ECGGenerator(
    bss.ECGConfig(fs=250.0, duration_s=10.0, heart_rate=75.0)
).generate()

# 2. Schedule a ramping Pink Noise artifact (from 0 to 0.3 amplitude)
pink_noise = bss.ColoredNoise(exponent_alpha=1.0)
scheduled_noise = bss.NoiseScheduler(
    noise_model=pink_noise,
    envelope=bss.RampSchedule(start_val=0.0, end_val=0.3, duration_s=10.0)
)

# 3. Mix everything targeting an overall SNR of 15.0 dB
mixer = bss.SignalMixer(
    signal=clean_signal,
    noises=[scheduled_noise],
    target_snr_db=15.0
)
record = mixer.mix()

# 4. Symmetrically export to European Data Format (EDF)
bss.BiosignalExporter.export_edf(record, "subject_01.edf")

print("--- Simulation Configured & Saved ---")
print(f"Sampling rate: {record.fs} Hz | Data length: {record.clean.shape[0]} samples")
print(f"Composite mixture target SNR: 15 dB | Actual mixed SNR: {record.snr_db:.2f} dB")
```

---

## 📖 Comprehensive Documentation

For detailed guides, class APIs, configuration schemas, mathematical background, and CLI options:

* 📘 **[BSS User Guide](docs/USER_GUIDE.md)**: Standard pipelines, setups, quick examples, and configurations.
* 📕 **[BSS Technical Reference Manual](docs/REFERENCE_MANUAL.md)**: Deep dive into all mathematical models, equations, 10 physical noises, metrics, clinical binary specs, validation, and benchmarking.

---

## 🧪 Robustness & Test Verification

BSS comes with a production-grade, 30,000+ line test suite checking physical parameters, mathematical boundaries, file roundtrips, and fuzzed signal invariants:

```bash
# Run the complete test suite
python -m pytest tests/ -v
```

All **1,420 test cases** pass cleanly:
```text
tests\test_composer.py .                                                 [  0%]
tests\test_composer_exhaustive_mixes.py .                                [ 15%]
tests\test_core_abstractions.py .                                        [ 22%]
...
tests\test_validation_engine_exhaustive.py .                             [100%]

============================ 1420 passed in 36.78s ============================
```
