# 🌟 BioSignal Simulator (BSS)

[![PyPI version](https://img.shields.io/pypi/v/biosignal-simulator.svg)](https://pypi.org/project/biosignal-simulator/)
[![Python support](https://img.shields.io/pypi/pyversions/biosignal-simulator.svg)](https://pypi.org/project/biosignal-simulator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Interactive Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/therajpoots/Biosim/blob/main/demo.ipynb)

**Stress-test biomedical ML models and digital signal processing (DSP) filters under highly realistic physical noise.**

Don't train physiological classifiers on pristine clinical databases and expect them to succeed in the wild. BSS generates clinical-grade clean bio-signals (ECG, EEG, EMG, PPG, EDA, respiration) and contaminates them with precise, parameterized physical noise engines (colored noise, electrode drift, motion artifacts, wearable packet loss).

---

## ⚡ Quick Start (Copy-Paste)

```bash
pip install biosignal-simulator
```

Generate a clean PPG signal, contaminate it with a ramping pink noise drift and motion artifacts, and export it to a clinical EDF file in under 10 lines of code:

```python
import biosignal_simulator as bss

# 1. Generate clean PPG signal with realistic dicrotic notch
ppg = bss.PPGGenerator(bss.PPGConfig(fs=125.0, duration_s=10.0, heart_rate=72.0)).generate()

# 2. Schedule a ramping Pink (1/f) noise drift
pink_noise = bss.NoiseScheduler(
    noise_model=bss.ColoredNoise(exponent_alpha=1.0),
    envelope=bss.RampSchedule(start_val=0.0, end_val=0.4, duration_s=10.0)
)

# 3. Mix targeting exactly 12 dB SNR
mixed = bss.SignalMixer(signal=ppg, noises=[pink_noise], target_snr_db=12.0).mix()

# 4. Symmetrically export to European Data Format (EDF)
bss.BiosignalExporter.export_edf(mixed, "ppg_stress_test.edf")
```

---

## 🎨 Interactive Visual Previews

### 1. Signal Composition Pipeline
The `SignalMixer` dynamically scales physical noise templates using scheduling envelopes (Ramps, Steps, Sigmoids) to meet your target SNR decibel constraint:

![Signal Mixer Pipeline](https://raw.githubusercontent.com/therajpoots/Biosim/main/docs/images/13_mixer_pipeline.png)

### 2. Contaminating Physical Noise Gallery
BSS models 10 distinct physical noise categories, including pink/brown flicker noise, electrode popcorn pops, motion transients, and wearable packet loss:

![Contaminating Noise Gallery](https://raw.githubusercontent.com/therajpoots/Biosim/main/docs/images/11_noise_gallery.png)

### 3. Interactive Metrics Dashboard
BSS automatically generates beautiful, diagnostic HTML dashboards with zoomable charts and Pan-Tompkins peak analysis to verify your DSP filters:

![Metrics Dashboard Preview](https://raw.githubusercontent.com/therajpoots/Biosim/main/docs/images/14_metrics_dashboard.png)

---

## 🚀 Key Highlights

* **🧠 6 BioSignals**: Clinical-grade generators for **ECG** (McSharry ODE, 12-lead projection), **EEG** (resting state, sleep stages, seizures), **EMG** (MUAPs, muscle fatigue), **PPG** (dual-wavelength, arterial stiffness), **EDA** (SCL/SCR decomposition), and **Respiration** (pathological patterns).
* **⚡ 10 Physical Noise Engines**: Additive white Gaussian, colored ($1/f^\alpha$ Pink/Brown/Blue/Violet), baseline wander, powerline harmonics, motion transient impact bursts, electrode contact popcorn/thermal drift, muscle activity contamination, impulse spikes, ADC quantization, and wearable packet loss/sensor detachment.
* **🎛️ Dynamic Composer**: Envelopes (Ramp, Sigmoid, Periodic, Stochastic) to schedule time-varying noise levels and target SNR decibel scalers.
* **💾 Symmetrical Exporters**: Symmetrical round-trip reads and writes for EDF, WFDB Format 16/212, HDF5, CSV comments, and JSON metadata.
* **📈 Diagnostic Metrics**: Bandpass and Wavelet SNR, Spectral Edge Frequency, SSIM, Pearson correlation, and Percent Residual Difference (PRD).

---

## 📖 Comprehensive Documentation

For detailed guides, class APIs, configuration schemas, mathematical background, and CLI options:

* 📘 **[BSS User Guide](https://github.com/therajpoots/Biosim/blob/main/docs/USER_GUIDE.md)**: Standard pipelines, setups, quick examples, and configurations.
* 📕 **[BSS Technical Reference Manual](https://github.com/therajpoots/Biosim/blob/main/docs/REFERENCE_MANUAL.md)**: Deep dive into all mathematical models, equations, 10 physical noises, metrics, clinical binary specs, validation, and benchmarking.

---

## 🧪 Robustness & Test Verification

BSS comes with a production-grade test suite checking physical parameters, mathematical boundaries, file roundtrips, and fuzzed signal invariants:

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
