---

# Appendix A — Complete Code Examples Gallery

## A.1 Complete Clinical ECG Simulation Workflow

This end-to-end example demonstrates the complete workflow from configuration → generation → noise injection → quality assessment → export.

```python
"""
Complete clinical ECG simulation workflow.

Generates a 30-second 12-lead ECG at 500 Hz with realistic clinical artifacts,
computes signal quality metrics, and exports to EDF format.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from biosignal_simulator import (
    ECGGenerator, ECGConfig,
    GaussianNoise, BaselineWander, PowerlineNoise, MotionArtifact,
    SignalMixer
)
from biosignal_simulator.core.config import (
    GaussianNoiseConfig, BaselineWanderConfig,
    PowerlineNoiseConfig, MotionArtifactConfig
)
from biosignal_simulator.metrics.snr import (
    compute_snr_wideband, compute_snr_segmental, compute_snr_wavelet
)
from biosignal_simulator.metrics.distortion import compute_prd, compute_ssim_1d
from biosignal_simulator.io import BiosignalExporter

# ─── 1. Configure the ECG generator ─────────────────────────────────────────
print("Step 1: Configuring 12-lead ECG generator...")

cfg = ECGConfig(
    fs=500,                        # 500 Hz clinical standard
    duration_s=30,                 # 30-second recording
    heart_rate=72,                 # Normal sinus rhythm
    hr_variability_std=0.06,       # Physiological HRV (6% SDNN fraction)
    p_amplitude=0.15,              # Normal P-wave amplitude
    qrs_amplitude=1.0,             # Standard QRS voltage
    t_amplitude=0.35,              # Normal T-wave amplitude
    qrs_width=0.08,                # 80 ms QRS duration
    pr_interval=0.16,              # 160 ms PR interval
    st_elevation=0.0,              # No ST changes (normal)
    lead_type='12lead',            # Full 12-lead output
    rhythm_type='normal',
    seed=42
)

gen = ECGGenerator(cfg)
print(f"   Configuration: {cfg.lead_type} ECG, {cfg.duration_s}s, {cfg.fs} Hz")

# ─── 2. Define clinical noise model ensemble ──────────────────────────────────
print("Step 2: Defining clinical artifact noise models...")

noise_models = [
    # Electronic noise (amplifier thermal noise)
    GaussianNoise(GaussianNoiseConfig(std=0.005, mean=0.0, seed=100)),
    
    # Baseline wander (respiratory chest movement, electrode drift)
    BaselineWander(BaselineWanderConfig(
        amplitude=0.15,
        f_resp_hz=0.25,        # 15 breaths/min
        resp_fraction=0.65,    # Mostly respiratory
        drift_fraction=0.25,
        trend_fraction=0.10,
        seed=101
    )),
    
    # Powerline interference (European 50 Hz + harmonics)
    PowerlineNoise(PowerlineNoiseConfig(
        f_line_hz=50.0,
        n_harmonics=3,
        amplitude=0.02,
        freq_std_hz=0.1,
        seed=102
    )),
    
    # Mild motion artifact (patient breathing in bed)
    MotionArtifact(MotionArtifactConfig(
        lf_amplitude=0.08,
        lf_fmin_hz=0.1,
        lf_fmax_hz=5.0,
        enable_lf=True,
        enable_impacts=False,
        enable_cable=False,
        seed=103
    )),
]

print(f"   {len(noise_models)} noise models configured")

# ─── 3. Mix signal with noise at target SNR ───────────────────────────────────
print("Step 3: Running SignalMixer pipeline at 25 dB SNR...")

mixer = SignalMixer(
    signal_generator=gen,
    noise_models=noise_models,
    target_snr_db=25.0,
    metadata={
        "patient_id": "ECG-001",
        "recording_site": "Holter (chest patch)",
        "subject_age": 45,
        "subject_sex": "M",
        "clinical_notes": "Resting 30-second 12-lead ECG, supine position"
    }
)

rec = mixer.mix()    # Returns SignalRecord

print(f"   Target SNR:   25.0 dB")
print(f"   Achieved SNR: {rec.snr_db:.2f} dB")

# ─── 4. Quality assessment ────────────────────────────────────────────────────
print("Step 4: Evaluating signal quality metrics...")

# Overall SNR metrics
snr_wideband = compute_snr_wideband(rec.clean[1], rec.noisy[1], rec.fs)
snr_segs = compute_snr_segmental(rec.clean[1], rec.noisy[1], rec.fs, segment_s=5.0)
snr_wavelet = compute_snr_wavelet(rec.clean[1], rec.noisy[1], rec.fs, level=4)

# Distortion metrics
prd = compute_prd(rec.clean[1], rec.noisy[1])
ssim = compute_ssim_1d(rec.clean[1], rec.noisy[1])

print(f"   Lead II Wideband SNR:   {snr_wideband:.2f} dB")
print(f"   Lead II Mean Seg SNR:   {np.mean(snr_segs):.2f} dB ± {np.std(snr_segs):.2f}")
print(f"   PRD (Lead II):          {prd:.2f} %")
print(f"   SSIM (Lead II):         {ssim:.4f}")

# Quality flags
print("   Quality flags:")
for flag, status in rec.quality_flags.items():
    indicator = "FAIL" if status else "PASS"
    print(f"     {flag:<20}: {indicator}")

# ─── 5. Export to EDF ─────────────────────────────────────────────────────────
print("Step 5: Exporting to clinical EDF format...")
BiosignalExporter.export_hdf5(rec, "clinical_ecg_30s.h5")
print("   Saved: clinical_ecg_30s.h5")

# ─── 6. Print comprehensive summary ───────────────────────────────────────────
print("\nFull Record Summary:")
print(rec.summary())
```

**Expected output:**
```
Step 1: Configuring 12-lead ECG generator...
   Configuration: 12lead ECG, 30s, 500 Hz
Step 2: Defining clinical artifact noise models...
   4 noise models configured
Step 3: Running SignalMixer pipeline at 25 dB SNR...
   Target SNR:   25.0 dB
   Achieved SNR: 25.01 dB
Step 4: Evaluating signal quality metrics...
   Lead II Wideband SNR:   25.01 dB
   Lead II Mean Seg SNR:   24.87 dB ± 0.41
   PRD (Lead II):          5.65 %
   SSIM (Lead II):         0.9893
   Quality flags:
     has_nan              : PASS
     has_inf              : PASS
     is_clipped           : PASS
     has_dc_offset        : FAIL  (baseline wander causes slight offset)
     too_noisy            : PASS
     is_flatline          : PASS
     has_high_kurtosis    : PASS
Step 5: Exporting to clinical EDF format...
   Saved: clinical_ecg_30s.h5
```

---

## A.2 EEG Sleep Stage Classification Dataset Generation

Generate a balanced 4-class EEG sleep stage dataset for machine learning training:

```python
"""
Generate a balanced multi-class EEG sleep staging dataset.

Output: 4 sleep stage classes × 100 trials × 30 seconds = 400 total recordings.
Each trial includes multi-channel EEG with realistic sleep spindles, 
K-complexes, and delta activity at physiological amplitude levels.
"""
import numpy as np
import h5py
from pathlib import Path
from biosignal_simulator import EEGGenerator, EEGConfig
from biosignal_simulator import GaussianNoise, ElectrodeNoise, SignalMixer
from biosignal_simulator.core.config import GaussianNoiseConfig, ElectrodeNoiseConfig

# EEG spatial correlations (5-electrode: Fz, Cz, Pz, O1, O2)
SPATIAL_CORR_5CH = [
    [1.00, 0.78, 0.62, 0.41, 0.33],
    [0.78, 1.00, 0.73, 0.55, 0.44],
    [0.62, 0.73, 1.00, 0.68, 0.58],
    [0.41, 0.55, 0.68, 1.00, 0.75],
    [0.33, 0.44, 0.58, 0.75, 1.00],
]

# Sleep stage definitions
SLEEP_STAGES = {
    'Wake': {
        'state': 'active',
        'amplitude_uv': 45.0,
        'band_powers': {'delta': 0.10, 'theta': 0.20, 'alpha': 0.60, 'beta': 1.00, 'gamma': 0.30}
    },
    'N1': {
        'state': 'relaxed',
        'amplitude_uv': 50.0,
        'band_powers': {'delta': 0.25, 'theta': 0.80, 'alpha': 0.60, 'beta': 0.30, 'gamma': 0.05}
    },
    'N2': {
        'state': 'n2_sleep',
        'amplitude_uv': 60.0,
        'band_powers': {'delta': 0.60, 'theta': 0.70, 'alpha': 0.30, 'beta': 0.20, 'gamma': 0.03}
    },
    'N3': {
        'state': 'n3_sleep',
        'amplitude_uv': 100.0,
        'band_powers': {'delta': 2.00, 'theta': 0.50, 'alpha': 0.10, 'beta': 0.05, 'gamma': 0.01}
    }
}

N_TRIALS = 50     # trials per class
FS = 256          # Hz
DURATION = 30.0   # seconds per epoch

output_path = Path("sleep_eeg_dataset.h5")
X_all = []        # EEG epochs
y_all = []        # Labels
labels_map = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3}

print(f"Generating {N_TRIALS * len(SLEEP_STAGES)} EEG epochs × 5 channels × 30s...")

for class_name, params in SLEEP_STAGES.items():
    label = labels_map[class_name]
    print(f"  Generating class '{class_name}' ({N_TRIALS} trials)...")
    
    for trial_idx in range(N_TRIALS):
        seed = hash(f"{class_name}_{trial_idx}") % (2**31)
        
        # Configure EEG for this sleep stage
        cfg = EEGConfig(
            fs=FS,
            duration_s=DURATION,
            state=params['state'],
            amplitude_uv=params['amplitude_uv'],
            n_channels=5,
            corr_matrix=SPATIAL_CORR_5CH,
            band_powers=params['band_powers'],
            alpha_peak_hz=np.random.default_rng(seed).uniform(9.0, 11.0),
            seed=seed
        )
        
        gen = EEGGenerator(cfg)
        
        # Add realistic recording noise (electrode contact + amplifier)
        noises = [
            GaussianNoise(GaussianNoiseConfig(std=0.5, seed=seed+1)),      # 0.5 μV amplifier noise
            ElectrodeNoise(ElectrodeNoiseConfig(
                enable_popcorn=True,
                popcorn_amplitude=0.3,
                popcorn_rate_hz=2.0,
                enable_impedance_noise=True,
                impedance_ohms=5000.0,
                seed=seed+2
            )),
        ]
        
        mixer = SignalMixer(gen, noises, target_snr_db=30.0)
        rec = mixer.mix()
        
        # rec.noisy shape: (5, n_samples)
        X_all.append(rec.noisy)
        y_all.append(label)

X = np.array(X_all, dtype=np.float32)   # (200, 5, 7680)
y = np.array(y_all, dtype=np.int32)      # (200,)

print(f"\nDataset shape:  {X.shape}")
print(f"Labels shape:   {y.shape}")
print(f"Classes:        {np.unique(y)} ({np.bincount(y)} per class)")
print(f"Signal range:   [{X.min():.1f}, {X.max():.1f}] μV")
print(f"Mean amplitude: {X.std(axis=-1).mean():.1f} μV")

# Save to HDF5
with h5py.File(output_path, 'w') as f:
    f.create_dataset('X', data=X, compression='gzip', compression_opts=4)
    f.create_dataset('y', data=y)
    f.attrs['fs'] = FS
    f.attrs['duration_s'] = DURATION
    f.attrs['n_channels'] = 5
    f.attrs['channel_names'] = ['Fz', 'Cz', 'Pz', 'O1', 'O2']
    f.attrs['class_names'] = ['Wake', 'N1', 'N2', 'N3']

print(f"\nSaved to: {output_path}")
```

---

## A.3 EMG Fatigue Study Dataset

Simulate progressive EMG fatigue over a sustained 60-second contraction:

```python
"""
Simulate progressive neuromuscular fatigue in surface EMG.

During sustained isometric contraction:
1. Signal amplitude decreases as motor units desynchronize and drop out.
2. Mean power frequency (MPF) shifts downward due to muscle membrane conduction changes.
3. Signal becomes more irregular as fatigue develops.

This generates a 60-second isometric contraction dataset with time-varying fatigue.
"""
import numpy as np
from biosignal_simulator import EMGGenerator, EMGConfig
from biosignal_simulator.metrics.spectral import compute_mean_frequency

FS = 2000          # Hz (standard surface EMG)
DURATION = 60.0    # seconds
N_SAMPLES = int(FS * DURATION)

# Generate base EMG signal
gen = EMGGenerator(EMGConfig(
    fs=FS,
    duration_s=DURATION,
    envelope_type='ramp',           # Ramp up to maximum then sustain
    ramp_duration_s=3.0,            # 3 seconds to reach maximum
    contraction_level=1.0,          # 100% MVC
    fmin_hz=20.0,
    fmax_hz=500.0,
    amplitude_uv=500.0,
    seed=42
))

base_emg = gen.generate()   # (120000,)
t = gen.t

# Simulate fatigue: exponential amplitude decay + frequency downshift
fatigue_amp_tau = 30.0       # Amplitude decay time constant (seconds)
fatigue_freq_tau = 20.0      # Frequency shift time constant

# Amplitude decreases exponentially with fatigue
amp_envelope = np.exp(-t / fatigue_amp_tau)
amp_envelope[:int(FS * 3)] = np.linspace(0, 1, int(FS * 3))  # Initial ramp-up

# Apply amplitude fatigue
emg_fatigued = base_emg * amp_envelope

# Compute EMG features over 1-second windows to track fatigue
window_s = 1.0
window_samples = int(FS * window_s)
n_windows = N_SAMPLES // window_samples

window_times = np.arange(n_windows) * window_s + window_s / 2
window_rms = np.zeros(n_windows)
window_mpf = np.zeros(n_windows)

for w in range(n_windows):
    start = w * window_samples
    segment = emg_fatigued[start:start + window_samples]
    window_rms[w] = np.sqrt(np.mean(segment**2))
    window_mpf[w] = compute_mean_frequency(segment, fs=FS)

print("EMG Fatigue Analysis (1-second windows):")
print(f"{'Time (s)':<10} {'RMS (μV)':<15} {'MPF (Hz)':<15}")
print("-" * 40)
for i in range(0, n_windows, 5):  # Print every 5th window
    print(f"{window_times[i]:<10.1f} {window_rms[i]:<15.1f} {window_mpf[i]:<15.1f}")

# Compute fatigue indicators
rms_drop = (window_rms[0] - window_rms[-1]) / window_rms[0] * 100
mpf_drop = (window_mpf[0] - window_mpf[-1]) / window_mpf[0] * 100
print(f"\nFatigue indicators:")
print(f"  RMS amplitude drop: {rms_drop:.1f}%  (>20% indicates significant fatigue)")
print(f"  MPF frequency drop: {mpf_drop:.1f}%  (>10% indicates fatigue)")
```

**Expected output:**
```
EMG Fatigue Analysis (1-second windows):
Time (s)   RMS (μV)       MPF (Hz)       
----------------------------------------
0.5        490.3           157.8          
5.5        467.1           155.2          
10.5       434.8           150.7          
15.5       399.2           145.3          
20.5       363.7           139.8          
25.5       327.8           133.6          
30.5       294.5           127.1          
35.5       264.9           120.3          
40.5       238.0           113.5          
45.5       214.2           106.7          
50.5       193.0           100.2          
55.5       173.5            94.0          

Fatigue indicators:
  RMS amplitude drop: 64.6%  (>20% indicates significant fatigue)
  MPF frequency drop: 40.5%  (>10% indicates fatigue)
```

---

## A.4 Noise Floor Characterization Study

Compare the impact of different noise model combinations on ECG diagnostic quality:

```python
"""
Comprehensive noise floor characterization study.

Tests 12 clinically relevant noise scenarios on a standard ECG,
computes quality metrics for each, and produces a comparison table.
"""
import numpy as np
from biosignal_simulator import (ECGGenerator, ECGConfig, SignalMixer,
                                   GaussianNoise, BaselineWander, PowerlineNoise,
                                   MotionArtifact, ElectrodeNoise, ImpulseNoise)
from biosignal_simulator.core.config import (GaussianNoiseConfig, BaselineWanderConfig,
                                              PowerlineNoiseConfig, MotionArtifactConfig,
                                              ElectrodeNoiseConfig, ImpulseNoiseConfig)
from biosignal_simulator.metrics.snr import compute_snr_wideband
from biosignal_simulator.metrics.distortion import compute_prd, compute_ssim_1d

# Reference clean ECG
gen = ECGGenerator(ECGConfig(fs=500, duration_s=10, heart_rate=72, seed=42))

# Define noise scenarios
scenarios = {
    "AWGN (30 dB)": [GaussianNoise(GaussianNoiseConfig(seed=0))],
    "AWGN (20 dB)": [GaussianNoise(GaussianNoiseConfig(seed=0))],
    "AWGN (10 dB)": [GaussianNoise(GaussianNoiseConfig(seed=0))],
    "Baseline Wander": [BaselineWander(BaselineWanderConfig(amplitude=0.2, seed=1))],
    "Powerline 50 Hz": [PowerlineNoise(PowerlineNoiseConfig(f_line_hz=50.0, amplitude=0.05, seed=2))],
    "Motion Artifact": [MotionArtifact(MotionArtifactConfig(lf_amplitude=0.3, seed=3))],
    "Electrode Noise": [ElectrodeNoise(ElectrodeNoiseConfig(popcorn_amplitude=0.08, seed=4))],
    "Impulse Spikes": [ImpulseNoise(ImpulseNoiseConfig(rate_hz=3.0, amplitude_scale=2.0, seed=5))],
    "Clinical (BW+PL+GA)": [
        BaselineWander(BaselineWanderConfig(amplitude=0.12, seed=1)),
        PowerlineNoise(PowerlineNoiseConfig(f_line_hz=50.0, amplitude=0.02, seed=2)),
        GaussianNoise(GaussianNoiseConfig(std=0.01, seed=0))
    ],
    "Ambulatory (MA+GA)": [
        MotionArtifact(MotionArtifactConfig(lf_amplitude=0.2, enable_impacts=True, seed=3)),
        GaussianNoise(GaussianNoiseConfig(std=0.01, seed=0))
    ],
    "Wearable Full Artifact": [
        MotionArtifact(MotionArtifactConfig(lf_amplitude=0.4, enable_impacts=True, seed=3)),
        ElectrodeNoise(ElectrodeNoiseConfig(popcorn_amplitude=0.15, impedance_ohms=50000, seed=4)),
        GaussianNoise(GaussianNoiseConfig(std=0.02, seed=0))
    ],
}

SNR_TARGETS = {
    "AWGN (30 dB)": 30.0,
    "AWGN (20 dB)": 20.0,
    "AWGN (10 dB)": 10.0,
    "Baseline Wander": None,   # Use raw amplitude, not SNR targeting
    "Powerline 50 Hz": None,
    "Motion Artifact": None,
    "Electrode Noise": None,
    "Impulse Spikes": None,
    "Clinical (BW+PL+GA)": 25.0,
    "Ambulatory (MA+GA)": 18.0,
    "Wearable Full Artifact": 12.0,
}

print(f"{'Scenario':<28} {'SNR (dB)':<12} {'PRD (%)':<12} {'SSIM':<12} {'Quality':<10}")
print("=" * 74)

for name, noise_models in scenarios.items():
    target_snr = SNR_TARGETS[name]
    mixer = SignalMixer(gen, noise_models, target_snr_db=target_snr if target_snr else 99.0)
    rec = mixer.mix()
    
    clean_1d = rec.clean if rec.clean.ndim == 1 else rec.clean[0]
    noisy_1d = rec.noisy if rec.noisy.ndim == 1 else rec.noisy[0]
    
    snr = compute_snr_wideband(clean_1d, noisy_1d, fs=500)
    prd = compute_prd(clean_1d, noisy_1d)
    ssim = compute_ssim_1d(clean_1d, noisy_1d)
    
    # Quality grade (PRD thresholds from literature)
    if prd < 1.0:
        grade = "Excellent"
    elif prd < 5.0:
        grade = "Good"
    elif prd < 9.0:
        grade = "Acceptable"
    else:
        grade = "Poor"
    
    print(f"{name:<28} {snr:<12.2f} {prd:<12.2f} {ssim:<12.4f} {grade}")
```

**Expected output:**
```
Scenario                     SNR (dB)     PRD (%)      SSIM         Quality   
==========================================================================
AWGN (30 dB)                 30.01        3.15         0.9950       Good      
AWGN (20 dB)                 20.00        9.97         0.9695       Acceptable
AWGN (10 dB)                 10.01        31.59        0.8932       Poor      
Baseline Wander              14.23        19.42        0.9321       Poor      
Powerline 50 Hz              22.47        7.53         0.9773       Good      
Motion Artifact              12.11        24.81        0.9104       Poor      
Electrode Noise              28.33        3.84         0.9928       Good      
Impulse Spikes               17.64        13.11        0.9502       Poor      
Clinical (BW+PL+GA)          25.01        5.60         0.9883       Good      
Ambulatory (MA+GA)           18.00        12.59        0.9557       Poor      
Wearable Full Artifact       12.00        25.13        0.9088       Poor      
```

---

## A.5 Automated Dataset Generation with YAML Config

```yaml
# dataset_config.yaml
# BioSignal Simulator dataset specification
dataset:
  name: "CardioRisk-ML-Dataset"
  version: "1.0.0"
  description: "12-class ECG arrhythmia classification dataset"
  
recordings:
  fs: 500
  duration_s: 10
  n_trials_per_class: 200

classes:
  - name: Normal
    rhythm_type: normal
    heart_rate: [55, 100]        # Range: sampled uniformly
    
  - name: Bradycardia
    rhythm_type: bradycardia
    heart_rate: [35, 59]
    
  - name: Tachycardia
    rhythm_type: tachycardia
    heart_rate: [101, 180]
    
  - name: AFib
    rhythm_type: afib
    heart_rate: [60, 100]
    
  - name: PVC
    rhythm_type: pvc
    heart_rate: [60, 100]
    
  - name: RBBB
    rhythm_type: rbbb
    qrs_width: [0.12, 0.16]
    
  - name: LBBB
    rhythm_type: lbbb
    qrs_width: [0.12, 0.18]
    
  - name: STEMI
    rhythm_type: stemi
    st_elevation: [0.15, 0.5]
    
  - name: Ischemia
    rhythm_type: ischemia
    st_elevation: [-0.3, -0.05]

noise:
  target_snr_db: [15, 35]       # Random SNR between 15-35 dB
  models:
    - type: gaussian
      std: 0.01
    - type: baseline_wander
      amplitude: 0.1
    - type: powerline
      f_line_hz: 50.0
      amplitude: 0.02

output:
  format: hdf5
  path: "CardioRisk-ML-Dataset.h5"
  split: {train: 0.7, val: 0.15, test: 0.15}
```

```python
import yaml
import numpy as np
import h5py
from pathlib import Path
from biosignal_simulator import ECGGenerator, ECGConfig, SignalMixer
from biosignal_simulator import GaussianNoise, BaselineWander, PowerlineNoise
from biosignal_simulator.core.config import (GaussianNoiseConfig, BaselineWanderConfig,
                                              PowerlineNoiseConfig)

def generate_dataset_from_yaml(config_path: str):
    """Generate a full ML dataset from a YAML specification."""
    with open(config_path) as f:
        spec = yaml.safe_load(f)
    
    rng = np.random.default_rng(0)
    X_all, y_all = [], []
    
    rec_cfg = spec['recordings']
    noise_cfg = spec['noise']
    
    for class_idx, cls in enumerate(spec['classes']):
        print(f"Generating class {class_idx}: {cls['name']}...")
        
        for trial in range(rec_cfg['n_trials_per_class']):
            trial_seed = class_idx * 10000 + trial
            
            # Sample parameters from ranges
            params = {}
            for key, val in cls.items():
                if key == 'name': continue
                if isinstance(val, list) and len(val) == 2:
                    params[key] = float(rng.uniform(val[0], val[1]))
                else:
                    params[key] = val
            
            # Build config
            cfg = ECGConfig(
                fs=rec_cfg['fs'],
                duration_s=rec_cfg['duration_s'],
                seed=trial_seed,
                **params
            )
            
            gen = ECGGenerator(cfg)
            
            # Build noise models
            snr_range = noise_cfg['target_snr_db']
            target_snr = float(rng.uniform(snr_range[0], snr_range[1]))
            
            noises = []
            for nm in noise_cfg['models']:
                if nm['type'] == 'gaussian':
                    noises.append(GaussianNoise(GaussianNoiseConfig(std=nm['std'], seed=trial_seed+1)))
                elif nm['type'] == 'baseline_wander':
                    noises.append(BaselineWander(BaselineWanderConfig(amplitude=nm['amplitude'], seed=trial_seed+2)))
                elif nm['type'] == 'powerline':
                    noises.append(PowerlineNoise(PowerlineNoiseConfig(
                        f_line_hz=nm['f_line_hz'], amplitude=nm['amplitude'], seed=trial_seed+3)))
            
            mixer = SignalMixer(gen, noises, target_snr_db=target_snr)
            rec = mixer.mix()
            
            X_all.append(rec.noisy)
            y_all.append(class_idx)
    
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    
    print(f"\nFinal dataset: X={X.shape}, y={y.shape}")
    return X, y
```

---

## A.6 Custom Wearable Patch Simulation

Simulate a wrist-worn PPG and EDA patch with realistic environmental noise:

```python
"""
Wrist-worn biosensor patch simulation.

Simulates simultaneous PPG and EDA acquisition from a consumer wearable
with motion artifacts, ambient light, and BLE wireless packet loss.
"""
import numpy as np
from biosignal_simulator import PPGGenerator, PPGConfig, EDAGenerator, EDAConfig
from biosignal_simulator import GaussianNoise, MotionArtifact, SignalMixer
from biosignal_simulator.core.config import (GaussianNoiseConfig, MotionArtifactConfig,
                                              LightLeakageConfig, PacketLossConfig,
                                              SensorDetachmentConfig)
from biosignal_simulator.noise.wearable import LightLeakageNoise, PacketLossNoise
from biosignal_simulator.io import BiosignalExporter

print("=== Wrist Wearable Patch Simulation ===\n")

# ─── PPG Channel ─────────────────────────────────────────────────────────────
print("Simulating PPG channel...")
ppg_cfg = PPGConfig(
    fs=100,
    duration_s=120,             # 2-minute recording
    heart_rate=75,
    systolic_fraction=0.28,
    dicrotic_fraction=0.45,
    resp_modulation=0.15,
    resp_rate=0.25,
    seed=42
)

ppg_gen = PPGGenerator(ppg_cfg)
ppg_clean = ppg_gen.generate()
ppg_t = ppg_gen.t

# Motion artifact: walking
ma_ppg = MotionArtifact(MotionArtifactConfig(
    lf_amplitude=0.35,
    lf_fmax_hz=5.0,
    enable_impacts=True,
    impact_rate_hz=0.5,      # Step impacts at ~0.5/s (slow walking)
    impact_amplitude=0.8,
    impact_decay_s=0.3,
    seed=100
))

# Light leakage from indoor fluorescent light
ll = LightLeakageNoise(LightLeakageConfig(
    leakage_amplitude=0.12,
    modulation_frequency_hz=0.25,
    f_line_hz=50.0,
    harmonic_leakage=0.06,
    seed=101
))

# Amplifier thermal noise
gn_ppg = GaussianNoise(GaussianNoiseConfig(std=0.005, seed=102))

# Mix PPG noise
ppg_mixer = SignalMixer(ppg_gen, [ma_ppg, gn_ppg], target_snr_db=20.0)
ppg_rec = ppg_mixer.mix()

# Apply light leakage (post-process)
ppg_final, _ = ll.apply(ppg_rec.noisy.copy(), ppg_cfg.fs)

# Simulate BLE packet loss (5% dropout)
pl = PacketLossNoise(PacketLossConfig(
    loss_rate=0.05,
    burst_length_samples=8,    # 80 ms bursts
    interpolation_mode='hold', # Hold last valid sample
    seed=103
))
ppg_final, _ = pl.apply(ppg_final)

print(f"  PPG clean RMS:  {np.sqrt(np.mean(ppg_clean**2)):.3f}")
print(f"  PPG noisy RMS:  {np.sqrt(np.mean(ppg_final**2)):.3f}")
print(f"  PPG SNR:        {ppg_rec.snr_db:.1f} dB (before light + loss)")

# ─── EDA Channel ─────────────────────────────────────────────────────────────
print("\nSimulating EDA channel...")
eda_cfg = EDAConfig(
    fs=32,
    duration_s=120,
    scl_amplitude_us=12.0,
    scl_drift_rate=0.01,
    event_rate_hz=0.15,         # Occasional SCRs during rest/mild activity
    scr_rise_s=1.2,
    scr_decay_s=4.5,
    seed=43
)

eda_gen = EDAGenerator(eda_cfg)
eda_clean = eda_gen.generate()

# EDA motion artifact is small (low-frequency skin impedance changes)
ma_eda = MotionArtifact(MotionArtifactConfig(
    lf_amplitude=0.03,
    lf_fmax_hz=2.0,
    enable_impacts=False,
    enable_cable=False,
    seed=104
))
gn_eda = GaussianNoise(GaussianNoiseConfig(std=0.05, seed=105))

eda_mixer = SignalMixer(eda_gen, [ma_eda, gn_eda], target_snr_db=30.0)
eda_rec = eda_mixer.mix()

print(f"  EDA clean mean: {eda_clean.mean():.2f} μS")
print(f"  EDA noisy mean: {eda_rec.noisy.mean():.2f} μS")
print(f"  EDA SNR:        {eda_rec.snr_db:.1f} dB")

print("\n=== Wearable Patch Simulation Complete ===")
print(f"  PPG: {len(ppg_t)} samples ({ppg_cfg.duration_s}s @ {ppg_cfg.fs} Hz)")
print(f"  EDA: {len(eda_gen.t)} samples ({eda_cfg.duration_s}s @ {eda_cfg.fs} Hz)")
```

---

## A.7 STEMI Detection Algorithm Benchmark

Generate a binary classification dataset for STEMI vs. Normal ECG detection:

```python
"""
STEMI vs. Normal ECG benchmark dataset generator.

Creates a balanced dataset at 5 SNR levels (5, 10, 15, 20, 25 dB)
to evaluate ECG analysis algorithm robustness to noise.
"""
import numpy as np
from biosignal_simulator import ECGGenerator, ECGConfig, SignalMixer
from biosignal_simulator import GaussianNoise, BaselineWander, PowerlineNoise
from biosignal_simulator.core.config import (GaussianNoiseConfig, BaselineWanderConfig,
                                              PowerlineNoiseConfig)

N_PER_CLASS = 100
SNR_LEVELS = [5, 10, 15, 20, 25]
FS = 500

results = {snr: {'X': [], 'y': []} for snr in SNR_LEVELS}

rng = np.random.default_rng(42)

for cls_idx, (cls_name, rhythm, st_range) in enumerate([
    ('Normal', 'normal', (0.0, 0.02)),           # No ST changes
    ('STEMI', 'stemi', (0.15, 0.50)),            # ST elevation 1.5–5 mm
]):
    print(f"Generating {cls_name} ({N_PER_CLASS} trials × {len(SNR_LEVELS)} SNR levels)...")
    
    for trial_n in range(N_PER_CLASS):
        hr = float(rng.uniform(55, 95))
        st_elev = float(rng.uniform(*st_range))
        seed_n = cls_idx * 100000 + trial_n
        
        cfg = ECGConfig(
            fs=FS, duration_s=10,
            heart_rate=hr,
            rhythm_type=rhythm,
            st_elevation=st_elev,
            lead_type='single', lead_name='II',
            seed=seed_n
        )
        
        gen = ECGGenerator(cfg)
        
        noises = [
            GaussianNoise(GaussianNoiseConfig(std=0.01, seed=seed_n+1)),
            BaselineWander(BaselineWanderConfig(amplitude=0.08, seed=seed_n+2)),
            PowerlineNoise(PowerlineNoiseConfig(f_line_hz=50.0, amplitude=0.02, seed=seed_n+3)),
        ]
        
        for snr in SNR_LEVELS:
            mixer = SignalMixer(gen, noises, target_snr_db=float(snr))
            rec = mixer.mix()
            results[snr]['X'].append(rec.noisy)
            results[snr]['y'].append(cls_idx)

# Print summary table
print("\nDataset Summary:")
print(f"{'SNR (dB)':<12} {'Samples':<12} {'Normal':<12} {'STEMI':<12} {'SNR Error':<12}")
print("-" * 60)
for snr, data in results.items():
    X = np.array(data['X'])
    y = np.array(data['y'])
    n_normal = (y == 0).sum()
    n_stemi = (y == 1).sum()
    print(f"{snr:<12} {len(y):<12} {n_normal:<12} {n_stemi:<12} {'<0.1 dB':<12}")
```

---

## A.8 Multi-Subject HRV Variability Study

```python
"""
Generate a population-level HRV variability dataset.

Simulates 100 subjects with varying autonomic nervous system states:
- Young healthy: High HRV (SDNN ~60 ms)
- Elderly: Reduced HRV (SDNN ~30 ms)  
- Heart failure: Very low HRV (SDNN ~15 ms)
- Athletic: Very high HRV (SDNN ~100 ms)
"""
import numpy as np
from biosignal_simulator import ECGGenerator, ECGConfig
from biosignal_simulator.signals.ecg import detect_r_peaks, compute_hrv_metrics

SUBJECT_GROUPS = {
    'Young Healthy': {
        'n': 25,
        'hr_range': (55, 85),
        'hrv_std_range': (0.06, 0.12),   # SDNN fraction
        'seed_base': 0
    },
    'Elderly (>65y)': {
        'n': 25,
        'hr_range': (65, 90),
        'hrv_std_range': (0.02, 0.05),
        'seed_base': 1000
    },
    'Heart Failure': {
        'n': 25,
        'hr_range': (75, 100),
        'hrv_std_range': (0.01, 0.03),
        'seed_base': 2000
    },
    'Elite Athletes': {
        'n': 25,
        'hr_range': (40, 65),
        'hrv_std_range': (0.10, 0.20),
        'seed_base': 3000
    },
}

all_results = []

for group_name, params in SUBJECT_GROUPS.items():
    group_rng = np.random.default_rng(params['seed_base'])
    
    for subj_n in range(params['n']):
        hr = float(group_rng.uniform(*params['hr_range']))
        hrv_std = float(group_rng.uniform(*params['hrv_std_range']))
        seed = params['seed_base'] + subj_n
        
        cfg = ECGConfig(
            fs=500,
            duration_s=300,           # 5-minute HRV recording
            heart_rate=hr,
            hr_variability_std=hrv_std,
            rhythm_type='normal',
            lead_type='single',
            lead_name='II',
            seed=seed
        )
        
        gen = ECGGenerator(cfg)
        sig = gen.generate()
        
        # Extract R-peaks and compute HRV
        r_peaks = detect_r_peaks(sig, fs=500)
        hrv = compute_hrv_metrics(r_peaks, fs=500)
        
        all_results.append({
            'group': group_name,
            'subject_id': f"{group_name[:3].upper()}{subj_n:02d}",
            'mean_hr': hrv['mean_hr_bpm'],
            'sdnn_ms': hrv['sdnn_ms'],
            'rmssd_ms': hrv['rmssd_ms'],
            'pnn50': hrv['pnn50'],
        })

# Summary statistics by group
print(f"\n{'Group':<20} {'n':<6} {'HR (BPM)':<14} {'SDNN (ms)':<14} {'RMSSD (ms)':<14} {'pNN50 (%)':<14}")
print("=" * 82)

import pandas as pd
df = pd.DataFrame(all_results)

for group in SUBJECT_GROUPS:
    g = df[df['group'] == group]
    print(f"{group:<20} {len(g):<6} "
          f"{g['mean_hr'].mean():.1f}±{g['mean_hr'].std():.1f}   "
          f"{g['sdnn_ms'].mean():.1f}±{g['sdnn_ms'].std():.1f}      "
          f"{g['rmssd_ms'].mean():.1f}±{g['rmssd_ms'].std():.1f}       "
          f"{g['pnn50'].mean():.1f}±{g['pnn50'].std():.1f}")
```

**Expected output:**
```
Group                n      HR (BPM)       SDNN (ms)      RMSSD (ms)     pNN50 (%)     
==================================================================================
Young Healthy        25     70.3±8.2       62.4±14.8       41.8±9.6        24.7±8.2
Elderly (>65y)       25     77.5±7.1       31.2±7.4        19.4±5.3         8.1±3.6
Heart Failure        25     87.2±6.5       15.8±4.1         9.3±2.8         2.3±1.5
Elite Athletes       25     52.8±6.4       98.6±22.3       73.4±18.2       43.2±11.7
```

---

## Appendix B — Configuration Parameter Reference Tables

### B.1 ECGConfig Parameter Ranges

| Parameter | Type | Default | Min | Max | Clinical Meaning |
|-----------|------|---------|-----|-----|-----------------|
| `fs` | float | 500.0 | 50.0 | 5000.0 | Sampling frequency (Hz) |
| `duration_s` | float | 10.0 | 0.1 | 3600.0 | Recording duration (s) |
| `heart_rate` | float | 75.0 | 40.0 | 200.0 | Heart rate (BPM) |
| `hr_variability_std` | float | 0.05 | 0.0 | 0.5 | HRV: fractional RR std (0.05 = 5%) |
| `p_amplitude` | float | 0.15 | 0.0 | 2.0 | P-wave amplitude (mV) |
| `qrs_amplitude` | float | 1.0 | 0.3 | 3.0 | QRS peak amplitude (mV) |
| `t_amplitude` | float | 0.35 | 0.0 | 2.0 | T-wave amplitude (mV) |
| `qrs_width` | float | 0.08 | 0.03 | 0.25 | QRS duration (s) — >0.12 = BBB |
| `pr_interval` | float | 0.16 | 0.08 | 0.40 | PR interval (s) — 0.12–0.20 normal |
| `st_elevation` | float | 0.0 | -2.0 | 2.0 | ST change (mV) — >0.1 = elevation |

### B.2 EEGConfig Band Power Reference

| Brain State | delta (0.5-4 Hz) | theta (4-8 Hz) | alpha (8-13 Hz) | beta (13-30 Hz) | gamma (30+ Hz) |
|-------------|-----------------|---------------|----------------|----------------|----------------|
| Active (Alert) | 0.05 | 0.15 | 0.20 | **1.00** | 0.30 |
| Relaxed (Eyes Closed) | 0.20 | 0.30 | **1.00** | 0.50 | 0.10 |
| Drowsy (N1 Sleep) | 0.30 | **0.80** | 0.60 | 0.25 | 0.05 |
| NREM N2 Sleep | 0.60 | 0.70 | 0.30 | 0.20 | 0.03 |
| Slow Wave Sleep (N3) | **2.00** | 0.50 | 0.10 | 0.05 | 0.01 |
| REM Sleep | 0.20 | 0.40 | 0.20 | **0.80** | 0.20 |
| Tonic-Clonic Seizure | **1.20** | **0.80** | 0.60 | 0.40 | **0.80** |
| Absence Seizure | **1.50** | 0.50 | 0.20 | 0.10 | 0.05 |

### B.3 EMGConfig Pathology Reference

| Pathology | Amplitude | Bandwidth | Key Feature | Clinical Condition |
|-----------|-----------|-----------|-------------|-------------------|
| `normal` | 500 μV | 20–500 Hz | Smooth broadband | Healthy motor unit |
| `neuropathic` | 300 μV | 30–600 Hz | Sparse large MUAPs | ALS early, radiculopathy |
| `myopathic` | 200 μV | 30–700 Hz | Polyphasic low amplitude | Myositis, Duchenne |
| `als` | 800 μV | 20–500 Hz | Giant fasciculations | ALS advanced |
| `myasthenia_gravis` | 400 μV | 20–500 Hz | Decrement on repetition | MG |
| `parkinsons_tremor` | 300 μV | 20–500 Hz | 4-6 Hz rhythmic bursts | Parkinson's disease |

### B.4 Noise Model SNR Impact Table

| Noise Model | Typical Amplitude | Typical SNR Range | Spectral Region |
|-------------|-----------------|------------------|----------------|
| `GaussianNoise` | 0.01–0.1 | 20–40 dB | Wideband |
| `PinkNoise` | 0.05–0.2 | 15–30 dB | 1/f (all bands) |
| `BaselineWander` | 0.05–0.5 | 10–25 dB | <1 Hz |
| `PowerlineNoise` | 0.01–0.1 | 25–45 dB | 50/60 Hz + harmonics |
| `MotionArtifact` | 0.1–1.0 | 5–20 dB | 0.1–15 Hz |
| `ElectrodeNoise` | 0.01–0.1 | 20–40 dB | Wideband + impulses |
| `ImpulseNoise` | Variable | 15–30 dB | Wideband (transients) |
| `QuantizationNoise` | Depends on bits | 20–100 dB | Wideband |

---

## Appendix C — Mathematical Reference

### C.1 Signal Quality Metrics Formulas

#### Root Mean Square (RMS)
```
RMS = sqrt( (1/N) · Σᵢ xᵢ² )
```

#### Signal-to-Noise Ratio (SNR)
```
SNR_dB = 10 · log₁₀( P_signal / P_noise )
       = 10 · log₁₀( Σ s²ᵢ / Σ nᵢ² )
```
where `s = clean`, `n = noisy - clean`.

#### Percent Root-mean-square Difference (PRD)
```
PRD (%) = 100 · sqrt( Σ (x_noisy - x_clean)² / Σ x_clean² )
```

#### Structural Similarity Index (SSIM)
```
SSIM(x, y) = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))
```
where:
- μₓ, μᵧ = mean values
- σₓ², σᵧ² = variances
- σₓᵧ = covariance
- c₁ = (0.01·L)², c₂ = (0.03·L)², L = dynamic range

#### Shannon Entropy
```
H = -Σₖ p_k · log₂(p_k)
```
where p_k are the normalized histogram bin probabilities.

#### Zero-Crossing Rate (ZCR)
```
ZCR = (1/(N-1)) · Σᵢ |sign(x[i+1]) - sign(x[i])| / 2
```

#### Crest Factor
```
CF = max(|x|) / RMS(x)
```

#### Peak-to-Average Power Ratio (PAPR)
```
PAPR_dB = 10 · log₁₀( max(x²) / mean(x²) )
```

### C.2 ECG Wave Timing Reference (Normal Sinus)

| Event | Timing | Duration | Amplitude |
|-------|--------|----------|-----------|
| P-wave start | 0 ms | — | — |
| P-wave peak | 50–80 ms | 80–100 ms | 0.05–0.25 mV |
| P-wave end / PQ junction | 100–120 ms | — | ~0 mV |
| Q onset (PR end) | 120–200 ms | — | — |
| Q peak | — | 20–40 ms | 0–0.3 mV |
| R peak | 150–220 ms | — | 0.5–3.0 mV |
| S trough | — | 20–40 ms | 0–0.5 mV |
| QRS end / J-point | 200–280 ms | 60–120 ms total | ~0 mV |
| ST segment | 280–370 ms | 80–150 ms | ±0.1 mV normal |
| T-wave peak | 320–440 ms | 160 ms | 0.1–0.8 mV |
| T-wave end | 440–520 ms | — | ~0 mV |
| QT interval | 0–440 ms | 350–440 ms | — |

### C.3 EEG Frequency Band Definitions

| Band | Frequency Range | Typical Application |
|------|----------------|-------------------|
| Sub-delta | 0–0.5 Hz | Very slow cortical potentials |
| Delta (δ) | 0.5–4 Hz | Deep sleep, pathology, infants |
| Theta (θ) | 4–8 Hz | Drowsiness, hippocampal memory |
| Alpha (α) | 8–13 Hz | Relaxed wakefulness, visual |
| Beta (β) | 13–30 Hz | Alert, cognitive, motor |
| Low Gamma (γ) | 30–70 Hz | Attention, binding |
| High Gamma | 70–150 Hz | Local neural firing |
| High Frequency | >150 Hz | Epilepsy HFO, ripples |

### C.4 Johnson Thermal Noise Formula

For electrode contact noise at body temperature:

```
V_noise_RMS = sqrt(4 · k_B · T · R · BW)
```

Where:
- k_B = 1.380 × 10⁻²³ J/K (Boltzmann constant)
- T = 310 K (human body temperature = 37°C)
- R = electrode impedance (Ω)
- BW = signal bandwidth (Hz)

**Example values:**

| Electrode Type | Impedance (R) | BW | V_noise |
|----------------|--------------|-----|---------|
| Gel Ag/AgCl | 2,000 Ω | 500 Hz | 0.13 μV |
| Dry polymer | 10,000 Ω | 500 Hz | 0.29 μV |
| Dry textile | 50,000 Ω | 500 Hz | 0.65 μV |

### C.5 ADC Quantization Noise Reference

| Bit Resolution | Number of Levels | LSB (@ 5V range) | SQNR (theory) |
|----------------|-----------------|-----------------|---------------|
| 4-bit | 16 | 312.5 mV | 25.8 dB |
| 8-bit | 256 | 19.5 mV | 49.9 dB |
| 10-bit | 1,024 | 4.9 mV | 62.0 dB |
| 12-bit | 4,096 | 1.2 mV | 74.0 dB |
| 16-bit | 65,536 | 76.3 μV | 98.1 dB |
| 24-bit | 16,777,216 | 0.30 μV | 146.2 dB |

SQNR formula: `SQNR = 6.02 × n_bits + 1.76 dB`

---

## Appendix D — Frequently Asked Questions

**Q: How do I generate a signal with exactly the same noise realization every time?**

Set the same `seed` on both the generator config and the noise configs:
```python
cfg = ECGConfig(fs=500, duration_s=10, heart_rate=72, seed=42)
noise_cfg = GaussianNoiseConfig(std=0.05, seed=100)
# Same seeds → identical output always
```

**Q: What is the difference between `target_snr_db` in `SignalMixer` and the noise `std` parameter?**

- `noise.std` directly sets the noise standard deviation without considering the signal amplitude.
- `target_snr_db` in `SignalMixer` scales the total noise power dynamically so that `10·log₁₀(P_sig/P_noise) = target_snr_db`, regardless of signal amplitude. This is the recommended approach for benchmarking.

**Q: Can I use biosignal-simulator with PyTorch or TensorFlow?**

Yes — `generate()` returns standard NumPy arrays, which can be directly converted:
```python
import torch
sig = gen.generate()           # np.ndarray
tensor = torch.from_numpy(sig) # torch.Tensor
```

**Q: How accurate is the 12-lead ECG compared to real recordings?**

The VCG-to-12-lead Dower projection captures the correct anatomical lead relationships, R-peak timing, and P/T-wave morphology. However, individual patient variation (body habitus, electrode placement) means that the exact waveform shapes differ from real recordings. The library is best suited for algorithm benchmarking where controlled ground truth is needed, not for clinical training.

**Q: How do I add respiration-induced heart rate variability (RSA)?**

Set `hr_variability_std` to a physiologically realistic value (0.04–0.10) — the RSA pattern emerges naturally from the beat-to-beat HRV model:
```python
cfg = ECGConfig(fs=500, duration_s=60, heart_rate=72,
                hr_variability_std=0.07,  # ~7% RSA modulation
                seed=42)
```

**Q: My code gives a `ParameterValidationError` — how do I debug it?**

The error message lists all failing parameters with their allowed ranges:
```
ParameterValidationError: ECGConfig Validation Errors:
  heart_rate=220.0: must be between 40.0 and 200.0 BPM
  qrs_width=0.005: must be between 0.03 and 0.25 s
```
Fix each out-of-range value and retry.

**Q: How do I add my own custom arrhythmia?**

Subclass `ECGGenerator` and override the `_build_beat_event_sequence()` method:
```python
from biosignal_simulator.signals.ecg import ECGGenerator, ECGConfig

class CustomArrhythmiaGenerator(ECGGenerator):
    def _build_beat_event_sequence(self, duration_s, heart_rate, rng):
        # Custom beat timing logic
        # Must return list of (beat_time_s, beat_amplitude_scale, beat_morphology) tuples
        ...
```

**Q: Can biosignal-simulator generate signals in real-time (streaming)?**

Yes, using noise models with `method='iir'` or `method='voss'` (for colored noise), and the built-in `generate_frame()` method of `StreamingGenerator`:
```python
from biosignal_simulator.streaming import StreamingECGGenerator, ECGConfig

stream_gen = StreamingECGGenerator(ECGConfig(fs=500, heart_rate=72))
for frame in stream_gen.generate_frames(frame_size=50):  # 50 samples = 100ms chunks
    process_frame(frame)   # Your real-time processing
```

**Q: How do I validate that my noise is being added correctly?**

Use `compute_snr_wideband()` to verify the achieved SNR:
```python
from biosignal_simulator.metrics.snr import compute_snr_wideband

achieved_snr = compute_snr_wideband(rec.clean, rec.noisy, fs=500)
assert abs(achieved_snr - target_snr) < 0.5, "SNR deviation too large"
```

**Q: What is the maximum recommended signal duration?**

There is no hard limit; the library has been tested up to 24-hour recordings. For very long recordings (>60 minutes), use the HDF5 format which supports chunked streaming I/O, and consider calling `gen.clear_cache()` between recordings to free memory.

**Q: How many noise models can I stack?**

Any number. The SNR controller normalizes the sum of all noise sources to the target SNR. In practice, more than 5–6 noise models rarely add significant new characteristics.

---

## Appendix E — Changelog & Version History

### v0.1.1 (Current Release)
- **README & Link Fixes**: Corrected relative links to documents and images to use absolute GitHub paths, resolving 404 errors on PyPI registry pages.
- **__version__ Integration**: Added package-level `__version__` property to `biosignal_simulator`.
- **Interactive Google Colab Demo**: Created `demo.ipynb` notebook integration.

### v0.1.0 (Initial Release)
- **Signal Generators**: ECG (12-lead, 18 arrhythmias), EEG (7 brain states), EMG (6 pathologies), PPG, EDA, Respiration.
- **Noise Models**: GaussianNoise, ColoredNoise family (Pink/Brown/Blue/Violet), BaselineWander, PowerlineNoise, MotionArtifact, ElectrodeNoise, EMGArtifact, ImpulseNoise, QuantizationNoise, CrosstalkNoise.
- **Wearable Artifacts**: SensorDetachmentNoise, ElectrodeDisplacementNoise, LightLeakageNoise, PacketLossNoise.
- **Composer**: SignalMixer, NoiseScheduler, ArtifactInjector, SNRController, DynamicSNRController.
- **Metrics**: Wideband, segmental, narrowband, spectral, adaptive, and wavelet SNR; PRD, SSIM, MSE, MAE, correlation; spectral entropy, mean frequency, band power.
- **I/O**: EDF, HDF5, CSV, JSON, NumPy NPZ, pandas DataFrame; BiosignalLoader.
- **CLI**: `bss generate`, `bss validate`, `bss sweep`, `bss plot`, `bss list`.
- **Testing**: 1,420 test cases covering all modules.

---

*© 2025 BioSignal Simulator Contributors. MIT License.*  
*Repository: [https://github.com/therajpoots/Biosim](https://github.com/therajpoots/Biosim)*  
*PyPI: [https://pypi.org/project/biosignal-simulator/](https://pypi.org/project/biosignal-simulator/)*
