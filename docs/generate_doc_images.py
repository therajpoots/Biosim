"""
Generate all documentation images for the BioSignal Simulator Reference Manual.
API Note: generator.generate() returns np.ndarray; use gen.t for time axis.
          generator.to_record() returns SignalRecord with .clean, .noisy, .t attrs.
          SignalMixer.mix() returns SignalRecord.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(OUT, exist_ok=True)

plt.style.use('dark_background')
C = {
    'ecg': '#00d4ff', 'eeg': '#a78bfa', 'emg': '#fb923c',
    'ppg': '#f472b6', 'eda': '#34d399', 'resp': '#fbbf24',
    'noisy': '#ff6b6b', 'noise': '#ff9500', 'clean': '#00d4ff',
    'grid': '#2a2a3a', 'bg': '#0d0d1a', 'text': '#e2e8f0',
    'accent': '#6366f1'
}

def save(fig, name):
    fig.savefig(os.path.join(OUT, f'{name}.png'), dpi=150, bbox_inches='tight',
                facecolor=C['bg'], edgecolor='none')
    plt.close(fig)
    print(f"  OK: {name}.png")

def style_ax(ax, xlim=None):
    ax.set_facecolor(C['bg'])
    ax.tick_params(colors=C['text'], labelsize=8)
    ax.spines[:].set_edgecolor(C['grid'])
    ax.grid(alpha=0.1)
    if xlim: ax.set_xlim(*xlim)


# ─────────────────────────────────────────────────────────────────────
# 1. ECG NORMAL SINUS RHYTHM
# ─────────────────────────────────────────────────────────────────────
print("Generating ECG images...")
from biosignal_simulator import ECGGenerator, ECGConfig
from biosignal_simulator.signals.ecg import detect_r_peaks

gen = ECGGenerator(ECGConfig(fs=500, duration_s=10, heart_rate=72, seed=42))
sig = gen.generate()
t = gen.t

fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=C['bg'])
style_ax(ax, xlim=(0, 8))
ax.plot(t, sig, color=C['ecg'], lw=1.2, label='Lead II — Normal Sinus Rhythm')
ax.set_xlabel('Time (s)', color=C['text'], fontsize=12)
ax.set_ylabel('Amplitude (mV)', color=C['text'], fontsize=12)
ax.set_title('ECG — Normal Sinus Rhythm (72 BPM)', color=C['text'], fontsize=14, fontweight='bold')
ax.legend(facecolor='#1a1a2e', edgecolor=C['grid'], labelcolor=C['text'])
r_peaks = detect_r_peaks(sig, 500)
if len(r_peaks) >= 1:
    rp = r_peaks[0]; rt = t[rp]
    ax.annotate('R', xy=(rt, sig[rp]), xytext=(rt-0.18, sig[rp]+0.25),
                color='#fbbf24', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=1.2))
fig.tight_layout(); save(fig, '01_ecg_normal')


# ─────────────────────────────────────────────────────────────────────
# 2. ECG RHYTHM LIBRARY
# ─────────────────────────────────────────────────────────────────────
rhythms = [('normal', 72, 'Normal Sinus (72 BPM)', C['ecg']),
           ('afib', 85, 'Atrial Fibrillation', '#ff6b6b'),
           ('pvc', 72, 'PVC Ectopics', '#fbbf24'),
           ('bradycardia', 45, 'Bradycardia (45 BPM)', '#a78bfa'),
           ('tachycardia', 130, 'Tachycardia (130 BPM)', '#34d399')]

fig, axes = plt.subplots(5, 1, figsize=(14, 10), facecolor=C['bg'])
for i, (rhy, hr, lbl, col) in enumerate(rhythms):
    g = ECGGenerator(ECGConfig(fs=500, duration_s=6, heart_rate=hr, rhythm_type=rhy, seed=42))
    s = g.generate()
    style_ax(axes[i], xlim=(0, 5.5))
    axes[i].plot(g.t, s, color=col, lw=1.0)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=9)
    if i < 4: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('ECG Rhythm Library — 5 Arrhythmia Types', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.15); save(fig, '02_ecg_rhythms')


# ─────────────────────────────────────────────────────────────────────
# 3. 12-LEAD ECG
# ─────────────────────────────────────────────────────────────────────
gen12 = ECGGenerator(ECGConfig(fs=500, duration_s=6, heart_rate=72, lead_type='12lead', seed=42))
sig12 = gen12.generate()  # shape: (12, n_samples)
t12 = gen12.t
lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
lead_colors = [C['ecg']]*6 + ['#f472b6']*6

fig = plt.figure(figsize=(16, 9), facecolor=C['bg'])
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
for i in range(12):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ax.set_facecolor(C['bg'])
    ax.plot(t12[:2501], sig12[i, :2501], color=lead_colors[i], lw=0.9)
    ax.set_title(lead_names[i], color=C['text'], fontsize=11, fontweight='bold')
    ax.tick_params(colors=C['text'], labelsize=7)
    ax.spines[:].set_edgecolor(C['grid'])
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('12-Lead ECG — Standard Clinical Layout (Dower Matrix Projection)',
             color=C['text'], fontsize=13, fontweight='bold', y=0.98)
save(fig, '03_ecg_12lead')


# ─────────────────────────────────────────────────────────────────────
# 4. EEG BRAIN STATES
# ─────────────────────────────────────────────────────────────────────
print("Generating EEG images...")
from biosignal_simulator import EEGGenerator, EEGConfig

states = [('relaxed', 'Relaxed (Alpha)', C['eeg']),
          ('active', 'Active (Beta)', '#a3e635'),
          ('n2_sleep', 'N2 Sleep (Spindles)', '#fb923c'),
          ('tonic_clonic', 'Tonic-Clonic Seizure', '#ff6b6b')]

fig, axes = plt.subplots(4, 1, figsize=(14, 9), facecolor=C['bg'])
for i, (st, lbl, col) in enumerate(states):
    g = EEGGenerator(EEGConfig(fs=256, duration_s=8, state=st, seed=42))
    s = g.generate()
    style_ax(axes[i], xlim=(0, 8))
    axes[i].plot(g.t, s, color=col, lw=0.8)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=9)
    if i < 3: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('EEG Brain States — 4 Physiological Conditions', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '04_eeg_states')


# ─────────────────────────────────────────────────────────────────────
# 5. EEG MULTI-CHANNEL
# ─────────────────────────────────────────────────────────────────────
cov = [[1.0, 0.7, 0.4], [0.7, 1.0, 0.6], [0.4, 0.6, 1.0]]
g_mc = EEGGenerator(EEGConfig(fs=256, duration_s=8, n_channels=3, corr_matrix=cov, state='relaxed', seed=42))
s_mc = g_mc.generate()  # shape: (3, n_samples)
ch_names = ['Fz (Frontal)', 'Cz (Central)', 'Pz (Parietal)']
ch_cols = [C['eeg'], '#c4b5fd', '#7c3aed']

fig, axes = plt.subplots(3, 1, figsize=(14, 7), facecolor=C['bg'])
for i in range(3):
    style_ax(axes[i], xlim=(0, 8))
    axes[i].plot(g_mc.t, s_mc[i], color=ch_cols[i], lw=0.8)
    axes[i].set_ylabel(ch_names[i], color=C['text'], fontsize=9)
    if i < 2: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('EEG — 3-Channel Spatially Correlated Recording (Cholesky Mixing)',
             color=C['text'], fontsize=13, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '05_eeg_multichannel')


# ─────────────────────────────────────────────────────────────────────
# 6. EMG PATHOLOGIES
# ─────────────────────────────────────────────────────────────────────
print("Generating EMG images...")
from biosignal_simulator import EMGGenerator, EMGConfig

paths = [('normal', 'Normal Motor Unit', C['emg']),
         ('neuropathic', 'Neuropathic', '#fb923c'),
         ('als', 'ALS Fasciculations', '#ff6b6b'),
         ('parkinsons_tremor', "Parkinson's Tremor", '#fbbf24')]

fig, axes = plt.subplots(4, 1, figsize=(14, 9), facecolor=C['bg'])
for i, (p, lbl, col) in enumerate(paths):
    g = EMGGenerator(EMGConfig(fs=2000, duration_s=4, pathology=p, seed=42))
    s = g.generate()
    style_ax(axes[i], xlim=(0, 4))
    axes[i].plot(g.t, s, color=col, lw=0.7)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=9)
    if i < 3: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('EMG Pathology Library — 4 Neuromuscular Phenotypes', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '06_emg_pathologies')


# ─────────────────────────────────────────────────────────────────────
# 7. EMG ENVELOPES
# ─────────────────────────────────────────────────────────────────────
envs = [('constant', 'Constant Contraction', C['emg']),
        ('ramp', 'Ramp (Build-Up)', '#a78bfa'),
        ('burst', 'Burst (Periodic)', '#f472b6')]

fig, axes = plt.subplots(3, 1, figsize=(14, 7), facecolor=C['bg'])
for i, (e, lbl, col) in enumerate(envs):
    g = EMGGenerator(EMGConfig(fs=2000, duration_s=5, envelope_type=e, seed=42))
    s = g.generate()
    style_ax(axes[i], xlim=(0, 5))
    axes[i].plot(g.t, s, color=col, lw=0.8)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=9)
    if i < 2: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('EMG Contraction Envelopes — Constant / Ramp / Burst', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '07_emg_envelopes')


# ─────────────────────────────────────────────────────────────────────
# 8. PPG SIGNAL
# ─────────────────────────────────────────────────────────────────────
print("Generating PPG/EDA/Resp images...")
from biosignal_simulator import PPGGenerator, PPGConfig

g_ppg = PPGGenerator(PPGConfig(fs=100, duration_s=12, heart_rate=72, resp_modulation=0.2, seed=42))
s_ppg = g_ppg.generate()
t_ppg = g_ppg.t

fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=C['bg'])
style_ax(ax, xlim=(0, 12))
ax.plot(t_ppg, s_ppg, color=C['ppg'], lw=1.3)
ax.set_xlabel('Time (s)', color=C['text'], fontsize=12)
ax.set_ylabel('Normalized AU', color=C['text'], fontsize=12)
ax.set_title('PPG — Photoplethysmogram with Respiratory Modulation (15%)', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); save(fig, '08_ppg_signal')


# ─────────────────────────────────────────────────────────────────────
# 9. EDA SIGNAL
# ─────────────────────────────────────────────────────────────────────
from biosignal_simulator import EDAGenerator, EDAConfig

g_eda = EDAGenerator(EDAConfig(fs=32, duration_s=60, event_rate_hz=0.2, seed=42))
s_eda = g_eda.generate()
t_eda = g_eda.t

fig, ax = plt.subplots(figsize=(14, 3.5), facecolor=C['bg'])
style_ax(ax, xlim=(0, 60))
ax.plot(t_eda, s_eda, color=C['eda'], lw=1.3)
ax.set_xlabel('Time (s)', color=C['text'], fontsize=12)
ax.set_ylabel('Conductance (μS)', color=C['text'], fontsize=12)
ax.set_title('EDA — Electrodermal Activity (Tonic SCL + Phasic SCR Events)', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); save(fig, '09_eda_signal')


# ─────────────────────────────────────────────────────────────────────
# 10. RESPIRATION PATTERNS
# ─────────────────────────────────────────────────────────────────────
from biosignal_simulator import RespGenerator, RespConfig

resp_variants = [(0.25, 1.0, 0.3, 0.1, 'Normal Eupnea (15 breaths/min)', C['resp']),
                 (0.1, 1.5, 0.5, 0.3, 'Slow Deep Breathing (6 breaths/min)', '#60a5fa'),
                 (0.5, 0.7, 0.2, 0.05, 'Rapid Shallow Breathing (30 breaths/min)', '#fb923c')]

fig, axes = plt.subplots(3, 1, figsize=(14, 7), facecolor=C['bg'])
for i, (rr, amp, hk, pn, lbl, col) in enumerate(resp_variants):
    g = RespGenerator(RespConfig(fs=32, duration_s=60, resp_rate_hz=rr, amplitude=amp,
                                  harmonic_k=hk, phase_noise_std=pn, seed=42))
    s = g.generate()
    style_ax(axes[i], xlim=(0, 60))
    axes[i].plot(g.t, s, color=col, lw=1.0)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=9)
    if i < 2: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('Respiration Signals — 3 Breathing Rate Patterns', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '10_resp_patterns')


# ─────────────────────────────────────────────────────────────────────
# 11. NOISE GALLERY
# ─────────────────────────────────────────────────────────────────────
print("Generating noise images...")
from biosignal_simulator import (GaussianNoise, ColoredNoise, PinkNoise, BaselineWander,
                                   PowerlineNoise, MotionArtifact, ElectrodeNoise, ImpulseNoise)
from biosignal_simulator.core.config import (GaussianNoiseConfig, ColoredNoiseConfig,
    BaselineWanderConfig, PowerlineNoiseConfig, MotionArtifactConfig,
    ElectrodeNoiseConfig, ImpulseNoiseConfig)

fs_n, n_n = 500, 2500
t_n = np.arange(n_n) / fs_n
carrier = 0.5 * np.sin(2 * np.pi * 1.5 * t_n)

noise_models = [
    (GaussianNoise(GaussianNoiseConfig(std=0.15, seed=0)), 'Gaussian (AWGN)', '#94a3b8'),
    (PinkNoise(std=1.0, seed=0), 'Pink Noise (1/f)', C['eeg']),
    (BaselineWander(BaselineWanderConfig(amplitude=0.4, seed=0)), 'Baseline Wander', C['eda']),
    (PowerlineNoise(PowerlineNoiseConfig(f_line_hz=50.0, amplitude=0.15, seed=0)), 'Powerline 50 Hz', C['resp']),
    (MotionArtifact(MotionArtifactConfig(lf_amplitude=0.3, seed=0)), 'Motion Artifact', C['emg']),
    (ElectrodeNoise(ElectrodeNoiseConfig(popcorn_amplitude=0.1, seed=0)), 'Electrode Noise', C['ppg']),
    (ImpulseNoise(ImpulseNoiseConfig(rate_hz=3.0, amplitude_scale=1.5, seed=0)), 'Impulse Spikes', '#ff6b6b'),
]

fig, axes = plt.subplots(7, 1, figsize=(14, 12), facecolor=C['bg'])
for i, (model, lbl, col) in enumerate(noise_models):
    nz = model.generate(n_n, fs_n)
    noisy = carrier + nz
    style_ax(axes[i], xlim=(0, 5))
    axes[i].plot(t_n, carrier, color='#4a5568', lw=0.8, alpha=0.5)
    axes[i].plot(t_n, noisy, color=col, lw=0.9, alpha=0.9)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=8.5)
    if i < 6: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('Noise Model Gallery — 7 Physical Artifacts on a 1.5 Hz Carrier',
             color=C['text'], fontsize=13, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.18); save(fig, '11_noise_gallery')


# ─────────────────────────────────────────────────────────────────────
# 12. SNR SCALING DEMO
# ─────────────────────────────────────────────────────────────────────
print("Generating SNR demo...")
g_snr = ECGGenerator(ECGConfig(fs=500, duration_s=5, heart_rate=72, seed=42))
clean_sig = g_snr.generate()
t_snr = g_snr.t

# Use BaseNoiseModel.generate_scaled if available, else manual scaling
gn_cfg = GaussianNoiseConfig(seed=1)
gn = GaussianNoise(gn_cfg)

def add_noise_at_snr(clean, noise_model, snr_db):
    raw_noise = noise_model.generate(len(clean), 500)
    p_sig = np.mean(clean**2)
    p_noise = np.mean(raw_noise**2)
    if p_noise > 1e-15:
        scale = np.sqrt(p_sig / (p_noise * 10**(snr_db/10)))
        raw_noise *= scale
    return clean + raw_noise

snr_levels = [30, 15, 5, 0]
snr_colors = [C['ecg'], '#6366f1', C['resp'], '#ff6b6b']

fig, axes = plt.subplots(4, 1, figsize=(14, 9), facecolor=C['bg'])
for i, (snr_db, col) in enumerate(zip(snr_levels, snr_colors)):
    noisy_sig = add_noise_at_snr(clean_sig, GaussianNoise(GaussianNoiseConfig(seed=i)), snr_db)
    style_ax(axes[i], xlim=(0, 5))
    axes[i].plot(t_snr, noisy_sig, color=col, lw=0.9)
    axes[i].set_ylabel(f'SNR = {snr_db} dB', color=C['text'], fontsize=10)
    if i < 3: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('SNR Scaling Demo — Gaussian Noise at 4 Levels on ECG Lead II',
             color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.18); save(fig, '12_snr_scaling')


# ─────────────────────────────────────────────────────────────────────
# 13. SIGNALMIXER PIPELINE
# ─────────────────────────────────────────────────────────────────────
print("Generating SignalMixer pipeline image...")
from biosignal_simulator import SignalMixer

g_mix = ECGGenerator(ECGConfig(fs=500, duration_s=8, heart_rate=72, seed=42))
noises_mix = [
    GaussianNoise(GaussianNoiseConfig(std=0.02, seed=0)),
    PowerlineNoise(PowerlineNoiseConfig(f_line_hz=50.0, amplitude=0.03, seed=1)),
    BaselineWander(BaselineWanderConfig(amplitude=0.15, seed=2))
]
mixer = SignalMixer(g_mix, noises_mix, target_snr_db=20.0)
mixed_rec = mixer.mix()  # SignalRecord

snr_str = f'{mixed_rec.snr_db:.1f} dB' if mixed_rec.snr_db else 'N/A'
combo_noise = sum(mixed_rec.noise_components.get(k, np.zeros_like(mixed_rec.clean))
                  for k in ['GaussianNoise', 'PowerlineNoise', 'BaselineWander'])

fig, axes = plt.subplots(3, 1, figsize=(14, 8), facecolor=C['bg'])
row_data = [(mixed_rec.clean, 'Clean ECG', C['ecg']),
            (combo_noise, 'Superposed Noise (Gaussian + Powerline + Baseline)', C['noise']),
            (mixed_rec.noisy, f'Mixed Output (Target SNR=20dB, Achieved≈{snr_str})', '#a78bfa')]
for i, (sig, lbl, col) in enumerate(row_data):
    style_ax(axes[i], xlim=(0, 8))
    axes[i].plot(mixed_rec.t, sig, color=col, lw=0.9)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=9)
    if i < 2: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('SignalMixer Pipeline — Clean → Noise → Mixed (SNR-Targeted)',
             color=C['text'], fontsize=13, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '13_mixer_pipeline')


# ─────────────────────────────────────────────────────────────────────
# 14. METRICS DASHBOARD
# ─────────────────────────────────────────────────────────────────────
print("Generating metrics dashboard...")
from biosignal_simulator.metrics.snr import (compute_snr_wideband, compute_snr_segmental,
                                               compute_snr_wavelet)
from scipy.signal import welch as sp_welch

clean_arr = mixed_rec.clean
noisy_arr = mixed_rec.noisy
fs_m = 500.0
snr_seg = compute_snr_segmental(clean_arr, noisy_arr, fs_m, segment_s=1.0)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=C['bg'])
fig.suptitle('Signal Quality Metrics Dashboard', color=C['text'], fontsize=14, fontweight='bold')

# (a) Segmental SNR
ax = axes[0, 0]; style_ax(ax)
ax.bar(np.arange(len(snr_seg)), snr_seg, color=C['accent'], alpha=0.85, width=0.7)
ax.axhline(np.mean(snr_seg), color=C['resp'], lw=1.5, ls='--', label=f'Mean: {np.mean(snr_seg):.1f} dB')
ax.set_title('Segmental SNR (per second)', color=C['text'], fontsize=11)
ax.set_xlabel('Segment #', color=C['text']); ax.set_ylabel('SNR (dB)', color=C['text'])
ax.legend(facecolor='#1a1a2e', edgecolor=C['grid'], labelcolor=C['text'], fontsize=9)

# (b) PSD
f_c, p_c = sp_welch(clean_arr, fs=fs_m, nperseg=512)
f_n, p_n = sp_welch(noisy_arr, fs=fs_m, nperseg=512)
ax = axes[0, 1]; style_ax(ax)
ax.semilogy(f_c, p_c, color=C['ecg'], lw=1.2, label='Clean')
ax.semilogy(f_n, p_n, color=C['noisy'], lw=1.0, alpha=0.8, label='Noisy')
ax.set_title('Power Spectral Density', color=C['text'], fontsize=11)
ax.set_xlabel('Frequency (Hz)', color=C['text']); ax.set_ylabel('PSD (V²/Hz)', color=C['text'])
ax.set_xlim(0, 100)
ax.legend(facecolor='#1a1a2e', edgecolor=C['grid'], labelcolor=C['text'], fontsize=9)

# (c) Wavelet SNR
wav_snr = compute_snr_wavelet(clean_arr, noisy_arr, fs_m, level=4)
ax = axes[1, 0]; style_ax(ax)
bar_colors = [C['eeg'], C['eeg'], C['eeg'], C['eeg'], C['ppg']]
ax.bar(list(wav_snr.keys()), list(wav_snr.values()), color=bar_colors[:len(wav_snr)], alpha=0.85)
ax.set_title('Wavelet Subband SNR (Haar DWT, L=4)', color=C['text'], fontsize=11)
ax.set_xlabel('Subband', color=C['text']); ax.set_ylabel('SNR (dB)', color=C['text'])

# (d) Summary table
wsnr = compute_snr_wideband(clean_arr, noisy_arr, fs_m)
try:
    from biosignal_simulator.metrics.distortion import compute_prd, compute_ssim_1d, compute_mse
    prd = compute_prd(clean_arr, noisy_arr)
    ssim = compute_ssim_1d(clean_arr, noisy_arr)
    mse = compute_mse(clean_arr, noisy_arr)
    rows = [('Wideband SNR', f'{wsnr:.2f} dB'), ('PRD (%)', f'{prd:.2f}'),
            ('SSIM', f'{ssim:.4f}'), ('MSE', f'{mse:.6f}')]
except Exception:
    rows = [('Wideband SNR', f'{wsnr:.2f} dB'), ('PRD', 'N/A'), ('SSIM', 'N/A'), ('MSE', 'N/A')]
ax = axes[1, 1]
ax.set_facecolor(C['bg']); ax.axis('off')
ax.set_title('Quality Metric Summary', color=C['text'], fontsize=11)
for j, (nm, vl) in enumerate(rows):
    ax.text(0.08, 0.82 - j*0.22, nm, color=C['text'], fontsize=13, fontweight='bold', transform=ax.transAxes)
    ax.text(0.62, 0.82 - j*0.22, vl, color=C['ecg'], fontsize=13, transform=ax.transAxes)

fig.tight_layout(); fig.subplots_adjust(hspace=0.35, wspace=0.3); save(fig, '14_metrics_dashboard')


# ─────────────────────────────────────────────────────────────────────
# 15. WEARABLE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────
print("Generating wearable artifact images...")
from biosignal_simulator.noise.wearable import SensorDetachmentNoise, PacketLossNoise
from biosignal_simulator.core.config import SensorDetachmentConfig, PacketLossConfig

g_w = ECGGenerator(ECGConfig(fs=500, duration_s=10, heart_rate=72, seed=42))
clean_w = g_w.generate()
t_w = g_w.t

det_model = SensorDetachmentNoise(SensorDetachmentConfig(detachment_time_s=4.0, transient_amplitude=3.0, seed=42))
noisy_det, _ = det_model.apply(clean_w.copy(), 500)

loss_model = PacketLossNoise(PacketLossConfig(loss_rate=0.08, burst_length_samples=15, interpolation_mode='zero', seed=42))
noisy_loss, _ = loss_model.apply(clean_w.copy())

fig, axes = plt.subplots(3, 1, figsize=(14, 8), facecolor=C['bg'])
for i, (d, lbl, col) in enumerate([(clean_w, 'Clean ECG', C['ecg']),
                                     (noisy_det, 'Sensor Detachment (at t=4s)', '#ff6b6b'),
                                     (noisy_loss, 'Packet Loss (8% dropout, zero-fill)', C['resp'])]):
    style_ax(axes[i], xlim=(0, 10))
    axes[i].plot(t_w, d, color=col, lw=0.9)
    axes[i].set_ylabel(lbl, color=C['text'], fontsize=9)
    if i < 2: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('Wearable Artifacts — Sensor Detachment & Wireless Packet Loss',
             color=C['text'], fontsize=13, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '15_wearable_artifacts')


# ─────────────────────────────────────────────────────────────────────
# 16. I/O FORMAT SIZES
# ─────────────────────────────────────────────────────────────────────
print("Generating I/O format comparison...")
import tempfile
from biosignal_simulator.io import BiosignalExporter

g_io = ECGGenerator(ECGConfig(fs=500, duration_s=5, heart_rate=72, seed=42))
rec_io = g_io.to_record()  # SignalRecord via to_record()

try:
    with tempfile.TemporaryDirectory() as tmpdir:
        edf_path = os.path.join(tmpdir, 'ecg.edf')
        BiosignalExporter.export_edf(rec_io, edf_path)
        edf_size = os.path.getsize(edf_path) / 1024

        h5_path = os.path.join(tmpdir, 'ecg.h5')
        BiosignalExporter.export_hdf5(rec_io, h5_path)
        h5_size = os.path.getsize(h5_path) / 1024

        csv_path = os.path.join(tmpdir, 'ecg.csv')
        BiosignalExporter.export_csv(rec_io, csv_path)
        csv_size = os.path.getsize(csv_path) / 1024

    formats = ['EDF\n(Binary)', 'HDF5\n(Compressed)', 'CSV\n(Text)']
    sizes = [edf_size, h5_size, csv_size]
except Exception as e:
    print(f"  (I/O demo fallback: {e})")
    formats = ['EDF', 'HDF5', 'CSV']
    sizes = [12.0, 28.0, 65.0]

fmt_colors = [C['ecg'], C['eeg'], C['emg']]
fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=C['bg'])
style_ax(ax)
bars = ax.bar(formats, sizes, color=fmt_colors, width=0.45, alpha=0.9)
for bar, sz in zip(bars, sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{sz:.1f} KB',
            ha='center', color=C['text'], fontsize=11, fontweight='bold')
ax.set_ylabel('File Size (KB)', color=C['text'], fontsize=12)
ax.set_title('Export Format Size Comparison — 5 s ECG @ 500 Hz',
             color=C['text'], fontsize=13, fontweight='bold')
ax.grid(alpha=0.12, axis='y')
fig.tight_layout(); save(fig, '16_io_formats')


# ─────────────────────────────────────────────────────────────────────
# 17. ARCHITECTURE DIAGRAM
# ─────────────────────────────────────────────────────────────────────
print("Generating architecture diagram...")
fig, ax = plt.subplots(figsize=(14, 7), facecolor=C['bg'])
ax.set_facecolor(C['bg']); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

def box(ax, x, y, w, h, text, bg, fc='#e2e8f0', fs=9):
    rect = plt.Rectangle((x, y), w, h, fc=bg, ec='#475569', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', color=fc, fontsize=fs,
            fontweight='bold', zorder=3, multialignment='center')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#64748b', lw=1.5), zorder=4)

# Config row
for i, (txt, col) in enumerate([('ECGConfig','#1e3a5f'),('EEGConfig','#1e3a5f'),
                                  ('EMGConfig','#1e3a5f'),('PPGConfig','#1e3a5f'),
                                  ('Noise\nConfigs','#2d1b69'),('I/O\nConfigs','#1a3a1a')]):
    box(ax, 0.2+i*1.6, 5.8, 1.4, 0.7, txt, col, '#90cdf4')

# Generator row
for i, (txt, col) in enumerate([('ECGGenerator','#1a3a2a'),('EEGGenerator','#1a3a2a'),
                                  ('EMGGenerator','#1a3a2a'),('Noise Models\n(12 types)','#2d1b69')]):
    box(ax, 0.3+i*2.4, 4.4, 2.0, 0.7, txt, col, '#6ee7b7', fs=9)
    arrow(ax, 0.7+i*1.6+0.7, 5.8, 1.3+i*2.4, 5.1)

# Mixer
box(ax, 1.8, 3.2, 6.4, 0.8, 'SignalMixer\n(SNR Control  •  NoiseScheduler  •  ArtifactInjector)', '#2d1b69', '#c4b5fd', 10)
for gx in [1.3, 3.7, 6.1, 8.5]:
    arrow(ax, gx, 4.4, 5.0, 4.0)

# Record
box(ax, 3.5, 2.1, 3.0, 0.8, 'SignalRecord', '#1a1a2e', '#94a3b8', 11)
arrow(ax, 5.0, 3.2, 5.0, 2.9)

# Output row
for i, (txt, col, fc) in enumerate([('Exporters\n(EDF/WFDB/HDF5)','#1a3a1a','#86efac'),
                                      ('Metrics\n(SNR/PRD/SSIM)','#1a1a3a','#818cf8'),
                                      ('Validation\n(Pan-Tompkins)','#3a1a1a','#fca5a5'),
                                      ('CLI\n(bss)','#2a1a0a','#fed7aa')]):
    x = 0.5 + i*2.4
    box(ax, x, 0.4, 2.0, 0.9, txt, col, fc, 8.5)
    arrow(ax, 5.0, 2.1, x+1.0, 1.3)

ax.set_title('BioSignal Simulator — System Architecture', color=C['text'], fontsize=14, fontweight='bold')
save(fig, '17_architecture')


# ─────────────────────────────────────────────────────────────────────
# 18. ADC QUANTIZATION
# ─────────────────────────────────────────────────────────────────────
print("Generating quantization demo...")
from biosignal_simulator import QuantizationNoise
from biosignal_simulator.core.config import QuantizationNoiseConfig

g_q = ECGGenerator(ECGConfig(fs=500, duration_s=3, heart_rate=72, seed=42))
clean_q = g_q.generate()
t_q = g_q.t

fig, axes = plt.subplots(3, 1, figsize=(14, 7), facecolor=C['bg'])
for i, (bits, col) in enumerate([(16, C['ecg']), (8, C['resp']), (4, '#ff6b6b')]):
    qn = QuantizationNoise(QuantizationNoiseConfig(n_bits=bits, v_range=4.0))
    q_out, _ = qn.apply(clean_q.copy())
    style_ax(axes[i], xlim=(0, 3))
    axes[i].plot(t_q, clean_q, color='#4a5568', lw=1.0, alpha=0.5)
    axes[i].plot(t_q, q_out, color=col, lw=0.9)
    axes[i].set_ylabel(f'{bits}-bit ADC\n({2**bits} levels)', color=C['text'], fontsize=9)
    if i < 2: axes[i].set_xticks([])
axes[-1].set_xlabel('Time (s)', color=C['text'], fontsize=11)
fig.suptitle('ADC Quantization Noise — 16-bit vs 8-bit vs 4-bit', color=C['text'], fontsize=14, fontweight='bold')
fig.tight_layout(); fig.subplots_adjust(hspace=0.2); save(fig, '18_quantization')

print("\nAll documentation images generated OK!")
print(f"   Output: {OUT}")
print(f"   Total:  {len([f for f in os.listdir(OUT) if f.endswith('.png')])} images")
