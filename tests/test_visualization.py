import os
import numpy as np
import pytest
import biosignal_simulator as bss
from biosignal_simulator.utils.visualization import (
    plot_record,
    plot_psd_comparison,
    plot_noise_characterization,
    plot_snr_sweep,
    plot_filter_response,
    plot_multi_lead_ecg,
    plot_eeg_spectrogram_and_bands,
    plot_emg_fatigue_indicators,
    plot_eda_decomposition,
    generate_interactive_html_dashboard,
    _check_matplotlib
)

@pytest.fixture
def sample_ecg_record():
    cfg = bss.ECGConfig(fs=100.0, duration_s=2.0)
    gen = bss.ECGGenerator(cfg)
    noise = bss.GaussianNoise(bss.GaussianNoiseConfig())
    mixer = bss.SignalMixer(signal_generator=gen, noise_models=[noise], target_snr_db=15.0)
    return mixer.mix()

@pytest.fixture
def sample_eeg_record():
    cfg = bss.EEGConfig(fs=100.0, duration_s=2.0)
    gen = bss.EEGGenerator(cfg)
    noise = bss.GaussianNoise(bss.GaussianNoiseConfig())
    mixer = bss.SignalMixer(signal_generator=gen, noise_models=[noise], target_snr_db=15.0)
    return mixer.mix()

@pytest.fixture
def sample_emg_record():
    cfg = bss.EMGConfig(fs=100.0, duration_s=2.0)
    gen = bss.EMGGenerator(cfg)
    noise = bss.GaussianNoise(bss.GaussianNoiseConfig())
    mixer = bss.SignalMixer(signal_generator=gen, noise_models=[noise], target_snr_db=15.0)
    return mixer.mix()

@pytest.fixture
def sample_eda_record():
    cfg = bss.EDAConfig(fs=100.0, duration_s=2.0)
    gen = bss.EDAGenerator(cfg)
    noise = bss.GaussianNoise(bss.GaussianNoiseConfig())
    mixer = bss.SignalMixer(signal_generator=gen, noise_models=[noise], target_snr_db=15.0)
    return mixer.mix()

def test_check_matplotlib():
    plt = _check_matplotlib()
    assert plt is not None

def test_plot_record(sample_ecg_record):
    plt = _check_matplotlib()
    fig = plot_record(sample_ecg_record, show_components=True)
    assert fig is not None
    plt.close(fig)

    fig_no_comp = plot_record(sample_ecg_record, show_components=False)
    assert fig_no_comp is not None
    plt.close(fig_no_comp)

def test_plot_psd_comparison():
    plt = _check_matplotlib()
    sig1 = np.random.randn(200)
    sig2 = np.random.randn(200)
    signals = {"signal1": sig1, "signal2": sig2}
    fig = plot_psd_comparison(signals, fs=100.0)
    assert fig is not None
    plt.close(fig)

def test_plot_noise_characterization():
    plt = _check_matplotlib()
    noise_model = bss.GaussianNoise(bss.GaussianNoiseConfig())
    fig = plot_noise_characterization(noise_model, n_samples=200, fs=100.0)
    assert fig is not None
    plt.close(fig)

    with pytest.raises(ValueError, match="does not have a generate method"):
        plot_noise_characterization(object(), n_samples=200, fs=100.0)

def test_plot_snr_sweep():
    plt = _check_matplotlib()
    results = {
        'snr_target_db': [5.0, 10.0, 15.0],
        'snr_out': [4.8, 9.9, 14.7],
        'rmse': [0.1, 0.05, 0.02],
        'correlation': [0.95, 0.98, 0.99]
    }
    fig = plot_snr_sweep(results)
    assert fig is not None
    plt.close(fig)

def test_plot_filter_response():
    plt = _check_matplotlib()
    b = np.array([0.5, 0.5])
    a = np.array([1.0, 0.0])
    fig = plot_filter_response(b, a, fs=100.0, fmax=40.0)
    assert fig is not None
    plt.close(fig)

def test_plot_multi_lead_ecg(sample_ecg_record):
    plt = _check_matplotlib()
    fig = plot_multi_lead_ecg(sample_ecg_record)
    assert fig is not None
    plt.close(fig)

def test_plot_eeg_spectrogram_and_bands(sample_eeg_record):
    plt = _check_matplotlib()
    fig = plot_eeg_spectrogram_and_bands(sample_eeg_record)
    assert fig is not None
    plt.close(fig)

def test_plot_emg_fatigue_indicators(sample_emg_record):
    plt = _check_matplotlib()
    fig = plot_emg_fatigue_indicators(sample_emg_record)
    assert fig is not None
    plt.close(fig)

def test_plot_eda_decomposition(sample_eda_record):
    plt = _check_matplotlib()
    fig = plot_eda_decomposition(sample_eda_record)
    assert fig is not None
    plt.close(fig)

def test_generate_interactive_html_dashboard(sample_ecg_record, tmp_path):
    out_file = tmp_path / "dashboard.html"
    generate_interactive_html_dashboard(sample_ecg_record, str(out_file))
    assert out_file.exists()
    
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "BioSignal Simulator Interactive Dashboard" in content
    assert "clean" in content
    assert "noisy" in content
