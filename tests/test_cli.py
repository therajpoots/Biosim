import os
import sys
import json
import pytest
from unittest.mock import patch
import numpy as np

import biosignal_simulator as bss
from biosignal_simulator.cli import main, prompt_user

def test_prompt_user_inputs():
    # Test normal input conversion
    with patch('builtins.input', return_value="120.0"):
        res = prompt_user("Enter HR", 75.0, float)
        assert res == 120.0

    # Test default value usage
    with patch('builtins.input', return_value=""):
        res = prompt_user("Enter HR", 75.0, float)
        assert res == 75.0

    # Test choices validation (first invalid, then valid)
    inputs = ["invalid", "ecg"]
    with patch('builtins.input', side_effect=inputs):
        res = prompt_user("Select type", "ecg", str, choices=["ecg", "eeg"])
        assert res == "ecg"

    # Test bool conversion
    with patch('builtins.input', return_value="y"):
        res = prompt_user("Add noise?", False, bool)
        assert res is True

    with patch('builtins.input', return_value="no"):
        res = prompt_user("Add noise?", True, bool)
        assert res is False

    # Test bad format recovery
    inputs_bad = ["abc", "100"]
    with patch('builtins.input', side_effect=inputs_bad):
        res = prompt_user("Enter int", 5, int)
        assert res == 100

def test_cli_help(capsys):
    # Empty args should print help and exit 0
    with patch.object(sys, 'argv', ['main']):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "BioSignal Platform CLI Engine" in captured.out

def test_cli_generate_basic(tmp_path):
    out_file = tmp_path / "test_gen.npz"
    # Basic ecg generation
    args = [
        'main', 'generate',
        '--type', 'ecg',
        '--duration', '2.0',
        '--fs', '100.0',
        '--hr', '80.0',
        '--rhythm', 'normal',
        '--output', str(out_file)
    ]
    with patch.object(sys, 'argv', args):
        main()
    assert out_file.exists()

    # Load and verify
    record = bss.BiosignalImporter.import_numpy(str(out_file))
    assert record.signal_type == 'ecg'
    assert record.fs == 100.0
    assert record.duration_s == 2.0

def test_cli_generate_with_noise_and_formats(tmp_path):
    # Test generating with multiple noise models and saving as CSV
    out_csv = tmp_path / "test_gen.csv"
    args = [
        'main', 'generate',
        '--type', 'eeg',
        '--duration', '2.0',
        '--fs', '100.0',
        '--noise', 'gaussian:15', 'pink',
        '--snr', '10.0',
        '--output', str(out_csv)
    ]
    with patch.object(sys, 'argv', args):
        main()
    assert out_csv.exists()

    # Test exporting to HDF5
    out_h5 = tmp_path / "test_gen.h5"
    args_h5 = [
        'main', 'generate',
        '--type', 'emg',
        '--duration', '2.0',
        '--fs', '100.0',
        '--output', str(out_h5)
    ]
    with patch.object(sys, 'argv', args_h5):
        main()
    assert out_h5.exists()

    # Test exporting to EDF
    out_edf = tmp_path / "test_gen.edf"
    args_edf = [
        'main', 'generate',
        '--type', 'ppg',
        '--duration', '2.0',
        '--fs', '100.0',
        '--output', str(out_edf)
    ]
    with patch.object(sys, 'argv', args_edf):
        main()
    assert out_edf.exists()

def test_cli_generate_from_yaml(tmp_path):
    yaml_config = """
signal:
  type: ecg
  params:
    fs: 100.0
    duration_s: 2.0
    heart_rate: 70.0
noise:
  - type: gaussian
    params:
      std: 0.1
  - type: colored
    params:
      exponent: 1.0
mixer:
  target_snr_db: 12.0
output:
  path: {path}
"""
    out_file = tmp_path / "yaml_sim.npz"
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        f.write(yaml_config.format(path=str(out_file).replace('\\', '/')))

    args = [
        'main', 'generate',
        '--config', str(config_file)
    ]
    with patch.object(sys, 'argv', args):
        main()
    assert out_file.exists()

def test_cli_validate(tmp_path, capsys):
    # First generate a record
    rec_file = tmp_path / "to_validate.npz"
    args_gen = [
        'main', 'generate',
        '--type', 'ecg',
        '--duration', '2.0',
        '--fs', '100.0',
        '--output', str(rec_file)
    ]
    with patch.object(sys, 'argv', args_gen):
        main()

    # Run validate
    html_report = tmp_path / "report.html"
    args_val = [
        'main', 'validate',
        str(rec_file),
        '--html', str(html_report)
    ]
    with patch.object(sys, 'argv', args_val):
        main()

    captured = capsys.readouterr()
    assert "SIGNAL VALIDATION REPORT" in captured.out
    assert html_report.exists()

def test_cli_sweep(tmp_path):
    out_csv = tmp_path / "sweep_results.csv"
    args = [
        'main', 'sweep',
        '--type', 'ecg',
        '--param', 'heart_rate',
        '--values', '[60, 80, 100]',
        '--fs', '100.0',
        '--duration', '2.0',
        '--output', str(out_csv)
    ]
    with patch.object(sys, 'argv', args):
        main()
    assert out_csv.exists()

    # Verify csv content
    with open(out_csv, 'r') as f:
        headers = f.readline().strip().split(',')
    assert 'sweep_value' in headers
    assert 'rmse' in headers

def test_cli_list(capsys):
    args = ['main', 'list']
    with patch.object(sys, 'argv', args):
        main()
    captured = capsys.readouterr()
    assert "PLATFORM ALGORITHMS & CAPABILITIES" in captured.out
    assert "Signal Waveform Generators" in captured.out

def test_cli_plot(tmp_path):
    # First generate a record
    rec_file = tmp_path / "to_plot.npz"
    args_gen = [
        'main', 'generate',
        '--type', 'ecg',
        '--duration', '2.0',
        '--fs', '100.0',
        '--output', str(rec_file)
    ]
    with patch.object(sys, 'argv', args_gen):
        main()

    # Test plot standard
    img_out = tmp_path / "plot_standard.png"
    args_plot = [
        'main', 'plot',
        str(rec_file),
        '--output', str(img_out),
        '--style', 'standard'
    ]
    with patch.object(sys, 'argv', args_plot):
        main()
    assert img_out.exists()

    # Test plot ecg-12
    img_out_12 = tmp_path / "plot_12.png"
    args_plot_12 = [
        'main', 'plot',
        str(rec_file),
        '--output', str(img_out_12),
        '--style', 'ecg-12'
    ]
    with patch.object(sys, 'argv', args_plot_12):
        main()
    assert img_out_12.exists()

    # Test plot dashboard
    html_out = tmp_path / "dashboard.html"
    args_plot_dash = [
        'main', 'plot',
        str(rec_file),
        '--output', str(html_out),
        '--style', 'dashboard'
    ]
    with patch.object(sys, 'argv', args_plot_dash):
        main()
    assert html_out.exists()

def test_cli_interactive(tmp_path):
    # Mock inputs for step-by-step interactive setup wizard
    # Select ECG, use defaults, no PVC, add gaussian, no others, target SNR, export to npz
    out_file = tmp_path / "interactive_sim.npz"
    user_inputs = [
        'ecg',          # Signal type
        '',             # FS (default 250)
        '2.0',          # Duration (2s instead of 10)
        '80.0',         # Heart rate
        'n',            # Inject PVCs? (no)
        'y',            # Add gaussian noise? (yes)
        'n',            # Add colored noise? (no)
        'n',            # Add baseline noise? (no)
        'n',            # Add powerline noise? (no)
        'n',            # Add motion noise? (no)
        'n',            # Add electrode noise? (no)
        'n',            # Add emg_artifact noise? (no)
        'n',            # Add sensor_detachment noise? (no)
        'y',            # Scale to target SNR? (yes)
        '18.0',         # Target SNR (dB)
        'npz',          # Export format
        str(out_file)   # Output filename
    ]

    args = ['main', 'interactive']
    with patch('builtins.input', side_effect=user_inputs):
        with patch.object(sys, 'argv', args):
            main()

    assert out_file.exists()

    # Let's test other wizard flows to cover EMG, EEG, EDA, PPG, Resp
    # Test EDA wizard flow
    out_file_eda = tmp_path / "wizard_eda.json"
    user_inputs_eda = [
        'eda',          # Signal type
        '',             # FS (default 250)
        '2.0',          # Duration
        '3.0',          # SCR frequency
        'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', # No noise models
        'n',            # Scale to target SNR? (no)
        'json',         # Export format
        str(out_file_eda)
    ]
    with patch('builtins.input', side_effect=user_inputs_eda):
        with patch.object(sys, 'argv', args):
            main()
    assert out_file_eda.exists()
