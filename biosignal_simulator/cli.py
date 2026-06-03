"""
BioSignal Simulator Command Line Interface (CLI) Suite.

This module provides a clinical-grade command-line interface for the BioSignal
Simulator, offering multi-command capabilities, parameter sweeps, signal
validation diagnostics, interactive configuration wizards, and plotting utilities.

Subcommands:
  - generate: Generate biosignals from direct arguments or YAML configs.
  - validate: Perform integrity and physiological validation on simulated files.
  - sweep: Execute parameter grid sweeps (e.g. sweeping SNR or heart rate).
  - interactive: Launch a step-by-step terminal configuration wizard.
  - list: Display all available signal generators, noise models, and metrics.
  - plot: Render and save graphs from existing output files.
"""

import os
import sys
import json
import argparse
import datetime
import traceback
from typing import List, Dict, Any, Optional

import numpy as np

import biosignal_simulator as bss
from biosignal_simulator.io.exporters import BiosignalExporter, BiosignalImporter
from biosignal_simulator.utils.validation import validate_signal, generate_validation_report_html
from biosignal_simulator.utils.visualization import (
    plot_record,
    plot_multi_lead_ecg,
    plot_eeg_spectrogram_and_bands,
    plot_emg_fatigue_indicators,
    plot_eda_decomposition,
    generate_interactive_html_dashboard,
    _check_matplotlib
)

ASCII_BANNER = r"""
========================================================================
   ____  _       _____ _                   _   ____  _
  |  _ \(_)     / ____(_)                 | | |  _ \(_)
  | |_) |_  ___| (___  _  __ _ _ __   __ _| | | |_) |_ _ __ ___
  |  _ <| |/ _ \\___ \| |/ _` | '_ \ / _` | | |  _ <| | '_ ` _ \
  | |_) | | (_) |___) | | (_| | | | | (_| | | | |_) | | | | | | |
  |____/|_|\___/_____/|_|\__, |_| |_|\__,_|_| |____/|_|_| |_| |_|
                          __/ |
                         |___/       BIOPOTENTIAL SIMULATION PLATFORM
========================================================================
"""

def print_banner():
    """Print the platform welcome banner."""
    print(ASCII_BANNER)


def prompt_user(text: str, default: Any, val_type: type = str, choices: Optional[List[Any]] = None) -> Any:
    """Prompt user for input in interactive mode with type checking and validation."""
    choice_str = f" ({'/'.join(map(str, choices))})" if choices else ""
    default_str = f" [default: {default}]" if default is not None else ""
    
    while True:
        try:
            inp = input(f"{text}{choice_str}{default_str}: ").strip()
            if not inp:
                if default is not None:
                    return default
                else:
                    print("Error: Input is required.")
                    continue
                    
            # Convert type
            if val_type == bool:
                if inp.lower() in ['y', 'yes', 'true', '1']:
                    return True
                elif inp.lower() in ['n', 'no', 'false', '0']:
                    return False
                else:
                    print("Error: Please enter y or n.")
                    continue
            else:
                converted = val_type(inp)
                
            # Verify choices
            if choices and converted not in choices:
                print(f"Error: Invalid choice. Select from {choices}.")
                continue
                
            return converted
        except ValueError:
            print(f"Error: Invalid format. Expected {val_type.__name__}.")


class CliCommandSuite:
    """
    Subcommand execution mapping handler.
    """
    
    @staticmethod
    def handle_generate(args: argparse.Namespace) -> None:
        """Execute biosignal generation."""
        print_banner()
        
        # Load from YAML configuration if provided
        if args.config:
            try:
                import yaml
            except ImportError:
                print("Error: PyYAML is required for parsing configuration files. Run `pip install pyyaml`.")
                sys.exit(1)
                
            print(f"Loading simulation script from '{args.config}'...")
            try:
                with open(args.config, 'r') as f:
                    cfg = yaml.safe_load(f)
            except Exception as e:
                print(f"Error reading configuration file: {e}")
                sys.exit(1)
                
            # Delegate parsing to legacy YAML loading flow
            CliCommandSuite._run_from_dict(cfg)
            return

        # Direct argument parsing
        sig_type = args.type.lower()
        print(f"Initializing {sig_type.upper()} Signal Generator...")
        
        # Instantiate configurations
        try:
            if sig_type == 'ecg':
                cfg_cls = bss.ECGConfig
                gen_cls = bss.ECGGenerator
                params = {
                    'heart_rate': args.hr,
                    'fs': args.fs,
                    'duration_s': args.duration,
                    'rhythm_type': args.rhythm
                }
            elif sig_type == 'eeg':
                cfg_cls = bss.EEGConfig
                gen_cls = bss.EEGGenerator
                params = {'fs': args.fs, 'duration_s': args.duration}
            elif sig_type == 'emg':
                cfg_cls = bss.EMGConfig
                gen_cls = bss.EMGGenerator
                params = {'fs': args.fs, 'duration_s': args.duration}
            elif sig_type == 'ppg':
                cfg_cls = bss.PPGConfig
                gen_cls = bss.PPGGenerator
                params = {'fs': args.fs, 'duration_s': args.duration}
            elif sig_type == 'eda':
                cfg_cls = bss.EDAConfig
                gen_cls = bss.EDAGenerator
                params = {'fs': args.fs, 'duration_s': args.duration}
            elif sig_type == 'resp':
                cfg_cls = bss.RespConfig
                gen_cls = bss.RespGenerator
                params = {'fs': args.fs, 'duration_s': args.duration}
            else:
                print(f"Error: Unknown signal type '{sig_type}'")
                sys.exit(1)
                
            sig_config = cfg_cls(**params)
            generator = gen_cls(sig_config)
        except Exception as e:
            print(f"Error creating signal generator configuration: {e}")
            sys.exit(1)
            
        # Parse Noise Models
        noise_models = []
        if args.noise:
            for noise_spec in args.noise:
                # Format: type:snr or just type (e.g. gaussian:15 or baseline)
                parts = noise_spec.split(':')
                n_type = parts[0].lower()
                snr = float(parts[1]) if len(parts) > 1 else None
                
                # Setup noise mapping
                noise_mapping = {
                    'gaussian': (bss.GaussianNoise, bss.GaussianNoiseConfig),
                    'colored': (bss.ColoredNoise, bss.ColoredNoiseConfig),
                    'pink': (bss.PinkNoise, bss.ColoredNoiseConfig),
                    'brown': (bss.BrownNoise, bss.ColoredNoiseConfig),
                    'blue': (bss.BlueNoise, bss.ColoredNoiseConfig),
                    'violet': (bss.VioletNoise, bss.ColoredNoiseConfig),
                    'baseline': (bss.BaselineWander, bss.BaselineWanderConfig),
                    'powerline': (bss.PowerlineNoise, bss.PowerlineNoiseConfig),
                    'motion': (bss.MotionArtifact, bss.MotionArtifactConfig),
                    'electrode': (bss.ElectrodeNoise, bss.ElectrodeNoiseConfig),
                    'emg_artifact': (bss.EMGArtifact, bss.EMGArtifactConfig),
                    'impulse': (bss.ImpulseNoise, bss.ImpulseNoiseConfig),
                    'quantization': (bss.QuantizationNoise, bss.QuantizationNoiseConfig),
                    'crosstalk': (bss.CrosstalkNoise, bss.CrosstalkNoiseConfig),
                    'sensor_detachment': (bss.SensorDetachmentNoise, bss.SensorDetachmentConfig),
                    'electrode_displacement': (bss.ElectrodeDisplacementNoise, bss.ElectrodeDisplacementConfig),
                    'light_leakage': (bss.LightLeakageNoise, bss.LightLeakageConfig),
                    'packet_loss': (bss.PacketLossNoise, bss.PacketLossConfig),
                }
                
                if n_type not in noise_mapping:
                    print(f"Warning: Skipping unknown noise type '{n_type}'")
                    continue
                    
                model_cls, config_cls = noise_mapping[n_type]
                try:
                    if n_type in ['pink', 'brown', 'blue', 'violet']:
                        model = model_cls()
                    else:
                        n_cfg = config_cls()
                        model = model_cls(n_cfg)
                    noise_models.append(model)
                except Exception as e:
                    print(f"Error building noise model '{n_type}': {e}")
                    sys.exit(1)
                    
        # Mix
        print("Synthesizing waveforms and injecting noise contamination...")
        try:
            mixer = bss.SignalMixer(
                signal_generator=generator,
                noise_models=noise_models,
                target_snr_db=args.snr
            )
            record = mixer.mix()
        except Exception as e:
            print(f"Error mixing signal: {e}")
            sys.exit(1)
            
        # Export
        out_path = args.output or "simulated_record.npz"
        print(f"Exporting biopotentials to '{out_path}'...")
        try:
            path_lower = out_path.lower()
            if path_lower.endswith('.npz'):
                BiosignalExporter.export_numpy(record, out_path)
            elif path_lower.endswith('.csv'):
                BiosignalExporter.export_csv(record, out_path)
            elif path_lower.endswith('.h5') or path_lower.endswith('.hdf5'):
                BiosignalExporter.export_hdf5(record, out_path)
            elif path_lower.endswith('.edf'):
                # Default to exporting noisy signal to EDF
                BiosignalExporter.export_edf_lite(record, out_path)
            else:
                BiosignalExporter.export_numpy(record, out_path)
            print("Signal generation and export completed successfully.")
        except Exception as e:
            print(f"Export failed: {e}")
            sys.exit(1)

    @staticmethod
    def _run_from_dict(cfg: Dict[str, Any]) -> None:
        """Legacy helper to run from YAML configuration dictionary."""
        sig_cfg = cfg.get('signal', {})
        sig_type = sig_cfg.get('type', '').lower()
        sig_params = sig_cfg.get('params', {})

        if sig_type == 'ecg':
            generator_cls = bss.ECGGenerator
            config_cls = bss.ECGConfig
        elif sig_type == 'eeg':
            generator_cls = bss.EEGGenerator
            config_cls = bss.EEGConfig
        elif sig_type == 'emg':
            generator_cls = bss.EMGGenerator
            config_cls = bss.EMGConfig
        elif sig_type == 'ppg':
            generator_cls = bss.PPGGenerator
            config_cls = bss.PPGConfig
        elif sig_type == 'eda':
            generator_cls = bss.EDAGenerator
            config_cls = bss.EDAConfig
        elif sig_type == 'resp':
            generator_cls = bss.RespGenerator
            config_cls = bss.RespConfig
        else:
            print(f"Error: Unknown or missing signal type '{sig_type}'")
            sys.exit(1)

        try:
            sig_config = bss.ConfigSerializer.from_dict(sig_params, config_cls)
            generator = generator_cls(sig_config)
        except Exception as e:
            print(f"Error initializing signal generator: {e}")
            sys.exit(1)

        # Parse Noise Models
        noise_list = cfg.get('noise', [])
        noise_models = []

        noise_mapping = {
            'gaussian': (bss.GaussianNoise, bss.GaussianNoiseConfig),
            'colored': (bss.ColoredNoise, bss.ColoredNoiseConfig),
            'pink': (bss.PinkNoise, bss.ColoredNoiseConfig),
            'brown': (bss.BrownNoise, bss.ColoredNoiseConfig),
            'blue': (bss.BlueNoise, bss.ColoredNoiseConfig),
            'violet': (bss.VioletNoise, bss.ColoredNoiseConfig),
            'baseline': (bss.BaselineWander, bss.BaselineWanderConfig),
            'powerline': (bss.PowerlineNoise, bss.PowerlineNoiseConfig),
            'motion': (bss.MotionArtifact, bss.MotionArtifactConfig),
            'electrode': (bss.ElectrodeNoise, bss.ElectrodeNoiseConfig),
            'emg_artifact': (bss.EMGArtifact, bss.EMGArtifactConfig),
            'impulse': (bss.ImpulseNoise, bss.ImpulseNoiseConfig),
            'quantization': (bss.QuantizationNoise, bss.QuantizationNoiseConfig),
            'crosstalk': (bss.CrosstalkNoise, bss.CrosstalkNoiseConfig),
            'sensor_detachment': (bss.SensorDetachmentNoise, bss.SensorDetachmentConfig),
            'electrode_displacement': (bss.ElectrodeDisplacementNoise, bss.ElectrodeDisplacementConfig),
            'light_leakage': (bss.LightLeakageNoise, bss.LightLeakageConfig),
            'packet_loss': (bss.PacketLossNoise, bss.PacketLossConfig),
        }

        for n_item in noise_list:
            n_type = n_item.get('type', '').lower()
            n_params = n_item.get('params', {})
            
            if n_type not in noise_mapping:
                print(f"Warning: Skipping unknown noise type '{n_type}'")
                continue
                
            model_cls, n_config_cls = noise_mapping[n_type]
            try:
                if n_type in ['pink', 'brown', 'blue', 'violet']:
                    model = model_cls(**n_params)
                else:
                    n_config = bss.ConfigSerializer.from_dict(n_params, n_config_cls)
                    model = model_cls(n_config)
                noise_models.append(model)
            except Exception as e:
                print(f"Error initializing noise model '{n_type}': {e}")
                sys.exit(1)

        # Mix
        mixer_cfg = cfg.get('mixer', {})
        target_snr = mixer_cfg.get('target_snr_db', None)
        
        try:
            mixer = bss.SignalMixer(
                signal_generator=generator,
                noise_models=noise_models,
                target_snr_db=target_snr
            )
            record = mixer.mix()
        except Exception as e:
            print(f"Error mixing signal and noise: {e}")
            sys.exit(1)

        # Export
        out_cfg = cfg.get('output', {})
        out_path = out_cfg.get('path', 'output.npz')

        try:
            path_lower = out_path.lower()
            if path_lower.endswith('.npz'):
                BiosignalExporter.export_numpy(record, out_path)
            elif path_lower.endswith('.csv'):
                BiosignalExporter.export_csv(record, out_path)
            elif path_lower.endswith('.h5') or path_lower.endswith('.hdf5'):
                BiosignalExporter.export_hdf5(record, out_path)
            elif path_lower.endswith('.edf'):
                BiosignalExporter.export_edf_lite(record, out_path)
            else:
                BiosignalExporter.export_numpy(record, out_path)
            print(f"Successfully generated and exported signal to '{out_path}'")
        except Exception as e:
            print(f"Error exporting signal: {e}")
            sys.exit(1)

    @staticmethod
    def handle_validate(args: argparse.Namespace) -> None:
        """Perform signal diagnostics and validation check."""
        print_banner()
        print(f"Loading signal record for verification: '{args.input}'...")
        
        # Load file using symmetrical importer
        try:
            path_lower = args.input.lower()
            if path_lower.endswith('.npz'):
                record = BiosignalImporter.import_numpy(args.input)
            elif path_lower.endswith('.csv'):
                record = BiosignalImporter.import_csv(args.input)
            elif path_lower.endswith('.h5') or path_lower.endswith('.hdf5'):
                record = BiosignalImporter.import_hdf5(args.input)
            elif path_lower.endswith('.edf'):
                record = BiosignalImporter.import_edf(args.input)
            elif path_lower.endswith('.hea') or path_lower.endswith('.dat'):
                base_path = os.path.splitext(args.input)[0]
                record = BiosignalImporter.import_wfdb(base_path)
            else:
                # Default fallback
                record = BiosignalImporter.import_numpy(args.input)
        except Exception as e:
            print(f"Error loading file: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        print("Record loaded successfully. Analyzing waveforms and metrics...")
        
        # Extract signal to validate (prefer noisy)
        sig_data = record.noisy
        if sig_data.ndim > 1:
            sig_data = sig_data[0] # analyze first channel
            
        report = validate_signal(
            signal=sig_data,
            fs=record.fs,
            expected_type=args.type or record.signal_type
        )
        
        # Print CLI summary
        print("\n" + "="*50)
        print("            SIGNAL VALIDATION REPORT")
        print("="*50)
        print(f"Signal Class:       {record.signal_type.upper()}")
        print(f"Sampling Frequency: {record.fs:.1f} Hz")
        print(f"Total Duration:     {record.t[-1]:.2f} s")
        print(f"Total Data Points:  {len(record.t)}")
        print(f"Overall Status:     " + ("\033[92mVALID\033[0m" if report.is_valid else "\033[93mWARNINGS PENDING\033[0m"))
        print("-"*50)
        
        print("Measured Metrics:")
        for k, v in report.metrics.items():
            if isinstance(v, float):
                print(f"  - {k:<25}: {v:.4f}")
            else:
                print(f"  - {k:<25}: {v}")
                
        print("-"*50)
        print("Active Warnings:")
        if report.warnings:
            for w in report.warnings:
                print(f"  [!] \033[93m{w}\033[0m")
        else:
            print("  None. Signal is physiologically coherent and artifact-free.")
        print("="*50 + "\n")
        
        # Export HTML report if requested
        if args.html:
            print(f"Exporting beautiful HTML validation report to '{args.html}'...")
            try:
                generate_validation_report_html(record, report, args.html)
                print("HTML report generated successfully.")
            except Exception as e:
                print(f"Failed to generate HTML report: {e}")

    @staticmethod
    def handle_sweep(args: argparse.Namespace) -> None:
        """Run parameter sweep experiments."""
        print_banner()
        print("Initializing Parameter Grid Sweep Experiment...")
        
        # Load grid parameters
        try:
            sweep_values = json.loads(args.values)
            if not isinstance(sweep_values, list):
                raise ValueError("Values must be a JSON list.")
        except Exception as e:
            print(f"Error parsing sweep values: {e}. Provide values as a JSON list, e.g. '[0, 5, 10, 15, 20]'")
            sys.exit(1)
            
        sig_type = args.type.lower()
        param_name = args.param
        
        print(f"Sweeping '{param_name}' over: {sweep_values}")
        print(f"Signal: {sig_type.upper()}, Noise standard: gaussian")
        
        results = {
            'sweep_value': [],
            'snr_target_db': [],
            'snr_achieved_db': [],
            'rmse': [],
            'correlation': []
        }
        
        # Determine generator config
        if sig_type == 'ecg':
            cfg_cls = bss.ECGConfig
            gen_cls = bss.ECGGenerator
        elif sig_type == 'eeg':
            cfg_cls = bss.EEGConfig
            gen_cls = bss.EEGGenerator
        elif sig_type == 'emg':
            cfg_cls = bss.EMGConfig
            gen_cls = bss.EMGGenerator
        else:
            print("Error: Parameter sweep currently supports ecg, eeg, and emg.")
            sys.exit(1)
            
        # Noise
        noise_model = bss.GaussianNoise(bss.GaussianNoiseConfig())
        
        for val in sweep_values:
            print(f"  Running simulation step for {param_name} = {val}...")
            
            # Setup params
            params = {'fs': args.fs, 'duration_s': args.duration}
            if param_name in ['heart_rate', 'hr'] and sig_type == 'ecg':
                params['heart_rate'] = float(val)
            else:
                # Custom injection into dictionary
                params[param_name] = val
                
            try:
                sig_config = cfg_cls(**params)
                generator = gen_cls(sig_config)
                
                # Check target SNR db or sweep target SNR itself!
                target_snr = args.snr
                if param_name == 'snr':
                    target_snr = float(val)
                    
                mixer = bss.SignalMixer(
                    signal_generator=generator,
                    noise_models=[noise_model],
                    target_snr_db=target_snr
                )
                record = mixer.mix()
                
                # Compute performance metrics
                from biosignal_simulator.metrics.snr import compute_snr_wideband
                from biosignal_simulator.metrics.distortion import compute_rmse, compute_correlation
                
                c_data = record.clean[0] if record.clean.ndim > 1 else record.clean
                n_data = record.noisy[0] if record.noisy.ndim > 1 else record.noisy
                
                achieved_snr = compute_snr_wideband(c_data, n_data, record.fs)
                rmse = compute_rmse(c_data, n_data)
                corr = compute_correlation(c_data, n_data)
                
                results['sweep_value'].append(val)
                results['snr_target_db'].append(target_snr if target_snr is not None else float('nan'))
                results['snr_achieved_db'].append(achieved_snr)
                results['rmse'].append(rmse)
                results['correlation'].append(corr)
                
            except Exception as e:
                print(f"    Simulation failed at step {val}: {e}")
                
        # Export stats
        out_csv = args.output or "sweep_results.csv"
        print(f"Sweep completed. Saving experimental metrics to '{out_csv}'...")
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(out_csv, index=False)
            
            # Display summary table in CLI
            print("\nSweep summary results:")
            print(df.to_string(index=False))
            print()
        except Exception as e:
            # Fallback direct formatting
            with open(out_csv, 'w') as f:
                f.write("sweep_value,snr_target_db,snr_achieved_db,rmse,correlation\n")
                for i in range(len(results['sweep_value'])):
                    f.write(
                        f"{results['sweep_value'][i]},{results['snr_target_db'][i]},"
                        f"{results['snr_achieved_db'][i]},{results['rmse'][i]},{results['correlation'][i]}\n"
                    )
            print(f"Results written to CSV. Matplotlib is required to render graph displays.")

    @staticmethod
    def handle_interactive(args: argparse.Namespace) -> None:
        """Launch the step-by-step terminal wizard configuration interface."""
        print_banner()
        print("Welcome to the BioSignal Simulator Wizard!")
        print("Follow the prompts below to configure and run your customized simulation.\n")
        
        # 1. Signal selection
        sig_type = prompt_user("Select biopotential signal type", "ecg", choices=["ecg", "eeg", "emg", "ppg", "eda", "resp"])
        
        fs = prompt_user("Enter sampling rate (Hz)", 250.0, float)
        duration = prompt_user("Enter signal duration (seconds)", 10.0, float)
        
        sig_params = {}
        if sig_type == "ecg":
            sig_params['heart_rate'] = prompt_user("Enter baseline heart rate (bpm)", 75.0, float)
            sig_params['add_pvc'] = prompt_user("Inject Premature Ventricular Contractions (PVCs)? (y/n)", False, bool)
            if sig_params['add_pvc']:
                sig_params['pvc_rate'] = prompt_user("PVC rate (ectopics/minute)", 5.0, float)
        elif sig_type == "eeg":
            sig_params['alpha_power'] = prompt_user("Alpha wave power (relaxation level)", 0.2, float)
            sig_params['inject_seizure'] = prompt_user("Inject epileptiform spike discharges? (y/n)", False, bool)
        elif sig_type == "emg":
            sig_params['firing_rate'] = prompt_user("Motor unit discharge rate (Hz)", 40.0, float)
            sig_params['fatigue_factor'] = prompt_user("Muscle fatigue scaling compression coefficient [0-1]", 0.0, float)
        elif sig_type == "ppg":
            sig_params['heart_rate'] = prompt_user("Heart rate/pulse rate baseline (bpm)", 70.0, float)
        elif sig_type == "eda":
            sig_params['scr_frequency'] = prompt_user("Skin conductance response peaks per minute", 3.0, float)
        elif sig_type == "resp":
            sig_params['breath_rate'] = prompt_user("Baseline respiration rate (breaths/min)", 15.0, float)

        # 2. Noise selection
        print("\n--- Noise Models Selection ---")
        noise_models = []
        
        noise_types = ["gaussian", "colored", "baseline", "powerline", "motion", "electrode", "emg_artifact", "sensor_detachment"]
        
        for n_t in noise_types:
            add = prompt_user(f"Add {n_t.replace('_', ' ')} noise model? (y/n)", False, bool)
            if add:
                # Instantiate configurations
                if n_t == "gaussian":
                    n_cfg = bss.GaussianNoiseConfig()
                    noise_models.append(bss.GaussianNoise(n_cfg))
                elif n_t == "colored":
                    exponent = prompt_user("Select spectral decay exponent (1=pink, 2=brownian)", 1.0, float)
                    n_cfg = bss.ColoredNoiseConfig(exponent=exponent)
                    noise_models.append(bss.ColoredNoise(n_cfg))
                elif n_t == "baseline":
                    n_cfg = bss.BaselineWanderConfig()
                    noise_models.append(bss.BaselineWander(n_cfg))
                elif n_t == "powerline":
                    n_cfg = bss.PowerlineNoiseConfig()
                    noise_models.append(bss.PowerlineNoise(n_cfg))
                elif n_t == "motion":
                    n_cfg = bss.MotionArtifactConfig()
                    noise_models.append(bss.MotionArtifact(n_cfg))
                elif n_t == "electrode":
                    n_cfg = bss.ElectrodeNoiseConfig()
                    noise_models.append(bss.ElectrodeNoise(n_cfg))
                elif n_t == "emg_artifact":
                    n_cfg = bss.EMGArtifactConfig()
                    noise_models.append(bss.EMGArtifact(n_cfg))
                elif n_t == "sensor_detachment":
                    n_cfg = bss.SensorDetachmentConfig()
                    noise_models.append(bss.SensorDetachmentNoise(n_cfg))

        # SNR
        add_snr = prompt_user("Scale noise to a target SNR ratio? (y/n)", True, bool)
        target_snr = None
        if add_snr:
            target_snr = prompt_user("Enter target wideband SNR (dB)", 15.0, float)
            
        # 3. Output exporter
        print("\n--- Output Configuration ---")
        out_format = prompt_user("Choose export format", "npz", choices=["npz", "csv", "hdf5", "edf", "wfdb", "json"])
        out_filename = prompt_user("Enter target filename", f"wizard_sim_{sig_type}.{out_format if out_format != 'wfdb' else 'hea'}")
        
        # Mix and synthesize
        print("\nSynthesizing waveforms according to specifications...")
        try:
            # Setup signal generator
            if sig_type == 'ecg':
                rhythm_type = 'pvc' if sig_params.get('add_pvc') else 'normal'
                sig_config = bss.ECGConfig(
                    fs=fs,
                    duration_s=duration,
                    heart_rate=sig_params.get('heart_rate', 75.0),
                    rhythm_type=rhythm_type
                )
                generator = bss.ECGGenerator(sig_config)
            elif sig_type == 'eeg':
                alpha_val = sig_params.get('alpha_power', 0.2)
                band_powers = {'delta': 0.2, 'theta': 0.3, 'alpha': alpha_val, 'beta': 0.5, 'gamma': 0.1}
                state = 'epileptiform_spikes' if sig_params.get('inject_seizure') else 'relaxed'
                sig_config = bss.EEGConfig(
                    fs=fs,
                    duration_s=duration,
                    band_powers=band_powers,
                    state=state
                )
                generator = bss.EEGGenerator(sig_config)
            elif sig_type == 'emg':
                pathology = 'fatigue' if sig_params.get('fatigue_factor', 0.0) > 0.0 else 'normal'
                sig_config = bss.EMGConfig(
                    fs=fs,
                    duration_s=duration,
                    pathology=pathology
                )
                generator = bss.EMGGenerator(sig_config)
            elif sig_type == 'ppg':
                sig_config = bss.PPGConfig(
                    fs=fs,
                    duration_s=duration,
                    heart_rate=sig_params.get('heart_rate', 70.0)
                )
                generator = bss.PPGGenerator(sig_config)
            elif sig_type == 'eda':
                event_rate_hz = sig_params.get('scr_frequency', 3.0) / 60.0
                sig_config = bss.EDAConfig(
                    fs=fs,
                    duration_s=duration,
                    event_rate_hz=event_rate_hz
                )
                generator = bss.EDAGenerator(sig_config)
            elif sig_type == 'resp':
                resp_rate_hz = sig_params.get('breath_rate', 15.0) / 60.0
                sig_config = bss.RespConfig(
                    fs=fs,
                    duration_s=duration,
                    resp_rate_hz=resp_rate_hz
                )
                generator = bss.RespGenerator(sig_config)
                
            mixer = bss.SignalMixer(
                signal_generator=generator,
                noise_models=noise_models,
                target_snr_db=target_snr
            )
            record = mixer.mix()
            
            # Export
            if out_format == 'npz':
                BiosignalExporter.export_numpy(record, out_filename)
            elif out_format == 'csv':
                BiosignalExporter.export_csv(record, out_filename)
            elif out_format == 'hdf5':
                BiosignalExporter.export_hdf5(record, out_filename)
            elif out_format == 'edf':
                BiosignalExporter.export_edf_lite(record, out_filename)
            elif out_format == 'wfdb':
                base_path = os.path.splitext(out_filename)[0]
                BiosignalExporter.export_wfdb(record, base_path)
            elif out_format == 'json':
                BiosignalExporter.export_json(record, out_filename)
                
            print(f"\n[+] Success! Signal exported to '{out_filename}'.")
            
            # Generate companion validation report
            report_name = f"wizard_sim_{sig_type}_validation.html"
            print(f"Generating diagnostic validation check report: '{report_name}'...")
            
            sig_data = record.noisy
            if sig_data.ndim > 1:
                sig_data = sig_data[0]
            report = validate_signal(sig_data, record.fs, record.signal_type)
            generate_validation_report_html(record, report, report_name)
            
            # Save companion interactive dashboard
            dashboard_name = f"wizard_sim_{sig_type}_dashboard.html"
            print(f"Generating premium interactive dashboard interface: '{dashboard_name}'...")
            generate_interactive_html_dashboard(record, dashboard_name)
            
            print("\nWizard completed. Open HTML dashboards to view zoomable wave traces.")
            
        except Exception as e:
            print(f"\n[-] Error occurred: {e}")
            traceback.print_exc()

    @staticmethod
    def handle_list(args: argparse.Namespace) -> None:
        """List all available algorithms, noise models, and metrics."""
        print_banner()
        print("="*60)
        print("          PLATFORM ALGORITHMS & CAPABILITIES")
        print("="*60)
        print("Signal Waveform Generators:")
        print("  - ecg : ECG cardiac rhythm engine (PVCs, Arrhythmias, Dower VCG)")
        print("  - eeg : EEG neural oscillations (Sleep spindles, Seizures)")
        print("  - emg : EMG intramuscular & surface motor units (Fatigue)")
        print("  - ppg : PPG pulse waveforms (Dicrotic notches, APG/VPG derivative)")
        print("  - eda : EDA electrodermal baseline (Tonic SCL + Phasic SCR jumps)")
        print("  - resp: Respiration chest volume waves (Apnea, Biot, Kussmaul)")
        print("-"*60)
        print("Noise Contaminations & Artifacts:")
        print("  - gaussian      : White noise distribution")
        print("  - colored       : Pink (1/f), Brownian, Blue, Violet spectral profiles")
        print("  - baseline      : low-frequency baseline drift and respiratory wander")
        print("  - powerline     : 50/60 Hz powerline line noise + harmonics")
        print("  - motion        : cable displacement transients & movement jumps")
        print("  - electrode     : POPCORN electrode biopotential telegraph shifts")
        print("  - emg_artifact  : muscle tremor interference bursts")
        print("  - sensor_detachment: sensor bounce & loose detachment clipping")
        print("  - packet_loss   : wireless biotelemetry Markov state packet drop")
        print("-"*60)
        print("Clinical & Quality Exporters:")
        print("  - npz  : Compressed NumPy array metadata storage")
        print("  - csv  : Text file columns with embedded JSON header comments")
        print("  - hdf5 : Hierarchical database tree with chunking and gzip")
        print("  - edf  : Compliant multi-channel European Data Format binary")
        print("  - wfdb : PhysioNet Format 16/212 binary header/data files")
        print("  - mat  : MATLAB struct compatibility format")
        print("="*60 + "\n")

    @staticmethod
    def handle_plot(args: argparse.Namespace) -> None:
        """Render and save signal figures."""
        print_banner()
        print(f"Loading '{args.input}' for visualization...")
        
        try:
            path_lower = args.input.lower()
            if path_lower.endswith('.npz'):
                record = BiosignalImporter.import_numpy(args.input)
            elif path_lower.endswith('.csv'):
                record = BiosignalImporter.import_csv(args.input)
            elif path_lower.endswith('.h5') or path_lower.endswith('.hdf5'):
                record = BiosignalImporter.import_hdf5(args.input)
            elif path_lower.endswith('.edf'):
                record = BiosignalImporter.import_edf(args.input)
            else:
                record = BiosignalImporter.import_numpy(args.input)
        except Exception as e:
            print(f"Failed to load file: {e}")
            sys.exit(1)
            
        out_img = args.output or "signal_plot.png"
        style = args.style.lower()
        
        print(f"Rendering graphic using visualization style: '{style}'...")
        
        try:
            _check_matplotlib()
            import matplotlib.pyplot as plt
            
            if style == "standard":
                fig = plot_record(record, show_components=True)
            elif style == "ecg-12":
                fig = plot_multi_lead_ecg(record)
            elif style == "eeg-bands":
                fig = plot_eeg_spectrogram_and_bands(record)
            elif style == "emg-fatigue":
                fig = plot_emg_fatigue_indicators(record)
            elif style == "eda-decomp":
                fig = plot_eda_decomposition(record)
            elif style == "dashboard":
                dashboard_name = out_img if out_img.endswith('.html') else "interactive_dashboard.html"
                print(f"Exporting self-contained interactive dashboard to '{dashboard_name}'...")
                generate_interactive_html_dashboard(record, dashboard_name)
                print("Dashboard generated successfully.")
                return
            else:
                print(f"Warning: Unknown style '{style}'. Falling back to standard plot.")
                fig = plot_record(record)
                
            fig.savefig(out_img, dpi=150)
            print(f"Successfully saved rendered graphic to '{out_img}'")
            plt.close(fig)
        except Exception as e:
            print(f"Visualization error occurred: {e}")
            traceback.print_exc()


def main():
    """CLI subcommands routing and parsing entryway."""
    parser = argparse.ArgumentParser(description="BioSignal Platform CLI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # 1. generate subparser
    parser_gen = subparsers.add_parser("generate", help="Synthesize clean and contaminated physiological waveforms")
    parser_gen.add_argument("--type", "-t", default="ecg", choices=["ecg", "eeg", "emg", "ppg", "eda", "resp"], help="Signal class type")
    parser_gen.add_argument("--fs", "-f", default=250.0, type=float, help="Sampling frequency (Hz)")
    parser_gen.add_argument("--duration", "-d", default=10.0, type=float, help="Signal length (seconds)")
    parser_gen.add_argument("--hr", default=75.0, type=float, help="Heart rate/pulse rate default (bpm)")
    parser_gen.add_argument("--noise", "-n", nargs="+", help="Contamination models (e.g. gaussian:15 powerline:20)")
    parser_gen.add_argument("--snr", type=float, help="Target wideband signal SNR scale (dB)")
    parser_gen.add_argument("--output", "-o", help="Target output file path")
    parser_gen.add_argument("--config", "-c", help="Path to YAML script configuration file")
    parser_gen.add_argument(
        "--rhythm", "-r",
        default="normal",
        choices=["normal", "bradycardia", "tachycardia", "afib", "aflutter",
                 "pvc", "pac", "vtach", "vfib", "av_block", "wenckebach",
                 "complete_av_block", "rbbb", "lbbb", "wpw", "long_qt", "stemi", "ischemia"],
        help="Cardiac rhythm / arrhythmia type (default: normal)"
    )
    
    # 2. validate subparser
    parser_val = subparsers.add_parser("validate", help="Execute signal integrity and biological verification")
    parser_val.add_argument("input", help="Simulated data file path to validate")
    parser_val.add_argument("--type", "-t", choices=["ecg", "eeg", "emg", "ppg", "eda", "resp"], help="Override biological type class")
    parser_val.add_argument("--html", help="Generate companion styled HTML quality validation report")
    
    # 3. sweep subparser
    parser_swp = subparsers.add_parser("sweep", help="Perform parameter grid sweep experiments")
    parser_swp.add_argument("--type", "-t", default="ecg", choices=["ecg", "eeg", "emg"], help="Override signal class type")
    parser_swp.add_argument("--param", "-p", default="heart_rate", help="Parameter variable name to sweep (e.g. heart_rate, snr)")
    parser_swp.add_argument("--values", "-v", default="[60, 80, 100, 120]", help="JSON formatted values list to sweep")
    parser_swp.add_argument("--fs", "-f", default=250.0, type=float, help="Sampling frequency (Hz)")
    parser_swp.add_argument("--duration", "-d", default=5.0, type=float, help="Signal length (seconds)")
    parser_swp.add_argument("--snr", default=15.0, type=float, help="Baseline SNR scaling limit (dB)")
    parser_swp.add_argument("--output", "-o", help="Target CSV output stats file path")
    
    # 4. interactive subparser
    subparsers.add_parser("interactive", help="Start terminal setup wizard")
    
    # 5. list subparser
    subparsers.add_parser("list", help="Print all signal classes, noise models, and exporters descriptions")
    
    # 6. plot subparser
    parser_plt = subparsers.add_parser("plot", help="Generate and save visual signal diagnostics")
    parser_plt.add_argument("input", help="Simulated record data path")
    parser_plt.add_argument("--output", "-o", help="Output file path (e.g. plot.png)")
    parser_plt.add_argument("--style", "-s", default="standard", choices=["standard", "ecg-12", "eeg-bands", "emg-fatigue", "eda-decomp", "dashboard"], help="Graphic template layout")
    
    # Help routing if empty parameters
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    
    # Command handlers mapping
    handlers = {
        "generate": CliCommandSuite.handle_generate,
        "validate": CliCommandSuite.handle_validate,
        "sweep": CliCommandSuite.handle_sweep,
        "interactive": CliCommandSuite.handle_interactive,
        "list": CliCommandSuite.handle_list,
        "plot": CliCommandSuite.handle_plot
    }
    
    if args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
