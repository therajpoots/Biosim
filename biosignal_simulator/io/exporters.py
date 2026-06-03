"""
BioSignal Simulator Input/Output (I/O) Engine.

This module provides exhaustive, clinical-grade utilities to export and import
physiological signals and their corresponding metadata. It supports compressed
NumPy archives, pandas DataFrames, formatted CSV files, structured HDF5,
European Data Format (EDF) binary files, PhysioNet WFDB format 16 and format 212
binary/header pairs, MATLAB workspace files, and raw/compressed JSON.

================================================================================
EDF (European Data Format) File Structure:
================================================================================
Header Record:
- 8 bytes   : ASCII version of data format (always "0       ")
- 80 bytes  : Local patient identification
- 80 bytes  : Local recording identification
- 8 bytes   : Start date of recording (dd.mm.yy)
- 8 bytes   : Start time of recording (hh.mm.ss)
- 8 bytes   : Number of bytes in header record
- 44 bytes  : Reserved
- 8 bytes   : Number of data records (-1 if unknown)
- 8 bytes   : Duration of a data record in seconds
- 4 bytes   : Number of signals (ns) in data record
Signal Headers (256 bytes per signal):
- 16 bytes  : Label (e.g. "ECG Lead I")
- 80 bytes  : Transducer type
- 8 bytes   : Physical dimension (e.g. "uV" or "mV")
- 8 bytes   : Physical minimum (e.g. "-5.0")
- 8 bytes   : Physical maximum (e.g. "5.0")
- 8 bytes   : Digital minimum (e.g. "-32768")
- 8 bytes   : Digital maximum (e.g. "32767")
- 80 bytes  : Prefiltering
- 8 bytes   : Number of samples in each data record
- 32 bytes  : Reserved
Data Records:
- Interleaved blocks containing 16-bit 2's complement integers for each signal.

================================================================================
WFDB (PhysioNet Waveform Database) File Structure:
================================================================================
Header File (.hea):
- ASCII file describing signal names, formats, gains, baseline offsets, etc.
Binary File (.dat):
- Format 16: Interleaved 16-bit 2's complement integers.
- Format 212: Three bytes represent two 12-bit samples (little-endian packed nibbles).

Mathematical Quantization Scaling:
For scaling a physical signal $x[n]$ to digital integer $d[n]$:
$$d[n] = d_{\\text{min}} + \\text{round}\\left( \\frac{x[n] - x_{\\text{phys,min}}}{x_{\\text{phys,max}} - x_{\\text{phys,min}}} \\times (d_{\\text{max}} - d_{\\text{min}}) \\right)$$
For reconstruction back to physical scale:
$$\\hat{x}[n] = x_{\\text{phys,min}} + \\frac{d[n] - d_{\\text{min}}}{d_{\\text{max}} - d_{\\text{min}}} \\times (x_{\\text{phys,max}} - x_{\\text{phys,min}})$$
"""

import os
import json
import gzip
import datetime
import warnings
from typing import Optional, Union, List, Dict, Tuple, Any
import numpy as np

from biosignal_simulator.core.record import SignalRecord

class BiosignalExporter:
    """
    Exhaustive signal exporter supporting clinical binary and structured text formats.
    """

    @staticmethod
    def export_numpy(record: SignalRecord, path: str) -> None:
        """
        Export SignalRecord as a compressed NumPy archive (.npz).
        
        Parameters
        ----------
        record : SignalRecord
            The signal record object to export.
        path : str
            Target file path.
            
        Raises
        ------
        ValueError
            If the record contains invalid shapes or empty data.
        """
        if record.clean is None or len(record.clean) == 0:
            raise ValueError("Cannot export empty SignalRecord.")
            
        metadata = {
            'signal_type': record.signal_type,
            'fs': record.fs,
            'snr_db': record.snr_db,
            'signal_params': record.signal_params,
            'noise_params': record.noise_params,
            'metadata': record.metadata
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        np.savez_compressed(
            path,
            t=record.t,
            clean=record.clean,
            noisy=record.noisy,
            metadata_json=json.dumps(metadata),
            **record.noise_components
        )

    @staticmethod
    def export_dataframe(
        record: SignalRecord,
        column_prefix: str = "",
        include_time_index: bool = True,
        absolute_time: bool = False,
        start_datetime: Optional[datetime.datetime] = None
    ) -> object:
        """
        Convert SignalRecord to a pandas DataFrame.
        
        Parameters
        ----------
        record : SignalRecord
            The signal record object to convert.
        column_prefix : str
            Optional prefix for signal column names.
        include_time_index : bool
            If True, assigns time as index. Otherwise, time is a regular column.
        absolute_time : bool
            If True, generates absolute timestamps instead of relative seconds.
        start_datetime : datetime.datetime
            The reference datetime for absolute timestamps. Defaults to UTC now.
            
        Returns
        -------
        pd.DataFrame
            The constructed pandas DataFrame.
            
        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for DataFrame export. Install it with `pip install pandas`."
            )
            
        data = {}
        
        # Setup channel dimensions
        clean_data = record.clean
        noisy_data = record.noisy
        
        if clean_data.ndim == 1:
            data[f"{column_prefix}clean"] = clean_data
            data[f"{column_prefix}noisy"] = noisy_data
            for key, val in record.noise_components.items():
                data[f"{column_prefix}noise_{key}"] = val
        else:
            n_ch = clean_data.shape[0]
            for c in range(n_ch):
                data[f"{column_prefix}clean_ch{c}"] = clean_data[c]
                data[f"{column_prefix}noisy_ch{c}"] = noisy_data[c]
                for key, val in record.noise_components.items():
                    if val.ndim == 2:
                        data[f"{column_prefix}noise_{key}_ch{c}"] = val[c]
                    else:
                        data[f"{column_prefix}noise_{key}"] = val
                        
        df = pd.DataFrame(data)
        
        # Construct index
        if absolute_time:
            ref_dt = start_datetime or datetime.datetime.now(datetime.timezone.utc)
            delta = pd.to_timedelta(record.t, unit='s')
            timestamps = ref_dt + delta
            if include_time_index:
                df.index = timestamps
                df.index.name = 'timestamp'
            else:
                df.insert(0, 'timestamp', timestamps)
                if include_time_index:
                    df.set_index('timestamp', inplace=True)
        else:
            if include_time_index:
                df.index = record.t
                df.index.name = 'time_s'
            else:
                df.insert(0, 'time_s', record.t)
                
        return df

    @staticmethod
    def export_csv(
        record: SignalRecord,
        path: str,
        delimiter: str = ",",
        timestamp_format: str = "relative",
        start_datetime: Optional[datetime.datetime] = None,
        write_header_comments: bool = True
    ) -> None:
        """
        Export SignalRecord to a CSV file with clinical metadata embedded in comments.
        
        Parameters
        ----------
        record : SignalRecord
            SignalRecord to export.
        path : str
            CSV output file path.
        delimiter : str
            Delimiter symbol (e.g. "," or ";").
        timestamp_format : str
            Either "relative" (seconds) or "absolute" (ISO 8601).
        start_datetime : Optional[datetime.datetime]
            Starting time for absolute timestamp calculations.
        write_header_comments : bool
            If True, prefixes the file with JSON-formatted metadata comments.
        """
        is_abs = timestamp_format.lower() == "absolute"
        df = BiosignalExporter.export_dataframe(
            record=record,
            include_time_index=True,
            absolute_time=is_abs,
            start_datetime=start_datetime
        )
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if write_header_comments:
                metadata = {
                    'signal_type': record.signal_type,
                    'fs': record.fs,
                    'snr_db': record.snr_db,
                    'signal_params': record.signal_params,
                    'noise_params': record.noise_params,
                    'metadata': record.metadata,
                    'export_time': datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
                f.write(f"# Metadata: {json.dumps(metadata)}\n")
                
            df.to_csv(f, sep=delimiter)

    @staticmethod
    def export_hdf5(
        record: SignalRecord,
        path: str,
        compression: str = "gzip",
        compression_opts: int = 4
    ) -> None:
        """
        Export SignalRecord to a structured HDF5 file with compression.
        
        Parameters
        ----------
        record : SignalRecord
            SignalRecord to export.
        path : str
            HDF5 file path.
        compression : str
            Compression filter name (e.g. "gzip", "szip", None).
        compression_opts : int
            Compression strength (1 to 9).
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 export. Install it with `pip install h5py`."
            )
            
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with h5py.File(path, 'w') as f:
            # Global Attributes
            f.attrs['file_format'] = "BioSignalSimulator_HDF5"
            f.attrs['fs'] = record.fs
            f.attrs['signal_type'] = record.signal_type
            f.attrs['snr_db'] = record.snr_db if record.snr_db is not None else float('nan')
            f.attrs['export_timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # Create core group
            sig_grp = f.create_group('signal')
            
            # Chunk shape: process in blocks of 1000 or the signal size, whichever is smaller
            n_samples = len(record.t)
            chunk_t = (min(10000, n_samples),)
            
            if record.clean.ndim == 1:
                chunk_sig = (min(10000, n_samples),)
            else:
                chunk_sig = (record.clean.shape[0], min(1000, n_samples))
                
            sig_grp.create_dataset(
                't', data=record.t, chunks=chunk_t,
                compression=compression, compression_opts=compression_opts
            )
            sig_grp.create_dataset(
                'clean', data=record.clean, chunks=chunk_sig,
                compression=compression, compression_opts=compression_opts
            )
            sig_grp.create_dataset(
                'noisy', data=record.noisy, chunks=chunk_sig,
                compression=compression, compression_opts=compression_opts
            )
            
            # Noise subcomponents group
            noise_grp = f.create_group('noise')
            for name, component in record.noise_components.items():
                k_grp = noise_grp.create_group(name)
                if component.ndim == 1:
                    c_chunk = (min(10000, n_samples),)
                else:
                    c_chunk = (component.shape[0], min(1000, n_samples))
                k_grp.create_dataset(
                    'samples', data=component, chunks=c_chunk,
                    compression=compression, compression_opts=compression_opts
                )
                
            # Metadata configuration subtrees
            meta_grp = f.create_group('metadata')
            meta_grp.create_dataset('signal_params', data=json.dumps(record.signal_params))
            meta_grp.create_dataset('noise_params', data=json.dumps(record.noise_params))
            meta_grp.create_dataset('custom_meta', data=json.dumps(record.metadata))

    @staticmethod
    def export_edf(
        record: SignalRecord,
        path: str,
        record_duration_s: float = 1.0,
        patient_id: str = "Patient_01",
        recording_id: str = "Recording_01"
    ) -> None:
        """
        Export multi-channel SignalRecord to a fully compliant European Data Format (EDF) file.
        
        Parameters
        ----------
        record : SignalRecord
            The signal record object containing multi-channel signals.
        path : str
            The output EDF file path.
        record_duration_s : float
            Duration of an EDF record block in seconds. Default is 1.0 s.
        patient_id : str
            EDF Local Patient Identification (max 80 chars).
        recording_id : str
            EDF Local Recording Identification (max 80 chars).
        """
        # Determine signals list to export: clean, noisy, and noise components
        # Order: 
        # Channels 0..C-1 of Clean
        # Channels 0..C-1 of Noisy
        # Noise component channels
        signals_to_write: List[Tuple[str, np.ndarray]] = []
        
        c_shape = record.clean.shape
        n_channels = 1 if record.clean.ndim == 1 else c_shape[0]
        n_samples = len(record.t)
        
        for c in range(n_channels):
            lbl = f"CLEAN_CH{c}" if record.clean.ndim > 1 else "CLEAN"
            val = record.clean[c] if record.clean.ndim > 1 else record.clean
            signals_to_write.append((lbl, val))
            
        for c in range(n_channels):
            lbl = f"NOISY_CH{c}" if record.noisy.ndim > 1 else "NOISY"
            val = record.noisy[c] if record.noisy.ndim > 1 else record.noisy
            signals_to_write.append((lbl, val))
            
        for name, comp in record.noise_components.items():
            for c in range(n_channels):
                lbl = f"N_{name.upper()}_CH{c}" if comp.ndim > 1 else f"N_{name.upper()}"
                val = comp[c] if comp.ndim > 1 else comp
                signals_to_write.append((lbl, val))
                
        ns = len(signals_to_write)
        fs = record.fs
        
        # Samples per block record for each signal
        samples_per_rec = int(np.round(record_duration_s * fs))
        if samples_per_rec <= 0:
            raise ValueError("Record duration is too small for sampling rate.")
            
        # Determine total records
        num_records = n_samples // samples_per_rec
        if num_records == 0:
            num_records = 1
            # Adjust padding if needed
            
        actual_samples_needed = num_records * samples_per_rec
        
        # Build headers
        def ascii_pad(s: str, width: int) -> bytes:
            s_clean = "".join([c if ord(c) < 128 else " " for c in s])
            return f"{s_clean:<{width}}".encode('ascii')[:width]
            
        now = datetime.datetime.now(datetime.timezone.utc)
        startdate_str = now.strftime("%d.%m.%y")
        starttime_str = now.strftime("%H.%M.%S")
        
        header_bytes = 256 + 256 * ns
        
        h_version = ascii_pad("0", 8)
        h_patient = ascii_pad(patient_id, 80)
        h_record = ascii_pad(recording_id, 80)
        h_startdate = ascii_pad(startdate_str, 8)
        h_starttime = ascii_pad(starttime_str, 8)
        h_header_len = ascii_pad(str(header_bytes), 8)
        h_reserved = ascii_pad("BioSignalSimulator", 44)
        h_n_records = ascii_pad(str(num_records), 8)
        h_dur_record = ascii_pad(f"{record_duration_s:.6f}", 8)
        h_n_signals = ascii_pad(str(ns), 4)
        
        header_part1 = (
            h_version + h_patient + h_record + h_startdate + h_starttime +
            h_header_len + h_reserved + h_n_records + h_dur_record + h_n_signals
        )
        
        # Signal headers arrays
        s_labels = []
        s_transducers = []
        s_dimensions = []
        s_phys_mins = []
        s_phys_maxs = []
        s_dig_mins = []
        s_dig_maxs = []
        s_prefilters = []
        s_nsamples = []
        s_reserveds = []
        
        dig_min = -32768
        dig_max = 32767
        
        # Calculate limits and scaling for each signal
        signals_processed = []
        phys_limits = []
        
        for label, val in signals_to_write:
            # Resample or pad/trim to match actual samples needed
            if len(val) >= actual_samples_needed:
                truncated = val[:actual_samples_needed]
            else:
                truncated = np.pad(val, (0, actual_samples_needed - len(val)), mode='edge')
            signals_processed.append(truncated)
            
            p_min = float(np.min(truncated))
            p_max = float(np.max(truncated))
            if np.isclose(p_max, p_min):
                p_max = p_min + 1.0
                
            phys_limits.append((p_min, p_max))
            
            s_labels.append(ascii_pad(label, 16))
            s_transducers.append(ascii_pad("Simulation", 80))
            s_dimensions.append(ascii_pad("mV" if "N_" not in label else "a.u.", 8))
            s_phys_mins.append(ascii_pad(f"{p_min:.4f}", 8))
            s_phys_maxs.append(ascii_pad(f"{p_max:.4f}", 8))
            s_dig_mins.append(ascii_pad(str(dig_min), 8))
            s_dig_maxs.append(ascii_pad(str(dig_max), 8))
            s_prefilters.append(ascii_pad("None", 80))
            s_nsamples.append(ascii_pad(str(samples_per_rec), 8))
            s_reserveds.append(ascii_pad("", 32))
            
        header_part2 = (
            b"".join(s_labels) + b"".join(s_transducers) + b"".join(s_dimensions) +
            b"".join(s_phys_mins) + b"".join(s_phys_maxs) + b"".join(s_dig_mins) +
            b"".join(s_dig_maxs) + b"".join(s_prefilters) + b"".join(s_nsamples) +
            b"".join(s_reserveds)
        )
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(header_part1 + header_part2)
            
            # Write data records
            for r in range(num_records):
                for s_idx in range(ns):
                    start = r * samples_per_rec
                    end = start + samples_per_rec
                    segment = signals_processed[s_idx][start:end]
                    
                    p_min, p_max = phys_limits[s_idx]
                    
                    # Convert to digital 16-bit integers
                    scaled = dig_min + (segment - p_min) / (p_max - p_min) * (dig_max - dig_min)
                    int16_data = np.clip(np.round(scaled), dig_min, dig_max).astype(np.int16)
                    f.write(int16_data.tobytes())

    @staticmethod
    def export_edf_lite(record: SignalRecord, path: str, use_clean: bool = False) -> None:
        """Export the single-channel signal to a European Data Format (EDF) file.
        
        Generates a valid EDF file from either the clean or noisy signal.
        """
        signal_data = record.clean if use_clean else record.noisy
        
        # We only support single-channel for EDF-lite
        if signal_data.ndim > 1:
            signal_data = signal_data[0]  # Take first channel
            
        n_samples = len(signal_data)
        dur = n_samples / record.fs

        phys_min = float(np.min(signal_data))
        phys_max = float(np.max(signal_data))
        if np.isclose(phys_max, phys_min):
            phys_max = phys_min + 1.0

        digital_min = -32768
        digital_max = 32767
        
        # Scale to 16-bit integers
        scaled = digital_min + (signal_data - phys_min) / (phys_max - phys_min) * (digital_max - digital_min)
        int16_data = np.round(scaled).astype(np.int16)

        def ascii_pad(val: str, length: int) -> bytes:
            return f"{val:<{length}}".encode('ascii')[:length]

        # Header fields
        h_version = ascii_pad("0", 8)
        h_patient = ascii_pad("PatientX", 80)
        h_record = ascii_pad("RecordX", 80)
        h_startdate = ascii_pad("01.01.26", 8)
        h_starttime = ascii_pad("00.00.00", 8)
        h_bytes = ascii_pad("512", 8)  # 256 (standard) + 256 (1 signal)
        h_reserved44 = ascii_pad("", 44)
        h_n_records = ascii_pad("1", 8)
        h_dur_record = ascii_pad(f"{dur:.6f}"[:8], 8)
        h_n_signals = ascii_pad("1", 4)

        # Signal-specific fields (256 bytes total)
        s_label = ascii_pad(record.signal_type.upper(), 16)
        s_transducer = ascii_pad("Sensor", 80)
        s_dimension = ascii_pad("mV", 8)
        s_phys_min = ascii_pad(f"{phys_min:.4f}"[:8], 8)
        s_phys_max = ascii_pad(f"{phys_max:.4f}"[:8], 8)
        s_dig_min = ascii_pad(str(digital_min), 8)
        s_dig_max = ascii_pad(str(digital_max), 8)
        s_prefilter = ascii_pad("None", 80)
        s_samples = ascii_pad(str(n_samples), 8)
        s_reserved32 = ascii_pad("", 32)

        header = (
            h_version + h_patient + h_record + h_startdate + h_starttime +
            h_bytes + h_reserved44 + h_n_records + h_dur_record + h_n_signals +
            s_label + s_transducer + s_dimension + s_phys_min + s_phys_max +
            s_dig_min + s_dig_max + s_prefilter + s_samples + s_reserved32
        )

        with open(path, 'wb') as f:
            f.write(header)
            f.write(int16_data.tobytes())

    @staticmethod
    def export_wfdb(
        record: SignalRecord,
        path: str,
        format_code: int = 16
    ) -> None:
        """
        Export multi-channel SignalRecord to PhysioNet WFDB format (.hea & .dat files).
        
        Supports Format 16 (16-bit 2's complement) and Format 212 (packed 12-bit).
        
        Parameters
        ----------
        record : SignalRecord
            SignalRecord to export.
        path : str
            The base file path without extension (e.g. "data/rec_001").
        format_code : int
            WFDB binary format code (16 or 212).
        """
        if format_code not in [16, 212]:
            raise ValueError("Supported WFDB formats: 16 and 212.")
            
        c_shape = record.clean.shape
        n_channels = 1 if record.clean.ndim == 1 else c_shape[0]
        n_samples = len(record.t)
        fs = record.fs
        
        # Collect signals to write (clean, noisy, and noise components)
        signals: List[Tuple[str, np.ndarray]] = []
        for c in range(n_channels):
            lbl = f"CLEAN_CH{c}" if record.clean.ndim > 1 else "CLEAN"
            signals.append((lbl, record.clean[c] if record.clean.ndim > 1 else record.clean))
        for c in range(n_channels):
            lbl = f"NOISY_CH{c}" if record.noisy.ndim > 1 else "NOISY"
            signals.append((lbl, record.noisy[c] if record.noisy.ndim > 1 else record.noisy))
            
        ns = len(signals)
        
        # Paths
        base_dir = os.path.dirname(os.path.abspath(path))
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)
            
        base_name = os.path.basename(path)
        hea_path = f"{path}.hea"
        dat_path = f"{path}.dat"
        
        # Scaling limits
        # Format 16: range [-32768, 32767]
        # Format 212: range [-2048, 2047]
        if format_code == 16:
            adc_min, adc_max = -32768, 32767
            resolution = 16
        else:
            adc_min, adc_max = -2048, 2047
            resolution = 12
            
        phys_ranges = []
        digital_signals = []
        gains = []
        baselines = []
        
        for name, sig in signals:
            p_min = float(np.min(sig))
            p_max = float(np.max(sig))
            if np.isclose(p_max, p_min):
                p_max = p_min + 1.0
                
            phys_ranges.append((p_min, p_max))
            
            # Standard WFDB gain mapping
            gain = (adc_max - adc_min) / (p_max - p_min)
            baseline = int(np.round(adc_min - p_min * gain))
            
            gains.append(gain)
            baselines.append(baseline)
            
            # Scaling
            scaled = np.round(sig * gain + baseline)
            digital_sig = np.clip(scaled, adc_min, adc_max).astype(np.int32)
            digital_signals.append(digital_sig)
            
        # Write .dat binary file
        with open(dat_path, 'wb') as f_bin:
            if format_code == 16:
                # Interleaved samples as int16
                interleaved = np.empty(n_samples * ns, dtype=np.int16)
                for ch in range(ns):
                    interleaved[ch::ns] = digital_signals[ch].astype(np.int16)
                f_bin.write(interleaved.tobytes())
            elif format_code == 212:
                # Format 212: 3 bytes represent two 12-bit samples (little endian packed)
                # Channel samples are interleaved
                interleaved = np.empty(n_samples * ns, dtype=np.int32)
                for ch in range(ns):
                    interleaved[ch::ns] = digital_signals[ch]
                    
                # We need to process pairs of samples
                n_pairs = len(interleaved) // 2
                packed = np.zeros(n_pairs * 3, dtype=np.uint8)
                
                # Unsigned offset mapping for packing bits cleanly
                u_vals = (interleaved & 0xFFF).astype(np.uint32)
                
                u0 = u_vals[0::2]
                u1 = u_vals[1::2]
                
                packed[0::3] = u0 & 0xFF
                packed[1::3] = ((u0 >> 8) & 0x0F) | (((u1 >> 8) & 0x0F) << 4)
                packed[2::3] = u1 & 0xFF
                
                f_bin.write(packed.tobytes())
                
                # Write trailing sample if odd count
                if len(interleaved) % 2 != 0:
                    u_last = interleaved[-1] & 0xFFF
                    f_bin.write(bytes([u_last & 0xFF, (u_last >> 8) & 0x0F, 0]))
                    
        # Write .hea ascii header file
        with open(hea_path, 'w', encoding='ascii') as f_hea:
            f_hea.write(f"{base_name} {ns} {fs:.6f} {n_samples}\n")
            for idx in range(ns):
                name, _ = signals[idx]
                gain_val = gains[idx]
                base_val = baselines[idx]
                f_hea.write(
                    f"{base_name}.dat {format_code} {gain_val:.4f}(mV)/{base_val} "
                    f"{resolution} 0 {digital_signals[idx][0]} 0 0 {name}\n"
                )

    @staticmethod
    def export_mat(record: SignalRecord, path: str) -> None:
        """
        Export SignalRecord variables into a MATLAB Workspace struct file.
        
        Parameters
        ----------
        record : SignalRecord
            SignalRecord to export.
        path : str
            MATLAB output path.
        """
        try:
            from scipy.io import savemat
        except ImportError:
            raise ImportError(
                "scipy is required for MATLAB export. Install it with `pip install scipy`."
            )
            
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        record_struct = {
            't': record.t,
            'clean': record.clean,
            'noisy': record.noisy,
            'fs': record.fs,
            'signal_type': record.signal_type,
            'snr_db': record.snr_db if record.snr_db is not None else -999.0,
            'noise_components': record.noise_components,
            'signal_params': json.dumps(record.signal_params),
            'noise_params': json.dumps(record.noise_params),
            'custom_metadata': json.dumps(record.metadata)
        }
        
        savemat(path, {'record': record_struct})

    @staticmethod
    def export_json(
        record: SignalRecord,
        path: str,
        indent: Optional[int] = 4,
        compress: bool = False
    ) -> None:
        """
        Serialize SignalRecord values and configuration dictionaries to JSON.
        
        Parameters
        ----------
        record : SignalRecord
            SignalRecord to serialize.
        path : str
            JSON target path.
        indent : Optional[int]
            Indentation level for formatting.
        compress : bool
            If True, saves as compressed GZIP format (.json.gz).
        """
        export_dict = {
            'signal_type': record.signal_type,
            'fs': record.fs,
            'snr_db': record.snr_db,
            't': record.t.tolist(),
            'clean': record.clean.tolist(),
            'noisy': record.noisy.tolist(),
            'noise_components': {k: v.tolist() for k, v in record.noise_components.items()},
            'signal_params': record.signal_params,
            'noise_params': record.noise_params,
            'metadata': record.metadata
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if compress:
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                json.dump(export_dict, f, indent=indent)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(export_dict, f, indent=indent)


class BiosignalImporter:
    """
    Symmetrical signal importer to load back SignalRecord objects from saved formats.
    """

    @staticmethod
    def import_numpy(path: str) -> SignalRecord:
        """
        Import a SignalRecord from a compressed NumPy archive (.npz).
        
        Parameters
        ----------
        path : str
            Path to the NumPy archive.
            
        Returns
        -------
        SignalRecord
            The reconstructed SignalRecord.
        """
        with np.load(path, allow_pickle=True) as data:
            t = data['t']
            clean = data['clean']
            noisy = data['noisy']
            meta_str = str(data['metadata_json'])
            meta_dict = json.loads(meta_str)
            
            # Load other arrays as noise components
            noise_components = {}
            reserved_keys = ['t', 'clean', 'noisy', 'metadata_json']
            for key in data.files:
                if key not in reserved_keys:
                    noise_components[key] = data[key]
                    
        return SignalRecord(
            t=t,
            clean=clean,
            noisy=noisy,
            fs=meta_dict['fs'],
            signal_type=meta_dict['signal_type'],
            snr_db=meta_dict['snr_db'],
            noise_components=noise_components,
            signal_params=meta_dict.get('signal_params', {}),
            noise_params=meta_dict.get('noise_params', {}),
            metadata=meta_dict.get('metadata', {})
        )

    @staticmethod
    def import_csv(path: str, delimiter: str = ",") -> SignalRecord:
        """
        Import a SignalRecord from a CSV file, parsing embedded JSON comment headers.
        
        Parameters
        ----------
        path : str
            The CSV file path.
        delimiter : str
            Separator used in the CSV.
            
        Returns
        -------
        SignalRecord
            The reconstructed SignalRecord.
        """
        metadata = {}
        header_lines = []
        
        # Read header lines beginning with '#'
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    header_lines.append(line.lstrip('#').strip())
                else:
                    break
                    
        # Parse metadata
        for item in header_lines:
            if "BioSignal_Simulator_Metadata_Start" in item or "BioSignal_Simulator_Metadata_End" in item:
                continue
            # Strip potential "Metadata:" prefix
            if item.startswith("Metadata:"):
                item = item[len("Metadata:"):].strip()
            try:
                metadata = json.loads(item)
                break
            except json.JSONDecodeError:
                continue
                
        # Load CSV using pandas
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for CSV import. Install it with `pip install pandas`."
            )
            
        # Re-read CSV skipping comment rows
        df = pd.read_csv(path, sep=delimiter, comment='#')
        
        # Extract time
        time_col = 'time_s' if 'time_s' in df.columns else df.columns[0]
        t_raw = df[time_col].values
        
        # If t_raw contains strings (absolute ISO 8601 timestamps), convert back to float relative seconds
        if len(t_raw) > 0 and isinstance(t_raw[0], str):
            try:
                times_parsed = pd.to_datetime(t_raw)
                t = (times_parsed - times_parsed[0]).total_seconds().values
            except Exception:
                t = np.arange(len(t_raw)) / metadata.get('fs', 250.0)
        else:
            t = t_raw.astype(np.float64)
        
        # Read channels
        clean_cols = [c for c in df.columns if c.startswith('clean')]
        noisy_cols = [c for c in df.columns if c.startswith('noisy')]
        
        if len(clean_cols) == 1:
            clean = df[clean_cols[0]].values
            noisy = df[noisy_cols[0]].values
        else:
            # Multi-channel
            # Sort by channel index
            clean_cols_sorted = sorted(clean_cols, key=lambda x: int(x.split('_ch')[-1]) if '_ch' in x else 0)
            noisy_cols_sorted = sorted(noisy_cols, key=lambda x: int(x.split('_ch')[-1]) if '_ch' in x else 0)
            
            clean = np.array([df[c].values for c in clean_cols_sorted])
            noisy = np.array([df[c].values for c in noisy_cols_sorted])
            
        # Read noise components
        noise_components = {}
        noise_cols = [c for c in df.columns if 'noise_' in c]
        
        # Group multi-channel noise components
        component_names = set()
        for col in noise_cols:
            # col looks like: noise_{component_name} or noise_{component_name}_ch{idx}
            parts = col.split('noise_')[-1]
            c_name = parts.split('_ch')[0]
            component_names.add(c_name)
            
        for name in component_names:
            matching = [c for c in noise_cols if f"noise_{name}" in c]
            if len(matching) == 1:
                noise_components[name] = df[matching[0]].values
            else:
                # Multi-channel component
                sorted_match = sorted(matching, key=lambda x: int(x.split('_ch')[-1]) if '_ch' in x else 0)
                noise_components[name] = np.array([df[c].values for c in sorted_match])
                
        return SignalRecord(
            t=t,
            clean=clean,
            noisy=noisy,
            fs=metadata.get('fs', 250.0),
            signal_type=metadata.get('signal_type', 'unknown'),
            snr_db=metadata.get('snr_db', None),
            noise_components=noise_components,
            signal_params=metadata.get('signal_params', {}),
            noise_params=metadata.get('noise_params', {}),
            metadata=metadata.get('metadata', {})
        )

    @staticmethod
    def import_hdf5(path: str) -> SignalRecord:
        """
        Import a SignalRecord from an HDF5 archive.
        
        Parameters
        ----------
        path : str
            The HDF5 file path.
            
        Returns
        -------
        SignalRecord
            The reconstructed SignalRecord.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 import. Install it with `pip install h5py`."
            )
            
        with h5py.File(path, 'r') as f:
            fs = f.attrs['fs']
            signal_type = f.attrs['signal_type']
            snr_db = f.attrs['snr_db']
            if np.isnan(snr_db):
                snr_db = None
                
            t = f['signal/t'][:]
            clean = f['signal/clean'][:]
            noisy = f['signal/noisy'][:]
            
            noise_components = {}
            for name in f['noise']:
                obj = f[f'noise/{name}']
                if isinstance(obj, h5py.Dataset):
                    noise_components[name] = obj[:]
                else:
                    noise_components[name] = obj['samples'][:]
                
            sig_params = json.loads(f['metadata/signal_params'][()]) if 'metadata/signal_params' in f else {}
            noise_params = json.loads(f['metadata/noise_params'][()]) if 'metadata/noise_params' in f else {}
            custom_meta = json.loads(f['metadata/custom_meta'][()]) if 'metadata/custom_meta' in f else {}
            
        return SignalRecord(
            t=t,
            clean=clean,
            noisy=noisy,
            fs=fs,
            signal_type=signal_type,
            snr_db=snr_db,
            noise_components=noise_components,
            signal_params=sig_params,
            noise_params=noise_params,
            metadata=custom_meta
        )

    @staticmethod
    def import_edf(path: str) -> SignalRecord:
        """
        Import and reconstruct a SignalRecord from a European Data Format (EDF) file.
        
        Parameters
        ----------
        path : str
            The binary EDF file path.
            
        Returns
        -------
        SignalRecord
            Reconstructed SignalRecord containing parsed clean and noisy signals.
        """
        with open(path, 'rb') as f:
            # Parse header record part 1
            h_version = f.read(8).decode('ascii').strip()
            h_patient = f.read(80).decode('ascii').strip()
            h_record = f.read(80).decode('ascii').strip()
            h_startdate = f.read(8).decode('ascii').strip()
            h_starttime = f.read(8).decode('ascii').strip()
            h_header_len = int(f.read(8).decode('ascii').strip())
            h_reserved = f.read(44).decode('ascii').strip()
            num_records = int(f.read(8).decode('ascii').strip())
            record_dur = float(f.read(8).decode('ascii').strip())
            ns = int(f.read(4).decode('ascii').strip())
            
            # Parse signal headers
            s_labels = [f.read(16).decode('ascii').strip() for _ in range(ns)]
            s_transducers = [f.read(80).decode('ascii').strip() for _ in range(ns)]
            s_dimensions = [f.read(8).decode('ascii').strip() for _ in range(ns)]
            s_phys_mins = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            s_phys_maxs = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            s_dig_mins = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            s_dig_maxs = [float(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            s_prefilters = [f.read(80).decode('ascii').strip() for _ in range(ns)]
            s_nsamples = [int(f.read(8).decode('ascii').strip()) for _ in range(ns)]
            s_reserveds = [f.read(32).decode('ascii').strip() for _ in range(ns)]
            
            # Jump to end of header to ensure alignment
            f.seek(h_header_len)
            
            # Total samples in each channel
            total_samples = [num_records * s_nsamples[s] for s in range(ns)]
            
            # Pre-allocate signal buffers
            digital_data = [np.zeros(total_samples[s], dtype=np.int16) for s in range(ns)]
            
            # Read records block-by-block
            for r in range(num_records):
                for s in range(ns):
                    n_s_bytes = s_nsamples[s] * 2
                    raw_block = f.read(n_s_bytes)
                    if len(raw_block) < n_s_bytes:
                        # File truncated, break out
                        break
                    block_data = np.frombuffer(raw_block, dtype=np.int16)
                    start_idx = r * s_nsamples[s]
                    end_idx = start_idx + len(block_data)
                    digital_data[s][start_idx:end_idx] = block_data
                    
        # Convert digital data back to physical
        physical_data = []
        for s in range(ns):
            dig = digital_data[s].astype(np.float64)
            p_min, p_max = s_phys_mins[s], s_phys_maxs[s]
            d_min, d_max = s_dig_mins[s], s_dig_maxs[s]
            
            phys = p_min + (dig - d_min) / (d_max - d_min) * (p_max - p_min)
            physical_data.append(phys)
            
        # Reconstruct clean, noisy, and noise components
        # Match labels
        clean_channels = []
        noisy_channels = []
        noise_components = {}
        
        for idx, lbl in enumerate(s_labels):
            lbl_upper = lbl.upper()
            if "CLEAN" in lbl_upper:
                clean_channels.append((lbl_upper, physical_data[idx]))
            elif "NOISY" in lbl_upper:
                noisy_channels.append((lbl_upper, physical_data[idx]))
            elif lbl_upper.startswith("N_"):
                # Component label format: N_GAUSSIAN or N_GAUSSIAN_CH0
                parts = lbl_upper[2:]
                comp_name = parts.split("_CH")[0].lower()
                if comp_name not in noise_components:
                    noise_components[comp_name] = []
                noise_components[comp_name].append(physical_data[idx])
                
        # Stack channels
        def stack_signals(chan_list):
            if not chan_list:
                return np.array([])
            # Sort by name/index
            chan_list_sorted = sorted(chan_list, key=lambda x: x[0])
            arrays = [x[1] for x in chan_list_sorted]
            stacked = np.stack(arrays)
            return stacked[0] if len(arrays) == 1 else stacked
            
        clean_sig = stack_signals(clean_channels)
        noisy_sig = stack_signals(noisy_channels)
        
        # Symmetrical default for legacy single-channel EDF files where labels are not labeled clean/noisy
        if clean_sig.size == 0 and noisy_sig.size == 0 and physical_data:
            clean_sig = physical_data[0]
            noisy_sig = physical_data[0]
        
        for name in list(noise_components.keys()):
            stacked_comp = np.stack(noise_components[name])
            noise_components[name] = stacked_comp[0] if len(stacked_comp) == 1 else stacked_comp
            
        # Sampling rate is parsed from first signal samples count per record divided by record duration
        fs = s_nsamples[0] / record_dur
        n_pts = len(physical_data[0])
        t = np.arange(n_pts) / fs
        
        # Attempt to deduce signal type from labels
        sig_type = "unknown"
        for lbl in s_labels:
            lbl_low = lbl.lower()
            for s_t in ["ecg", "eeg", "emg", "ppg", "eda", "resp"]:
                if s_t in lbl_low:
                    sig_type = s_t
                    break
                    
        return SignalRecord(
            t=t,
            clean=clean_sig,
            noisy=noisy_sig,
            fs=fs,
            signal_type=sig_type,
            snr_db=None,
            noise_components=noise_components
        )

    @staticmethod
    def import_wfdb(path: str) -> SignalRecord:
        """
        Import a SignalRecord from a PhysioNet WFDB format (.hea & .dat files).
        
        Parameters
        ----------
        path : str
            Base file path without extension (e.g. "data/rec_001").
            
        Returns
        -------
        SignalRecord
            Reconstructed SignalRecord.
        """
        hea_path = f"{path}.hea"
        dat_path = f"{path}.dat"
        
        if not os.path.exists(hea_path) or not os.path.exists(dat_path):
            raise FileNotFoundError("Header (.hea) or Data (.dat) file is missing.")
            
        # Parse .hea file
        with open(hea_path, 'r', encoding='ascii') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        # Parse first line: record_name ns fs n_samples [starttime] [startdate]
        first_line_parts = lines[0].split()
        ns = int(first_line_parts[1])
        fs = float(first_line_parts[2])
        n_samples = int(first_line_parts[3])
        
        # Parse signal channels details
        channels_info = []
        for idx in range(1, ns + 1):
            parts = lines[idx].split()
            # File format code is parts[1]
            fmt_code = int(parts[1])
            # Gain and baseline: e.g. 200(mV)/0 or 200/0
            gain_baseline = parts[2]
            if '/' in gain_baseline:
                gain_part, base_part = gain_baseline.split('/')
                baseline = int(base_part)
                # Gain might have units in parenthesis
                if '(' in gain_part:
                    gain = float(gain_part.split('(')[0])
                else:
                    gain = float(gain_part)
            else:
                gain = float(gain_baseline)
                baseline = 0
                
            resolution = int(parts[3])
            init_val = int(parts[5])
            lbl = parts[-1] if len(parts) >= 9 else f"CH{idx-1}"
            
            channels_info.append({
                'format': fmt_code,
                'gain': gain,
                'baseline': baseline,
                'label': lbl
            })
            
        fmt = channels_info[0]['format']
        
        # Read binary file
        with open(dat_path, 'rb') as f_bin:
            binary_content = f_bin.read()
            
        digital_signals = [np.zeros(n_samples, dtype=np.int32) for _ in range(ns)]
        
        if fmt == 16:
            # 16-bit signed interleaved samples
            samples = np.frombuffer(binary_content, dtype=np.int16)
            # Reshape
            total_elements = n_samples * ns
            if len(samples) < total_elements:
                warnings.warn("WFDB binary file is truncated.")
                n_samples_read = len(samples) // ns
                samples = samples[:n_samples_read * ns]
            else:
                n_samples_read = n_samples
                
            for ch in range(ns):
                digital_signals[ch][:n_samples_read] = samples[ch::ns]
        elif fmt == 212:
            # Packed 12-bit format.
            # Three bytes represent two 12-bit samples.
            bytes_array = np.frombuffer(binary_content, dtype=np.uint8)
            n_pairs = len(bytes_array) // 3
            
            b0 = bytes_array[0::3].astype(np.int32)
            b1 = bytes_array[1::3].astype(np.int32)
            b2 = bytes_array[2::3].astype(np.int32)
            
            u0 = b0 | ((b1 & 0x0F) << 8)
            u1 = b2 | (((b1 >> 4) & 0x0F) << 8)
            
            # Map 12-bit unsigned to signed offset (WFDB Format 212 represents signed 12-bit)
            val0 = np.where(u0 >= 2048, u0 - 4096, u0)
            val1 = np.where(u1 >= 2048, u1 - 4096, u1)
            
            # Interleave into single array
            interleaved = np.empty(n_pairs * 2, dtype=np.int32)
            interleaved[0::2] = val0
            interleaved[1::2] = val1
            
            n_samples_read = len(interleaved) // ns
            for ch in range(ns):
                digital_signals[ch][:n_samples_read] = interleaved[ch:n_samples_read*ns:ns]
        else:
            raise NotImplementedError(f"WFDB Format {fmt} is not supported.")
            
        # Reconstruct physical signals
        physical_data = []
        for ch in range(ns):
            info = channels_info[ch]
            gain = info['gain']
            baseline = info['baseline']
            phys = (digital_signals[ch] - baseline) / gain
            physical_data.append(phys)
            
        # Extract channels into SignalRecord structure
        clean_channels = []
        noisy_channels = []
        noise_components = {}
        
        for idx, info in enumerate(channels_info):
            lbl_upper = info['label'].upper()
            phys = physical_data[idx]
            
            if "CLEAN" in lbl_upper:
                clean_channels.append((lbl_upper, phys))
            elif "NOISY" in lbl_upper:
                noisy_channels.append((lbl_upper, phys))
            elif lbl_upper.startswith("N_"):
                parts = lbl_upper.split("N_")[-1]
                comp_name = parts.split("_CH")[0].lower()
                if comp_name not in noise_components:
                    noise_components[comp_name] = []
                noise_components[comp_name].append(phys)
                
        # Stack channels
        def stack_signals(chan_list):
            if not chan_list:
                return np.array([])
            chan_list_sorted = sorted(chan_list, key=lambda x: x[0])
            arrays = [x[1] for x in chan_list_sorted]
            stacked = np.stack(arrays)
            return stacked[0] if len(arrays) == 1 else stacked
            
        clean_sig = stack_signals(clean_channels)
        noisy_sig = stack_signals(noisy_channels)
        
        for name in list(noise_components.keys()):
            stacked_comp = np.stack(noise_components[name])
            noise_components[name] = stacked_comp[0] if len(stacked_comp) == 1 else stacked_comp
            
        t = np.arange(n_samples) / fs
        
        sig_type = "unknown"
        for info in channels_info:
            lbl_low = info['label'].lower()
            for s_t in ["ecg", "eeg", "emg", "ppg", "eda", "resp"]:
                if s_t in lbl_low:
                    sig_type = s_t
                    break
                    
        return SignalRecord(
            t=t,
            clean=clean_sig,
            noisy=noisy_sig,
            fs=fs,
            signal_type=sig_type,
            snr_db=None,
            noise_components=noise_components
        )

    @staticmethod
    def import_mat(path: str) -> SignalRecord:
        """
        Import a SignalRecord from a MATLAB Workspace workspace struct file (.mat).
        
        Parameters
        ----------
        path : str
            MATLAB file path.
            
        Returns
        -------
        SignalRecord
            Reconstructed SignalRecord.
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError(
                "scipy is required for MATLAB import. Install it with `pip install scipy`."
            )
            
        mat_data = loadmat(path)
        if 'record' not in mat_data:
            raise KeyError("MATLAB workspace file does not contain 'record' structure.")
            
        rec_struct = mat_data['record'][0, 0]
        
        t = rec_struct['t'].squeeze()
        clean = rec_struct['clean']
        noisy = rec_struct['noisy']
        
        # Ensure dimensions match
        if clean.ndim == 2 and clean.shape[1] == len(t) and clean.shape[0] != len(t):
            pass  # correctly shaped
        elif clean.ndim == 2 and clean.shape[0] == len(t):
            clean = clean.T
            noisy = noisy.T
            
        fs = float(rec_struct['fs'][0, 0])
        signal_type = str(rec_struct['signal_type'][0])
        snr_db_val = float(rec_struct['snr_db'][0, 0])
        snr_db = None if snr_db_val == -999.0 else snr_db_val
        
        # Extract noise components
        noise_components = {}
        comp_struct = rec_struct['noise_components'][0, 0]
        for field in comp_struct.dtype.names:
            arr = comp_struct[field]
            if arr.ndim == 2 and arr.shape[0] == len(t):
                arr = arr.T
            noise_components[field] = arr.squeeze()
            
        # Parse params
        sig_params = json.loads(str(rec_struct['signal_params'][0])) if 'signal_params' in rec_struct.dtype.names else {}
        noise_params = json.loads(str(rec_struct['noise_params'][0])) if 'noise_params' in rec_struct.dtype.names else {}
        custom_meta = json.loads(str(rec_struct['custom_metadata'][0])) if 'custom_metadata' in rec_struct.dtype.names else {}
        
        return SignalRecord(
            t=t,
            clean=clean,
            noisy=noisy,
            fs=fs,
            signal_type=signal_type,
            snr_db=snr_db,
            noise_components=noise_components,
            signal_params=sig_params,
            noise_params=noise_params,
            metadata=custom_meta
        )

    @staticmethod
    def import_json(path: str, compressed: bool = False) -> SignalRecord:
        """
        Import and reconstruct a SignalRecord from a JSON file.
        
        Parameters
        ----------
        path : str
            JSON target path.
        compressed : bool
            If True, reads from gzip format (.json.gz).
            
        Returns
        -------
        SignalRecord
            Reconstructed SignalRecord.
        """
        if compressed or path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        t = np.array(data['t'])
        clean = np.array(data['clean'])
        noisy = np.array(data['noisy'])
        
        noise_components = {}
        for k, v in data.get('noise_components', {}).items():
            noise_components[k] = np.array(v)
            
        return SignalRecord(
            t=t,
            clean=clean,
            noisy=noisy,
            fs=data['fs'],
            signal_type=data['signal_type'],
            snr_db=data['snr_db'],
            noise_components=noise_components,
            signal_params=data.get('signal_params', {}),
            noise_params=data.get('noise_params', {}),
            metadata=data.get('metadata', {})
        )
