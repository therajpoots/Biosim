from biosignal_simulator.signals.ecg import ECGGenerator
from biosignal_simulator.signals.eeg import EEGGenerator
from biosignal_simulator.signals.emg import EMGGenerator
from biosignal_simulator.signals.ppg import (
    PPGGenerator,
    make_ppg_normal,
    make_ppg_tachycardia,
    make_ppg_bradycardia,
    make_vpg,
    make_apg,
    make_ppg_motion_artifact,
    make_ppg_light_leakage
)
from biosignal_simulator.signals.eda import EDAGenerator
from biosignal_simulator.signals.resp import RespGenerator

