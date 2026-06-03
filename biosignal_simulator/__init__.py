from biosignal_simulator.core import (
    BaseSignal,
    BaseNoiseModel,
    SignalRecord,
    ECGConfig,
    EEGConfig,
    EMGConfig,
    PPGConfig,
    EDAConfig,
    RespConfig,
    GaussianNoiseConfig,
    ColoredNoiseConfig,
    BaselineWanderConfig,
    PowerlineNoiseConfig,
    MotionArtifactConfig,
    ElectrodeNoiseConfig,
    EMGArtifactConfig,
    ImpulseNoiseConfig,
    QuantizationNoiseConfig,
    CrosstalkNoiseConfig,
    SensorDetachmentConfig,
    ElectrodeDisplacementConfig,
    LightLeakageConfig,
    PacketLossConfig,
    ConfigSerializer,
    BenchmarkSuite
)
from biosignal_simulator.signals import (
    ECGGenerator,
    EEGGenerator,
    EMGGenerator,
    PPGGenerator,
    EDAGenerator,
    RespGenerator,
    make_ppg_normal,
    make_ppg_tachycardia,
    make_ppg_bradycardia,
    make_vpg,
    make_apg,
    make_ppg_motion_artifact,
    make_ppg_light_leakage
)
from biosignal_simulator.noise import (
    GaussianNoise,
    ColoredNoise,
    PinkNoise,
    BrownNoise,
    BlueNoise,
    VioletNoise,
    BaselineWander,
    PowerlineNoise,
    MotionArtifact,
    ElectrodeNoise,
    EMGArtifact,
    ImpulseNoise,
    QuantizationNoise,
    CrosstalkNoise,
    SensorDetachmentNoise,
    ElectrodeDisplacementNoise,
    LightLeakageNoise,
    PacketLossNoise
)
from biosignal_simulator.composer import (
    SignalMixer,
    SNRController,
    CompositeSNRController,
    NoiseScheduler,
    StepSchedule,
    RampSchedule,
    PeriodicSchedule,
    ArtifactInjector
)
from biosignal_simulator.io import BiosignalExporter, BiosignalImporter
from biosignal_simulator.utils import (
    plot_record,
    plot_psd_comparison,
    plot_noise_characterization,
    plot_snr_sweep,
    plot_filter_response
)

list_generators = BaseSignal.list_generators
get_generator = BaseSignal.get_generator

try:
    from importlib.metadata import version as _version
    __version__ = _version("biosignal-simulator")
except Exception:
    __version__ = "0.1.1"

