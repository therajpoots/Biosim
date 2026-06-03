from biosignal_simulator.utils.validation import (
    validate_config,
    validate_signal,
    validate_physiological_bounds,
    generate_validation_report_html,
    SignalIntegrityChecker,
    ValidationReport
)
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
    generate_interactive_html_dashboard
)

