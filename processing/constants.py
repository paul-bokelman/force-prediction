from typing import Callable

# ---------------------------------- general --------------------------------- #
mvc_levels = [5, 10, 20, 40, 60, 100] # the total list of mvc levels to be considered

# ------------------------------- sanitization ------------------------------- #
min_neurons = 6 # minimum number of neurons required for a trial to be considered valid (N)
spike_smoothing_window = 500 # the window size for the median filter used to smooth the spike data
md_region_window = 2000 # the window size for region gradient calculation and comparison
md_median_gradient_scaler = 3 # the scaling factor for the median gradient used in flat region detection
md_region_tolerance = 0.7 # the relative tolerance for flat region gradient comparison
md_symmetric_zeros_tolerance = 0.07 # the relative tolerance for areas in symmetries that are close to zero (relative to max value)
gmd_maximum_nil_activation_windows = {
    5: 10000,
    10: 10000,
    20: 15000,
    40: 25000,
    60: 25000,
    100: 10000
}  # maximum window size for sequence of empty neuron cols to be acceptable in time-steps (by mvc)
md_mvc_gradient_scaling = {
    5: 5,
    10: 3,
    20: 2,
    40: 2,
    60: 2,
    100: 1
} # the scaling factor for artificial gradient computation based on mvc level

manually_purged_trials = [
    'nastos-jr.10.2',
    'nastos-jr.10.3',
    'nikos.60.2',
    'nikos.60.3',
    'dim_tselios.20.1',
    'dim_tselios.20.3',
    'dim_tselios.60.1',
    'iatroudelis.5.3',
    'iatroudelis.10.3',
    'dwrotheos.60.3',
    'anestis.40.3',
    'manos.10.1',
    'leonidas.5.1',
    'leonidas.5.2',
    'leonidas.5.3',
    'leonidas.10.1',
    'leonidas.10.2',
    'leonidas.20.3',
    'thanasis.40.1',
    'giannatos.40.2',
    'mpardas.5.2',
    'konstantopoulos.40.1',
    'paliaxanis.10.3',
    'paliaxanis.20.2',
    'pavlidis.60.3',
    'mavrokefalidis.10.2',
    'mavrokefalidis.60.3'
]

# -------------------------------- conversion -------------------------------- #
data_dir = "data" # directory containing all subject data
processed_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/processed" # directory containing subject-specific processed data
raw_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/raw" # directory containing subject-specific raw data
base_output_dir = "out" # directory to save output files
user_output_dir: Callable[[str], str] = lambda name: f"{base_output_dir}/{name}" # directory to save user-specific output files
min_spike_interval = 33 # minimum amount of time between spikes in milliseconds

# ------------------------------- preprocessing ------------------------------ #

sequence_length = 200 # length of the sliding window for preprocessing
stride = sequence_length // 2 # stride for sliding window
train_split_percentage, val_split_percentage, test_split_percentage = 0.8, 0.1, 0.1 # split percentages for training, validation, and test sets