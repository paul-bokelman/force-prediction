from typing import Callable

# ---------------------------------- shared ---------------------------------- #
mvc_levels = [5, 10, 20, 40, 60] # the total list of mvc levels to be considered (missing 100)
data_dir = "processing/data" # directory containing all subject data
dataset_path = f"{data_dir}/dataset.pkl" # path to the dataset file, which is modified by the various data processing steps

# ------------------------------- sanitization ------------------------------- #
manually_selected_valid_trials_path = f"{data_dir}/valid-trials.json"
sanitization_output_dir = "processing/out" # directory for saving sanitized data raster plots
min_neurons = 6 # minimum number of neurons required for a trial to be considered valid (N)
gmd_maximum_nil_activation_windows = {
    5: 10000,
    10: 10000,
    20: 15000,
    40: 25000,
    60: 25000,
    100: 10000
}  # maximum window size for sequence of empty neuron cols to be acceptable in time-steps (by mvc)

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
processed_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/processed" # directory containing subject-specific processed data
raw_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/raw" # directory containing subject-specific raw data
min_spike_interval = 33 # minimum amount of time between spikes in milliseconds

# ------------------------------- preprocessing ------------------------------ #
sequence_length = 200 # length of the sliding window for preprocessing
stride = sequence_length // 2 # stride for sliding window
train_split_percentage, val_split_percentage, test_split_percentage = 0.8, 0.1, 0.1 # split percentages for training, validation, and test sets