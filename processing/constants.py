from typing import Callable
from globals import constants

# ---------------------------------- shared ---------------------------------- #
mvc_levels = [5, 10, 20, 40, 60] # the total list of mvc levels to be considered (missing 100)
data_dir = "processing/data" # directory containing all subject data
subject_data_dir = f"{data_dir}/subjects" # directory containing all subject data
subject_mappings_path = f"{data_dir}/subject-mappings.json" # path to the subject mappings file
dataset_path = f"{data_dir}/dataset.pkl" # dataset path that is manipulated by conversion and sanitization

# ------------------------------- sanitization ------------------------------- #
manually_selected_valid_trials_path = f"{data_dir}/valid-trials.json"
sanitization_output_dir = "processing/out" # directory for saving sanitized data raster plots
min_neurons = 6 # minimum number of neurons required for a trial to be considered valid (N)

# -------------------------------- conversion -------------------------------- #
processed_data_dir: Callable[[str], str] = lambda name: f"{subject_data_dir}/{name}/processed" # directory containing subject-specific processed data
raw_data_dir: Callable[[str], str] = lambda name: f"{subject_data_dir}/{name}/raw" # directory containing subject-specific raw data
min_spike_interval = 33 # minimum amount of time between spikes in milliseconds

# ------------------------------- preprocessing ------------------------------ #
preprocessed_dataset_path = lambda identifier: f"{data_dir}/preprocessed/{identifier}.pkl" # path to a specific preprocessed dataset file

bin_size = int(constants.sampling_frequency / 100) # bin size for binning the neuron data (2048 -> 1 second -> 1000ms*0.01 = 10ms)
exponential_decay_lifetime = 20 # memory decay rate for neuronal spike data (in milliseconds), 20 -> signal gone in 20ms
size_amplification_factor = 0.3 # amplification factor for increasing neuron size (based on first activation) ln(x)^fac