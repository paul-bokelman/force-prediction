from typing import Callable

# ---------------------------------- general --------------------------------- #
mvc_levels = [5, 10, 20, 40, 60, 100] # the total list of mvc levels to be considered

# ------------------------------- sanitization ------------------------------- #
min_neurons = 6 # minimum number of neurons required for a trial to be considered valid (N)
spike_smoothing_window = 500 # the window size for the median filter used to smooth the spike data
md_flat_region_detection_step = 0.25 # the step percentage for the flat region detection algorithm in the measurement decorrelation handler
md_flat_region_error_threshold = 0.1 # the pass/fail threshold for identifying flat regions in the measurement decorrelation handler

# -------------------------------- conversion -------------------------------- #
data_dir = "data" # directory containing all subject data
processed_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/processed" # directory containing subject-specific processed data
raw_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/raw" # directory containing subject-specific raw data
base_output_dir = "out" # directory to save output files
user_output_dir: Callable[[str], str] = lambda name: f"{base_output_dir}/{name}" # directory to save user-specific output files
min_spike_interval = 33 # minimum amount of time between spikes in milliseconds