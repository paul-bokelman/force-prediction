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
md_mvc_gradient_scaling = {
    5: 5,
    10: 3,
    20: 2,
    40: 2,
    60: 2,
    100: 1
} # the scaling factor for artificial gradient computation based on mvc level

# -------------------------------- conversion -------------------------------- #
data_dir = "data" # directory containing all subject data
processed_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/processed" # directory containing subject-specific processed data
raw_data_dir: Callable[[str], str] = lambda name: f"{data_dir}/{name}/raw" # directory containing subject-specific raw data
base_output_dir = "out" # directory to save output files
user_output_dir: Callable[[str], str] = lambda name: f"{base_output_dir}/{name}" # directory to save user-specific output files
min_spike_interval = 33 # minimum amount of time between spikes in milliseconds