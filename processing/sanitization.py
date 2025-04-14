from typing import Union, cast
from processing.types import ISIStatistics
import os
import shutil
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import globals.constants
from globals.visualization import visualize_trial
from globals.utils import Log, format_subject
import processing.constants as constants

class Sanitization:
    """Remove or correct any errors in the data. This includes removing outliers, filling in missing values, and correcting any other errors in the data. Some entries are not recoverable and will be "purged"."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_df = df.copy()
        self._purge_insufficient_neuron_data() # N
        self._compute_global_statistics()

    @staticmethod
    def _compute_neuron_isi(neuron: np.ndarray) -> Union[np.ndarray, None]:
        activation_indices = np.where(neuron == 1)[0] # get indices of activated neurons
        if len(activation_indices) > 1: # compute ISI if there are more than 1 activation
            return np.diff(activation_indices) # return differences between activations (distance)
        else:
            Log.warn("Neuron does not have enough activations to compute ISI.")
            return None
        
    @staticmethod
    def _compute_isi_statistics(neuron_set: np.ndarray) -> ISIStatistics:
        """Computes the mean and standard deviation of the inter-spike intervals (ISI) for the given neuron data."""
        mean_distances = [] # distances between activations for each neuron
        max_distances = [] # max distances between activations for each neuron

        # compute isi mean and max for each neuron in the sets
        for neuron in neuron_set:
            # skip the neuron if it contains no activations
            if np.isscalar(neuron) or np.all(neuron == 0) or neuron is None:
                continue
            
            isi = Sanitization._compute_neuron_isi(neuron)

            # skip if there is not enough data to compute ISI
            if isi is None:
                continue

            max_distances.append(np.max(isi))
            mean_distances.append(np.mean(isi))

        max_distance = np.max(max_distances) # mean max distance between activations
        mean_max_distance = np.mean(max_distances) # mean max distance between activations
        mean_distance = np.mean(mean_distances) # mean distance between activations
        std_activation_distance = np.std(mean_distances) # standard deviation of distances between activations
        coefficient_of_variation = std_activation_distance / mean_distance # coefficient of variation

        return ISIStatistics(max_distance, mean_max_distance, mean_distance, std_activation_distance, coefficient_of_variation)

    def _compute_global_statistics(self):
        """Computes various data-wide statistics for sanitization purposes."""
        Log.info(f"Computing global statistics...")

        # mean max firing distance across all neuron sets            
        self.global_mean_max_distance = np.mean([self._compute_isi_statistics(ns).max for ns in self.df["neuron_data"] if ns is not None])
        Log.info(f"\tAverage max activation distance: {self.global_mean_max_distance}")

        # compute max shape (force & neuron data)
        self.max_force_len = max(len(x) for x in self.df["force_data"])
        self.max_n_neurons = max(x.shape[0] for x in self.df["neuron_data"] if x is not None)
        self.max_activations = max(x.shape[1] for x in self.df["neuron_data"] if x is not None)

        Log.info(f"\tMax force data length: {self.max_force_len}")
        Log.info(f"\tMax number of neurons: {self.max_n_neurons}")
        Log.info(f"\tMax number of activations: {self.max_activations}")

        Log.success("Global statistics computed.")

    def _normalize_and_scale(self):
        """Scale force data and normalize neuron data"""

        Log.info("Normalizing and scaling neuron and force data...")

        neuron_data, force_data = np.array(self.df['neuron_data']), np.array(self.df['force_data'])

        # pad and scale force data -> (max_force_len,)
        for i in range(len(force_data)):
            force_data[i] = np.pad( force_data[i], (0, self.max_force_len - len(force_data[i])), mode='constant')

            # scale force data
            force_data[i] = MinMaxScaler().fit_transform(force_data[i].reshape(-1, 1)).reshape(-1)

        # pad neuron data -> (max_neurons, max_activations)
        for i in range(len(neuron_data)):
            padding = ((0, self.max_n_neurons - neuron_data[i].shape[0]), # pad rows (neurons)
                        (0, self.max_activations - neuron_data[i].shape[1])) # pad columns (activations)
            neuron_data[i] = np.pad(neuron_data[i], padding, mode='constant', constant_values=0) # pad each neuron data array

        self.df['neuron_data'] = neuron_data # update the dataframe with the normalized neuron data
        self.df['force_data'] = force_data # update the dataframe with the scaled force data

        Log.info(f"Normalized Neuron Shape: {neuron_data[0].shape}")
        Log.info(f"Scaled Force Data Shape: {force_data[0].shape}")
        Log.success("Neuron and force data normalized and scaled.")

    def _purge_insufficient_neuron_data(self):
        """Remove entries that have insufficient number of neurons in the data."""
        Log.info("Purging entries with insufficient neuron data...")
        prev_len = len(self.df)
        
        # filter function to check if the number of neurons is valid
        def valid_number_of_neurons(x: np.ndarray) -> bool:
            return x is not None and x.ndim > 1 and x.shape[0] >= constants.min_neurons

        # purge any entries that do not have a valid number of neurons
        self.df = self.df[self.df["neuron_data"].map(valid_number_of_neurons)] 

        Log.success(f"Purged {prev_len - len(self.df)} entries with insufficient neuron data. {len(self.df)} entries remaining.")

    def _purge_neuron_inconsistencies(self):
        """Check if there are any outlandish inconsistencies in the neuron data."""
        Log.info("Purging entries with inconsistent neuron data...")

        def filter_by_firing_distance(neuron_data: np.ndarray) -> bool:
            """Filter function to check if the average firing distance is within a certain threshold of the global average."""

            # no data or average distance is less than 2 times the global average -> keep the entry
            if neuron_data is None or self._compute_isi_statistics(neuron_data).mean_max < self.global_mean_max_distance * 2:
                return True

            return False
            
        prev_len = len(self.df)

        # filter out rows where average firing distance is too different from global average
        self.df = self.df[self.df['neuron_data'].apply(filter_by_firing_distance)]
        
        Log.success(f"Purged {prev_len - len(self.df)} entries with inconsistent firing distances")

    def _handle_measurement_decorrelation(self):
        """Handle measurement decorrelation (MD) by adding artificial neuron activation data in sparse regions and clamping force dat beyond measures. Unrecoverable de-correlations (general measurement decorrelation (GMD)) will also be identified and purged."""
        Log.info("Handling measurement correlation...")
        
        subjects = self.df['subject'].unique()
        self.df['force_inflection_indices'] = None

        # detect measurement decorrelation
        for subject in subjects:
            # trial-wise (full trial neuron set) computation of ISI and CV 
            trials: pd.DataFrame = self.df[self.df['subject'] == subject]
            for index, trial in trials.iterrows():
                mvc_level, trial_number, neuron_data, force_data = trial["mvc_level"], trial["trial_number"], trial['neuron_data'], trial['force_data']
                # no neuron data -> purge
                if neuron_data is None:
                    Log.warn(f"{format_subject(trial)} Neuron data is missing for trial {trial_number} @ {mvc_level}%.")
                    continue

                # find first and last activations
                first_activation_index: int = min(np.argmax(neuron_data == 1, axis=1))
                last_activation_index: int = max(neuron_data.shape[1] - np.argmax(neuron_data[:, ::-1] == 1, axis=1) - 1)

                # clip leading/trailing force data that has or will hit zero
                left_symmetric: np.ndarray = force_data[:int(len(force_data) / 2)]
                right_symmetric: np.ndarray = force_data[int(len(force_data) / 2):]

                # clip force data below zero
                np.clip(left_symmetric, a_min=0, a_max=np.inf, out=left_symmetric)
                np.clip(right_symmetric, a_min=0, a_max=np.inf, out=right_symmetric)

                # find zeros in symmetric regions
                left_symmetric_zeros = np.where(left_symmetric <= np.max(left_symmetric) * constants.md_symmetric_zeros_tolerance)[0]
                right_symmetric_zeros = np.where(right_symmetric <= np.max(right_symmetric * constants.md_symmetric_zeros_tolerance))[0]

                # track modifications for additional processing
                left_symmetric_modified = False
                right_symmetric_modified = False
                
                # clip regions before and after zeros respectively (relative to first and last activation)
                if left_symmetric_zeros.size > 0:
                    left_zero_closest_to_activation = left_symmetric_zeros[np.argmin(np.abs(left_symmetric_zeros - first_activation_index))]
                    left_symmetric[:left_zero_closest_to_activation] = 0
                    left_symmetric_modified = True
                if right_symmetric_zeros.size > 0:
                    right_zero_closest_to_activation = right_symmetric_zeros[np.argmin(np.abs(right_symmetric_zeros + len(left_symmetric) - last_activation_index))]
                    right_symmetric[right_zero_closest_to_activation:] = 0
                    right_symmetric_modified = True

                force_data = np.concatenate((left_symmetric, right_symmetric))

                # both regions modified -> skip gradient modification
                if left_symmetric_modified and right_symmetric_modified:
                    Log.warn(f"\tBoth regions modified -> Skipping", trial=trial)
                    continue

                force_data_median_index = int(len(force_data) / 2)
                median_region_gradient = np.mean(np.gradient(force_data[force_data_median_index - constants.md_region_window * constants.md_median_gradient_scaler:force_data_median_index * constants.md_median_gradient_scaler + constants.md_region_window]))

                # compute full gradient and second derivative of force data
                force_gradient = np.gradient(force_data)
                force_second_derivative = np.gradient(force_gradient)

                # define thresholds based on gradients
                gradient_threshold = np.mean(np.abs(force_gradient)) * 0.9
                second_derivative_threshold = np.mean(np.abs(force_second_derivative)) * 4

                # identify potential turning points between in threshold
                turning_points = np.where(
                    (np.abs(force_second_derivative) > second_derivative_threshold) & 
                    (np.abs(force_gradient) < gradient_threshold)
                )[0]

                # no turning points -> warn and skip
                if len(turning_points) == 0:
                    Log.warn(f"No turning points found between first activation and median for trial {trial_number} @ {mvc_level}%")
                    continue

                # find closest turning point to relative activations
                leading_turn_index = turning_points[np.argmin(np.abs(turning_points - first_activation_index))]
                trailing_turn_index = turning_points[np.argmin(np.abs(turning_points - last_activation_index))]

                Log.debug(f"\tLeading turn: {leading_turn_index / globals.constants.sampling_frequency} | Trailing turn: {trailing_turn_index / globals.constants.sampling_frequency}", trial=trial)

                region_gradients: dict[int, float] = {}

                # compute mean gradient for each window in the force data
                for window in range(leading_turn_index, trailing_turn_index, constants.md_region_window):
                    region_mean_gradient = np.mean(np.gradient(force_data[window:window + constants.md_region_window]))
                    region_gradients[window] = region_mean_gradient

                inflection_indices: list[int] = [0, 0] # left and right inflection points (edge of flat region) respectively

                # find furthest gradient from median region gradient within error threshold
                for region_start_index, region_gradient in region_gradients.items():
                    # right inflection point already found -> skip 
                    if region_start_index <= force_data_median_index and inflection_indices[0] != 0:
                        continue
                    
                    # within acceptable threshold of median region gradient -> potential inflection point
                    if math.isclose(region_gradient, median_region_gradient, rel_tol=constants.md_region_tolerance):
                        is_left_inflection = region_start_index < force_data_median_index
                        inflection_index = 0 if is_left_inflection else 1
                        inflection_indices[inflection_index] = region_start_index if is_left_inflection else region_start_index + constants.md_region_window
                
                # no valid inflection indices -> skip 
                if inflection_indices[0] == 0 or inflection_indices[1] == 0:
                    Log.warn(f"Could not find valid inflection indices for trial {trial_number} @ {mvc_level}%")
                    continue

                # store inflection indices for future processing
                self.df.at[index, 'force_inflection_indices'] = (inflection_indices[0], inflection_indices[1])

                Log.debug(f"\tFlat Region Interval (Inflection Points): ({inflection_indices[0] / globals.constants.sampling_frequency}, {inflection_indices[1] / globals.constants.sampling_frequency})", trial=trial)

                if inflection_indices[0] > first_activation_index + 1:
                    mean_leading_gradient = np.mean(np.gradient(force_data[first_activation_index:inflection_indices[0]]))
                else:
                    mean_leading_gradient = 0  # Default value if gradient cannot be computed

                if last_activation_index > inflection_indices[1] + 1:
                    mean_trailing_gradient = np.mean(np.gradient(force_data[inflection_indices[1]:last_activation_index]))
                else:
                    mean_trailing_gradient = 0  # Default value if gradient cannot be computed
                mean_edge_gradient = np.mean([mean_leading_gradient, -mean_trailing_gradient])

                # left symmetric region not modified -> apply gradient modification before first activation
                if not left_symmetric_modified:
                    if inflection_indices[0] == 0:
                        Log.warn(f"Required left inflection index not found for trial {trial_number} @ {mvc_level}%, skipping...", )
                        continue
                    
                    # replace the data before the first activation with scaled mean-gradient generated data 
                    gradient_values = np.arange(len(force_data[:first_activation_index])) * constants.md_mvc_gradient_scaling[mvc_level] * mean_edge_gradient + force_data[first_activation_index]
                    force_data[:first_activation_index:] = np.clip(gradient_values, a_min=0, a_max=np.inf) # clip negative values

                # right symmetric region not modified -> apply gradient modification past last activation
                if not right_symmetric_modified:
                    if inflection_indices[1] == 0:
                        Log.warn(f"Required right inflection index not found for trial {trial_number} @ {mvc_level}%, skipping...", )
                        continue

                    # replace the data after the last activation with scaled mean-gradient generated data 
                    gradient_values = np.arange(len(force_data[last_activation_index:])) * constants.md_mvc_gradient_scaling[mvc_level] * -mean_edge_gradient + force_data[last_activation_index]
                    force_data[last_activation_index:] = np.clip(gradient_values, a_min=0, a_max=np.inf) # clip negative values

                self.df.at[index, 'force_data'] = force_data

        # detect and purge gmd. This happens after main decorrelation detection because gmd detection relies on 'reliable' force profiles
        for subject in subjects:
            trials: pd.DataFrame = self.df[self.df['subject'] == subject]
            for index, trial in trials.iterrows():
                mvc_level, trial_number, neuron_data, force_data = trial["mvc_level"], trial["trial_number"], trial['neuron_data'], trial['force_data']

                # gmd params and calculations
                first_positive_read = np.argmax(force_data > 0)
                last_positive_read = len(force_data) - 1 - np.argmax(force_data[::-1] > 0)
                transposed_neuron_data = neuron_data.T
                gmd_empty_sequence_count = 0

                # detect gmd by finding unacceptable regions of no activations
                for i in range(first_positive_read, last_positive_read):
                    # column has no activations -> increment empty sequence count
                    if not transposed_neuron_data[i].any():
                        gmd_empty_sequence_count += 1
                    else:
                        gmd_empty_sequence_count = 0

                    # too many empty activation cols -> purge
                    if gmd_empty_sequence_count >= constants.gmd_maximum_nil_activation_windows[mvc_level]:
                        Log.warn(f"\t{format_subject(trial)} Purging for GMD ({(i - gmd_empty_sequence_count)/2048}-{i/2048})")
                        self.df.drop(index, inplace=True)
                        break

        Log.success("Measurement decorrelation mitigated.")

    def show_purged(self):
        """Show the difference between the given dataframe and the current dataframe."""
        # Compare rows based on subject, mvc_level, and trial_number
        merged = self.original_df.merge(
            self.df,
            on=['subject', 'mvc_level', 'trial_number'],
            how='outer',
            indicator=True
        )
        
        # Show rows that are different between original and current dataframes
        return merged[merged['_merge'] != 'both'][['subject', 'mvc_level', 'trial_number']]

    def sanitize(self, export = False):
        """Sanitize the given pandas dataframe with feature-specific methods."""
        self._purge_neuron_inconsistencies() # NI
        self._handle_measurement_decorrelation() # MD
        self._normalize_and_scale()

        if export:
            Log.info(f"Exporting {len(self.df)} entries and deleting previous")

            # Ensure the output directory exists and is wiped
            output_dir = "processing/out"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)  # Remove the directory and its contents
            os.makedirs(output_dir)  # Create a fresh directory

            for trial in self.df.itertuples():
                subject, trial_number, mvc_level = cast(str, trial.subject), cast(int, trial.trial_number), cast(int, trial.mvc_level)

                if trial.neuron_data is None or trial.force_data is None:
                    Log.warn(f"{format_subject(subject=subject, trial_number=trial_number, mvc_level=mvc_level)} Data missing, not exporting...")
                    continue

                export_path = f"{output_dir}/{subject}.{mvc_level}.{trial_number}.png"
                visualize_trial(self.df, subject=subject, trial_number=trial_number, mvc_level=mvc_level, export_path=export_path)

        return self.df
    