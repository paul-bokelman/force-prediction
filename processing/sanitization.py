from typing import Union, Any
from processing.types import ISIStatistics
import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
import globals.constants
from globals.utils import Log, format_subject
import processing.constants as constants

class Sanitization:
    """Remove or correct any errors in the data. This includes removing outliers, filling in missing values, and correcting any other errors in the data. Some entries are not recoverable and will be "purged"."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_df = df.copy()
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
            if np.all(neuron == 0):
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
        """Reshape, normalize, and scale the neuron and force data for consistency."""
        Log.info("Normalizing data...")

        # Normalize neuron data shapes
        def normalize_neuron_data(data: np.ndarray) -> Any:
            if data is None:
                return None
            
            normalized = np.zeros((self.max_n_neurons, self.max_activations))
            h, w = data.shape
            normalized[:h, :w] = data
            return normalized

        self.df.loc[:, "neuron_data"] = self.df["neuron_data"].apply(normalize_neuron_data)

        Log.info(f"\tNormalized neuron shape to ({self.max_n_neurons}, {self.max_activations})")

        # Normalize force data shapes
        def normalize_force_data(data) -> Any:
            normalized = np.zeros(self.max_force_len)
            length = len(data)
            normalized[:length] = data
            return normalized

        self.df["force_data"].apply(normalize_force_data)

        Log.info(f"\tNormalized force shape to ({self.max_force_len},)")
         
        # convert negative force values to 0 (reLU)
        self.df.loc[:, "force_data"] = self.df["force_data"].map(lambda x: np.maximum(x, 0))

        # scale given data to values between 0 and 1
        def scale(x):
            return (x - x.min()) / (x.max() - x.min())

        # normalize force to values between 0 and 1
        self.df.loc[:, "force_data"] = self.df["force_data"].apply(scale)

        Log.success("Data normalized.")

    def _purge_insufficient_neuron_data(self):
        """Remove entries that have insufficient number of neurons in the data."""
        Log.info("Purging entries with insufficient neuron data...")
        prev_len = len(self.df)
        
        # filter function to check if the number of neurons is valid
        def valid_number_of_neurons(x: np.ndarray) -> bool:
            return x is not None and x.ndim > 1 and x.shape[1] >= constants.min_neurons

        # purge any entries that do not have a valid number of neurons
        self.df = self.df[self.df["neuron_data"].map(valid_number_of_neurons)] 

        Log.success(f"Purged {prev_len - len(self.df)} entries with insufficient neuron data. {len(self.df)} entries remaining.")

    def _smooth_force_spikes(self):
        """Smooth out force spikes in the data."""
        Log.info("Checking for force spikes...")

        # apply median_filter with large window to smooth out force spikes
        self.df['force_data'] = [median_filter(x, size=constants.spike_smoothing_window) for x in self.df['force_data']]
        
        Log.success(f"Processed force spikes in {len(self.df)} entries.")

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

        def isi_gamma(mean_isi: np.floating, cv: np.floating, n_spikes: int) -> np.ndarray:
            """Generate ISIs from a gamma distribution for bursty firing patterns."""
            k = 1 / (cv ** 2)  # shape parameter
            theta = mean_isi / k  # scale parameter
            return np.random.gamma(shape=k, scale=theta, size=n_spikes)
        
        def isi_expo(mean_isi: np.floating, n_spikes: int) -> np.ndarray:
            """Generate ISIs from an exponential distribution for Poisson-like firing. """
            return np.random.exponential(scale=mean_isi, size=n_spikes)

        def isi_normal(mean_isi: np.floating, isi_std: np.floating, n_spikes: int) -> np.ndarray:
            """Generate ISIs from a normal distribution for regular firing patterns."""
            return np.abs(np.random.normal(loc=mean_isi, scale=isi_std, size=n_spikes))
        
        subjects = self.df['subject'].unique()

        # detect measurement decorrelation
        for subject in subjects:
            # trial-wise (full trial neuron set) computation of ISI and CV 
            trials: pd.DataFrame = self.df[self.df['subject'] == subject]
            for _, trial in trials.iterrows():
                mvc_level, trial_number, neuron_data, force_data = trial["mvc_level"], trial["trial_number"], trial['neuron_data'], trial['force_data']
                # no neuron data -> purge
                if neuron_data is None:
                    Log.warn(f"{format_subject(trial)} Neuron data is missing for trial {trial_number} @ {mvc_level}%.")
                    continue

                neuron_stats = self._compute_isi_statistics(neuron_data)

                region_gradients: dict[float, tuple[int, int]] = {}

                # compute mean rate of change for region and append if within 'flat' threshold
                for step_percentage in range(int(1/constants.md_flat_region_detection_step)):
                    start_index = int(step_percentage * constants.md_flat_region_detection_step * len(force_data))
                    end_index = int((step_percentage + 1) * constants.md_flat_region_detection_step * len(force_data))
                    region = force_data[start_index:end_index if end_index < len(force_data) else len(force_data)]
                    region_mean_gradient = np.mean(np.gradient(region))

                    region_gradients[region_mean_gradient] = (start_index, end_index)
                    
                    # region is relatively flat -> add to flat region intervals
                    # if abs(0 - np.abs(region_mean_gradient)) <= constants.md_flat_region_error_threshold:
                    #     Log.info(f"Identified flat region @ {start_index / globals.constants.sampling_frequency} - {end_index / globals.constants.sampling_frequency}: {region_mean_gradient}", trial=trial)
                    #     flat_region_intervals.append((start_index / globals.constants.sampling_frequency, end_index))
                
                # no flat regions -> no leading and trailing MDs can be correct -> warn and skip
                # if len(flat_region_intervals) == 0:
                #     Log.warn(f"No flat regions found for trial {trial_number} @ {mvc_level}%")
                #     continue
                
                # Log.debug(f"\tFlat regions: {flat_region_intervals}", trial=trial)

                # compute slopes of rising and falling edges

                # calculate first and second derivatives
                gradient = np.gradient(force_data)
                second_derivative = np.gradient(gradient)

                # define thresholds based on gradients, higher threshold -> less likely to be a turning point
                gradient_threshold = np.mean(np.abs(gradient)) * 0.5 # look for sharp changes in force
                second_derivative_threshold = np.mean(np.abs(second_derivative)) * 3 # look for very sharp changes in gradient

                # identify potential turning points
                turning_points = np.where(
                    (np.abs(second_derivative) > second_derivative_threshold) & 
                    (np.abs(gradient) < gradient_threshold)
                )[0]

                # no turning points -> no leading and trailing MDs can be correct -> warn and skip
                if len(turning_points) == 0:
                    Log.warn(f"No turning points found for trial {trial_number} @ {mvc_level}%")
                    continue

                # identify the turning points closest to the first and last activations
                first_activation_index: int = min(np.argmax(neuron_data == 1, axis=1))
                last_activation_index: int = max(neuron_data.shape[1] - np.argmax(neuron_data[:, ::-1] == 1, axis=1) - 1)

                leading_turn_index: float = turning_points[np.argmin(np.abs(turning_points - first_activation_index))]
                trailing_turn_index: float = turning_points[np.argmin(np.abs(turning_points - last_activation_index))]


                Log.debug(f"\tLeading turn: {leading_turn_index / globals.constants.sampling_frequency} | Trailing turn: {trailing_turn_index / globals.constants.sampling_frequency}", trial=trial)

                # calculate gradient between leading turn and the proceeding activation

                # adjust all points before first activation to follow the leading gradient
                # start, end = (first_activation_index, leading_turn_index) if first_activation_index <= leading_turn_index else (leading_turn_index, first_activation_index)
                # leading_gradient: float = np.mean(np.gradient(force_data[start:end]))
                # force_data[:first_activation_index] = (force_data[first_activation_index] - leading_gradient * np.arange(first_activation_index)[::-1])
                # print(force_data[:first_activation_index])
                # get indices where force is 0 before first activation
                # leading_zero = np.where(force_data[:first_activation_index] == 0)[0]

                # # if there are any zeros before first activation
                # if len(leading_zero) > 0:
                #     Log.info(f"Leading zeros found for trial {trial_number} @ {mvc_level}%")
                #     # replace all values before the first zero with zeros
                #     first_zero_index = leading_zero[0]
                #     force_data[:first_zero_index] = 0

                # adjust all points after last activation to follow the trailing gradient
                # start, end = (last_activation_index, trailing_turn_index) if last_activation_index <= trailing_turn_index else (trailing_turn_index, last_activation_index)
                # trailing_gradient: float = np.mean(np.gradient(force_data[start:end]))
                # points_after = len(force_data) - last_activation_index
                # force_data[last_activation_index:] = trailing_gradient * np.arange(points_after)

                # force_data[:] = np.clip(force_data, a_min=0, a_max=np.inf)

                #! 2. Purge GMDs



                #! 3. Correct inner MDs by adding artificial neuron activation data in sparse regions and dropping bad neurons

                
                # conditionally add artificial neural activation data in sparse regions based on CV (relative model functions) 
                # compute and compare each neurons ISI to the global mean max distance
                # for i, neuron in enumerate(neuron_data):
                #     neuron_isi = self._compute_neuron_isi(neuron)
                    
                #     # skip if there is not enough data to compute ISI
                #     if neuron_isi is None:
                #         continue

                #     Log.warn(f"Neuron {i + 1} | Trial {trial_number} @ {mvc_level}% | {np.max(neuron_isi)}")

                #     # activations are sparse or don't conform to trend -> include artificial activations
                #     if np.max(neuron_isi) > self.global_mean_max_distance:
                #         Log.error(f"Neuron {i + 1} | Trial {trial_number} @ {mvc_level}% | {np.max(neuron_isi)}")

                #     # print(max_distance, self.global_mean_max_distance)

                # #! detect md with isi_stats.max, purge gmd
                # #? could absorb _purge_neuron_inconsistencies

                # # MD is too great (GMD) -> purge entry

                #     # calculate ISI and CV for each subject
                # isi, std, cv = neuron_stats.isi, neuron_stats.std, neuron_stats.cv
                # spike_train_len = neuron_data.shape[0]

                # # compute the new isi based on the CV
                # computed_isi = isi_normal(isi, cv, spike_train_len) if cv < 1 else isi_gamma(isi, std, spike_train_len) if cv > 1 else isi_expo(isi, spike_train_len)

                # # build artificial spike train based on computed ISI
                # spike_indices = np.cumsum(computed_isi).astype(int)  # convert to discrete indices
                # spike_indices = spike_indices[spike_indices < spike_train_len] # remove indices that exceed the length of the spike train
                # binary_spike_train = np.zeros(spike_train_len, dtype=int) # initialize binary spike train with 0s
                # binary_spike_train[spike_indices] = 1  # set spikes to 1 


        Log.success("Measurement decorrelation mitigated.")
        

    def _check_trend(self):
        """Check to see if trend is significantly different average trend."""
        pass

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

    def sanitize(self):
        """Sanitize the given pandas dataframe with feature-specific methods."""
        self._purge_insufficient_neuron_data() # N
        # self._purge_neuron_inconsistencies() # NI
        # self._smooth_force_spikes() # S
        self._handle_measurement_decorrelation() # MD
        # self._normalize_and_scale()
        return self.df
    