import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
from globals.utils import Log
import sanitization.constants as constants

class Sanitization:
    """Remove or correct any errors in the data. This includes removing outliers, filling in missing values, and correcting any other errors in the data. Some entries are not recoverable and will be "purged"."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_df = df.copy()
        self._compute_global_statistics()

    @staticmethod
    def _compute_average_max_firing_distance(neuron_set: np.ndarray) -> np.floating:
        """Computes the average of the max firing distances for each neuron in the given neuron set. A higher number means there is greater distance between firings."""
        max_distances: list[np.signedinteger] = [] 
        
        # record max distances between firings for each neuron in set
        for neuron in neuron_set.T:
            fire_indices = np.where(neuron == 1)[0] # get indices of fired neurons
            if len(fire_indices) > 1: # only compute distance if there are more than 1 firing
                max_distances.append(np.max(np.diff(fire_indices))) # get max distance between firings
        
        # return the average of the max distances
        return np.mean(max_distances)

    def _compute_global_statistics(self):
        """Computes various data-wide statistics for sanitization purposes."""
        Log.info(f"Computing global statistics...")

        # compute global average firing distance across all neuron sets
        firing_distances = []
        for neuron_set in self.df["neuron_data"]:
            if neuron_set is not None:
                firing_distances.append(self._compute_average_max_firing_distance(neuron_set))
            
        self.average_firing_distance = np.mean(firing_distances) # average firing distance across all neuron sets
        Log.info(f"Global average firing distance: {self.average_firing_distance}")
        Log.success("Global statistics computed.")

    def _normalize_and_scale(self):
        """Normalize and scale the force data to values between 0 and 1."""
        Log.info("Normalizing data...")
        
        # convert negative force values to 0 (reLU)
        self.df["force_data"] = self.df["force_data"].map(lambda x: np.maximum(x, 0))

        # scale given data to values between 0 and 1
        def scale(x):
            return (x - x.min()) / (x.max() - x.min())

        # normalize force to values between 0 and 1
        self.df["force_data"] = self.df["force_data"].apply(scale)

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
            neuron_avg_firing_distance = self._compute_average_max_firing_distance(neuron_data)

            # no data or average distance is less than 2 times the global average -> keep the entry
            if neuron_data is None or neuron_avg_firing_distance < self.average_firing_distance * 2:
                return True

            return False
            
        prev_len = len(self.df)

        # filter out rows where average firing distance is too different from global average
        self.df = self.df[self.df['neuron_data'].apply(filter_by_firing_distance)]
        
        Log.success(f"Purged {prev_len - len(self.df)} entries with inconsistent firing distances")

    def _check_measurement_decorrelation(self):
        """Check for discrepancies between measured neuron activations and force readings. An abundance of discrepancies (MD) in the dataset leads to General Measurement Decorrelation (GMD) and marshals a purge."""
        pass

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
        self._purge_neuron_inconsistencies() # NI
        self._smooth_force_spikes() # S
        self._normalize_and_scale()
        return self.df
    