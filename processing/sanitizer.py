from typing import Union
from processing.types import ISIStatistics
import os
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from processing import constants
from globals.utils import Log, format_subject

class Sanitizer:
    """Remove or correct any errors in the data. This includes removing outliers, filling in missing values, and correcting any other errors in the data. Some entries are not recoverable and will be "purged"."""
    def __init__(self, df: pd.DataFrame):
        self.df = df
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
            
            isi = Sanitizer._compute_neuron_isi(neuron)

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

        # compute shape metrics (force & neuron data)
        self.max_force_len = max(len(x) for x in self.df["force_data"])
        self.max_n_neurons = max(x.shape[0] for x in self.df["neuron_data"] if x is not None)
        self.max_activations = max(x.shape[1] for x in self.df["neuron_data"] if x is not None)

        Log.info(f"\tAverage max activation distance: {self.global_mean_max_distance}")
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

    def _trim_decorrelation(self):
        """Trims all leading and trailing force values past the first and last neuron activation respectively"""
        subjects = self.df['subject'].unique()

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

                # trim force data to the first and last neuron activation
                force_data = force_data[first_activation_index:last_activation_index + 1]  # +1 to include the last activation
                self.df.at[index, 'force_data'] = force_data  # update the force data in the dataframe

                # trim neuron data to the first and last activation
                neuron_data = neuron_data[:, first_activation_index:last_activation_index + 1]
                self.df.at[index, 'neuron_data'] = neuron_data


    def sanitize(self, override: bool = False) -> pd.DataFrame:
        """Sanitize the current dataframe and optionally export the modified dataframe as a pkl file."""
        self._purge_insufficient_neuron_data() # remove entries that do not have enough neurons
        self._purge_neuron_inconsistencies() # remove entries that don't align with the global average firing distance
        self._trim_decorrelation() # trim all leading and trailing force values past the first and last neuron activation respectively

        # save the sanitized dataframe if specified
        if override:
            self.df.reset_index(drop=True, inplace=True)
            self.df.to_pickle(constants.dataset_path)

        Log.success(f"Sanitized dataset saved to {constants.dataset_path}")
        Log.info(f"Sanitized dataset contains {len(self.df)} entries after sanitization.")
        return self.df