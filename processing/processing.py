from typing import Literal, Generator, cast
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from globals.utils import Log
from processing import constants
import prediction.constants
from scipy.signal import resample
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy().sample(frac=1, random_state=42).reset_index(drop=True) # shuffle trials 
        self.preprocessed = False
        
        # perform splits at the trial level
        total_entries = len(self.df)
        self.train_end = int(constants.train_percentage * total_entries)
        self.val_end = self.train_end + int(constants.val_percentage * total_entries)

    @staticmethod
    def _bin_spikes(trains: np.ndarray , bin_size: int = constants.bin_size) -> np.ndarray:
        """Bins and normalizes incoming spike train data"""
        time_steps, neurons = trains.shape # T is the number of time steps, N is the number of neurons
        n_bins = time_steps // bin_size # Number of bins

        binned_spikes: np.ndarray = trains[:n_bins * bin_size].reshape(n_bins, bin_size, neurons).sum(axis=1) # sum spikes in each bin
        return binned_spikes / bin_size  # normalize by bin size
    
    @staticmethod
    def _decay_spikes(trains: np.ndarray, tau: int = constants.exponential_decay_lifetime) -> np.ndarray:
        """Applies an exponential decay filter to the incoming spike train data."""
        filtered_train = np.zeros_like(trains)
        alpha = np.exp(-1 / tau)
        filtered_train[0, :] = trains[0, :]
        for t in range(1, trains.shape[0]):
            filtered_train[t, :] = filtered_train[t-1, :] * alpha + trains[t, :]

        return filtered_train
    
    @staticmethod
    def _normalize_spikes(spike_trains: np.ndarray):
        """Z-score normalizes the spike trains across all neurons."""
        # mean = spike_trains.mean(axis=1, keepdims=True)
        # std = spike_trains.std(axis=1, keepdims=True)
        # std[std == 0] = 1
        # # return (spike_trains - mean) / std
        # return MinMaxScaler().fit_transform(spike_trains)

        X_min = spike_trains.min()
        X_max = spike_trains.max()

        return (spike_trains - X_min) / (X_max - X_min)
    
    @staticmethod
    def _amplify_spikes(spike_trains: np.ndarray, factor: float = constants.size_amplification_factor) -> np.ndarray:
        """Amplifies the spike trains by a given factor."""
        amplification_factors = np.log(np.arange(1, spike_trains.shape[0] + 1)) ** factor # amplification based on neuron index
        first_activations = np.argmax(spike_trains > 0, axis=1)
        neuron_order = np.argsort(first_activations)
        sorted_spike_trains = spike_trains[neuron_order]  # sort neurons by first activation
        return sorted_spike_trains * amplification_factors[:, np.newaxis]   # apply amplification factor to each neuron

    @staticmethod
    def _process_spike_trains(spike_trains: np.ndarray) -> np.ndarray:
        """Bins, applies exponential decay filter, and normalizes the spike trains."""
        binned_train = Preprocessor._bin_spikes(spike_trains.T) #/ should've been transposed before getting here...
        filtered_train = Preprocessor._decay_spikes(binned_train)
        amplified_train = Preprocessor._amplify_spikes(filtered_train.T)
        return Preprocessor._normalize_spikes(amplified_train)

    @staticmethod
    def _process_force(force_data: np.ndarray, bin_size: int = constants.bin_size) -> np.ndarray:
        """Down-samples and normalizes the force data."""
        if len(force_data) < bin_size:
            Log.warn("Force data is shorter than bin size, returning original force data.")
            return force_data
        
        # down-sample the force data
        ds_force: np.ndarray = cast(np.ndarray, resample(force_data, num=len(force_data) // bin_size))
        
        # normalize the force data
        max_force = np.max(ds_force)
        if max_force > 0:
            ds_force /= max_force
        
        return ds_force
    
    def generator(self, dataset_id: Literal['train', 'val', 'test']) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        assert self.preprocessed, "Data must be preprocessed before using generator"

        # conditionally choose data based on the dataset type
        dataset = self.df[:self.train_end] if dataset_id == 'train' else \
               self.df[self.train_end:self.val_end] if dataset_id == 'val' else \
               self.df[self.val_end:]
        
        x_batch, y_batch = [], [] # store current batches

        while True:
            for trial in dataset.itertuples():

                # extract neuron data, force data, and MVC from the trial
                neuron_data = np.array(trial.neuron_data).T  # (time_steps, neurons)
                force_data = np.array(trial.force_data)  # (time_steps,)
                time_steps, neurons = neuron_data.shape

                # ensure the trial has enough data for at least one sliding window
                if time_steps < constants.sequence_length:
                    Log.warn(f"Skipping trial {trial.Index} due to insufficient data length.")
                    continue

                # generate sliding windows
                x = np.squeeze(sliding_window_view(neuron_data, window_shape=(constants.sequence_length, neurons))[::constants.stride], axis=1)
                y = sliding_window_view(force_data, window_shape=constants.sequence_length)[::constants.stride]

                for x_window, y_window in zip(x, y):
                    # add windows to the batch
                    x_batch.append(x_window)
                    y_batch.append(y_window)

                    # yield batch when size is reached
                    if len(x_batch) == prediction.constants.batch_size:
                        indices = np.arange(len(x_batch))
                        np.random.shuffle(indices) # shuffle the indices for randomness
                        yield np.array(x_batch)[indices], np.array(y_batch)[indices]
                        x_batch, y_batch = [], [] # reset the batch

            # handle remaining data in the batch
            if x_batch:
                indices = np.arange(len(x_batch))
                np.random.shuffle(indices)  # shuffle the indices for randomness
                yield np.array(x_batch)[indices], np.array(y_batch)[indices]
                x_batch, y_batch = [], [] # reset the batch

    def preprocess(self):
        """Preprocess the sanitized data in the following ways: bin, apply exponential decay filter, and normalize spike trains, down-sample and normalize force and include MVC as a feature"""
        modified_trains = self.df["neuron_data"].map(self._process_spike_trains)
        modified_forces = self.df["force_data"].map(self._process_force)
        max_neurons = max([t.shape[0] for t in modified_trains]) + 1 # add +1 for mvc level addition

        for index, (spike_train, mvc_level) in enumerate(zip(modified_trains, self.df["mvc_level"].values)):
            neurons, time_steps = spike_train.shape
            neuron_padding = np.zeros((max(0, max_neurons - neurons - 1), time_steps))
            mvc = np.full((1, time_steps), mvc_level / 100) # add MVC as a feature to the neuron data (basically an extra neuron with )
            constituents = [spike_train, mvc] + ([neuron_padding] if neurons != max_neurons else [])
            modified_trains[index] = np.concatenate(constituents, axis=0)

        self.df["neuron_data"] = modified_trains
        self.df["force_data"] = modified_forces
        self.preprocessed = True # mark the data as preprocessed
        return self.df
    
    def get_generators(self) -> tuple[Generator[tuple[np.ndarray, np.ndarray], None, None], ...]:
        """Returns all generators in the order: train, val, test"""
        return (self.generator('train'), self.generator('val'), self.generator('test'))
    
    @staticmethod
    def preprocess_trial(neuron_data: np.ndarray, mvc: int) -> np.ndarray:
        """Preprocess a single trial of neuron data to be used as model input."""
        return np.array([])

class Postprocessor:
    @staticmethod
    def postprocess_prediction(neuron_data: np.ndarray, predicted_force: np.ndarray) -> np.ndarray:
        """Postprocess the model prediction to match the original trial data."""

        def butter_lowpass_filter(data, cutoff=4, fs=1100, order=4):
            nyq = 0.5 * fs  # nyquist Frequency
            normal_cutoff = cutoff / nyq
            b, a = cast(tuple[np.ndarray, ...], butter(order, normal_cutoff, btype='low', analog=False))
            return filtfilt(b, a, data)

        # find first and last activations
        first_activation_index: int = min([n for n in np.argmax(neuron_data == 1, axis=1) if n != 0])
        last_activation_index: int = max(neuron_data.shape[1] - np.array([n for n in np.argmax(neuron_data[:, ::-1] == 1, axis=1) if n != 0]) - 1)

        predicted_force = butter_lowpass_filter(predicted_force)

        # clip leading/trailing predicted force data that has or will hit zero
        np.clip(predicted_force, a_min=0, a_max=np.inf, out=predicted_force)
        
        # normalize the predicted force to the range [0, 1]
        max_force = np.max(predicted_force)
        if max_force > 0: # avoid division by zero
            predicted_force = predicted_force / max_force

        # set all force values before the first activation and after the last activation to zero
        predicted_force[:first_activation_index] = 0
        predicted_force[last_activation_index + 1:] = 0

        return predicted_force