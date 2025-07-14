from typing import Literal, cast
from processing.types import DataGenerator
from prediction.tuning import Hyperparameters
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numpy.lib.stride_tricks import sliding_window_view
from globals.utils import Log
from processing import constants
from processing.utils import get_subject_mappings, get_dataframe

type DatasetId = Literal['train', 'val', 'test']

class Preprocessor:
    def __init__(self, hyperparameters: Hyperparameters) -> None:
        self.hash = hyperparameters.preprocessing.hash() # get the hash associated with the preprocessing parameters
        self.df = get_dataframe(constants.preprocessed_dataset_path(self.hash))

        self.sequence_length = hyperparameters.training.sequence_length
        self.stride = hyperparameters.training.stride
        self.train_percentage = hyperparameters.training.train_percentage
        self.test_percentage = hyperparameters.training.test_percentage
        self.validation_percentage = hyperparameters.training.validation_percentage
        self.batch_size = hyperparameters.training.batch_size
        self.bin_size = hyperparameters.preprocessing.bin_size
        self.exponential_decay_lifetime = hyperparameters.preprocessing.exponential_decay_lifetime
        self.size_amplification_factor = hyperparameters.preprocessing.size_amplification_factor

    @staticmethod
    def _bin_spikes(trains: np.ndarray , bin_size: int = constants.bin_size) -> np.ndarray:
        """Bins and normalizes incoming spike train data"""
        time_steps, neurons = trains.shape # T is the number of time steps, N is the number of neurons
        n_bins = time_steps // bin_size # Number of bins

        binned_spikes: np.ndarray = trains[:n_bins * bin_size].reshape(n_bins, bin_size, neurons).sum(axis=1) # sum spikes in each bin
        return binned_spikes / bin_size  # normalize by bin size
    
    @staticmethod
    def _decay_spikes(trains: np.ndarray, tau: float = constants.exponential_decay_lifetime) -> np.ndarray:
        """Applies an exponential decay filter to the incoming spike train data."""
        filtered_train = np.zeros_like(trains)
        alpha = np.exp(-1 / tau) if tau > 0 else 0 # alpha is the decay factor, 0 means no decay
        filtered_train[0, :] = trains[0, :]
        for t in range(1, trains.shape[0]):
            filtered_train[t, :] = filtered_train[t-1, :] * alpha + trains[t, :]

        return filtered_train
    
    @staticmethod
    def _normalize_spikes(spike_trains: np.ndarray):
        """Z-score normalizes the spike trains across all neurons."""
        X_min, X_max = spike_trains.min(), spike_trains.max()
        return (spike_trains - X_min) / (X_max - X_min)
    
    @staticmethod
    def _amplify_spikes(spike_trains: np.ndarray, factor: float = constants.size_amplification_factor) -> np.ndarray:
        """Amplifies the spike trains by a given factor."""
        amplification_factors = np.log(np.arange(1, spike_trains.shape[0] + 1)) ** factor # amplification based on neuron index
        first_activations = np.argmax(spike_trains > 0, axis=1)
        neuron_order = np.argsort(first_activations)
        sorted_spike_trains = spike_trains[neuron_order]  # sort neurons by first activation
        return sorted_spike_trains * amplification_factors[:, np.newaxis]   # apply amplification factor to each neuron

    def _process_spike_trains(self, spike_trains: np.ndarray) -> np.ndarray:
        """Bins, applies exponential decay filter, and normalizes the spike trains."""
        binned_train = Preprocessor._bin_spikes(spike_trains.T, self.bin_size) #/ should've been transposed before getting here...
        filtered_train = Preprocessor._decay_spikes(binned_train, self.exponential_decay_lifetime)
        amplified_train = Preprocessor._amplify_spikes(filtered_train.T, self.size_amplification_factor)
        return Preprocessor._normalize_spikes(amplified_train)
    
    @staticmethod
    def _normalize_force(force_data: np.ndarray) -> np.ndarray:
        """Normalizes the force data to the range [0, 1] based on the maximum value."""
        max_force: np.floating = np.max(force_data)
        assert max_force > 0, "Max force must be greater than 0 to avoid division by zero"
        return force_data / max_force

    def process_force(self, force_data: np.ndarray) -> np.ndarray:
        """Down-samples and normalizes the force data."""
        if len(force_data) < self.bin_size:
            Log.warn("Force data is shorter than bin size, returning original force data.")
            return force_data
        
        # down-sample the force data
        ds_force: np.ndarray = cast(np.ndarray, resample(force_data, num=len(force_data) // self.bin_size))
        return Preprocessor._normalize_force(ds_force)
    
    def generator(self, dataset_id: DatasetId) -> DataGenerator:
        assert self.previously_preprocessed, "Data must be preprocessed before using generator"
        dataset = self._get_dataset(dataset_id) # conditionally choose data based on the dataset type
        
        x_batch: dict[str, list] = {"neural_input": [], "subject_id": []} # batch of neuron data windows and subject ids
        y_batch: list[np.ndarray] = [] # batch of force data windows

        def reset_batch():
            """Returns empty batch representations."""
            return {"neural_input": [], "subject_id": []}, []
        
        def shuffled_batches(x_batch: dict[str, list], y_batch: list[np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
            """Shuffles the batches and returns them as numpy arrays."""
            indices = np.arange(len(x_batch["neural_input"]))
            np.random.shuffle(indices)
            shuffled_neural_inputs = np.array(x_batch["neural_input"])[indices]
            shuffled_subject_ids = np.array(x_batch["subject_id"])[indices]
            shuffled_force_data = np.array(y_batch)[indices]
            return {"neural_input": shuffled_neural_inputs, "subject_id": shuffled_subject_ids}, shuffled_force_data

        while True:
            for trial in dataset.itertuples():
                # extract neuron data, force data, and MVC from the trial
                neuron_data = np.array(trial.neuron_data).T  # (time_steps, neurons)
                force_data = np.array(trial.force_data)  # (time_steps,)
                subject_id = trial.subject
                time_steps, neurons = neuron_data.shape

                # ensure the trial has enough data for at least one sliding window
                if time_steps < self.sequence_length:
                    Log.warn(f"Skipping trial {trial.Index} due to insufficient data length.")
                    continue

                # generate sliding windows
                x = np.squeeze(sliding_window_view(neuron_data, window_shape=(self.sequence_length, neurons))[::self.stride], axis=1)
                y = sliding_window_view(force_data, window_shape=self.sequence_length)[::self.stride]

                for x_window, y_window in zip(x, y):
                    # add windows to the batch
                    x_batch["neural_input"].append(x_window)
                    x_batch["subject_id"].append(subject_id)  # assuming subject is a scalar id
                    y_batch.append(y_window)

                    # yield batch when size is reached
                    if len(x_batch) == self.batch_size:
                        yield shuffled_batches(x_batch, y_batch) # yield a shuffled version of the batch
                        x_batch, y_batch = reset_batch()  # reset the batch after yielding

            # handle remaining data in the batch
            if x_batch:
                yield shuffled_batches(x_batch, y_batch)
                x_batch, y_batch = reset_batch()  # reset the batch after yielding

    def preprocess(self, compute_baseline: bool = False, overwrite: bool = False):
        """Preprocess the sanitized data in the following ways: bin, apply exponential decay filter, and normalize spike trains, down-sample and normalize force and include MVC as a feature"""
        # already processed and no overwrite -> skip
        if self.previously_preprocessed and not overwrite:
            return Log.info(f"[{self.hash}] Preprocessed data already exists")
        
        Log.info(f"[{self.hash}] Preprocessing data...")
        
        data = get_dataframe(constants.dataset_path)  # ensure the original dataset is loaded
        subject_mappings = get_subject_mappings()

        subjects = data["subject"].map(lambda x: subject_mappings[x])  # map subjects to their ids
        modified_trains = data["neuron_data"].map(lambda x: self._process_spike_trains(x))
        modified_forces = data["force_data"].map(lambda x: self.process_force(x))
        max_neurons = max([t.shape[0] for t in modified_trains]) + 1 # add +1 for mvc level addition
        self.n_subjects = len(subject_mappings)

        # padding and concatenating the neuron data with MVC level
        for index, (spike_train, mvc_level) in enumerate(zip(modified_trains, data["mvc_level"].values)):
            neurons, time_steps = spike_train.shape
            neuron_padding = np.zeros((max(0, max_neurons - neurons - 1), time_steps))
            mvc = np.full((1, time_steps), mvc_level / 100) # add MVC as a feature to the neuron data (basically an extra neuron with just mvc level)
            constituents = [spike_train, mvc] + ([neuron_padding] if neurons != max_neurons else [])
            modified_trains[index] = np.concatenate(constituents, axis=0)

        # create a new dataframe with preprocessed data and save to disk
        preprocessed_df = pd.DataFrame({"subject": subjects, "neuron_data": modified_trains, "force_data": modified_forces })

        # optionally compute baseline metrics for the preprocessed data
        if compute_baseline:
            force_data = preprocessed_df["force_data"].values
            mean_forces: list[float] = [np.mean(force, axis=0) for force in force_data]

            baseline_metrics = {'mae': [], 'mse': [], 'r2': []}

            for average, actual in zip(mean_forces, force_data):
                predicted = np.full_like(actual, average)
                baseline_metrics['mae'].append(mean_absolute_error(actual, predicted))
                baseline_metrics['mse'].append(mean_squared_error(actual, predicted))
                baseline_metrics['r2'].append(r2_score(actual, predicted))
            
            # Compute and store baseline metrics
            for key, value in baseline_metrics.items():
                preprocessed_df.attrs[f'baseline_{key}'] = np.mean(value)
        
        preprocessed_df.to_pickle(constants.preprocessed_dataset_path(self.hash))

        self.df = preprocessed_df  # update the internal dataframe
        Log.info(f"[{self.hash}] Preprocessing complete")
    
    def get_generators(self) -> tuple[DataGenerator, DataGenerator, DataGenerator]:
        """Returns all generators in the order: train, val, test"""
        return (self.generator('train'), self.generator('val'), self.generator('test'))
    
    def _get_dataset(self, dataset_id: DatasetId) -> pd.DataFrame:
        """Returns the dataset for the given dataset_id."""
        assert self.previously_preprocessed, "Data must be preprocessed before accessing dataset"

        # perform splits at the trial level
        total_entries = len(self.df)
        self.train_end = int(self.train_percentage * total_entries)
        self.val_end = self.train_end + int(self.validation_percentage * total_entries)

        return self.df[:self.train_end] if dataset_id == 'train' else \
               self.df[self.train_end:self.val_end] if dataset_id == 'val' else \
               self.df[self.val_end:]
    
    def compute_dataset_windows(self, dataset_id: DatasetId) -> int:
        dataset = self._get_dataset(dataset_id)
        total_samples = sum([len(f) for f in dataset["force_data"]])
        return (total_samples - self.sequence_length) // self.stride + 1
    
    def preprocess_neuron_data(self, neuron_data: np.ndarray, mvc_level: int) -> np.ndarray:
        """Preprocess a single trial of neuron data to be used as model input."""
        _, n_features = self.input_shape
        modified_train = self._process_spike_trains(neuron_data)

        # add mvc as feature and pad to max neurons
        n_neurons, n_time_steps = modified_train.shape
        neuron_padding = np.zeros((max(0, n_features - n_neurons - 1), n_time_steps))
        mvc = np.full((1, n_time_steps), mvc_level / 100)
        return np.concatenate([modified_train, mvc] + ([neuron_padding] if n_neurons != n_features else []), axis=0).T
    
    @staticmethod
    def window_neuron_data(neuron_data: np.ndarray, sequence_length: int, stride: int) -> np.ndarray:
        """Generates sliding windows from the neuron data."""

        # ensure the neuron data has enough time steps for at least one sliding window
        if neuron_data.shape[0] < sequence_length:
            Log.warn("Neuron data is shorter than sequence length, returning original neuron data.")
            return neuron_data
        
        # generate sliding windows
        return np.squeeze(sliding_window_view(neuron_data, window_shape=(sequence_length, neuron_data.shape[1]))[::stride], axis=1)

    @property
    def previously_preprocessed(self) -> bool:
        """Returns whether the data has been preprocessed."""
        return os.path.exists(constants.preprocessed_dataset_path(self.hash)) and not self.df.empty
    
    @property 
    def input_shape(self) -> tuple[int, int]:
        """Returns the input shape of the preprocessed data."""
        assert self.previously_preprocessed, "Data must be preprocessed before accessing max neurons"
        n_features = max([x.shape[0] for x in self.df["neuron_data"]])
        return (self.sequence_length, n_features)

class Postprocessor:
    @staticmethod
    def overlap_average(sequence: np.ndarray, time_steps: int, stride: int) -> np.ndarray:
        """Reconstructs the full prediction from overlapping window predictions using overlap averaging."""
        sequence = sequence.squeeze() # ensure the sequence is 2D
        averaged_sequence = np.zeros(time_steps)
        averaged_sequence[:stride] = sequence[0, :stride] # copy the first window directly

        # iterate over the sequence windows and average the overlapping parts
        for index, stride_start in enumerate(range(stride, time_steps - stride, stride)):
            # last window -> copy the remaining values directly (no averaging)
            if index == sequence.shape[0] - 1:
                averaged_sequence[stride_start:] = sequence[index, -(time_steps - stride_start):]
                break
            averaged_sequence[stride_start:stride_start+stride] = (sequence[index][stride:] + sequence[index+1][:stride]) / 2

        return averaged_sequence