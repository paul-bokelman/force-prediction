from typing import Literal, Generator, cast
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from globals.utils import Log
from processing import constants
import prediction.constants

class Processing:
    """Preprocesses the neuronal and force data into sequences for model training."""
    def __init__(self, df: pd.DataFrame, sequence_length: int = constants.sequence_length, stride: int = constants.stride) -> None:
        self.df = df.sample(frac=1, random_state=42) # shuffle the incoming df for training

        self.max_n_neurons: int = max([d.shape[0] for d in df['neuron_data']])
        self.max_activations: int = max([d.shape[1] for d in df['neuron_data']])
        self.max_force_len: int = max([len(d) for d in df['force_data']])

        self.sequence_length = sequence_length
        self.stride = stride

        # split dataset into train, val, and test sets
        total_entries = len(self.df)
        train_end = int(constants.train_split_percentage * total_entries)
        val_end = train_end + int(constants.val_split_percentage * total_entries)
        self.train = self.df[:train_end]
        self.val = self.df[train_end:val_end]
        self.test = self.df[val_end:]

        Log.debug(f"Splits: train={len(self.train)}, val={len(self.val)}, test={len(self.test)}")

        # Ensure activations and force data lengths match
        if self.max_activations != self.max_force_len:
            raise ValueError("The number of activations and force data lengths must be equal.")
        
        Log.debug(f"sl={self.sequence_length}, s={self.stride}")

    @staticmethod
    def _create_sliding_windows(data: np.ndarray, sequence_length = constants.sequence_length, stride = constants.stride) -> np.ndarray:
        """Create sliding windows from the data."""
        num_windows = (data.shape[0] - sequence_length) // stride + 1
        windows = np.array([
            data[start:start + sequence_length]
            for start in range(0, num_windows * stride, stride)
        ])
        return windows
    
    def generator(self, 
                dataset_id: Literal['train', 'val', 'test'], 
                batch_size=prediction.constants.batch_size
        ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Sliding window generator for a given dataset (train, test, val)."""

        dataset: pd.DataFrame = getattr(self, dataset_id)  # Get the correct dataset for generation
        x_batch, y_batch = [], []

        while True:
            for trial in dataset.itertuples():
                # extract neuron data, force data, and MVC from the trial
                neuron_data = np.array(trial.neuron_data).T  # (time_steps, neurons)
                force_data = np.array(trial.force_data)  # (time_steps,)
                mvc = trial.mvc_level  # Scalar (e.g., 5, 10, 20, etc.)

                # add MVC as an additional feature to neuron data
                mvc_column = np.full((neuron_data.shape[0], 1), mvc)  # Shape: (time_steps, 1)
                trial_data = np.concatenate([neuron_data, mvc_column], axis=1)  # Shape: (time_steps, neurons + 1)

                # Ensure the trial has enough data for at least one sliding window
                if trial_data.shape[0] < self.sequence_length:
                    Log.warn(f"Skipping trial {trial.Index} due to insufficient data length.")
                    continue

                # generate sliding windows
                num_windows = (trial_data.shape[0] - self.sequence_length) // self.stride + 1
                for start in range(0, num_windows * self.stride, self.stride):
                    end = start + self.sequence_length
                    x_window = trial_data[start:end]
                    y_window = force_data[start:end]

                    # ensure the window has the correct shape
                    if x_window.shape[0] == self.sequence_length and y_window.shape[0] == self.sequence_length:
                        x_batch.append(x_window)
                        y_batch.append(y_window)

                    # yield batch when size is reached
                    if len(x_batch) == batch_size:
                        yield np.array(x_batch), np.array(y_batch)
                        x_batch, y_batch = [], []  # Reset batches for the next windows

            # handle remaining data in the batch
            if x_batch:
                yield np.array(x_batch), np.array(y_batch)
                x_batch, y_batch = [], []

    def get_generators(self) -> tuple[Generator[tuple[np.ndarray, np.ndarray], None, None], ...]:
        """Returns all generators in the order: train, val, test"""
        return (self.generator('train'), self.generator('val'), self.generator('test'))

    def preprocess_trial(self, neuron_data: np.ndarray, mvc: float) -> np.ndarray:
        """Preprocess a single trial of neuron data to be used as model input."""
        # transpose neuron data to shape (time_steps, neurons)
        neuron_data = neuron_data.T  # (time_steps, neurons)

        # add MVC as an additional feature
        mvc_column = np.full((neuron_data.shape[0], 1), mvc)  # time_steps, 1)
        trial_data = np.concatenate([neuron_data, mvc_column], axis=1)  # (time_steps, neurons + 1)

        # creating sliding windows
        trial_windows = Processing._create_sliding_windows(trial_data, sequence_length=self.sequence_length, stride=self.stride)

        return trial_windows  # (num_windows, sequence_length, features)
    
    def postprocess_prediction(self, neuron_data: np.ndarray, predicted_force: np.ndarray) -> np.ndarray:
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