from processing.types import PreprocessedData, PreprocessedDataSplit
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from globals.utils import Log
import processing.constants as constants
import prediction.constants

class Processing:
    """Preprocesses the neuronal and force data into sequences for model training."""
    def __init__(self, df: pd.DataFrame, model_info: dict = prediction.constants.models[prediction.constants.default_model_name]) -> None:
        self.df = df
        self.max_n_neurons: int = max([d.shape[0] for d in df['neuron_data']])
        self.max_activations: int = max([d.shape[1] for d in df['neuron_data']])
        self.max_force_len: int = max([len(d) for d in df['force_data']])
        self.sequence_length, self.stride = model_info['sequence_length'], model_info['stride']

        # Ensure activations and force data lengths match
        if self.max_activations != self.max_force_len:
            raise ValueError("The number of activations and force data lengths must be equal.")
        
        if model_info['name'] != prediction.constants.default_model_name:
            Log.info(f"Using {model_info['name']} presets (sl={self.sequence_length}, s={self.stride})")
        else:
            Log.info("Using default model presets ")
        
    @staticmethod
    def _create_sliding_windows(data: np.ndarray, sequence_length = constants.sequence_length, stride = constants.stride) -> np.ndarray:
        """Create sliding windows from the data."""
        num_windows = (data.shape[0] - sequence_length) // stride + 1
        windows = np.array([
            data[start:start + sequence_length]
            for start in range(0, num_windows * stride, stride)
        ])
        return windows

    def preprocess(self) -> PreprocessedData:
        """Preprocess the neuronal and force data into sequences for model training."""
        x_windows, y_windows = [], []

        for trial in self.df.itertuples():
            # extract neuron data and force data from the trial
            neuron_data = np.array(trial.neuron_data).T  # shape: (time_steps, neurons)
            force_data = np.array(trial.force_data)  # shape: (time_steps,)
            mvc = trial.mvc_level

            # add mvc as an additional feature
            mvc_column = np.full((neuron_data.shape[0], 1), mvc)  # shape: (time_steps, 1)
            trial_data = np.concatenate([neuron_data, mvc_column], axis=1)  # shape: (time_steps, neurons + 1)

            # create sliding windows for neuron data and force data
            neuron_windows = Processing._create_sliding_windows(trial_data, self.sequence_length, self.stride)
            force_windows = Processing._create_sliding_windows(force_data[:, np.newaxis], self.sequence_length, self.stride)

            x_windows.extend(neuron_windows)
            y_windows.extend(force_windows.squeeze(axis=-1))

        X = np.array(x_windows)  # shape: (samples, sequence_length, features)
        y = np.array(y_windows)  # shape: (samples, sequence_length)

        Log.info(f"preprocessed data shapes: X {X.shape}, y {y.shape}")

        # split data without shuffling to maintain time-series integrity
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # define input shape and output shape
        input_shape = (X_train.shape[1], X_train.shape[2])

        # return preprocessed data in formatted data container
        return PreprocessedData(
            X=PreprocessedDataSplit(train=X_train, test=X_test, val=X_val),
            y=PreprocessedDataSplit(train=y_train, test=y_test, val=y_val),
            input_shape=input_shape,
            output_dim=1
        )

    @staticmethod
    def preprocess_trial(neuron_data: np.ndarray, mvc: float) -> np.ndarray:
        """Preprocess a single trial of neuron data to be used as model input."""
        # Transpose neuron data to shape (time_steps, neurons)
        neuron_data = neuron_data.T  # Shape: (time_steps, neurons)

        # Add MVC as an additional feature
        mvc_column = np.full((neuron_data.shape[0], 1), mvc)  # Shape: (time_steps, 1)
        trial_data = np.concatenate([neuron_data, mvc_column], axis=1)  # Shape: (time_steps, neurons + 1)

        # Create sliding windows
        trial_windows = Processing._create_sliding_windows(trial_data)

        return trial_windows  # Shape: (num_windows, sequence_length, features)
    
    def postprocess_prediction(self, trial: pd.Series, predicted_force: np.ndarray) -> np.ndarray:
        """Postprocess the model prediction to match the original trial data."""
        neuron_data = np.array(trial['neuron_data'])

        def butter_lowpass_filter(data, cutoff=5, fs=1000, order=4):
            nyq = 0.5 * fs  # Nyquist Frequency
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)

        # find first and last activations
        first_activation_index: int = min([n for n in np.argmax(neuron_data == 1, axis=1) if n != 0])
        last_activation_index: int = max(neuron_data.shape[1] - np.array([n for n in np.argmax(neuron_data[:, ::-1] == 1, axis=1) if n != 0]) - 1)

        print(first_activation_index, last_activation_index)
        predicted_force = butter_lowpass_filter(predicted_force)

        # clip leading/trailing predicted force data that has or will hit zero
        np.clip(predicted_force, a_min=0, a_max=np.inf, out=predicted_force)

        # Set all force values before the first activation and after the last activation to zero
        predicted_force[:first_activation_index] = 0
        predicted_force[last_activation_index + 1:] = 0


        return predicted_force