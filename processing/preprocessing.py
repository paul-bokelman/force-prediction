from processing.types import PreprocessedData, PreprocessedDataSplit
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from sklearn.model_selection import train_test_split
from globals.utils import Log
import processing.constants as constants

class Preprocessing:
    """Preprocesses the neuronal and force data into sequences for model training."""
    def __init__(self, neuron_data_series: pd.Series, force_data_series: pd.Series, mvc_series: pd.Series) -> None:
        self.neuron_data_series: pd.Series = neuron_data_series
        self.force_data_series: pd.Series = force_data_series
        self.mvc_series: pd.Series = mvc_series
        self.max_n_neurons: int = max([d.shape[0] for d in neuron_data_series])
        self.max_activations: int = max([d.shape[1] for d in neuron_data_series])
        self.max_force_len: int = max([len(d) for d in force_data_series])

        #/ praying to our lord and savior this is never raised
        if self.max_activations != self.max_force_len:
            raise ValueError("The number of activations and force data lengths must be equal.")

    def preprocess(self) -> PreprocessedData:
        """Preprocess the neuronal and force data into sequences for model training."""
        neuron_data, force_data = np.array(list(self.neuron_data_series)), np.array(list(self.force_data_series)) #/ stupid casting hack for dims

        X_list = []
        y_list = []

        for neuron_data, force_data, mvc in zip(neuron_data, force_data, self.mvc_series):
            T = force_data.shape[0]
            mvc_row = np.full((1, T), mvc) # [1, T]
            input_matrix = np.vstack([neuron_data, mvc_row]) # [N+1, T]
            input_matrix = input_matrix.T # [T, N+1]

            X_list.append(input_matrix.astype(np.float32)) # [T, N+1]
            y_list.append(force_data.astype(np.float32)) # [T]

        X = np.stack(X_list) # [num_trials, T, N+1]
        y = np.stack(y_list) # [num_trials, T]

        Log.info(f"Preprocessed data shapes: X {X.shape}, y {y.shape}")

        # split data without shuffling to maintain time-series integrity
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # define input shape and output shape
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # return preprocessed data in formatted data container
        return PreprocessedData(X=PreprocessedDataSplit(train=X_train, test=X_test, val=X_val),
                                y=PreprocessedDataSplit(train=y_train, test=y_test, val=y_val),
                                input_shape=input_shape, output_dim=1)
    
    @staticmethod
    def preprocess_trial(neuron_data: np.ndarray, mvc: float) -> np.ndarray:
        """Preprocess a single trial of neuron data to be used as model input."""
        T = neuron_data.shape[1]

        mvc_row = np.full((1, T), mvc)
        input_matrix = np.vstack([neuron_data, mvc_row]).T  # [T, N+1]
        return input_matrix[np.newaxis, :, :]  # shape [1, T, N+1]