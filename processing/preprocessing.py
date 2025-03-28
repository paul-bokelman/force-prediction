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

        # construct X (trials, columns, features[...activations, mvc]) and y (trials, force_data)
        x = np.array([neuron_data[i].T for i in range(len(neuron_data))]) # (trials, columns, features[activations])
        encoded_mvc = np.array(self.mvc_series)[:, np.newaxis].repeat(self.max_activations, axis=1)[...,np.newaxis] # (trials, max_activations, 1)
        x = np.concatenate((x, encoded_mvc), axis=2) # (trials, columns, features[...activations, mvc])

        X_windows = []
        y_targets = []

        for trial_idx in range(len(x)):
            # create sliding window views of the data per trial
            trial_x = sliding_window_view(x[trial_idx], constants.sequence_length, axis=0) 
            
            # corresponding force data
            trial_y = force_data[trial_idx, constants.sequence_length:]
            
            assert len(trial_x) == len(trial_y) + 1, "Mismatch in trial sequence lengths" # verify alignment
            
            trial_x = trial_x[:-1] # trim X to match y length
            
            X_windows.append(trial_x)
            y_targets.append(trial_y)

        # final additional shaping
        X = np.transpose(np.vstack(X_windows), (0, 2, 1)) # (samples, sequence_length, features)
        y = np.concatenate(y_targets).reshape(-1, 1) # (samples, 1)

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
        """
        Preprocess a single trial of neuron data to be used as model input.
        
        Args:
            neuron_data: Array of shape (neurons, activations)
            mvc: Maximum voluntary contraction value
            
        Returns:
            Processed data of shape (n_windows, sequence_length, features)
        """
        # Transpose to get (activations, neurons)
        x = neuron_data.T
        
        # Add MVC as additional feature
        encoded_mvc = np.full((x.shape[0], 1), mvc)
        x = np.concatenate((x, encoded_mvc), axis=1)
        
        # Create sliding windows
        windows = sliding_window_view(x, constants.sequence_length, axis=0)
        
        # Shape to match model input format (n_windows, sequence_length, features)
        windows = np.transpose(windows, (0, 2, 1))
        
        return windows