from processing.types import PreprocessedData, PreprocessedDataSplit
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from globals.utils import Log

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

    def _normalize_and_scale(self):
            """Scale force data and normalize neuron data"""
            neuron_data, force_data = np.array(self.neuron_data_series), np.array(self.force_data_series)

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

            #/ weird conversion to list and back to array to avoid numpy broadcasting issues
            return np.array(list(neuron_data)), np.array(list(force_data))

    def preprocess(self) -> PreprocessedData:
        """Preprocess the neuronal and force data into sequences for model training."""
        neuron_data, force_data = self._normalize_and_scale()

        # construct X (trials, columns, features[...activations, mvc]) and y (trials, force_data)
        X = np.array([neuron_data[i].T for i in range(len(neuron_data))]) # (trials, columns, features[activations])
        encoded_mvc = np.array(self.mvc_series)[:, np.newaxis].repeat(self.max_activations, axis=1)[...,np.newaxis] # (trials, max_activations, 1)
        X = np.concatenate((X, encoded_mvc), axis=2) # (trials, columns, features[...activations, mvc])
        y = force_data # combine all force data

        Log.info(f"Preprocessed data shapes: X {X.shape}, y {y.shape}")

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # define input shape and output shape
        input_shape, output_shape = (X_train.shape[1], X_train.shape[2]), y_train.shape[1]
        
        # return preprocessed data in formatted data container
        return PreprocessedData(X=PreprocessedDataSplit(train=X_train, test=X_test, val=X_val),
                                y=PreprocessedDataSplit(train=y_train, test=y_test, val=y_val),
                                input_shape=input_shape, output_shape=output_shape)