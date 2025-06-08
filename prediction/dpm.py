from typing import cast
from prediction.types import ArchitectureName
import os
import pandas as pd
import numpy as np
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout, Conv1D, Reshape
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from processing.processing import Processing
from globals.utils import Log
from prediction import constants

class DirectPredictionModel:
    """Direct prediction model (DPM) architecture (LSTM) and methods for training and evaluation."""
    def __init__(self, processor: Processing, architecture_name: ArchitectureName, overwrite: bool = False):
        self.processor = processor
        self.input_shape = (self.processor.sequence_length, self.processor.max_n_neurons + 1)

        self.model_identifier = f'{architecture_name}-{self.processor.sequence_length}-{self.processor.stride}'
        self.model_path = os.path.join('prediction/', constants.saves_directory, self.model_identifier + ".keras")

        # create the saves directory if it does not exist
        if not os.path.exists('prediction/' + constants.saves_directory):
            os.makedirs('prediction/' + constants.saves_directory)

        self.trained = False

        # different model architectures
        self.architectures: dict[ArchitectureName, Sequential] = {
            "single-lstm": Sequential([
                Input(shape=self.input_shape),
                LSTM(128, return_sequences=True),
                Dense(1)
            ]),

            "2x-lstm": Sequential([
                Input(shape=self.input_shape),
                LSTM(128, return_sequences=True),
                LSTM(128, return_sequences=True),
                Dense(1)
            ]),

            "2x-bi-lstm": Sequential([
                Input(shape=self.input_shape),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.2),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ]),

            "conv-bi-lstm": Sequential([
                Input(shape=self.input_shape),
                Conv1D(32, kernel_size=5, activation='relu', padding='same'),
                Conv1D(32, kernel_size=5, activation='relu', padding='same'),
                Bidirectional(LSTM(64, return_sequences=True)),
                Dense(1),
                Reshape((-1,))  
            ])
        }

        self.architecture = self.architectures[architecture_name]
        self.get_model(overwrite=overwrite) # get the current model

    def _create_model(self):
        """Create an LSTM model for neuronal to force prediction."""
        self.architecture.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = self.architecture
        return self.model
    
    def get_model(self, overwrite: bool = False):
        """Get the saved model if it exists, otherwise create a new model."""

        # overwrite or model not saved -> create a new model
        if overwrite or not os.path.exists(self.model_path):
            Log.warn(f"Training new model: {self.model_identifier}")
            self.trained = False
            self.model = self._create_model()
        else:
            Log.info(f"Loading existing model: {self.model_identifier}")
            self.trained = True
            self.model = cast(Sequential, load_model(self.model_path))

        return self.model
    
    def optionally_train(self):
        """Train the model if it is not already trained."""
        if not self.trained:
            return self._train()

    def _train(self):
        """Train and evaluate the LSTM model using a data generator."""

        Log.info(f"Training {self.model_identifier} | {constants.epochs} Epochs | {constants.batch_size} Batch Size")

        # training callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=constants.early_stopping_patience, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss')
        ]

        windows_per_trial = (self.processor.max_activations - self.processor.sequence_length) // self.processor.stride + 1
        train_windows = windows_per_trial * len(self.processor.train)
        validation_windows = windows_per_trial * len(self.processor.val)

        # train the model with callbacks and save the training history
        history = self.model.fit(
            self.processor.generator('train'),
            steps_per_epoch=train_windows // constants.batch_size,
            validation_data=self.processor.generator('val'),
            validation_steps=validation_windows // constants.batch_size,
            epochs=constants.epochs,
            callbacks=callbacks,
        )

        return history
    
    def _overlap_average(self, predicted_windows: np.ndarray, trial_length: int):
        """Reconstructs the full prediction from overlapping window predictions using overlap averaging."""
        full_pred = np.zeros(trial_length)
        weight_sum = np.zeros(trial_length)

        for i, window in enumerate(predicted_windows):
            start = i * self.processor.stride
            end = start + self.processor.sequence_length

            full_pred[start:end] += window.squeeze()
            weight_sum[start:end] += 1

        weight_sum[weight_sum == 0] = 1 # avoid division by zero
        return full_pred / weight_sum
    
    def predict(self, neuron_data: np.ndarray, mvc_level: int) -> np.ndarray:
        """Predict a force profile given neuron data, accounting for stride and window size."""

        # neuron_data = np.ndarray(list(neuron_data))

        # preprocess and predict windows
        x = self.processor.preprocess_trial(np.array(list(neuron_data)), mvc_level)
        predicted_windows = self.model.predict(x)

        # stitch and postprocess the predictions
        prediction = self._overlap_average(predicted_windows, trial_length=neuron_data.shape[1])
        prediction = self.processor.postprocess_prediction(neuron_data, prediction)
        
        return prediction
