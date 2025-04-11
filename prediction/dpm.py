from typing import cast
from processing.types import PreprocessedData
import os
import pandas as pd
import numpy as np
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Input, Bidirectional, Dropout
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from processing.processing import Processing
from globals.utils import Log
import prediction.constants as constants

class DirectPredictionModel:
    """Direct prediction model (DPM) architecture (LSTM) and methods for training and evaluation."""
    def __init__(self, data: PreprocessedData, overwrite: bool = False, model_info: dict = constants.models[constants.default_model_name]):
        self.data = data
        self.model_info = model_info
        self.sequence_length, self.stride = model_info['sequence_length'], model_info['stride']

        if model_info['name'] != constants.default_model_name:
            Log.info(f"Using {model_info['name']} presets (sl={self.sequence_length}, s={self.stride})")
        else:
            Log.info("Using default model presets ")

        self.model_path = os.path.join('prediction/', constants.saves_directory, self.model_info['name'] + ".keras")

        # create the saves directory if it does not exist
        if not os.path.exists('prediction/' + constants.saves_directory):
            os.makedirs('prediction/' + constants.saves_directory)

        self.trained = False
        self.get_model(overwrite=overwrite) # get the current model

    def _create_model(self):
        """Create an LSTM model for neuronal to force prediction."""
       
        model = Sequential([
            Input(shape=self.data.input_shape),

            # 2x-lstm
            # LSTM(128, return_sequences=True),
            # LSTM(128, return_sequences=True),
            # Dense(1),

            # 2x-bi-lstm
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)

            # single-lstm
            # LSTM(256, return_sequences=True),
            # Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def get_model(self, overwrite: bool = False):
        """Get the saved model if it exists, otherwise create a new model."""

        # overwrite or model not saved -> create a new model
        if overwrite or not os.path.exists(self.model_path):
            Log.info('Model not found or overwrite flag is set. Creating a new model...')
            self.trained = False
            self.model = self._create_model()
        else:
            Log.info('Loading the existing model...')
            # load the existing model (pre-trained)
            self.trained = True
            self.model = cast(Sequential, load_model(self.model_path))

        return self.model
    
    def optionally_train(self, epochs: int = constants.training_epochs, batch_size: int = constants.training_batch_size):
        """Train the model if it is not already trained."""
        if not self.trained:
            Log.info('Training the model...')
            return self._train(epochs=epochs, batch_size=batch_size)
        else:
            Log.info('Model is already trained. Skipping training...')

    def _train(self, epochs: int = constants.training_epochs, batch_size: int = constants.training_batch_size):
        """Train and evaluate the LSTM model using a data generator."""

        # training callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=constants.early_stopping_patience, restore_best_weights=True),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss')
        ]

        # train the model with callbacks and save the training history
        history = self.model.fit(
            self.data.X.train, self.data.y.train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.data.X.val, self.data.y.val),
            callbacks=callbacks,
        )

        return history
    
    def overlap_average(self, predicted_windows: np.ndarray, trial_length: int):
        """Reconstructs the full prediction from overlapping window predictions using overlap averaging."""
        full_pred = np.zeros(trial_length)
        weight_sum = np.zeros(trial_length)

        for i, window in enumerate(predicted_windows):
            start = i * self.stride
            end = start + self.sequence_length

            full_pred[start:end] += window.squeeze()
            weight_sum[start:end] += 1

        weight_sum[weight_sum == 0] = 1 # avoid division by zero
        return full_pred / weight_sum
    
    def predict(self, trial: pd.Series) -> np.ndarray:
        """Predict a force profile given neuron data, accounting for stride and window size."""
        neuron_data, mvc = trial['neuron_data'], trial['mvc_level']
        
        # preprocess, predict, and stitch windows
        x = Processing.preprocess_trial(np.array(list(neuron_data)), mvc)
        predicted_windows = self.model.predict(x)
        full_prediction = self.overlap_average(predicted_windows, trial_length=neuron_data.shape[1])
        
        return full_prediction
