from typing import cast
from processing.types import PreprocessedData
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import Sequential, load_model
from keras.api.layers import Bidirectional, LSTM, Conv1D, Dense, Input, Reshape
from keras.api.callbacks import EarlyStopping, ModelCheckpoint
from processing.preprocessing import Preprocessing
from globals.utils import Log
import prediction.constants as constants

class DirectPredictionModel:
    """Direct prediction model (DPM) architecture (LSTM) and methods for training and evaluation."""
    def __init__(self, preprocessed_data: PreprocessedData, overwrite: bool = False):
        self.data = preprocessed_data
        self.saved_model_path = os.path.join('prediction/', constants.saves_directory, constants.saved_model_name)

        # create the saves directory if it does not exist
        if not os.path.exists('prediction/' + constants.saves_directory):
            os.makedirs('prediction/' + constants.saves_directory)

        self.trained = False
        self.get_model(overwrite=overwrite) # get the current model

    def _create_model(self):
        """Create an LSTM model for neuronal to force prediction."""
       
        model = Sequential([
            Input(shape=self.data.input_shape), # input_shape: (T, N+1)

            # short-term pattern detection
            Conv1D(32, kernel_size=5, activation='relu', padding='same'),
            Conv1D(32, kernel_size=5, activation='relu', padding='same'),

            # long term context
            Bidirectional(LSTM(128, return_sequences=True)), 
            Dense(64, activation='relu'),

            # output and reshape
            Dense(1),
            Reshape((-1,)) # shape: (T,)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae']) # compile the model
        
        return model
    
    def get_model(self, overwrite: bool = False):
        """Get the saved model if it exists, otherwise create a new model."""

        # overwrite or model not saved -> create a new model
        if overwrite or not os.path.exists(self.saved_model_path):
            Log.info('Model not found or overwrite flag is set. Creating a new model...')
            self.trained = False
            self.model = self._create_model()
        else:
            Log.info('Loading the existing model...')
            # load the existing model (pre-trained)
            self.trained = True
            self.model = cast(Sequential, load_model(self.saved_model_path))

        # self.model.summary() # print the model summary
        
        return self.model
    
    def optionally_train(self, epochs: int = constants.training_epochs, batch_size: int = constants.training_batch_size):
        """Train the model if it is not already trained."""
        if not self.trained:
            Log.info('Training the model...')
            return self._train(epochs=epochs, batch_size=batch_size)
        else:
            Log.info('Model is already trained. Skipping training...')

    def _train(self, epochs: int = constants.training_epochs, batch_size: int = constants.training_batch_size):
        """Train and evaluate the LSTM model."""
        
        # training callbacks
        callbacks = [
            # stop training if validation loss does not improve for x epochs
            EarlyStopping(monitor='val_loss', patience=constants.early_stopping_patience, restore_best_weights=True),
            ModelCheckpoint(self.saved_model_path, save_best_only=True, monitor='val_loss') # save the best model based on validation loss
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
    
    def predict(self, neuron_data: pd.Series, mvc: int) -> np.ndarray:
        """Predict a force profile given neuron data"""
        x = Preprocessing.preprocess_trial(np.array(list(neuron_data)), mvc)
        force = self.model.predict(x) # preprocess the trial data to match model input
        return force.flatten()

    def visualize_results(self):
        """Visualize the model predictions vs actual force profiles."""
        predictions = self.model.predict(self.data.X.test)

        plt.figure(figsize=(12, 6))
        
        # plot training history
        plt.plot(self.data.y.test[:200], label='True Force')
        plt.plot(predictions[:200], label='Predicted Force')
        plt.legend()
        plt.title(f'Predicted Force')
        plt.xlabel('Time Steps')
        plt.ylabel('Force Value')
        
        plt.tight_layout()
        plt.show()