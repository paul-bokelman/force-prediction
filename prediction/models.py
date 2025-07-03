from typing import cast
import os
import numpy as np
import pandas as pd
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras._tf_keras.keras.saving import register_keras_serializable
from keras._tf_keras.keras.metrics import MeanSquaredError, MeanAbsoluteError, R2Score
from processing.processors import Preprocessor, Postprocessor
from prediction.tuning import ModelCandidate, Architectures
from globals.utils import Log
from prediction import constants

@register_keras_serializable(package='prediction')
class ModifiedSequential(Sequential):
    def __init__(self, *args, **kwargs):
        super(ModifiedSequential, self).__init__(*args, **kwargs)
    
    def predict_force(self, neuron_data: np.ndarray, sequence_length: int, stride: int) -> np.ndarray:
        """Predict the force sequence of a given neuronal sequence."""
        time_steps = neuron_data.shape[1] # get original number of time steps
        neuron_data = Preprocessor.window_neuron_data(neuron_data.T, sequence_length, stride)
        raw_force_predictions = np.array(self.call(neuron_data))
        force_averages = Postprocessor.overlap_average(raw_force_predictions, time_steps, sequence_length, stride)
        # return Postprocessor.smooth_and_normalize(neuron_data, force_aversages)
        return force_averages
    
def get_weights_path(hash: str) -> str:
    """Get the path to the saved model weights."""
    return os.path.join(constants.candidate_out_dir(hash), "weights.keras")

def obtain(hash: str) -> ModifiedSequential | None:
    """Get the saved model if it exists, otherwise create a new model."""
    weights_path = get_weights_path(hash)
    if os.path.exists(weights_path):
        return cast(ModifiedSequential, load_model(weights_path))
    
def train(preprocessor: Preprocessor, candidate: ModelCandidate) -> tuple[ModifiedSequential, dict[str, list[float]]]:
    """Train and evaluate the LSTM model using a data generator."""
    candidate_hash = candidate.hash() # get the unique hash for the candidate

    Log.info(f"[{candidate_hash}] Starting Training")

    # create the candidate directory if it does not exist
    if not os.path.exists(constants.candidate_out_dir(candidate_hash)):
        os.makedirs(constants.candidate_out_dir(candidate_hash))

    # setup and compile model architecture
    model = ModifiedSequential(Architectures.construct(input_shape=preprocessor.input_shape, architecture=candidate.architecture))
    model.compile(
        optimizer='adam', 
        loss=candidate.hyperparameters.training.loss, 
        metrics=[MeanSquaredError(name='mse'), MeanAbsoluteError(name='mae'), R2Score(name='r2')]
    )

    # training callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=candidate.hyperparameters.training.early_stopping_patience, restore_best_weights=True),
        ModelCheckpoint(get_weights_path(candidate_hash), save_best_only=True, monitor='val_loss'),
        TensorBoard(log_dir=constants.tensorboard_log_dir, histogram_freq=1)
    ]

    train_steps_per_epoch = preprocessor.compute_dataset_windows("train") // candidate.hyperparameters.preprocessing.batch_size
    val_steps_per_epoch = preprocessor.compute_dataset_windows("val") // candidate.hyperparameters.preprocessing.batch_size

    Log.debug(f"Training steps per epoch: {train_steps_per_epoch}, Validation steps per epoch: {val_steps_per_epoch}")

    # train the model with callbacks and save the training history
    history = model.fit(
        preprocessor.generator('train'),
        steps_per_epoch=train_steps_per_epoch,
        batch_size=candidate.hyperparameters.preprocessing.batch_size,
        validation_data=preprocessor.generator('val'),
        validation_steps=val_steps_per_epoch,
        epochs=candidate.hyperparameters.training.epochs,
        callbacks=callbacks,
    )

    # save history to disk
    pd.DataFrame(history.history).to_pickle(os.path.join(constants.candidate_out_dir(candidate_hash), "history.pkl"))

    Log.info(f"[{candidate_hash}] Training completed successfully")
    
    return model, history.history