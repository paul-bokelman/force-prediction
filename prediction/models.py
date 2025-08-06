from typing import cast
import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from keras._tf_keras.keras.models import Model, load_model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.saving import register_keras_serializable
from keras._tf_keras.keras.metrics import MeanSquaredError, MeanAbsoluteError, R2Score
from processing.processors import Preprocessor, Postprocessor
from optimization.params import ModelCandidate, Architecture
from globals.utils import Log
from prediction import constants
from processing.utils import get_subject_mappings

@register_keras_serializable(package='prediction')
class ModifiedModel(Model):
    def __init__(self, *args, **kwargs):
        super(ModifiedModel, self).__init__(*args, **kwargs)
    
    def predict_force(self, neuron_data: np.ndarray, subject_id: int, sequence_length: int, stride: int) -> np.ndarray:
        """Predict the force sequence of a given neuronal sequence."""
        time_steps = neuron_data.shape[1] # get original number of time steps
        neuron_data = Preprocessor.window_neuron_data(neuron_data.T, sequence_length, stride)
        subject_ids = np.array([subject_id] * neuron_data.shape[0], dtype="int8") # create an array of subject IDs for the batch
        raw_force_predictions = np.array(self.call({"neural_input": neuron_data, "subject_id": subject_ids})) # call the model with the input data
        averaged_force = Postprocessor.overlap_average(raw_force_predictions, time_steps, stride)
        return savgol_filter(averaged_force, stride, 8) # apply Savitzky-Golay filter to smooth the predictions
    
def get_weights_path(hash: str) -> str:
    """Get the path to the saved model weights."""
    return os.path.join(constants.candidate_out_dir(hash), "weights.keras")

def obtain(hash: str) -> ModifiedModel | None:
    """Get the saved model if it exists, otherwise create a new model."""
    weights_path = get_weights_path(hash)
    if os.path.exists(weights_path):
        return cast(ModifiedModel, load_model(weights_path))

def train(
        preprocessor: Preprocessor, 
        candidate: ModelCandidate, 
        save: bool = True, 
        train_on_val: bool = False,
        verbose: str | int = "auto"
    ) -> tuple[ModifiedModel, dict[str, list[float]]]:
    """Train and evaluate the LSTM model using a data generator."""
    candidate_hash = candidate.hash() # get the unique hash for the candidate

    Log.debug(f"[{candidate_hash}] Starting Training")

    # create the candidate directory if it does not exist
    if not os.path.exists(constants.candidate_out_dir(candidate_hash)) and save:
        os.makedirs(constants.candidate_out_dir(candidate_hash))

    # construct the model architecture
    inputs, output = Architecture.construct(
        neural_input_shape=preprocessor.input_shape, 
        n_subjects=len(get_subject_mappings()), #/ unnecessary computation when not using subject embeddings
        architecture=candidate.architecture,
        subject_embedding_dimension=candidate.hyperparameters.training.subject_embedding_dimension
    )
    model = ModifiedModel(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer='adam', 
        loss=candidate.hyperparameters.training.loss, 
        metrics=[MeanSquaredError(name='mse'), MeanAbsoluteError(name='mae'), R2Score(name='r2')]
    )

    # training callbacks
    callbacks: list = [
        EarlyStopping(monitor='val_loss', patience=candidate.hyperparameters.training.early_stopping_patience, restore_best_weights=True)
    ]

    if save: # only save checkpoints if specified
        callbacks.append(
            ModelCheckpoint(get_weights_path(candidate_hash), save_best_only=True, monitor='val_loss')
        )

    train_windows = preprocessor.compute_dataset_windows("train&val" if train_on_val else "train")
    train_steps_per_epoch = train_windows // candidate.hyperparameters.training.batch_size
    val_steps_per_epoch = (preprocessor.compute_dataset_windows("val") // candidate.hyperparameters.training.batch_size) if not train_on_val else None

    # train the model with callbacks and save the training history
    history = model.fit(
        preprocessor.generator('train&val' if train_on_val else 'train'),
        steps_per_epoch=train_steps_per_epoch,
        batch_size=candidate.hyperparameters.training.batch_size,
        validation_data=None if train_on_val else preprocessor.generator('val'),
        validation_steps=val_steps_per_epoch,
        epochs=candidate.hyperparameters.training.epochs,
        callbacks=callbacks,
        verbose=cast(str, verbose) #/ could be int or str, but Keras expects a string for verbosity
    )

    # save training history only if specified and not training on validation data
    if save and not train_on_val:
        pd.DataFrame(history.history).to_pickle(os.path.join(constants.candidate_out_dir(candidate_hash), "history.pkl"))

    Log.debug(f"[{candidate_hash}] Training completed successfully")
    return model, history.history