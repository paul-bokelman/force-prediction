from typing import cast, Any, Optional
from optimization.types import LossFunction, ArchitectureIdentifier
from dataclasses import dataclass, field
import os
import hashlib
import json
from keras._tf_keras.keras.layers import Layer, LSTM, Dense, Input, GRU, Bidirectional, TimeDistributed, Embedding, RepeatVector, Concatenate
import prediction.constants
import processing.constants

losses: list[LossFunction] = ["mse", "mae", "huber"]
architectures: list[ArchitectureIdentifier] = ["LSTM", "DualLSTM"]

@dataclass
class PreprocessingHyperparameters:
    bin_size: int = processing.constants.bin_size
    exponential_decay_lifetime: int = processing.constants.exponential_decay_lifetime
    size_amplification_factor: float = processing.constants.size_amplification_factor
    
    def hash(self) -> str:
        """Return a full hash representation of the preprocessing hyperparameters."""
        string_representation = json.dumps(vars(self), sort_keys=True)
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]

@dataclass
class TrainingHyperparameters:
    epochs: int = prediction.constants.epochs
    early_stopping_patience: int = prediction.constants.early_stopping_patience
    sequence_length: int = prediction.constants.sequence_length
    stride: int = prediction.constants.stride
    train_percentage: float = prediction.constants.train_percentage
    test_percentage: float = prediction.constants.test_percentage
    validation_percentage: float = prediction.constants.validation_percentage
    subject_embedding_dimension: int | None = prediction.constants.subject_embedding_dimension
    batch_size: int = prediction.constants.batch_size
    train_on_val: bool = False
    loss: LossFunction = 'mse'

    def hash(self) -> str:
        """Return a hash representation of the training hyperparameters."""
        string_representation = json.dumps(vars(self), sort_keys=True)
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]

@dataclass
class Hyperparameters:
    preprocessing: 'PreprocessingHyperparameters' = field(default_factory=PreprocessingHyperparameters)
    training: 'TrainingHyperparameters' = field(default_factory=TrainingHyperparameters)

@dataclass
class PartialArchitecture:
    """A partial architecture that can be used to construct a full model."""
    identifier: ArchitectureIdentifier
    units: int
    layers: list[Layer]

@dataclass
class ModelCandidate:
    architecture: PartialArchitecture
    identifier: str = "null"
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    version: int = 1 # used to change hash for same model representation

    def __str__(self) -> str:
        formatted_string = f"identifier: {self.identifier}\narchitecture: {self.architecture}\ntraining: \n{"\n".join([f'   {k}: {v}' for k, v in self.hyperparameters.training.__dict__.items()])}\npreprocessing: \n{"\n".join([f'   {k}: {v}' for k, v in self.hyperparameters.preprocessing.__dict__.items()])}"

        return formatted_string
    
    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the model candidate."""
        return {
            "identifier": self.identifier,
            "architecture": Architecture.string_representation(self.architecture),
            "hyperparameters": {
                "preprocessing": vars(self.hyperparameters.preprocessing),
                "training": vars(self.hyperparameters.training)
            },
            "version": self.version
        }
    
    def hash(self) -> str:
        """Return a hash representation of the model candidate."""
        string_representation = (
            self.identifier
            + Architecture.hash(self.architecture)
            + self.hyperparameters.preprocessing.hash()
            + self.hyperparameters.training.hash()
        )
        
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]
    
    @property
    def already_computed(self) -> bool:
        """Check if the model candidate has already been computed."""
        candidate_dir = prediction.constants.candidate_out_dir(self.hash())
        required_files = ["history.pkl", "weights.keras", "metrics.pkl"]
        if not os.path.exists(candidate_dir):
            return False
        return all(os.path.exists(os.path.join(candidate_dir, fname)) for fname in required_files)

class Architecture:
    """Reuseable model architectures for different candidates."""
    @staticmethod
    def LSTM(units: int) -> PartialArchitecture:
        return PartialArchitecture("LSTM", units, [LSTM(units, return_sequences=True)])
    
    @staticmethod 
    def DualLSTM(units: int) -> PartialArchitecture:
        return PartialArchitecture("DualLSTM", units, [LSTM(units, return_sequences=True), LSTM(units, return_sequences=True)])
    
    @staticmethod
    def from_config(identifier: str, units: int) -> PartialArchitecture:
        """Configures a partial architecture based on the identifier and units."""
        assert identifier in Architecture.__dict__, f"Unknown architecture identifier: {identifier}"
        return cast(PartialArchitecture, Architecture.__dict__[identifier](units))
    
    @staticmethod
    def construct(
        neural_input_shape: tuple[int,...], 
        n_subjects: int, 
        architecture: PartialArchitecture, 
        subject_embedding_dimension: Optional[int]
    ) -> tuple:
        """Constructs the full model architecture. If subject_embedding_dimension is provided, it adds an embedding layer for subjects."""
        use_subject_embedding = subject_embedding_dimension is not None and n_subjects > 1
        sequence_length, features = neural_input_shape
        
        neural_input = Input(shape=(sequence_length, features), name='neural_input')
        subject_input = Input(shape=(), dtype='int8', name='subject_id') # scalar id per sample

        # optionally construct subject embedding layer
        if use_subject_embedding:
            # subject embedding layer
            subject_embedding = Embedding(input_dim=n_subjects, output_dim=subject_embedding_dimension)(subject_input)
            subject_embedding = RepeatVector(sequence_length)(subject_embedding)  # shape: (batch_size, seq_len, embedding_dim)

            # concatenate embedding with neural input
            x = Concatenate(axis=-1)([neural_input, subject_embedding])  # shape: (batch_size, seq_len, input_dim + embedding_dim)
            inputs = [neural_input, subject_input]
        else:
            x = neural_input
            inputs = [neural_input]

        # apply the architecture layers
        for layer in architecture.layers:
            assert isinstance(layer, Layer), f"Invalid layer type: {type(layer)}. Expected a Keras Layer."
            x = layer(x)

        output = TimeDistributed(Dense(1))(x)
        return inputs, output #/ subject input does nothing when not using subject embeddings
    
    @staticmethod
    def hash(architecture: PartialArchitecture) -> str:
        """Return a hash representation of the architecture."""
        return hashlib.sha256(Architecture.string_representation(architecture).encode()).hexdigest()[:8]
    
    @staticmethod
    def string_representation(architecture: PartialArchitecture) -> str:
        """Return a string representation of the architecture."""
        #/ this will fail if a layer in the architecture does not have a 'units' property
        return ' -> '.join([f"{layer.__class__.__name__.lower()}({layer.units})" for layer in architecture.layers])