from dataclasses import dataclass, field
import os
import hashlib
import json
from keras._tf_keras.keras.layers import Layer, LSTM, Dense, Input, GRU, Bidirectional
import prediction.constants
import processing.constants

@dataclass
class PreprocessingHyperparameters:
    sequence_length: int = processing.constants.sequence_length
    stride: int = processing.constants.stride
    train_percentage: float = processing.constants.train_percentage
    test_percentage: float = processing.constants.test_percentage
    validation_percentage: float = processing.constants.validation_percentage
    bin_size: int = processing.constants.bin_size
    exponential_decay_lifetime: float = processing.constants.exponential_decay_lifetime
    size_amplification_factor: float = processing.constants.size_amplification_factor
    batch_size: int = processing.constants.batch_size

    def hash(self) -> str:
        """Return a hash representation of the preprocessing hyperparameters."""

        # only consider the long-term storage hyperparameters for hashing
        string_representation = json.dumps({ "bin_size": self.bin_size,
            "exponential_decay_lifetime": self.exponential_decay_lifetime,
            "size_amplification_factor": self.size_amplification_factor,
            "batch_size": self.batch_size },
            sort_keys=True
        )
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]
    
    def full_hash(self) -> str:
        """Return a full hash representation of the preprocessing hyperparameters."""
        string_representation = json.dumps(vars(self), sort_keys=True)
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]

@dataclass
class TrainingHyperparameters:
    epochs: int = prediction.constants.epochs
    early_stopping_patience: int = prediction.constants.early_stopping_patience
    use_tensorboard: bool = True
    loss: str = 'mse'

    def hash(self) -> str:
        """Return a hash representation of the training hyperparameters."""
        string_representation = json.dumps(vars(self), sort_keys=True)
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]

@dataclass
class Hyperparameters:
    preprocessing: 'PreprocessingHyperparameters' = field(default_factory=PreprocessingHyperparameters)
    training: 'TrainingHyperparameters' = field(default_factory=TrainingHyperparameters)

@dataclass
class ModelCandidate:
    architecture: list[Layer]
    identifier: str = "null"
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    version: int = 1 # used to change hash for same model representation

    def __str__(self) -> str:
        formatted_string = f"identifier: {self.identifier}\narchitecture: {self.architecture}\ntraining: \n{"\n".join([f'   {k}: {v}' for k, v in self.hyperparameters.training.__dict__.items()])}\npreprocessing: \n{"\n".join([f'   {k}: {v}' for k, v in self.hyperparameters.preprocessing.__dict__.items()])}"

        return formatted_string
    
    def hash(self) -> str:
        """Return a hash representation of the model candidate."""
        string_representation = (
            self.identifier
            + Architectures.hash(self.architecture)
            + self.hyperparameters.preprocessing.full_hash()
            + self.hyperparameters.training.hash()
        )
        
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]
    
    @property
    def already_computed(self) -> bool:
        """Check if the model candidate has already been computed."""
        candidate_dir = prediction.constants.candidate_out_dir(self.hash())
        required_files = ["history.pkl", "weights.keras", "metrics.pkl", "report.html"]
        if not os.path.exists(candidate_dir):
            return False
        return all(os.path.exists(os.path.join(candidate_dir, fname)) for fname in required_files)

class Architectures:
    """Reuseable model architectures for different candidates."""

    @staticmethod
    def LSTM(units: int) -> list[Layer]:
        return [LSTM(units, return_sequences=True)]
    
    @staticmethod
    def GRU(units: int) -> list[Layer]:
        return [GRU(units, return_sequences=True)]
    
    @staticmethod
    def BidirectionalLSTM(units: int) -> list[Layer]:
        return [Bidirectional(LSTM(units, return_sequences=True))]
    
    @staticmethod
    def BidirectionalGRU(units: int) -> list[Layer]:
        return [Bidirectional(GRU(units, return_sequences=True))]
    
    @staticmethod
    def GRU_LSTM(gru_units: int, lstm_units: int) -> list[Layer]:
        return [GRU(gru_units, return_sequences=True), LSTM(lstm_units, return_sequences=True)]
    
    @staticmethod
    def construct(input_shape: tuple[int,...], architecture: list[Layer]) -> list[Layer]:
        """Constructs the full model architecture."""
        return [
            Input(shape=input_shape), *architecture, Dense(1)]
    
    @staticmethod
    def hash(architecture: list[Layer]) -> str:
        """Return a hash representation of the architecture."""
        return hashlib.sha256(''.join([layer.__class__.__name__.lower() for layer in architecture]).encode()).hexdigest()[:8]
    
candidates: list['ModelCandidate'] = [ #todo: unique architectures need to change hash
    ModelCandidate(version=1, architecture=Architectures.LSTM(units=32)),
    ModelCandidate(version=2, architecture=Architectures.LSTM(units=64)),
    ModelCandidate(version=3, architecture=Architectures.LSTM(units=128)),
    ModelCandidate(
        architecture=Architectures.LSTM(units=64),
        hyperparameters=Hyperparameters(
            preprocessing=PreprocessingHyperparameters(sequence_length=100, stride=50)
        )
    ),
    ModelCandidate(
        architecture=Architectures.LSTM(units=64),
        hyperparameters=Hyperparameters(
            preprocessing=PreprocessingHyperparameters(sequence_length=400, stride=200)
        )
    ),
    ModelCandidate(
        architecture=Architectures.LSTM(units=64),
        hyperparameters=Hyperparameters(
            preprocessing=PreprocessingHyperparameters(sequence_length=100, stride=10)
        )
    ),
]