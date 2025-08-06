from typing import Iterator
from optimization.types import LossFunction, ArchitectureIdentifier
from optimization.params import ModelCandidate, Architecture, Hyperparameters, TrainingHyperparameters, PreprocessingHyperparameters, architectures, losses
from dataclasses import dataclass
import json
import hashlib
import numpy as np
import processing.constants
import prediction.constants

@dataclass
class StateSpaceRange:
    """Range of possible values for a gene"""
    start: float | int
    end: float | int
    step: float | int = 1
    optional: bool = False

    def __iter__(self) -> Iterator[int | float]:
        if self.optional:
            yield -1
        current = self.start
        if self.step > 0:
            while current < self.end:
                # Round to nearest significant value of step if float
                if isinstance(self.step, float):
                    decimals = abs(int(np.floor(np.log10(abs(self.step))))) if self.step != 0 else 0
                    yield round(current, decimals)
                else:
                    yield current
                current += self.step
            # Ensure end is included
            if isinstance(self.step, float):
                decimals = abs(int(np.floor(np.log10(abs(self.step))))) if self.step != 0 else 0
                yield round(self.end, decimals)
            else:
                yield self.end
        elif self.step < 0:
            while current > self.end:
                if isinstance(self.step, float):
                    decimals = abs(int(np.floor(np.log10(abs(self.step))))) if self.step != 0 else 0
                    yield round(current, decimals)
                else:
                    yield current
                current += self.step
            # Ensure end is included
            if isinstance(self.step, float):
                decimals = abs(int(np.floor(np.log10(abs(self.step))))) if self.step != 0 else 0
                yield round(self.end, decimals)
            else:
                yield self.end
        else:
            raise ValueError("Step cannot be zero")

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start value must be less than or equal to end value.")

@dataclass
class GeneSpace:
    """Possible values (ranges/items) for each gene within the genome"""
    architecture_identifier: list[ArchitectureIdentifier]
    units: StateSpaceRange
    subject_embedding_dimension: StateSpaceRange
    sequence_length: StateSpaceRange
    stride_divisor: StateSpaceRange
    bin_size: StateSpaceRange
    exponential_decay_lifetime: StateSpaceRange
    size_amplification_factor: StateSpaceRange
    loss: list[LossFunction]

    def parameterize(self) -> list[list[int | float]]:
        """Convert the gene space dataclass into an acceptable format for pygad"""
        return [ 
            list(g) if isinstance(g, StateSpaceRange) else [i for i in range(len(g))]
            for g in self.__dict__.values()
        ]
    
    def count(self) -> int:
        """Count the total number of genes in the gene space"""
        return len(self.__dict__.keys())

@dataclass
class Genes:
    """Dataclass representing the genes of a genome with helper methods for conversion and hashing."""
    architecture_identifier: ArchitectureIdentifier = "LSTM"
    units: int = 64
    subject_embedding_dimension: int | None = prediction.constants.subject_embedding_dimension
    sequence_length: int = prediction.constants.sequence_length
    stride_divisor: int = 2 # stride is calculated as sequence_length / stride_divisor
    bin_size: int = processing.constants.bin_size
    exponential_decay_lifetime: int = processing.constants.exponential_decay_lifetime
    size_amplification_factor: float = processing.constants.size_amplification_factor
    loss: LossFunction = "mse"

    def hash(self) -> str:
        """Return a hash representation of the genes."""
        string_representation = json.dumps(vars(self), sort_keys=True)
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]
    
    @staticmethod
    def from_array(array: np.ndarray) -> 'Genes':
        """Construct a Genes instance from an array, ensuring all properties exist and are cast."""
        if len(array) != 9:
            raise ValueError("Array must have exactly 9 elements corresponding to all genes.")

        return Genes(
            architecture_identifier= architectures[int(array[0])],
            units=int(array[1]),
            subject_embedding_dimension=None if array[2] == -1 else int(array[2]), # -1 means no subject embedding,
            sequence_length=int(array[3]),
            stride_divisor=int(array[4]),
            bin_size=int(array[5]),
            exponential_decay_lifetime=int(array[6]),
            size_amplification_factor=array[7],
            loss=losses[int(array[8])]
        )
    
    def to_array(self) -> np.ndarray:
        """Convert the genes to a numpy array for use in pygad"""
        return np.array([
            architectures.index(self.architecture_identifier),
            self.units,
            -1 if self.subject_embedding_dimension is None else self.subject_embedding_dimension, # subject embedding dimension, -1 -> no embedding
            self.sequence_length,
            self.stride_divisor,
            self.bin_size,
            self.exponential_decay_lifetime,
            self.size_amplification_factor,
            losses.index(self.loss)
        ], dtype=float)
    
    @staticmethod
    def from_model_candidate(candidate: ModelCandidate) -> 'Genes':
        """Construct a Genes instance from a ModelCandidate, ensuring all properties exist and are cast."""
        if not isinstance(candidate, ModelCandidate):
            raise TypeError("candidate must be an instance of ModelCandidate.")
        
        return Genes(
            architecture_identifier=candidate.architecture.identifier,
            units=candidate.architecture.units,
            subject_embedding_dimension=candidate.hyperparameters.training.subject_embedding_dimension,
            sequence_length=candidate.hyperparameters.training.sequence_length,
            stride_divisor=candidate.hyperparameters.training.sequence_length // candidate.hyperparameters.training.stride,
            bin_size=candidate.hyperparameters.preprocessing.bin_size,
            exponential_decay_lifetime=candidate.hyperparameters.preprocessing.exponential_decay_lifetime,
            size_amplification_factor=candidate.hyperparameters.preprocessing.size_amplification_factor,
            loss=candidate.hyperparameters.training.loss
        )
    
    def to_model_candidate(self) -> ModelCandidate:
        """Convert the genome to a ModelCandidate for metrics and evaluation"""
        return ModelCandidate( #/ properties that aren't specified use constant defaults
            architecture=Architecture.from_config(self.architecture_identifier, self.units),
            hyperparameters=Hyperparameters(
                training=TrainingHyperparameters(
                    sequence_length=self.sequence_length,
                    stride=self.sequence_length // self.stride_divisor,
                    subject_embedding_dimension=self.subject_embedding_dimension,
                    loss=self.loss
                ),
                preprocessing=PreprocessingHyperparameters(
                    bin_size=self.bin_size,
                    exponential_decay_lifetime=self.exponential_decay_lifetime,
                    size_amplification_factor=self.size_amplification_factor
                )
            )
        )
    
    def __str__(self) -> str:
        return json.dumps(vars(self), indent=2)