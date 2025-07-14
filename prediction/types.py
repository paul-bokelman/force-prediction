from typing import cast, Literal
import processing.constants
import prediction.constants
import json
import hashlib
import dataclasses

# ---------------------------------- models ---------------------------------- #

type LossFunction = Literal["mse", "mae", "huber"]

# --------------------------------- genetics --------------------------------- #

type ArchitectureIdentifier = Literal["LSTM", "DualLSTM"]

@dataclasses.dataclass
class StateSpaceRange:
    """Range of possible values for a gene"""
    start: float | int
    end: float | int
    step: int = 1

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start value must be less than or equal to end value.")

@dataclasses.dataclass
class GeneStateSpace:
    """Possible values (ranges/items) for each gene within the genome"""
    # ---------------------------------- general --------------------------------- #
    architecture_identifier: list[ArchitectureIdentifier]
    units: StateSpaceRange
    # ------------------------------- preprocessing ------------------------------ #
    bin_size_divisor: StateSpaceRange
    exponential_decay_lifetime: StateSpaceRange
    size_amplification_factor: StateSpaceRange
    # --------------------------------- training --------------------------------- #
    sequence_length: StateSpaceRange
    stride_divisor: StateSpaceRange
    subject_embedding_dimension: StateSpaceRange
    loss: list[LossFunction]

@dataclasses.dataclass
class Genes:
    """All editable genes in genome, exactly the same as state space in keys."""
    # ---------------------------------- general --------------------------------- #
    architecture_identifier: ArchitectureIdentifier = "LSTM"
    units: int = 64
    # ------------------------------- preprocessing ------------------------------ #
    bin_size_divisor: int = 100 # proportion of sampling frequency -> sampling frequency / bin_size_divisor
    exponential_decay_lifetime: int = processing.constants.exponential_decay_lifetime
    size_amplification_factor: int = int(processing.constants.size_amplification_factor * 10)
    # --------------------------------- training --------------------------------- #
    sequence_length: int = prediction.constants.sequence_length
    stride_divisor: int = 2 # stride is calculated as sequence_length / stride_divisor
    subject_embedding_dimension: int = prediction.constants.subject_embedding_dimension
    loss: LossFunction = "mse"

    def hash(self) -> str:
        """Return a hash representation of the genes."""
        string_representation = json.dumps(vars(self), sort_keys=True)
        return hashlib.sha256(string_representation.encode()).hexdigest()[:10]

    @staticmethod
    def from_dict(data: dict[str, str | int]):
        """Construct a Genes instance from a dictionary, ensuring all properties exist and are cast."""
        required_keys = [
            "architecture_identifier",
            "units",
            "subject_embedding_dimension",
            "sequence_length",
            "stride_divisor",
            "bin_size_divisor",
            "exponential_decay_lifetime",
            "size_amplification_factor",
            "loss",
        ]

        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required gene property: {key}")

        return Genes(
            architecture_identifier=cast(ArchitectureIdentifier, data["architecture_identifier"]),
            units=cast(int, data["units"]),
            subject_embedding_dimension=cast(int, data["subject_embedding_dimension"]),
            sequence_length=cast(int, data["sequence_length"]),
            stride_divisor=cast(int, data["stride_divisor"]),
            bin_size_divisor=cast(int, data["bin_size_divisor"]),
            exponential_decay_lifetime=cast(int, data["exponential_decay_lifetime"]),
            size_amplification_factor=cast(int, data["size_amplification_factor"]),
            loss=cast(LossFunction, data["loss"]),
        )
