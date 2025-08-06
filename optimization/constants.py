from optimization.genetics.genes import GeneSpace, StateSpaceRange
from optimization.params import ModelCandidate, Architecture, Hyperparameters, TrainingHyperparameters, PreprocessingHyperparameters, architectures, losses

# ---------------------------------- params ---------------------------------- #

# manually defined set of model candidates with different architectures and hyperparameters
candidates: list['ModelCandidate'] = [
    ModelCandidate(
        architecture=Architecture.LSTM(units=16),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=6),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
    ModelCandidate(
        architecture=Architecture.LSTM(units=32),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=2),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
    ModelCandidate(
        architecture=Architecture.LSTM(units=8),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=8),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
    ModelCandidate(
        architecture=Architecture.LSTM(units=32),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=4),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
    ModelCandidate(
        architecture=Architecture.LSTM(units=40),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=2),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
    ModelCandidate(
        architecture=Architecture.LSTM(units=64),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=2),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
    ModelCandidate(
        architecture=Architecture.LSTM(units=42),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=4),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
    ModelCandidate(
        architecture=Architecture.LSTM(units=32),
        hyperparameters=Hyperparameters(
            training=TrainingHyperparameters(subject_embedding_dimension=6),
            preprocessing=PreprocessingHyperparameters(size_amplification_factor=0.8)
        )
    ),
]

# --------------------------------- genetics --------------------------------- #

use_candidates_as_initial_population = True # use the manually defined candidates as initial population

solution_save_dir = f"optimization/out/solutions" # directory for saving GA solutions
fitness_cache_dir = "optimization/out/cache" # directory for caching fitness values of solutions

# define default gene space for the genetic algorithm (overrides the constants in genes.py)
gene_space = GeneSpace(
    # ---------------------------------- general --------------------------------- #
    architecture_identifier = architectures, # single lstm or stacked lstm architecture (same units for both layers)
    units = StateSpaceRange(start=4, end=80, step=4), # range of units for LSTM layers
    # ------------------------------- preprocessing ------------------------------ #
    bin_size = StateSpaceRange(start=10, end=30, step=10), # bin size for preprocessing, measured in milliseconds
    exponential_decay_lifetime = StateSpaceRange(start=10, end=30, step=10), # measured in milliseconds, 0 means instant decay
    size_amplification_factor = StateSpaceRange(start=0.6, end=1, step=0.1), # factor for increasing neuron influence based on 'size', 0 means no amplification
    # --------------------------------- training --------------------------------- #
    sequence_length = StateSpaceRange(start=100, end=400, step=50), # length of the input sequence
    stride_divisor = StateSpaceRange(start=1, end=4, step=1), # stride is calculated as sequence_length / stride_divisor
    subject_embedding_dimension = StateSpaceRange(start=2, end=16, step=2, optional=True), # dimension of the subject embedding layer
    loss = losses # possible loss functions
)