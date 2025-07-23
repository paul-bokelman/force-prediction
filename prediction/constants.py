from prediction.types import StateSpaceRange, GeneStateSpace

# ---------------------------------- shared ---------------------------------- #

output_dir = "prediction/out/candidates" # directory for saving metrics comparison visualizations
candidate_out_dir = lambda hash: f"{output_dir}/{hash}"

# ---------------------------------- models ---------------------------------- #

epochs = 50 # number of training epochs
batch_size = 32 # batch size for training
early_stopping_patience = 5 # patience for early stopping callback during training

subject_embedding_dimension = 8 # dimension of the subject embedding layer
sequence_length = 200 # length of the sliding window for preprocessing
stride = sequence_length // 2 # stride for sliding window
train_percentage, validation_percentage, test_percentage = 0.8, 0.1, 0.1 # split percentages for training, validation, and test sets

tensorboard_log_dir = 'prediction/logs' # directory for saving TensorBoard logs

# --------------------------------- genetics --------------------------------- #

population_size = 10 # number of candidates in each generation
generations = 3 # number of generations to evolve the population
base_mutation_probability = 0.4 # base mutation probability for each candidate -> combined with desperation factor
gene_mutation_probability = 0.1 # probability of mutation for each gene in a genome

fitness_cache_dir = "prediction/out/cache" # directory for caching fitness values of candidates for faster evaluation
genome_save_dir = f"prediction/out/genomes/" # directory for saving genome files

# default gene state space for genetics
gene_state_space = GeneStateSpace(
    # ---------------------------------- general --------------------------------- #
    architecture_identifier = ["LSTM", "DualLSTM"], # single lstm or stacked lstm architecture (same units for both layers)
    units= StateSpaceRange(start=16, end=256, step=16), # range of units for LSTM layers
    # ------------------------------- preprocessing ------------------------------ #
    bin_size_divisor = StateSpaceRange(start=50, end=200, step=50), # proportion of sampling frequency, 0 means no binning
    exponential_decay_lifetime = StateSpaceRange(start=0, end=40, step=5), # measured in milliseconds, 0 means instant decay
    size_amplification_factor = StateSpaceRange(start=0, end=10, step=1), # factor for increasing neuron influence based on 'size', 0 means no amplification
    # --------------------------------- training --------------------------------- #
    sequence_length = StateSpaceRange(start=100, end=400, step=50),
    stride_divisor = StateSpaceRange(start=1, end=10), # stride is calculated as sequence_length / stride_divisor
    subject_embedding_dimension = StateSpaceRange(start=8, end=16, step=4), # dimension of the subject embedding layer
    loss = ["mse", "mae", "huber"] # possible loss functions
)