from prediction.types import GeneticAlgorithmConfig
import processing.constants

# ----------------------- direct prediction model (dpm) ---------------------- #

epochs = 5 # number of training epochs
batch_size = 16 # batch size for training
saves_directory = 'saves/' # directory for saving trained models
default_model_name = 'model' # default name of the saved model
early_stopping_patience = 10 # patience for early stopping callback during training

models = {
    # LSTM(256, seqs) -> Dense(1)
    "single-lstm": {"name": "single-lstm", "sequence_length": 200, "stride": 100},
    
    # LSTM(128, seqs) -> LSTM(128, seqs) -> Dense(1)
    "2x-lstm": {"name": "2x-lstm", "sequence_length": 200, "stride": 100},

    # Bi-LSTM(64, seqs) -> Bi-LSTM(64, seqs) -> Dense(1)
    "2x-bi-lstm": {"name": "2x-bi-lstm", "sequence_length": 200, "stride": 100},
    
    # test architecture -> uses default params
    default_model_name: {"name": default_model_name, "sequence_length": processing.constants.sequence_length, "stride": processing.constants.stride}
}

# -------------------- inferential prediction model (ipm) -------------------- #

neuronal_sum_error = 0.05

# genetic algorithm configuration
ga_generations = 10
genetics = GeneticAlgorithmConfig(n_organisms=100, tournament_proportion=0.8, mutation_probability=0.4)
