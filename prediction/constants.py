from prediction.types import GeneticAlgorithmConfig

# ----------------------- direct prediction model (dpm) ---------------------- #

training_epochs = 1 # number of training epochs
training_batch_size = 32 # batch size for training
saves_directory = 'saves/' # directory for saving trained models
saved_model_name = 'model.keras' # name of the saved model
early_stopping_patience = 10 # patience for early stopping callback during training

# -------------------- inferential prediction model (ipm) -------------------- #

neuronal_sum_error = 0.05

# genetic algorithm configuration
ga_generations = 10
genetics = GeneticAlgorithmConfig(n_organisms=100, tournament_proportion=0.8, mutation_probability=0.4)
