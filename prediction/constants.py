# ---------------------------------- shared ---------------------------------- #

output_dir = "prediction/out" # directory for saving metrics comparison visualizations
candidate_out_dir = lambda hash: f"{output_dir}/{hash}"

# ---------------------------------- models ---------------------------------- #

epochs = 50 # number of training epochs
early_stopping_patience = 10 # patience for early stopping callback during training
tensorboard_log_dir = 'prediction/logs' # directory for saving TensorBoard logs
