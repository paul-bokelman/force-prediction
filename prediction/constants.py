# ---------------------------------- shared ---------------------------------- #

output_dir = "prediction/out/candidates" # directory for saving metrics comparison visualizations
candidate_out_dir = lambda hash: f"{output_dir}/{hash}"

# ---------------------------------- models ---------------------------------- #

epochs = 1 # number of training epochs
batch_size = 32 # batch size for training
early_stopping_patience = 5 # patience for early stopping callback during training

subject_embedding_dimension = 8 # dimension of the subject embedding layer
sequence_length = 200 # length of the sliding window for preprocessing
stride = sequence_length // 2 # stride for sliding window
train_percentage, validation_percentage, test_percentage = 0.8, 0.1, 0.1 # split percentages for training, validation, and test sets