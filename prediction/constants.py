from prediction.types import ArchitectureName, ArchitectureTemporalContextWindows
import os

# ----------------------- direct prediction model (dpm) ---------------------- #

epochs = 10 # number of training epochs
batch_size = 16 # batch size for training
saves_directory = 'saves/' # directory for saving trained models
early_stopping_patience = 5 # patience for early stopping callback during training

architecture_names: list[ArchitectureName] = ['single-lstm', '2x-bi-lstm', '2x-lstm', 'conv-bi-lstm']

architecture_temporal_context_windows: ArchitectureTemporalContextWindows = {
    "large": {"sequence_length": 4000, "stride": 2000},
    "medium": {"sequence_length": 1500, "stride": 750},
    "small": {"sequence_length": 200, "stride": 100},
}

# ---------------------------------- metics ---------------------------------- #

metrics_output_dir = "prediction/out" # directory for saving metrics comparison visualizations
metrics_saved_tables_dir = f"{metrics_output_dir}/tables" # directory for saving metrics comparison tables

print(os.getcwd())