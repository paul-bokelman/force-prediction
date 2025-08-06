from typing import Literal
import pygad

# ---------------------------------- shared ---------------------------------- #

type LossFunction = Literal["mse", "mae", "huber"]
type ArchitectureIdentifier = Literal["LSTM", "DualLSTM"]