from typing import Literal
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np

# -------------------------------- conversion -------------------------------- #

type DataTypeKeys = Literal["mvc", "force"]

type TrialShapedData = dict[int, dict[int, NDArray[np.float64]]] # {mvc_level: {trial_number: data}}
type UnifiedSubjectData = dict[int, dict[int, dict[DataTypeKeys, NDArray[np.float64]]]] # {mvc_level: {trial_number: (mvc_data, force_data)}}

# ------------------------------- sanitization ------------------------------- #

@dataclass
class ISIStatistics:
    max: np.floating
    mean_max: np.floating
    isi: np.floating
    std: np.floating
    cv: np.floating