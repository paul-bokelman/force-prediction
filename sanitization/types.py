from typing import Literal
from numpy.typing import NDArray
import numpy as np

type DataTypeKeys = Literal["mvc", "force"]

type TrialShapedData = dict[int, dict[int, NDArray[np.float64]]] # {mvc_level: {trial_number: data}}
type UnifiedSubjectData = dict[int, dict[int, dict[DataTypeKeys, NDArray[np.float64]]]] # {mvc_level: {trial_number: (mvc_data, force_data)}}