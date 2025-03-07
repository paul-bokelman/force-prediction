from typing import Union
import numpy as np

def scale_to_interval(data: np.ndarray, min: int, max: int):
    """Scale data to a specific interval [min, max]"""
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (max - min) +  min

def scale_to_data(x: float, data: list[Union[float, int]]) -> float:
    """Scale a single value relative to the range of the given data array."""
    data_min, data_max = min(data), max(data)
    return (x - data_min) / (data_max - data_min)