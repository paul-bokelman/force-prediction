from typing import Literal
import numpy as np
import pandas as pd
import prediction.constants as constants
import prediction.genetics as ga

def _scaling_distribution(dist: Literal['linear', 'exponential'], n: int, r: int = 2, max_scale: float = 1.0) -> list[float]:
    """Generate a scaling distribution for neuron data. Currently supports linear and exponential distributions."""
    if dist == 'exponential' and r is None:
        raise ValueError("r must be provided for exponential distribution")

    values = list(range(1, n + 1)) if dist == 'linear' else [r**i for i in range(n)]
    total = sum(values)
    normalized = [v / total for v in values]
    
    return [v * max_scale for v in normalized]

def neuron_column_inference(col: np.ndarray, force: float, s: list[float]) -> np.ndarray:
    """Infer the rest of the neuron column given the produced force, S, and current activations"""
    remaining = force
    artificial_activations: list[int] = [0] * len(col)

    for i, scaler in enumerate(sorted(s)):
        print(scaler)
        remaining -= scaler
        
        if remaining <= constants.neuronal_sum_error:
            break
        
        artificial_activations[i] = 1

    print(f"Recruited {sum(artificial_activations)} neurons for force {force}")
    print(f"{sum([a * s for a,s in zip(artificial_activations, sorted(s))])} sum for force {force}")

    return np.array(artificial_activations)

def neuron_inference(trial: pd.Series) -> tuple[np.ndarray, list[float]]:
    """Infer neuron data from force data"""
    force_profile: np.ndarray = trial['force_data']
    neurons: np.ndarray = trial['neuron_data']

    n = len(neurons)
    # s = _scaling_distribution('linear', n, max_scale=np.max(force_profile)) # neuron force scaling distribution
    # print(np.mean(np.std(neurons, axis=1)))

    pop = ga.Population(neurons, force_profile)
    for _ in range(constants.ga_generations):
        pop.evolve()

    s = list(pop.best().genome)
    # iterate over each neuron column (that already has activations) and infer the rest of the column
    for i, (col, force) in enumerate(zip(neurons.T, force_profile)):
        # no force or no activations in the column, skip
        if force == 0 or sum(col) == 0:
            continue
    
        artificial_activations = neuron_column_inference(col, force, s)
        neurons.T[i] = artificial_activations

    return neurons, s

def force_inference(activations: np.ndarray, force: np.ndarray, s: list[float]) -> np.ndarray:
    """Infer force data from neuron data"""
    return force