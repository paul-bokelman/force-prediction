from dataclasses import dataclass

@dataclass
class GeneticAlgorithmConfig:
    n_organisms: int
    tournament_proportion: float
    mutation_probability: float