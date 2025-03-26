from prediction.types import GeneticAlgorithmConfig

neuronal_sum_error = 0.05

# genetic algorithm configuration
ga_generations = 10
genetics = GeneticAlgorithmConfig(n_organisms=100, tournament_proportion=0.8, mutation_probability=0.4)
