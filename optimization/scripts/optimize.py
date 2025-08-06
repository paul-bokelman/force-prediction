# python -m optimization.scripts.optimize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all tf except fatal errors

import pygad
from optimization import constants
from optimization.genetics.functions import fitness_function, on_generation, on_start, on_stop
from optimization.genetics.genes import Genes

initial_population = [Genes.from_model_candidate(c).to_array() for c in constants.candidates] if constants.use_candidates_as_initial_population else None

ga = pygad.GA(
    fitness_func=fitness_function, # fitness function to evaluate the solutions, defined in functions.py
    on_generation=on_generation, # callbacks for logging and saving solutions after each generation
    on_start=on_start, # callbacks for logging and saving solutions 
    on_stop=on_stop, # callbacks for logging and saving solutions
    num_genes=constants.gene_space.count(), # number of genes in the gene space, defined in genes.py
    gene_space=constants.gene_space.parameterize(), # use the gene space defined in genes.py
    num_generations=2, # 50-100 generations is a good starting point for most problems
    num_parents_mating=2, # 10-20 parents per generation
    sol_per_pop=4, # 20-50 solutions per population
    initial_population=initial_population, # use the manually defined candidates as initial population
    crossover_probability=0.8, # probability of crossover between parents
    mutation_type="adaptive", # adaptive mutation based on the fitness of the population
    mutation_probability=[0.4, 0.05], # probability of mutation for each gene as generations progresses
    mutation_by_replacement=True, # replace the mutated genes with new ones instead of adding them
    allow_duplicate_genes=True, # allow duplicate genes in the population
    keep_elitism=1, # number of best solutions to keep in the next generation
)

ga.run()