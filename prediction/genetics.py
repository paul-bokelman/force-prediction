from typing import Optional
import random
import math
import numpy as np
import prediction.constants as constants
from prediction.utils import chance, random_exclude
import pandas as pd

type OrganismGenome = np.ndarray # type alias for genome

class Organism:
    def __init__(self, genome_length: int, initial_genome: Optional[OrganismGenome] = None) -> None:
        self.internal_mutation_probability = 0.3 # mutation probability for gene in genome (should be in config)

        if initial_genome is not None and len(initial_genome) != genome_length:
            raise ValueError("Initial genome must have the same length as genome length")

        # initialize genome
        self.genome: OrganismGenome = self.random_genome(genome_length) if initial_genome is None else initial_genome

    def fitness(self, neurons: np.ndarray, force: np.ndarray) -> float:
        """Computes the fitness of this organism. Fitness is the sum of the distances between the organism's applied phenotype (scalers) and actual force. Lower fitness -> better organism."""
        # filter out zero columns and corresponding force values
        mask = (neurons.sum(axis=0) != 0) & (force != 0)
        nonzero_neurons = neurons[:, mask]
        nonzero_force = force[mask]
        
        # calculate all distances at once using numpy
        distances = np.abs(nonzero_force - np.sum(nonzero_neurons * np.array(self.genome)[:, np.newaxis], axis=0))
        
        return np.sum(distances)
    
    def random_genome(self, n: int) -> OrganismGenome:
        """Generate a random genome of a given length."""
        return np.random.random(n)
    
    # mutate genome by randomly changing bits in genome
    def mutate(self):
        # apply mutation probability to each gene in genome
        mutation_mask = np.random.random(len(self.genome)) < self.internal_mutation_probability
        self.genome[mutation_mask] += np.random.uniform(-0.1, 0.1, size=mutation_mask.sum())

    def __str__(self) -> str:
        return f"[Organism] | Genome: {self.genome}"

class Population:
    def __init__(self, neurons: np.ndarray, force: np.ndarray) -> None:
        """Initialize a population of organisms with a given genome length."""
        self.neurons = neurons
        self.force = force
        self.genome_length = len(neurons)
        self.genesis()

    def evolve(self):
        """Evolve the population to the next generation via tournament selection and crossover"""
        # tournament selection -> crossover -> mutation
        self.generation += 1
        candidates: list[Organism] = []

        # tournament selection for crossover candidates
        for _ in range(math.floor((constants.genetics.n_organisms * constants.genetics.tournament_proportion) / 2)):
            p1_index = random.randint(0, len(self.organisms) - 1)
            participant1 = self.organisms[p1_index] 
            p2_index = random_exclude(p1_index, min=0, max=len(self.organisms) - 1)
            participant2 = self.organisms[p2_index]

            # whoever has better fitness is added to candidate pool, loser is removed from species
            if(participant1.fitness(self.neurons, self.force) < participant2.fitness(self.neurons, self.force)):
                candidates.append(participant1)
                self.organisms.pop(p2_index)
            else:
                candidates.append(participant2)
                self.organisms.pop(p1_index)

        # uneven number of candidates, add 1 
        if(len(candidates) % 2 != 0):
            candidates.append(self.organisms[random.randint(0, len(self.organisms) - 1)])

        candidate_middle_index = int(len(candidates) / 2)

        # crossover for all pairs of candidates
        for (parent1, parent2) in zip(candidates[:candidate_middle_index], candidates[candidate_middle_index:]):
            self._crossover(parent1, parent2)
    
    def _crossover(self, p1: Organism, p2: Organism):
        """Crossover 2 organisms to create 2 children with chance of mutation."""
        position = random.randint(2, self.genome_length)  # choose random split position
        
        # convert parent genomes to numpy arrays and split
        p1_genome = np.array(p1.genome)
        p2_genome = np.array(p2.genome)
        
        # create children genomes using numpy concatenate
        child1_genome = np.concatenate((p1_genome[:position], p2_genome[position:]))
        child2_genome = np.concatenate((p2_genome[:position], p1_genome[position:]))
        
        # create and potentially mutate children
        for combined_genome in (child1_genome, child2_genome):
            child = Organism(self.genome_length, combined_genome)
            if chance(constants.genetics.mutation_probability):
                child.mutate()

            self.organisms.append(child)

    def best(self):
        """Find the best organism in the population."""
        fitnesses = np.array([o.fitness(self.neurons, self.force) for o in self.organisms])
        return self.organisms[np.argmin(fitnesses)]
    
    def _average_fitness(self):
        """Calculate the average fitness of the population."""
        # compute all fitness values at once and take mean
        return np.mean([o.fitness(self.neurons, self.force) for o in self.organisms])
    
    def _compare_organisms(self, o1: Organism, o2: Organism) -> bool:
        """Compare 2 organisms, return True if o1 is better than o2 (lower fitness is better)."""
        return o1.fitness(self.neurons, self.force) < o2.fitness(self.neurons, self.force)

    def genesis(self):
        """Initialize the population with random organisms."""
        self.generation = 0
        self.organisms = [Organism(self.genome_length) for _ in range(constants.genetics.n_organisms)]

    def gather(self) -> pd.DataFrame:
        """Return a DataFrame of the population's fitness and genomes."""
        data = [{
            'fitness': org.fitness(self.neurons, self.force),
            'genome': org.genome
        } for org in self.organisms]
        
        df = pd.DataFrame(data)
        df = df.sort_values('fitness')  # sort by fitness (lower is better)
        return df

    def __str__(self) -> str:
        return f"[Population] | G{self.generation} | {len(self.organisms)} organisms | Avg: {self._average_fitness()} | Best: {self.best().fitness(self.neurons, self.force)}"