from typing import cast, Optional
import os
import hashlib
import random
import pickle
from datetime import datetime
import diskcache as dc
import globals.constants
from globals.utils import Log
from processing import processors
from prediction import constants, models
from prediction.tuning import ModelCandidate, Hyperparameters, TrainingHyperparameters, PreprocessingHyperparameters, Architecture
from prediction.types import StateSpaceRange, Genes

cache = dc.Cache(constants.fitness_cache_dir)  # cache for fitness values of candidates

# initial genes for the ga -> leads to better start convergence
initial_genes = [
    Genes(),
    Genes(units=32),
    Genes(units=128),
    Genes(sequence_length=100),
    Genes(sequence_length=400),
    Genes(sequence_length=100, stride_divisor=10),
]

def search(obliterate_cache: bool = False) -> 'Genome':
    """Run the genetic algorithm to evolve model candidates."""
    pop = Population(initial_genomes=[Genome(g) for g in initial_genes])
    try:
        # evolve for the specified number of generations
        for _ in range(constants.generations): 
            pop.evolve()
    except (KeyboardInterrupt, Exception) as e:
        Log.warn(f"Interrupted or error occurred - Saving best genome of generation {pop.generation}.")
    finally:
        saved_best = pop.save_best() # save the best genome of the current generation

        # optionally obliterate the cache if specified
        if obliterate_cache:
            Log.warn("Obliterating fitness cache.")
            cache.clear()

        cache.close() # close the cache to ensure changes are saved
        return saved_best

class Genome:
    def __init__(self, genes: Optional[Genes] = None) -> None:
        """Initialize the genome with the provided genes or random genes. Mutation will take place if genes are provided unless otherwise specified."""
        self.genes = genes if genes else Genome.random_genes()

    @staticmethod
    def random_genes(gss = constants.gene_state_space) -> Genes:
        """Generate a random genome based on the given gene state space."""
        randomized_range_genes = {k: Genome._random_range_gene(r) for k, r in gss.__dict__.items() if isinstance(r, StateSpaceRange)}

        return Genes(
            architecture_identifier=random.choice(gss.architecture_identifier),
            units=cast(int, randomized_range_genes["units"]),
            loss=random.choice(gss.loss),
            subject_embedding_dimension=cast(int, randomized_range_genes["subject_embedding_dimension"]),
            sequence_length=cast(int, randomized_range_genes["sequence_length"]),
            stride_divisor=cast(int, randomized_range_genes["stride_divisor"]),
            bin_size_divisor=cast(int, randomized_range_genes["bin_size_divisor"]),
            exponential_decay_lifetime=cast(int, randomized_range_genes["exponential_decay_lifetime"]),
            size_amplification_factor=cast(int, randomized_range_genes["size_amplification_factor"]),
        )
    
    @staticmethod
    def _random_range_gene(ssr: StateSpaceRange) -> float | int:
        """Generate a random value within the given range, supporting optional step (only for int range)."""
        start, end, step = ssr.start, ssr.end, ssr.step
        if isinstance(start, int) and isinstance(end, int):
            return random.randrange(start, end + 1, step)
        return random.uniform(start, end)
    
    def mutate(self) -> None:
        """Chance mutate each gene in the genome independently based on the gene mutation probability."""
        for key, value in self.genes.__dict__.items():
            if random.random() < constants.gene_mutation_probability: # independent mutation probability for each gene
                if isinstance(value, StateSpaceRange):
                    # mutate the range by randomly selecting a new value within the range
                    setattr(self, key, self._random_range_gene(value))
                elif isinstance(value, list):
                    # mutate the list by randomly selecting a new value from the state space
                    setattr(self, key, random.choice(constants.gene_state_space.__dict__[key]))

    def save(self) -> None:
        """Save the genome to a file."""
        with open(os.path.join(constants.genome_save_dir, f"{self.genes.hash()}.pkl"), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(hash: str) -> 'Genome':
        """Load a genome from a file."""
        with open(os.path.join(constants.genome_save_dir, f"{hash}.pkl"), 'rb') as f:
            return pickle.load(f)
        
    def to_model_candidate(self) -> ModelCandidate:
        """Convert the genome to a ModelCandidate for metrics and evaluation"""
        return ModelCandidate( #/ properties that aren't specified use constant defaults
            architecture=Architecture.from_config(self.genes.architecture_identifier, self.genes.units),
            hyperparameters=Hyperparameters(
                training=TrainingHyperparameters(
                    sequence_length=self.genes.sequence_length,
                    stride=self.genes.sequence_length // self.genes.stride_divisor, # compute stride as proportion of sequence length
                    subject_embedding_dimension=self.genes.subject_embedding_dimension,
                    loss=self.genes.loss
                ),
                preprocessing=PreprocessingHyperparameters(
                    bin_size=globals.constants.sampling_frequency // self.genes.bin_size_divisor, # compute bin size as proportion of sampling frequency
                    exponential_decay_lifetime=self.genes.exponential_decay_lifetime,
                    size_amplification_factor=round(self.genes.size_amplification_factor / 10, 2) # convert to proportion 
                )
            )
        )
        
    @property
    def fitness(self) -> float | None:
        """Calculate the fitness of the genome based on the target r² score."""
        candidate = self.to_model_candidate()
        hash = self.genes.hash() # get the unique hash for the genome
        cached_fitness = cast(float | None | str, cache.get(hash, "null"))

        # cached fitness exists -> don't recompute it -> return it
        if cached_fitness != "null":
            assert isinstance(cached_fitness, (float, type(None))), "Cached fitness must be a float or None."
            return cached_fitness
        
        computed_fitness: float | None = None

        try:
            # setup the preprocessor and preprocess the dataset for this candidate
            preprocessor = processors.Preprocessor(candidate.hyperparameters)
            preprocessor.preprocess(compute_baseline=True, overwrite=False)
            model, _ = models.train(preprocessor, candidate, save=False, verbose="0")

            # evaluate the model on the test set
            evaluation_metrics: dict[str, float] = model.evaluate(
                preprocessor.generator('test'), 
                steps=preprocessor.compute_dataset_windows('test') // candidate.hyperparameters.training.batch_size,
                return_dict=True,
                verbose="0"
            )

            computed_fitness = evaluation_metrics.get('r2', None)
        except Exception as e:
            Log.error(f"Error evaluating genome {hash}: {e}")

        cache[hash] = computed_fitness # cache the computed fitness value
        return computed_fitness

class Population:
    def __init__(self, size: int = constants.population_size, initial_genomes: list[Genome] = []) -> None:
        assert size > 5 and size % 2 == 0, "Population size must be a positive even integer greater than 5."
        assert len(initial_genomes) <= size, "Seed candidates cannot exceed population size."
        self.size = size # number of candidates in each generation
        self.identifier = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:8]
        self.average_fitness: float | None = None # average fitness of the population, computed after each evolution step

        # initialize the population with the initial genomes and fill the rest with random genomes
        self.genomes = [Genome() for _ in range(size - len(initial_genomes))] + initial_genomes
        self.generation = 0 # current generation number

        Log.info(f"Initialized population {self.identifier} with {len(self.genomes)} genomes.")
        
    def crossover(self, p1: Genome, p2: Genome, desperation: float) -> tuple[Genome, Genome]:
        """Perform crossover between two genomes to create a new genome with chance of mutation."""
        # randomly select genes from each parent with a 50% chance for each gene
        p1_dict= p1.genes.__dict__
        p2_dict = p2.genes.__dict__
        keys = p1_dict.keys()
        split_map = {k: random.randint(1, 2) for k in keys} # generate gene split map -> 1 for p1, 2 for p2

        # split genes based on the split map -> 1 for p1, 2 for p2 then reverse the split map for the second child
        c1 = Genome(Genes.from_dict({k: p1_dict[k] if split_map[k] == 1 else p2_dict[k] for k in keys}))
        c2 = Genome(Genes.from_dict({k: p2_dict[k] if split_map[k] == 1 else p1_dict[k] for k in keys}))

        mutation_probability = constants.base_mutation_probability * (1 + desperation) # probability increases with desperation

        # chance of mutation for each child genome (mutates in place)
        if random.random() < mutation_probability:
            c1.mutate()
        if random.random() < mutation_probability:
            c1.mutate()

        return c1, c2

    def evolve(self) -> None:
        """Evolve to the next generation by performing elitist selection, crossover, and mutation."""

        #/ measure fitness of each genome
        scored_genomes = sorted(
            [(index, genome.fitness) for index, genome in enumerate(self.genomes)],
            key=lambda x: x[1] if x[1] is not None else float('-inf'),
            reverse=True
        )

        # computed average fitness of the population
        self.average_fitness = sum(c[1] for c in scored_genomes if c[1] is not None) / len(scored_genomes)
        total_failed = sum(1 for c in scored_genomes if c[1] is None)
        Log.info(f"G{self.generation} | best: {scored_genomes[0][1]} | average fitness: {self.average_fitness:.4f} | failed: {total_failed} / {len(scored_genomes)} | size: {self.size}")

        # select a proportion of elites to carry to the next generation
        elite_proportion = 0.2  # proportion of elites to select
        scored_candidates = [c for c in scored_genomes[:int(self.size * elite_proportion)]] 
        genome_pool = scored_genomes[int(self.size * elite_proportion):] # remove elites from the pool (already selected)
        
        # tournament selection until population is half full
        while len(scored_candidates) < (self.size // 2):
            # randomly select two candidates then remove them from the pool
            c1, c2 = random.sample(genome_pool, k=2)
            genome_pool.remove(c1) #? should remove? 
            genome_pool.remove(c2)

            c1_fitness, c2_fitness = c1[1], c2[1]

            # skip if both parents have no fitness
            if c1_fitness is None and c2_fitness is None: 
                continue #/ could cause problems if too many candidates have no fitness

            # if one parent has no fitness, select the other parent
            if c1_fitness is None or (c2_fitness is not None and c2_fitness > c1_fitness):
                scored_candidates.append(c2)
            else:
                scored_candidates.append(c1)

        next_generation = [self.genomes[c[0]] for c in scored_candidates]
        parents = random.sample(scored_candidates, k=len(scored_candidates))

        # crossover adjacent pairs of parents to create new genomes
        for i in range(0, len(parents) - 1,  2):
            (p1_index, p1_fitness), (p2_index, p2_fitness) = parents[i], parents[i + 1]
            # compute desperation based on combined fitness distance from top fitness
            averaged_fitness = max((cast(float, p1_fitness) + cast(float, p2_fitness)) / 2, 0)
            desperation = min(max(abs(cast(float, scored_candidates[0][1]) - averaged_fitness), 0), 1) 

            # perform crossover and mutation then add to next generation
            c1, c2 = self.crossover(self.genomes[p1_index], self.genomes[p2_index], desperation) 
            next_generation.extend([c1, c2])

        # ensure no duplicates -> remove duplicates based on the genome's hash
        next_generation = list({g.genes.hash(): g for g in next_generation}.values()) 
        if len(next_generation) > self.size: # more genomes than the population size -> truncate the list
            next_generation = next_generation[:self.size]
        elif len(next_generation) < self.size: #fewer genomes than the population size, fill with random genomes
            while len(next_generation) < self.size:
                next_generation.append(Genome())

        self.genomes = next_generation
        self.generation += 1

    def save_best(self) -> Genome:
        """Save the best genome of the current generation to a file."""
        if not self.genomes:
            raise ValueError("No genomes available to save the best genome.")

        best_genome = self.best
        best_genome.save() # save the genome to disk
        Log.info(f"Saved best genome of generation {self.generation} with fitness {best_genome.fitness}.")
        return best_genome

    @property
    def best(self) -> Genome:
        """Return the best genome in the population based on fitness. Only considers genomes with cached fitness values."""
        if not self.genomes:
            raise ValueError("No genomes available to determine the best genome.")

        cached = [(g, cache.get(g.genes.hash(), "null")) for g in self.genomes]
        cached = [(g, f) for g, f in cached if f != "null" and f is not None]

        if not cached:
            raise ValueError("No cached genomes available to determine the best genome.")

        return max(cached, key=lambda x: cast(float, x[1]))[0]