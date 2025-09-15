from typing import cast
import os
import hashlib
import secrets
import numpy as np
import pygad
from diskcache import Cache
from prediction import models
from processing.processors import Preprocessor
from optimization.genetics.genes import Genes
from globals.utils import Log
from optimization import constants

cache = Cache(constants.fitness_cache_dir) # cache for fitness values of solutions

def _save_solution(instance: pygad.GA, solution: np.ndarray, name: str) -> None:
    """Save the solution to a file in the respective solutions directory."""
    hash = getattr(instance, 'hash', None)  # get the unique hash for this run
    assert hash is not None, "Hash must be set on the instance before saving solutions."
    solution_path = os.path.join(constants.solution_save_dir, hash, name + ".npy")
    np.save(solution_path, solution)  # save the solution as a .npy file

def fitness_function(instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> float:
    """Calculate fitness by training and evaluating the model. Fitness the resulting r² score."""
    genes = Genes.from_array(solution) # convert the solution to a Genes instance
    candidate, hash = genes.to_model_candidate(), genes.hash()
    fitness: float = float('-inf')  # default fitness value if an error occurs
    cached_fitness = cast(float | None, cache.get(hash, None))

    # cached fitness exists -> don't recompute it, return it
    if cached_fitness:
        assert isinstance(cached_fitness, float), "Cached fitness must be a float."
        Log.info(f"[{hash}] {round(cached_fitness, 2)}")
        return cached_fitness
    
    try:
        # setup the preprocessor and preprocess the dataset for this candidate
        preprocessor = Preprocessor(candidate.hyperparameters)
        preprocessor.preprocess(compute_baseline=True, overwrite=False)
        model, _ = models.train(preprocessor, candidate, save=False, verbose=0)

        # evaluate the model on the test set
        evaluation_metrics: dict[str, float] = model.evaluate(
            preprocessor.generator('test'), 
            steps=preprocessor.compute_dataset_windows('test') // candidate.hyperparameters.training.batch_size,
            return_dict=True,
            verbose=cast(str, 0) # keras expects a string for verbosity
        )

        fitness = evaluation_metrics.get('r2', float('-inf')) # get the r² score, default to -inf if not present
        Log.info(f"[{hash}] {round(fitness, 2)}")
    except Exception as e:
        Log.error(f"[{hash}] Error: {e}")
    
    cache.set(hash, fitness)
    return fitness

def on_generation(instance: pygad.GA):
    """Callback function called at the end of each generation, exports the best solution and logs fitness."""
    solution, solution_fitness, solution_idx = instance.best_solution()
    _save_solution(instance, solution, f"G{instance.generations_completed}") # save the best solution of this generation
    Log.info("\n" + "=" * 70)
    Log.info(f"[{getattr(instance, 'hash', 'unknown')}] G{instance.generations_completed}")
    Log.info(f"Population fitness: {round(cast(np.ndarray, instance.last_generation_fitness).mean(), 2)}")
    Log.info(f"Best solution: {round(solution_fitness, 2)}")
    Log.info(f"Best solution genome: {solution}")
    Log.info(("=" * 70) + "\n")

def on_start(instance: pygad.GA):
    """Callback function called at the start of the GA run. Cleans up any previous runs, initialize save directory, and logs GA parameters."""
    hash = hashlib.sha256(secrets.token_bytes(16)).hexdigest()[:10] # generate a unique hash for this run
    setattr(instance, 'hash', hash) # include a unique hash for this run
    respective_solutions_directory = os.path.join(constants.solution_save_dir, hash)
    Log.info(f"GA [{hash}]")
    os.mkdir(respective_solutions_directory) # create a directory for saving solutions specific to this run
    Log.info(f"Created respective solution directory: {respective_solutions_directory}")
    instance.summary()
    print()

def on_stop(instance: pygad.GA, population_fitness: np.ndarray):
    """Callback function called at the end of the GA run. Exports the best solution, logs final fitness, and plots fitness."""
    solution, solution_fitness, solution_idx = instance.best_solution()
    _save_solution(instance, solution, "best")  # save the best solution
    Log.info(f"[{getattr(instance, 'hash', 'unknown')}] Final solution: G{instance.generations_completed}")
    Log.info(f"Best solution ({round(solution_fitness, 2)}): ")
    Log.info(f"\n{Genes.from_array(solution)}\n")