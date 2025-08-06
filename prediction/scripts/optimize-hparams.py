# python -m prediction.scripts.optimize-hparams
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all tf except fatal errors

from prediction.types import StateSpaceRange, Genes
import pygad
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from prediction import constants
from prediction.tuning import ModelCandidate, Hyperparameters, TrainingHyperparameters, PreprocessingHyperparameters, Architecture

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)    

def fitness_func(instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> float:
    genes = Genes.from_array(solution)
    neurons = int(genes.units)              # hidden layer size

    # Create Keras model
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X.shape[1],), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_train, y_train, epochs=2, verbose=0, batch_size=32) # type: ignore

    # Evaluate on validation set
    _, accuracy = model.evaluate(X_val, y_val, verbose="0")
    return accuracy

last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

gene_space = [
    {"low": g.start, "high": g.end, "step": g.step } if isinstance(g, StateSpaceRange) else [i for i in range(len(g))]
    for g in constants.gene_state_space.__dict__.values()
]

if __name__ == "__main__":
    
    ga = pygad.GA(
        num_generations=10,
        num_parents_mating=4,
        fitness_func=fitness_func,
        on_generation=on_generation,
        sol_per_pop=10, # members in the population
        num_genes=len(gene_space),
        initial_population=None,
        gene_space=gene_space,
        mutation_percent_genes=50, # type: ignore #/ doesn't understand that it likes numbers...
        mutation_type="random",
        random_mutation_min_val=0,
        random_mutation_max_val=1,
        allow_duplicate_genes=False,
    )


    # ==== 4. Run the GA ====
    ga.run()

    ga.plot_fitness()

    # ==== 5. Best solution ====
    solution, solution_fitness, solution_idx = ga.best_solution()
    print("Best solution:", solution)
    print("Best fitness (accuracy):", solution_fitness)