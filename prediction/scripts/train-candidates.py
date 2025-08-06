# python -m prediction.scripts.train-candidates
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all tf except fatal errors

from prediction import models
from processing import processors
from analysis import metrics
import optimization.constants
from globals.utils import Log

# train, evaluate, and measure all candidate models
for candidate in optimization.constants.candidates:
    candidate_hash = candidate.hash()

    # skip candidates that have already been computed
    if candidate.already_computed:
        Log.warn(f"Candidate {candidate_hash} already computed, skipping...")
        continue

    model = models.obtain(candidate_hash) # get existing model if it exists, or None to train a new one later
    preprocessor = processors.Preprocessor(candidate.hyperparameters)
    preprocessor.preprocess(compute_baseline=True, overwrite=False) # optionally preprocess the data based on the candidate's hyperparameters
    
    # train the model based on it's configuration if it does not exist
    if not model:
        # train with validation for training history
        model, history = models.train(preprocessor, candidate, train_on_val=False) 

        # fully train on all data if specified in the candidate's hyperparameters
        if candidate.hyperparameters.training.train_on_val:
            Log.info(f"Training on validation set for candidate {candidate_hash}...")
            model, _ = models.train(preprocessor, candidate, train_on_val=True)
            
    else:
        Log.warn(f"Model {candidate_hash} already exists, skipping training...")

    assert model is not None, f"Model {candidate_hash} is None after training."

    # measure the model and export the results
    evaluation_metrics = model.evaluate(
        preprocessor.generator('test'), 
        steps=preprocessor.compute_dataset_windows('test') // candidate.hyperparameters.training.batch_size,
        return_dict=True
    )

    metrics.save(candidate_hash, evaluation_metrics) # save evaluation metrics to disk

    Log.info(f"Generating report for candidate {candidate_hash}...")
    metrics.generate_report(candidate)

    Log.success(f"Completed training and evaluation for model {candidate_hash}\n")