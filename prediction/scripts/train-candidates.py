# python -m prediction.scripts.train-candidates
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all tf except fatal errors

from prediction import models, tuning
from processing import processors
from analysis import metrics
from globals.utils import Log

# train, evaluate, and measure all candidate models
for candidate in tuning.candidates:
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
        model, history = models.train(preprocessor, candidate)
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