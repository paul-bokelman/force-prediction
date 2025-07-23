from typing import cast
from analysis.types import CandidateReport, CandidateInfo, ComparisonMetrics, MetricsGroup, Plots, MetricsPlot, PredictionsPlot
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from globals.utils import Log
from prediction import models, tuning
from analysis import plotting
import prediction.constants
import processing.constants
import analysis.constants
from processing.utils import get_dataframe

def get_metrics_path(hash: str) -> str:
    """Get the path to the saved metrics file."""
    return os.path.join(prediction.constants.candidate_out_dir(hash), "metrics.pkl")

def save(hash: str, metrics: dict[str, list[float]]) -> None:
    """Export the computed metrics to a pickle file."""
    pd.DataFrame(metrics, index=[0]).to_pickle(get_metrics_path(hash))
    Log.info(f"Saved metrics for '{hash}'")

def predict_by_index(candidate: tuning.ModelCandidate, row_index: int) -> None:
    """Predict the force sequence for a specific row index in the preprocessed dataset."""
    candidate_hash = candidate.hash()

    dataset = get_dataframe(processing.constants.dataset_path)
    preprocessed_dataset = get_dataframe(processing.constants.preprocessed_dataset_path(candidate.hyperparameters.preprocessing.hash()))
    model = models.obtain(candidate_hash)

    assert model is not None, f"Model for candidate '{candidate_hash}' does not exist. Cannot predict force."

    row = preprocessed_dataset.iloc[row_index]
    neuron_data, force_data, subject = row["neuron_data"], row["force_data"], row["subject"]
    original_neuron_data = dataset.iloc[cast(int, row_index)]["neuron_data"]

    predicted_force = model.predict_force(
        neuron_data, 
        subject,
        candidate.hyperparameters.training.sequence_length, 
        candidate.hyperparameters.training.stride
    )

    score = r2_score(force_data, predicted_force)

    plotting.visualize_trial(
        title=f"Prediction {row_index} (r² = {score:.4f})",
        neuron_data=original_neuron_data,
        force_data=force_data, 
        predicted_force=predicted_force,
    )

def export_all_predictions(candidate: tuning.ModelCandidate) -> None:
    """Export all predictions for the given candidate to png files within the candidate's output directory."""
    candidate_hash = candidate.hash()

    dataset = get_dataframe(processing.constants.dataset_path)
    preprocessed_dataset = get_dataframe(processing.constants.preprocessed_dataset_path(candidate.hyperparameters.preprocessing.hash()))
    model = models.obtain(candidate_hash)

    assert model is not None, f"Model for candidate '{candidate_hash}' does not exist. Cannot export predictions."
    predictions_export_dir = os.path.join(prediction.constants.candidate_out_dir(candidate_hash), "predictions/")

    # create the predictions export directory if it does not exist
    if not os.path.exists(predictions_export_dir):
        os.makedirs(predictions_export_dir)

    # export all predictions for each row in the preprocessed dataset
    for row_index, row in preprocessed_dataset.iterrows():
        neuron_data, force_data, subject = row["neuron_data"], row["force_data"], row["subject"]
        original_neuron_data = dataset.iloc[cast(int, row_index)]["neuron_data"]
        predicted_force = model.predict_force(
            neuron_data, 
            subject,
            candidate.hyperparameters.training.sequence_length, 
            candidate.hyperparameters.training.stride
        )

        score = r2_score(force_data, predicted_force)

        plotting.visualize_trial(
            title=f"Prediction {row_index} (r² = {score:.4f})",
            neuron_data=original_neuron_data,
            force_data=force_data, 
            predicted_force=predicted_force, 
            export_path=os.path.join(predictions_export_dir, f"pred{row_index}.png")
        )

def generate_report(candidate: tuning.ModelCandidate) -> None:
    """Generate a json report for the given candidate, including metrics and plots. Report is visualized in the web interface."""
    candidate_hash = candidate.hash()
    model = models.obtain(candidate_hash)
    assert model is not None, "Model is not present. Cannot generate report without a trained model."
    
    candidate_information_dict = candidate.to_dict() # modified dict conversion to include architecture representation

    candidate_information = CandidateInfo(
        hash=candidate_hash,
        identifier=candidate_information_dict["identifier"],
        architecture=candidate_information_dict["architecture"],
        version=candidate_information_dict["version"],
        hyperparameters={
            "preprocessing": candidate_information_dict["hyperparameters"]["preprocessing"],
            "training": candidate_information_dict["hyperparameters"]["training"]
        }
    )

    comparison_metrics: ComparisonMetrics = {}

    # get the metrics and baseline metrics
    dataset = get_dataframe(processing.constants.dataset_path)
    metrics = get_dataframe(get_metrics_path(candidate_hash))
    preprocessed_dataset = get_dataframe(processing.constants.preprocessed_dataset_path(candidate.hyperparameters.preprocessing.hash()))
    baseline_metrics = preprocessed_dataset.attrs

    # compile evaluation metrics
    for fullname, key in [("mean absolute error", "mae"), ("mean squared error", "mse"), ("r²", "r2")]:
        model_value = metrics.iloc[0][key] if key in metrics.columns else None
        baseline_key = f"baseline_{key}"
        baseline_value = baseline_metrics.get(baseline_key, None)
        assert baseline_value is not None and model_value is not None, f"Missing metric {key} for model {candidate.hash()} or baseline."
        improvement = True if (key == "r2" and model_value > baseline_value) or (key != "r2" and model_value < baseline_value) else False
        
        comparison_metrics[fullname] = MetricsGroup(
            baseline=round(baseline_value, 4),
            candidate=round(model_value, 4),
            improvement=improvement
        )

    plots = Plots(
        training_history="",
        metrics=MetricsPlot(
            data_volume="",
            number_of_neurons="",
        ),
        predictions=PredictionsPlot(
            best="",
            average="",
            worst=""
        )
    )

    # construct training history plot
    training_history = pd.read_pickle(os.path.join(prediction.constants.candidate_out_dir(candidate_hash), "history.pkl"))
    plots['training_history'] = cast(str, plotting.visualize_training(training_history, encode=True, dark_mode=True))

    def encode_candidate_prediction(key: str, data_index: int, score: float, predicted_force: np.ndarray) -> str:
        """Encode a candidate prediction plot for the report."""
        prediction_entry = preprocessed_dataset.iloc[data_index]
        neuron_data, force_data = dataset.iloc[data_index]["neuron_data"], prediction_entry["force_data"]

        # visualize the trial and encode the plot
        return cast(str, plotting.visualize_trial(
            title=f"{key.capitalize()} Prediction (r² = {score:.4f})",
            neuron_data=neuron_data, 
            force_data=force_data, 
            predicted_force=predicted_force, 
            encode=True,
            dark_mode=True,
        ))

    # compute predictions for all data
    all_predictions: list[dict] = []
    for row_index, row in preprocessed_dataset.iterrows():
        neuron_data, force_data, subject = row["neuron_data"], row["force_data"], row["subject"]
        predicted_force = model.predict_force(
            neuron_data, 
            subject,
            candidate.hyperparameters.training.sequence_length, 
            candidate.hyperparameters.training.stride
        )
        all_predictions.append({
            "row_index": cast(int, row_index),
            "score": r2_score(force_data, predicted_force),
            "predicted_force": predicted_force
        })

    # find best, worst, and average predictions
    scores = np.array([result["score"] for result in all_predictions])

    # Remove outliers from scores using IQR method
    Q1 = np.quantile(scores, 0.25)
    Q3 = np.quantile(scores, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_indices = np.where((scores >= lower_bound) & (scores <= upper_bound))[0]

    # Print outliers
    outlier_indices = np.where((scores < lower_bound) | (scores > upper_bound))[0]
    if len(outlier_indices) > 0:
        print("Outlier indices:", outlier_indices)
        print("Outlier scores:", scores[outlier_indices])

    # Remove outliers from all_predictions
    all_predictions = [all_predictions[i] for i in filtered_indices]
    scores = scores[filtered_indices]

    baseline_r2 = baseline_metrics["baseline_r2"]
    best_idx, worst_idx, avg_idx = int(np.argmax(scores)), int(np.argmin(scores)), int(np.argmin(np.abs(scores - baseline_r2)))

    plots["predictions"] = PredictionsPlot( # encode the best, worst, and average predictions
        best=encode_candidate_prediction("best", best_idx, all_predictions[best_idx]["score"], all_predictions[best_idx]["predicted_force"]),
        average=encode_candidate_prediction("average", avg_idx, all_predictions[avg_idx]["score"], all_predictions[avg_idx]["predicted_force"]),
        worst=encode_candidate_prediction("worst", worst_idx, all_predictions[worst_idx]["score"], all_predictions[worst_idx]["predicted_force"])
    )

    predictions_for_metrics: list[tuple[int, float]] = [(result["row_index"], result["score"]) for result in all_predictions] # reduce data for metrics plots

    plots["metrics"] = MetricsPlot(
        data_volume=cast(str, plotting.visualize_data_volume_impact(dataset, predictions_for_metrics, encode=True, dark_mode=True)),
        number_of_neurons=cast(str, plotting.visualize_number_of_neurons_impact(dataset, predictions_for_metrics, encode=True, dark_mode=True)),
    )

    candidate_report = CandidateReport(
        candidate=candidate_information,
        metrics=comparison_metrics,
        plots=plots,
    )

    # export the report to a JSON file in the views directory
    with open(os.path.join(analysis.constants.views_candidates_dir, f"{candidate_hash}.json"), "w") as f:
        json.dump(candidate_report, f)

    # update the views registry to reflect the new report
    reports = [r for r in os.listdir(analysis.constants.views_candidates_dir) if r.endswith(".json")]
    with open(analysis.constants.views_registry_path, "w") as f:
        json.dump(reports, f)

    Log.success(f"Generated and exported report for candidate '{candidate_hash}'")