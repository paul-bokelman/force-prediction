from typing import cast
import os
import io
import html
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from globals.utils import Log
from prediction import models, tuning
from analysis import plotting
import prediction.constants
import processing.constants
from processing.utils import get_dataframe
from analysis.partials.utils import replace_placeholders, generate_attribute_list

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
        predicted_force=predicted_force
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
    """Generate an HTML report for all aspects of the model"""
    candidate_hash = candidate.hash()
    candidate_information_dict = candidate.to_dict() # modified dict conversion to include architecture representation

    candidate_information_partial_fields = {
        "base_information": generate_attribute_list({k: html.escape(str(v)) for k, v in candidate_information_dict.items() if k != 'hyperparameters'}),
        "preprocessing": generate_attribute_list(candidate_information_dict["hyperparameters"]["preprocessing"]),
        "training": generate_attribute_list(candidate_information_dict["hyperparameters"]["training"]),
    }
    candidate_information_partial = replace_placeholders("candidate_information", candidate_information_partial_fields)
    training_history = pd.read_pickle(os.path.join(prediction.constants.candidate_out_dir(candidate_hash), "history.pkl"))

    # construct training history plot
    plt.figure(figsize=(14, 5))
    plt.plot(training_history["loss"], label="Training Loss")
    plt.plot(training_history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss and Validation Loss over Epochs")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    training_history_plot = base64.b64encode(buf.read()).decode("utf-8")

    # get the metrics and baseline metrics
    dataset = get_dataframe(processing.constants.dataset_path)
    metrics = get_dataframe(get_metrics_path(candidate_hash))
    preprocessed_dataset = get_dataframe(processing.constants.preprocessed_dataset_path(candidate.hyperparameters.preprocessing.hash()))
    baseline_metrics = preprocessed_dataset.attrs

    evaluation_metrics = {} # mae, mse, r2

    # compile evaluation metrics
    for fullname, key in [("mean absolute error", "mae"), ("mean squared error", "mse"), ("r²", "r2")]:
        model_value = metrics.iloc[0][key] if key in metrics.columns else None
        baseline_key = f"baseline_{key}"
        baseline_value = baseline_metrics.get(baseline_key, None)
        assert baseline_value is not None and model_value is not None, f"Missing metric {key} for model {candidate.hash()} or baseline."
        improvement = True if (key == "r2" and model_value > baseline_value) or (key != "r2" and model_value < baseline_value) else False
        evaluation_metrics[fullname] = {
            "quantity": f"{round(baseline_value, 4)} → {round(model_value, 4)}", 
            "improvement": improvement
        }

    # include plots of best and worst trial predictions
    model = models.obtain(candidate_hash)
    prediction_indices: dict[str, tuple[int, float, np.ndarray]] = {
        "best": (0, float('-inf'), np.array([])),
        "average": (0, 0.0, np.array([])),
        "worst": (0, float('inf'), np.array([]))
    }

    assert model is not None, "Model is not present. Cannot generate report without a trained model."

    # find the best and worst predictions
    for row_index, row in preprocessed_dataset.iterrows():
        neuron_data, force_data, subject = row["neuron_data"], row["force_data"], row["subject"]
        predicted_force = model.predict_force(
            neuron_data, 
            subject,
            candidate.hyperparameters.training.sequence_length, 
            candidate.hyperparameters.training.stride
        )

        prediction_score = r2_score(force_data, predicted_force)
        
        # replace the best and worst indices if the current mse is better or worse
        for key, (_, current_score, _) in prediction_indices.items():
            if key == "best" and prediction_score > current_score:
                prediction_indices[key] = (cast(int, row_index), prediction_score, predicted_force)
            elif key == "worst" and prediction_score < current_score:
                prediction_indices[key] = (cast(int, row_index), prediction_score, predicted_force)
            elif key == "average" and abs(prediction_score - baseline_metrics["baseline_r2"] < abs(current_score - baseline_metrics["baseline_r2"])):
                prediction_indices[key] = (cast(int, row_index), prediction_score, predicted_force)

    # b64 encode the best and worst predictions
    encoded_predictions = {}
    for key, (data_index, score, predicted_force) in prediction_indices.items():
        prediction_entry = preprocessed_dataset.iloc[data_index]
        neuron_data, force_data = dataset.iloc[data_index]["neuron_data"], prediction_entry["force_data"]

        # visualize the trial and encode the plot
        prediction_encoding = plotting.visualize_trial(
            title=f"{key.capitalize()} Prediction (r² = {score:.4f})",
            neuron_data=neuron_data, 
            force_data=force_data, 
            predicted_force=predicted_force, 
            encode=True
        )

        assert isinstance(prediction_encoding, str), f"Failed to encode {key} prediction plot."
        encoded_predictions[key] = prediction_encoding
    
    root = replace_placeholders("root", {
        "candidate_identifier": candidate_hash, 
        "candidate_information": candidate_information_partial, 
        "evaluation_metrics": generate_attribute_list({ k: v["quantity"] for k, v in evaluation_metrics.items() }, flags={key: value["improvement"] for key, value in evaluation_metrics.items()}),
        "training_history": replace_placeholders("plot", {"img": training_history_plot}),
        "predictions": "".join([replace_placeholders("plot", {"img": data}) for data in encoded_predictions.values()])
    })

    # export the HTML report to the candidate's output directory
    with open(os.path.join(prediction.constants.candidate_out_dir(candidate_hash), f"report.html"), "w") as f:
        f.write(root)

    Log.info(f"Exported HTML report for '{candidate_hash}'")

