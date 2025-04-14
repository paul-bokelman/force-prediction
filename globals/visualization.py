from typing import cast, Optional
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals.constants as constants

def visualize_trial(
        df: Optional[pd.DataFrame] = None, 
        subject: Optional[str] = None,
        mvc_level: Optional[int] = None, 
        trial_number: Optional[int] = None, 
        neuron_data: Optional[np.ndarray] = None,
        force_data: Optional[np.ndarray] = None,
        predicted_force: Optional[np.ndarray] = None,
        export_path: Optional[str] = None, 
        vmarkers: Optional[list[float]] = None, 
        hmarkers: Optional[list[float]] = None,
    ) -> None:
    """Plot grid of raster plots with force overlays for a given trial. Optionally save the plot to a file and or apply markers. Visualization also removes neuron rows with no data."""

    # either neuron_data or force_data is None, we need df and all trial identifiers
    if (neuron_data is None or force_data is None) and (df is None or subject is None or mvc_level is None or trial_number is None):
        raise ValueError("If either neuron_data or force_data is None, then df, subject, mvc_level, and trial_number are all required.")
    
    # ensure we have at least one data source (DataFrame or direct data)
    if df is None and (neuron_data is None and force_data is None):
        raise ValueError("Either a DataFrame or both neuron_data and force_data must be provided.")
    
    # filter data for the given subject, trial, and mvc level
    trial = df[(df["subject"] == subject) & (df["trial_number"] == trial_number) & (df["mvc_level"] == mvc_level)].iloc[0] if df is not None else None

    # setting up the plot
    plt.figure(figsize=(18, 6))
    plt.ylabel("Motor Neuron")
    plt.xlabel("Time (s)")

    # conditionally set title based on whether trial data is available
    if trial is not None:
        plt.title(f"{trial['subject']} | Trial {trial['trial_number']} | {trial['mvc_level']}% MVC")
    else:
        plt.title(f"Visualizing Manual Data")

    force_data = cast(pd.Series, trial)["force_data"] if force_data is None else force_data
    neuron_data = cast(pd.Series, trial)["neuron_data"].copy() if neuron_data is None else neuron_data

    # remove all rows with no activations from neuron_data
    if neuron_data is not None and neuron_data.ndim > 1:
        neuron_data = neuron_data[~np.all(neuron_data == 0, axis=1)]

    def plot_force_only(force_data: np.ndarray, time_values: np.ndarray) -> None:
        """Plots force data only"""
        plt.plot(time_values, force_data, label="Force Profile", color='red')
        plt.ylabel("Force", color='red')

    def plot_raster(neuron_data: np.ndarray, time_values: np.ndarray) -> None:
        """Plots raster representation of neuron data"""
        if neuron_data.ndim == 1:
            plt.plot(time_values, neuron_data, '|')
            
        # Sort and plot neurons based on first spike time
        neuron_order = np.argsort(np.argmax(neuron_data, axis=1))
        sorted_data = neuron_data[neuron_order]
        for i, neuron in enumerate(sorted_data):
            plt.plot(time_values, neuron * (i + 1), '|')

    def plot_force_overlay(force_data: np.ndarray, time_values: np.ndarray, force_axis: Axes, color: str):
        """Plots force overlay on additional axis"""
        force_axis.set_ylabel("Force", color='red')
        force_axis.plot(time_values, force_data, label="Force Profile", color=color)
        
        # add legend
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        plt.legend(lines1, labels1, loc='upper right')

    # handle force-only case
    if neuron_data.size == 0 and force_data.size > 0:
        time_values = np.linspace(0, len(force_data)/constants.sampling_frequency, len(force_data))
        plot_force_only(force_data, time_values)
        return

    # setup for combined plot
    total_time = neuron_data.shape[1] / constants.sampling_frequency
    time_values = np.linspace(0, total_time, neuron_data.shape[1])
    
    # plot the raster representation of the neuron data
    plot_raster(neuron_data, time_values)
    
    # add force overlay if force data is present
    if force_data.size > 0:
        force_axis = cast(Axes, plt.twinx())
        force_time = np.linspace(0, total_time, len(force_data))
        plot_force_overlay(force_data, force_time, force_axis, 'red')
        
        # add predicted force overlay if present
        if predicted_force is not None and predicted_force.size > 0:
            # Use same time scale as actual force
            pred_force_time = np.linspace(0, total_time, len(predicted_force))
            force_axis.plot(pred_force_time, predicted_force, label="Predicted Force", color='blue')
            # Update legend to include both force profiles
            lines, labels = force_axis.get_legend_handles_labels()
            force_axis.legend(lines, labels, loc='upper right')

    # add vertical markers if present
    if vmarkers:
        for marker in vmarkers:
            plt.axvline(marker, color='black', linestyle='--')

    # add vertical lines for force inflection points if present
    if trial is not None and trial['force_inflection_indices']:
        for inflection_index in [idx for idx in trial['force_inflection_indices'] if idx is not None]:
            plt.axvline(inflection_index / constants.sampling_frequency, color='blue', linestyle='-')

    # add horizontal markers if present
    if hmarkers:
        for marker in hmarkers:
            plt.axhline(marker, color='black', linestyle='--')
    if export_path:
        plt.savefig(export_path)

def visualize_subject(df: pd.DataFrame, subject: str):
    """Visualize all trials for a given subject"""
    subject_data = df[df['subject'] == subject]
    for _, trial in subject_data.iterrows():
        visualize_trial(df, subject, trial['mvc_level'], trial['trial_number'])
        plt.show()