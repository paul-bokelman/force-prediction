from typing import cast, Optional
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import globals.constants as constants

def visualize_trial(
        df: pd.DataFrame, 
        subject: str, 
        mvc_level: int, 
        trial_number: int, 
        output: Optional[str] = None, 
        vmarkers: Optional[list[float]] = None, 
        hmarkers: Optional[list[float]] = None,
    ) -> None:
    """Plot grid of raster plots with force overlays for a given trial. Optionally save the plot to a file and or apply markers."""
        
    # filter data for the given subject, trial, and mvc level
    trial = df[(df["subject"] == subject) & (df["trial_number"] == trial_number) & (df["mvc_level"] == mvc_level)].iloc[0]

    # setting up the plot
    plt.figure(figsize=(18, 6))
    plt.title(f"{trial["subject"]} | Trial {trial["trial_number"]} | {trial["mvc_level"]}% MVC")
    plt.ylabel("Motor Neuron")
    plt.xlabel("Time (s)")

    force_data = trial["force_data"]
    neuron_data = trial["neuron_data"]

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

    def plot_force_overlay(force_data: np.ndarray, time_values: np.ndarray, force_axis: Axes):
        """Plots force overlay on additional axis"""
        force_axis.set_ylabel("Force", color='red')
        force_axis.plot(time_values, force_data, label="Force Profile", color='red')
        
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
        plot_force_overlay(force_data, force_time, force_axis)

    # add vertical markers if present
    if vmarkers:
        for marker in vmarkers:
            plt.axvline(marker, color='black', linestyle='--')

    # add horizontal markers if present
    if hmarkers:
        for marker in hmarkers:
            plt.axhline(marker, color='black', linestyle='--')
    if output:
        plt.savefig(output)