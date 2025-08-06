from typing import cast, Optional
import io
import base64
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from globals import constants

def b64_encode_plot():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def visualize_trial(
        df: Optional[pd.DataFrame] = None, 
        subject: Optional[str] = None,
        mvc_level: Optional[int] = None, 
        trial_number: Optional[int] = None, 
        neuron_data: Optional[np.ndarray] = None,
        force_data: Optional[np.ndarray] = None,
        predicted_force: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        export_path: Optional[str] = None, 
        ax: Optional[Axes] = None,
        show_legend: bool = False,
        encode: Optional[bool] = False,
        dark_mode: bool = False
):
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
    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, 6))
    else:
        fig = ax.figure
        
        if ax is not None:
            for extra_ax in fig.axes[1:]:
                fig.delaxes(extra_ax)
            ax.clear()

    if dark_mode:
        fig.patch.set_facecolor('#161616')
        ax.set_facecolor('#161616')

    ax.set_ylabel("Motor Neuron")
    ax.set_xlabel("Time (s)")

    assert ax is not None, "Invalid Axes"

    if title is not None:
        ax.set_title(title)
    else:
        # conditionally set title based on whether trial data is available
        if trial is not None:
            ax.set_title(f"{trial['subject']} | Trial {trial['trial_number']} | {trial['mvc_level']}% MVC")
        else:
            ax.set_title(f"Visualizing Manual Data")

    force_data = cast(pd.Series, trial)["force_data"] if force_data is None else force_data
    neuron_data = cast(pd.Series, trial)["neuron_data"].copy() if neuron_data is None else neuron_data

    assert neuron_data is not None, "Neuron data must be provided either directly or through the DataFrame."
    assert force_data is not None, "Force data must be provided either directly or through the DataFrame."

    # remove all rows with no activations from neuron_data
    if neuron_data.ndim > 1:
        neuron_data = neuron_data[~np.all(neuron_data == 0, axis=1)]

    def plot_force_only(force_data: np.ndarray, time_values: np.ndarray) -> None:
        """Plots force data only"""
        ax.plot(time_values, force_data, label="Force Profile", color=("#E5E5E5" if dark_mode else "red"))

    def plot_raster(neuron_data: np.ndarray, time_values: np.ndarray) -> None:
        """Plots raster representation of neuron data"""
        if neuron_data.ndim == 1:
            raise ValueError("Neuron data must be a 2D array with shape (num_neurons, num_time_points).")

        # Sort and plot neurons based on first spike time
        neuron_order = np.argsort(np.argmax(neuron_data, axis=1))
        sorted_data = neuron_data[neuron_order]
        for i, neuron in enumerate(sorted_data):
            if i == len(sorted_data) - 1:
                ax.plot(time_values, neuron * (i + 1), '|', alpha=1.0, color=("#161616" if dark_mode else "white"), linewidth=2)
            else:
                ax.plot(time_values, neuron * (i), '|', alpha=(0.5 if dark_mode else 1.0))

    def plot_force_overlay(force_data: np.ndarray, time_values: np.ndarray, force_axis: Axes):
        """Plots force overlay on additional axis"""
        force_axis.plot(time_values, force_data, label="Force Profile", color= ("#BABABA" if dark_mode else "red"))
        
        # optionally show legend for force overlay
        if show_legend:
            lines, labels = force_axis.get_legend_handles_labels()
            force_axis.legend(lines, labels, loc='upper right')

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
        force_axis = cast(Axes, ax.twinx())
        force_time = np.linspace(0, total_time, len(force_data))
        plot_force_overlay(force_data, force_time, force_axis)
        
        # add predicted force overlay if present
        if predicted_force is not None and predicted_force.size > 0:
            # Use same time scale as actual force
            pred_force_time = np.linspace(0, total_time, len(predicted_force))
            force_axis.plot(pred_force_time, predicted_force, label="Predicted Force", color=("#FF6900" if dark_mode else 'blue'))
            # Update legend to include both force profiles
            lines, labels = force_axis.get_legend_handles_labels()
            force_axis.legend(lines, labels, loc='upper right')

    if export_path:
        plt.savefig(export_path)
    if encode:
        return b64_encode_plot()
    else:
        plt.show()

def visualize_subject(df: pd.DataFrame, subject: str):
    """Visualize all trials for a given subject"""
    subject_data = df[df['subject'] == subject]
    for _, trial in subject_data.iterrows():
        visualize_trial(df, subject, trial['mvc_level'], trial['trial_number'])
        plt.show()

def visualize_spike_trains(spike_trains: np.ndarray, section: Optional[tuple[int, int]] = None, bw: bool = False, line_height: float = 0.5, dark_mode: bool = False):
    """Visualize spike trains as an image, scaling x axis by 2048."""
    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    fig, ax = plt.subplots(figsize=(20, 6))
    
    if dark_mode:
        fig.patch.set_facecolor('#161616')
        ax.set_facecolor('#161616')

    # section provided -> zoom into range
    if section is not None:
        start, end = section
        spike_trains = spike_trains[:, start:end]

    # sort neurons by first activation (lowest index = earliest activation)
    first_activations = np.argmax(spike_trains > 0, axis=1)
    # neuron never fires -> set to large value so it sorts last
    first_activations[np.all(spike_trains == 0, axis=1)] = spike_trains.shape[1] + 1
    neuron_order = np.argsort(first_activations)
    spike_trains = spike_trains[neuron_order]

    for i in range(spike_trains.shape[0]):
        times = np.where(spike_trains[i] >= 0)[0]
        times_scaled = times / 2048
        magnitudes = spike_trains[i, times]
        cmap = cm.get_cmap('plasma')
        colors = [cmap(mag) for mag in magnitudes] if not bw else [(0, 0, 0, mag) for mag in magnitudes]

        ax.eventplot(positions=[times_scaled], lineoffsets=i, colors=[colors], linelengths=line_height) # type: ignore

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons")
    ax.set_title("Spike Trains")
    
    plt.show()

def visualize_training(training_history: dict[str, list[float]], export_path: Optional[str] = None, encode: Optional[bool] = False, dark_mode: bool = False):
    """Visualize training and validation loss/accuracy over epochs. Optionally encode plot as base64."""
    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    _, ax = plt.subplots(figsize=(14, 5))
    
    if dark_mode:
        plt.gcf().patch.set_facecolor('#161616')
        ax.set_facecolor('#161616')
    
    ax.plot(training_history["loss"], label="Training Loss")
    ax.plot(training_history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Loss and Validation Loss over Epochs")

    if export_path:
        plt.savefig(export_path)
    if encode:
        return b64_encode_plot()
    else:
        plt.show()

def visualize_number_of_neurons_impact(
    dataset: pd.DataFrame,
    predictions: list[tuple[int, float]], 
    export_path: Optional[str] = None, 
    encode: Optional[bool] = False, 
    dark_mode: bool = False):
    """Visualize the impact of varying number of neurons on model performance as a histogram."""

    # extract neuron counts and scores from predictions
    neuron_counts = np.array([
        dataset.iloc[index]["neuron_data"].shape[0] if isinstance(dataset.iloc[index]["neuron_data"], np.ndarray) else 0
        for index, _ in predictions
    ])
    scores = np.array([score for _, score in predictions])

    min_neurons, max_neurons = neuron_counts.min(), neuron_counts.max()

    neuron_nums = np.arange(min_neurons + 1, max_neurons + 1)
    mean_scores = np.array([
        scores[neuron_counts == n].mean() if np.any(neuron_counts == n) else 0
        for n in neuron_nums
    ])
    counts_per_bin = np.array([
        np.count_nonzero(neuron_counts == n) for n in neuron_nums
    ])

    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, ax = plt.subplots(figsize=(14, 5))
    if dark_mode:
        fig.patch.set_facecolor('#161616')
        ax.set_facecolor('#161616')
    ax.set_title("Impact of Number of Neurons on Model Performance")
    ax.set_xlabel("Number of Neurons")
    ax.set_ylabel("Average Performance Metric (R² Score)")
    ax.grid(axis='y', alpha=0.25)
    bar = ax.bar(neuron_nums, mean_scores, color='#f0ebd8')

    bar_labels = [f"{count} ({score:.2f})" if count > 0 else "" for count, score in zip(counts_per_bin, mean_scores)]
    ax.bar_label(bar, labels=bar_labels, padding=2, fontsize=10, color='black' if not dark_mode else '#f0ebd8')

    if export_path:
        plt.savefig(export_path)
    if encode:
        return b64_encode_plot()
    else:
        plt.show()

def visualize_data_volume_impact(
    dataset: pd.DataFrame,
    predictions: list[tuple[int, float]], 
    export_path: Optional[str] = None, 
    encode: Optional[bool] = False, 
    dark_mode: bool = False):
    """Visualize the impact of varying data volume on model performance as a histogram."""

    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    data_point_counts, performance_metrics = zip(*[
        (len(dataset.iloc[i]['neuron_data'].reshape(-1)), score)
        for i, score in predictions
    ])
    
    data_point_counts = np.array(data_point_counts)
    performance_metrics = np.array(performance_metrics)

    bins = np.linspace(np.min(data_point_counts), np.max(data_point_counts), 8)
    bin_indices = np.digitize(data_point_counts, bins) - 1

    binned_values = [performance_metrics[bin_indices == i] if np.any(bin_indices == i) else np.nan for i in range(len(bins)-1)]
    bin_means = np.array([np.nanmean(values) if isinstance(values, np.ndarray) and values.size > 0 else np.nan for values in binned_values])
    bin_counts = [len(values) if isinstance(values, np.ndarray) else 0 for values in binned_values]

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    bar_labels = [
        f"{count} ({mean:.2f})" if count > 0 else ""
        for count, mean in zip(bin_counts, bin_means)
    ]
    
    _, ax = plt.subplots(figsize=(14, 5))
    if dark_mode:
        plt.gcf().patch.set_facecolor('#161616')
    ax.set_facecolor('#161616')
    ax.set_title("Impact of Neuronal Volume on Model Performance")
    ax.set_xlabel("Data Point Count (Number of Neuronal Activations)")
    ax.set_ylabel("Average Performance Metric (R² Score)")
    ax.grid(axis='y', alpha=0.25)
    bar = ax.bar(bin_centers, bin_means, width=np.diff(bins), color="#FF6900", edgecolor="black")
    ax.bar_label(bar, labels=bar_labels, padding=2, fontsize=10, color='black' if not dark_mode else 'white')
    if export_path:
        plt.savefig(export_path)
    if encode:
        return b64_encode_plot()
    else:
        plt.show()
