from typing import cast, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import dataframe_image as dfi
from prediction.dpm import DirectPredictionModel
from globals.utils import Log
from prediction import constants

class Metrics:
    def __init__(self, reset: bool = False) -> None:
        self.metric_names = [f.__metric_name__ for f in self.__class__.__dict__.values() if hasattr(f, '__metric_name__')]
        base_columns = ['model', 'subject', 'mvc_level', 'trial_number'] + self.metric_names
        self.metrics_save_path = f"{constants.metrics_output_dir}/saved_metrics.csv"
        
        if reset:
            Log.info("Reset flag is True. Starting with a new metrics table.")
            self.metrics = pd.DataFrame(columns=base_columns)
        else:
            existing_metrics = self._load_existing_metrics()
            if not existing_metrics.empty and all(col in existing_metrics.columns for col in base_columns):
                self.metrics = existing_metrics
            else:
                if not existing_metrics.empty:
                    Log.warn("Existing metrics file is missing required columns. Starting with a new metrics table.")
                self.metrics = pd.DataFrame(columns=base_columns)

    def _mean_absolute_error(self, true: np.ndarray, predicted: np.ndarray) -> np.floating:
        """Compute mean absolute error (MAE)"""
        return np.mean(np.abs(true - predicted))
    
    _mean_absolute_error.__metric_name__ = 'MAE'
    
    def _mean_squared_error(self, true: np.ndarray, predicted: np.ndarray) -> np.floating:
        """Compute mean squared error (MSE)"""
        return np.mean(np.square(true - predicted))
    
    _mean_squared_error.__metric_name__ = 'MSE'
    
    def _r_squared(self, true: np.ndarray, predicted: np.ndarray) -> np.floating:
        """Compute R-squared (R2)"""
        return self._pearson_r(true, predicted) ** 2
    
    _r_squared.__metric_name__ = 'R2'
    
    def _pearson_r(self, true: np.ndarray, predicted: np.ndarray) -> np.floating:
        """Compute Pearson correlation coefficient"""
        return cast(np.floating, cast(Any, pearsonr(true, predicted)).statistic)
    
    _pearson_r.__metric_name__ = 'Pearson R'
    
    def _variance_accounted_for(self, true: np.ndarray, predicted: np.ndarray) -> np.floating:
        """Compute variance accounted for (VAF)"""
        return 1 - np.var(true - predicted) / np.var(true)
    
    _variance_accounted_for.__metric_name__ = 'VAF'
    
    def compute_metrics(self, model: DirectPredictionModel):
        """Compute mean of all measurements across test data for this model and add to table. This action replaces the previously recorded metrics."""
        Log.info(f"Computing metrics for {model.model_identifier} with {len(model.processor.test)} test trials...")

        # compute metrics for each trial in the models processor test set
        for _, trial in model.processor.test.iterrows():
            subject, mvc_level, trial_number = trial['subject'], trial['mvc_level'], trial['trial_number']
            sequence_length, stride = model.processor.sequence_length, model.processor.stride

            predicted = model.predict(neuron_data=trial['neuron_data'], mvc_level=trial['mvc_level'])
            true = trial['force_data']

            # only consider non-zero values in the true/predicted data
            non_zero_indices = np.where(true != 0)[0]
            if len(non_zero_indices) > 0:
                start_index = non_zero_indices[0]
                end_index = non_zero_indices[-1]
                true = true[start_index : end_index + 1]
                predicted = predicted[start_index : end_index + 1]
            else:
                Log.warn(f"Trial {trial['trial_id']} has no non-zero values in true data. Skipping...")
                continue 

            # ensure arrays are not empty after trimming
            if true.size == 0:
                continue

            metric_entry = pd.Series({
                'model': model.model_identifier,
                'subject': subject,
                'mvc_level': mvc_level,
                'trial_number': trial_number,
                'window': f"{sequence_length}-{stride}"
            })
            
            # get and measure all metrics (class methods with __metric_name__ attr)
            for f in self.__class__.__dict__.values():
                if hasattr(f, '__metric_name__'):
                    metric_name = cast(str, f.__metric_name__)
                    metric_entry[metric_name] = f(self, true, predicted)

            self.metrics = pd.concat([self.metrics, metric_entry.to_frame().T], ignore_index=True)

    def _optionally_export_df_as_png(self, df: pd.DataFrame, filename: str, export: bool = False):
        """Export dataframe to a file"""
        if export:
            dfi.export(df, f"{constants.metrics_output_dir}/{filename}-table.png", table_conversion='matplotlib')

    def metrics_by_model(self, export: bool = False) -> pd.DataFrame:
        """Return mean metrics grouped by a model column (drops others)"""
        if self.metrics.empty:
            Log.warn("Metrics table is empty. Run measure() first.")
            return pd.DataFrame()
        
        grouped = self.metrics.copy().drop(columns=['subject', 'mvc_level', 'trial_number', 'window']).groupby("model").mean().reset_index()
        self._optionally_export_df_as_png(grouped, 'metrics_by_model', export=export)
        return grouped
    
    def metrics_by_mvc(self, export: bool = False) -> pd.DataFrame:
        """Return mean metrics grouped by a mvc column (drops others)"""
        if self.metrics.empty:
            Log.warn("Metrics table is empty. Run measure() first.")
            return pd.DataFrame()
        
        grouped = self.metrics.copy().drop(columns=['subject', 'model', 'trial_number', 'window']).groupby("mvc_level").mean().reset_index()
        self._optionally_export_df_as_png(grouped, 'metrics_by_mvc', export=export)
        return grouped
    
    def metrics_by_window(self, export: bool = False) -> pd.DataFrame:
        """Return mean metrics grouped by a window column (drops others)"""
        if self.metrics.empty:
            Log.warn("Metrics table is empty. Run measure() first.")
            return pd.DataFrame()
        
        grouped = self.metrics.copy().drop(columns=['subject', 'model', 'mvc_level']).groupby("window").mean().reset_index()
        self._optionally_export_df_as_png(grouped, 'metrics_by_window', export=export)
        return grouped
    
    def _visualize_general_metrics(self, df: pd.DataFrame, feature: str, formatted_feature_name: str, export: bool = False):
        """Plot metrics for a given dataframe"""

        num_metrics = len(df.columns) - 1
        if num_metrics == 0:
            Log.warn("No metrics found in the table.")
            return

        # determine layout for subplots
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
        axes = axes.flatten() # flatten in case of single row/column

        metric_columns = [col for col in df.columns if col in self.metric_names]

        # plot each metric (subplot)
        for i, metric in enumerate(metric_columns):
            ax = axes[i]
            sns.barplot(x=feature, y=metric, data=df, ax=ax, palette='tab20', hue=feature, dodge=False, legend=False)
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel(formatted_feature_name)
            ax.set_ylabel(metric)
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor") 

        # delete unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Metrics by {formatted_feature_name}", fontsize=16)
        plt.tight_layout()

        # optionally export the plot before showing
        if export:
            export_path = f"{constants.metrics_output_dir}/metrics_by_{feature}.png"
            plt.savefig(export_path, dpi=300, bbox_inches='tight')
            Log.success(f"Metrics plot saved to {export_path}")

        plt.show()

    def visualize_by_model(self, export: bool = False):
        """Plot comparison of metrics by model"""
        self._visualize_general_metrics(self.metrics_by_model(), "model", "Model", export=export)

    def visualize_by_mvc(self, export: bool = False):
        """Plot comparison of metrics by mvc level"""
        self._visualize_general_metrics(self.metrics_by_mvc(), "mvc_level", "MVC Level", export=export)

    def visualize_by_window(self, export: bool = False):
        """Plot comparison of metrics by window size"""
        self._visualize_general_metrics(self.metrics_by_window(), "window", "Window Size", export=export)

    def visualize(self, export: bool = False):
        """Run all visualizations"""
        self.visualize_by_model(export=export)
        self.visualize_by_mvc(export=export)
        self.visualize_by_window(export=export)

    def save_metrics(self, export: bool = False):
        """Export metrics table to a file"""
        if self.metrics.empty:
            Log.warn("Metrics table is empty. Run measure() first.")
            return

        self.metrics.to_csv(self.metrics_save_path, index=False)
        Log.success(f"Metrics table saved to {self.metrics_save_path}")

    def _load_existing_metrics(self) -> pd.DataFrame:
        """Load metrics from a file"""
        try:
            df = pd.read_csv(self.metrics_save_path)
            Log.success(f"Metrics loaded from {self.metrics_save_path}")
            return df
        except FileNotFoundError:
            Log.warn(f"Metrics file not found at {self.metrics_save_path}")
            return pd.DataFrame()
        
    def exists(self) -> bool:
        """Check if the metrics table exists"""
        return not self.metrics.empty
