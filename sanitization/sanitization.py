import pandas as pd
import numpy as np
from globals.utils import Log
import sanitization.constants as constants

class Sanitization:
    """Remove or correct any errors in the data. This includes removing outliers, filling in missing values, and correcting any other errors in the data. Some entries are not recoverable and will be "purged"."""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _purge(self):
        """Purge the given pandas dataframe of any entries that are not recoverable."""
        pass

    def _purge_insufficient_neuron_data(self):
        """Remove entries that have insufficient number of neurons in the data."""
        Log.info("Purging entries with insufficient neuron data...")
        prev_len = len(self.df)
        
        # filter function to check if the number of neurons is valid
        def valid_number_of_neurons(x: np.ndarray) -> bool:
            return x is not None and x.ndim > 1 and x.shape[1] >= constants.min_neurons

        # purge any entries that do not have a valid number of neurons
        self.df = self.df[self.df["neuron_data"].map(valid_number_of_neurons)] 

        Log.success(f"Purged {prev_len - len(self.df)} entries with insufficient neuron data. {len(self.df)} entries remaining.")

    def _check_spikes(self):
        """Check if there are any major force spikes in the data."""
        pass

    def _check_neuron_inconsistencies(self):
        """Check if there are any outlandish inconsistencies in the neuron data."""
        pass

    def _check_measurement_decorrelation(self):
        """Check for discrepancies between measured neuron activations and force readings. An abundance of discrepancies (MD) in the dataset leads to General Measurement Decorrelation (GMD) and marshals a purge."""
        pass

    def _check_trend(self):
        """Check to see if trend is significantly different average trend."""
        pass

    def sanitize(self):
        """Sanitize the given pandas dataframe with feature-specific methods."""
        self._purge_insufficient_neuron_data() # N
        return self.df