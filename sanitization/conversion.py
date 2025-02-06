from typing import Union
from numpy.typing import NDArray
from sanitization.types import DataTypeKeys, UnifiedSubjectData
import os
import shutil
import numpy as np
import pandas as pd
import scipy.io
import sanitization.constants as constants
from globals.utils import Log

class Conversion:
    """Converts and pre-processes the original data files into a format that is easier to work with. The data is cleaned up and converted into a single file present in the designated data directory."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.data_dir = os.path.join(prefix, constants.data_dir)
        self.processed_dir = lambda s: os.path.join(prefix, constants.processed_data_dir(s))
        self.raw_data_dir = lambda s: os.path.join(prefix, constants.raw_data_dir(s))

    def cleanup_data(self):
        """Cleans up the data directory by moving all files from the data electrode folder to a raw folder in the subject's directory. All other files and folders in the subject's directory are removed. The subject folder is then renamed to be all lowercase with dashes in place of spaces."""

        for subject in os.listdir(self.data_dir):
            subject_dir = os.path.join(self.data_dir, subject)

            if os.path.isdir(subject_dir):
                raw_dir = self.raw_data_dir(subject)

                # skip this subject if the raw folder already exists
                if os.path.exists(raw_dir):
                    Log.warn(f"Skipping {subject} as raw folder already exists")
                    continue
                
                # find the electrode folder (Array1...) (case insensitive)
                electrode_path = None
                for folder in os.listdir(subject_dir):
                    if folder.lower().startswith("array1"):
                        electrode_path = os.path.join(subject_dir, folder)
                        break

                if electrode_path is None:
                    Log.error(f"No electrode folder found for {subject}")
                    break

                # create the raw folder if it doesn't exist
                os.makedirs(raw_dir, exist_ok=True)

                # move all files from Array1_proximal electrode to raw
                if os.path.exists(electrode_path):
                    for file_name in os.listdir(electrode_path):

                        if "PF" in file_name:
                            Log.warn(f"Skipping {file_name} as it contains PF")
                            continue

                        new_file_name = file_name

                        # replace spaces with periods, remove underscores, and remove the "DF" string
                        new_file_name = new_file_name.replace("DF", "").replace(" ", ".").replace("_", "")

                        content_start_index = 0

                        # add start if we hit a number or the start of the "MVC" string
                        for i, c in enumerate(new_file_name):
                            if c.isdigit() or c == "M": 
                                content_start_index = i
                                break
                        
                        # remove the prefix and suffix surrounding the mvc level
                        new_file_name = new_file_name[content_start_index:]
                        first_dot_index = new_file_name.index(".")
                        new_file_name = new_file_name[:new_file_name.index(".", first_dot_index+1)] + ".mat"

                        # move the file to the raw folder
                        file_path = os.path.join(electrode_path, file_name)
                        new_file_path = os.path.join(raw_dir, new_file_name)
                        if os.path.isfile(file_path):
                            shutil.move(file_path, new_file_path)

                # remove all other files and folders in the subject's directory
                for item in os.listdir(subject_dir):
                    item_path = os.path.join(subject_dir, item)
                    if item_path != raw_dir:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)

                # rename the subject folder to be all lowercase and have dashes in place of spaces
                new_subject_folder = subject.lower().replace(" ", "-")
                new_subject_dir = os.path.join(self.data_dir, new_subject_folder)
                os.rename(subject_dir, new_subject_dir)

                Log.success(f"Cleaned up and renamed {subject} to {new_subject_folder}")

    @staticmethod
    def _convert_mv_to_kgf(mv_profile: np.ndarray) -> np.ndarray:
        # Convert mV to volts and then to kg-force
        v_profile = mv_profile / 1000
        kg_profile = v_profile * 200 / (2001 * 5 * 0.5)

        # Define the baseline value
        baseline = kg_profile[0]

        # Translate all points so that the first point is at y=0
        translated_kg_profile = kg_profile + baseline

        # Flip the translated data about the baseline
        mirrored_translated_kg_profile = 2 * baseline - translated_kg_profile

        return mirrored_translated_kg_profile

    def _process(self, subject: str):
        """Loads and parses the .mat files from the given directory and extracts the MVC and force data into separate files (category and trial wise)."""
        raw_dir = self.raw_data_dir(subject)
        processed_dir = self.processed_dir(subject)

        for file in os.listdir(raw_dir):
            if not file.endswith(".mat"):
                Log.warn(f"Skipping {file} as it is not a .mat file")
                continue

            mvc_level: Union[int, None] = None
            trial_number: Union[int, None] = None

            # "MVC" is present -> set mvc_level to 100, else extract the number
            mvc_level = 100 if "MVC" in file else int(file.split(".")[0])
            trial_number = int(file.split(".")[1].split(".")[0])

            mvc_file_path = os.path.join(processed_dir, f"MVC.{mvc_level}.{trial_number}.txt")
            force_file_path = os.path.join(processed_dir, f"FORCE.{mvc_level}.{trial_number}.txt")

            if mvc_level is None or trial_number is None:
                Log.error(f"Could not extract MVC level or trial number from {file}")
                continue

            if os.path.exists(mvc_file_path) and os.path.exists(force_file_path):
                Log.warn(f"Skipping {file} as it has already been processed")
                continue

            if mvc_level not in constants.mvc_levels:
                Log.warn(f"Skipping {file} as the MVC level {mvc_level} is not in the list of valid MVC levels")
                continue

            Log.info(f"Processing {file} (MVC: {mvc_level}, Trial: {trial_number})...")

            raw_data = scipy.io.loadmat(os.path.join(raw_dir, file)) # load the .mat file

            # extract the MVC and force data
            mvc_data = raw_data["MUPulses"]

            # no MVC data was found and the MVC level is not 100 (only force) -> skip the file
            if len(mvc_data) == 0 and mvc_level != 100:
                Log.warn(f"Skipping {file} as no MVC data was found")
                continue
            
            force_data = self._convert_mv_to_kgf(raw_data["ref_signal"][0])
            np.savetxt(force_file_path, force_data)

            # only process force data if the MVC level is 100
            if mvc_level != 100:
                trial_time = raw_data["SIGlength"][0][0]
                compiled_mvc_data = np.concatenate(mvc_data, axis=0)
                
                time_length = np.round(2048 * trial_time)
                shape = (int(time_length), len(compiled_mvc_data))
                activation_time = np.zeros(shape)

                for neuron_index, neuron_firings in enumerate(compiled_mvc_data):
                    # iterate over each firing time within the motor neuron's firing array
                    neuron_firings=neuron_firings.flatten()
                    for firing_time in neuron_firings:
                        index = int(np.round(firing_time)) # calculate the corresponding index
                        activation_time[index, neuron_index] = firing_time # assign the firing time to the corresponding index

                # save mvc data
                np.savetxt(mvc_file_path, activation_time)


    def process_subjects(self):
        """Processes the data by performing some operation on each file in the raw folder of each subject's directory."""
        subjects = os.listdir(self.data_dir)

        for subject in subjects:
            Log.info(f"\nProcessing data for {subject}\n")

            # skip this subject if the raw folder doesn't exist
            if not os.path.exists(self.raw_data_dir(subject)):
                Log.warn(f"Skipping {subject} as raw folder does not exist")
                continue
            
            # create the 'processed' folder if it doesn't exist
            if not os.path.exists(self.processed_dir(subject)):
                os.makedirs(self.processed_dir(subject), exist_ok=True)

            self._process(subject) # process the subject's data and export to their processed folder

            Log.success(f"Successfully processed data for {subject}")


    @staticmethod
    def _sanitize_mvc_data(data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Replace all non-zero values with 1. 'Spikes' that are within a given range (33ms) of each other are eliminated as this is an impossibility."""
        
        # replace all non-zero values with 1 in all the data
        trial_data = np.where(data != 0, 1, data) # 0 > -> 1

        if trial_data.ndim != 2:
            spike_indices = np.where(trial_data == 1)[0]  # Find indices of spikes
            
            if len(spike_indices) == 0:  # Check if spike_indices is empty
                return trial_data  # Skip to the next neuron if no spikes are found
            
            last_valid_spike = spike_indices[0]  # Initialize the last valid spike index

            for k in range(1, len(spike_indices)):  # Start from the second spike
                if spike_indices[k] - last_valid_spike <= constants.min_spike_interval:
                    trial_data[spike_indices[k]] = 0  # Eliminate the spike
                else:
                    last_valid_spike = spike_indices[k]  # Update the last valid spike index 

            return trial_data


        for i in range(trial_data.shape[1]):  # Iterate over each neuron (column)
            spike_indices = np.where(trial_data[:, i] == 1)[0]  # Find indices of spikes
            
            if len(spike_indices) == 0:  # Check if spike_indices is empty
                continue  # Skip to the next neuron if no spikes are found
            
            last_valid_spike = spike_indices[0]  # Initialize the last valid spike index

            for k in range(1, len(spike_indices)):  # Start from the second spike
                if spike_indices[k] - last_valid_spike <= constants.min_spike_interval:
                    trial_data[spike_indices[k], i] = 0  # Eliminate the spike
                else:
                    last_valid_spike = spike_indices[k]  # Update the last valid spike index 

        return trial_data

    def _load_subject_data(self, subject: str):
        """Load all the subject data from the processed data directory"""
        Log.info(f"Loading data for {subject}...")

        data = {}

        # load all the subjects mvc and force data
        for filename in os.listdir(self.processed_dir(subject)):
            # file name format: MVC.{mvc_level}.{trial_number}.txt
            if "MVC" or "FORCE" in filename:
                # extract mvc level and trial number from the file name
                mvc_level, trial_number = map(int, filename.replace("MVC.", "").replace("FORCE.", "").replace(".txt", "").split("."))
                
                # ensure mvc level is valid
                if mvc_level not in constants.mvc_levels:
                    Log.warn(f"Skipping {filename} as the MVC level {mvc_level} is not in the list of valid MVC levels")
                    continue
                
                # no present mvc key -> create new key
                if mvc_level not in data:
                    data[mvc_level] = {}
                
                # no present trial key -> create new key
                if trial_number not in data[mvc_level]:
                    data[mvc_level][trial_number] = {}
                
                try:
                    # process the force or mvc data
                    data_key: DataTypeKeys = "mvc" if "MVC" in filename else "force"
                    trial_data = np.loadtxt(os.path.join(self.processed_dir(subject), filename))

                    # mvc data must be sanitized before being stored
                    if data_key == "mvc":
                        trial_data = self._sanitize_mvc_data(trial_data)

                    data[mvc_level][trial_number][data_key] = trial_data
                except Exception as e:
                    raise ValueError(f"Error loading data from file {filename}: {e}")
            else:
                Log.warn(f"Skipping {filename} as it does not contain 'MVC' or 'FORCE'")

        # Sort the data by keys (mvc_level and trial_number)
        data = {mvc_level: dict(sorted(trials.items())) for mvc_level, trials in sorted(data.items())}

        return data

    def get_dataframe(self):
        """Get all the data from the processed files as a single pandas dataframe."""
        data: dict[str, UnifiedSubjectData] = {}

        subjects = os.listdir(self.data_dir)

        # load all the data for each subject
        for subject in subjects:
            subject_data = self._load_subject_data(subject)
            data[subject] = subject_data

        entries = [] 

        # flatten the data into a list of entries
        for subject, levels in data.items():
            for mvc_level, trials in levels.items():
                for trial_number, datums in trials.items():
                    mvc_data = datums["mvc"] if "mvc" in datums else None
                    force_data = datums["force"] if "force" in datums else None
                    entries.append([subject, mvc_level, trial_number, mvc_data, force_data])

        return pd.DataFrame(entries, columns=["subject", "mvc_level", "trial_number", "mvc_data", "force_data"])