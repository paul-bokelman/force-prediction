"""Manually sanitize trials with a GUI using on-the-fly visualization. Modifies the existing sanitized dataset by removing trials that are not marked as valid."""

from typing import cast
from numpy.typing import NDArray
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Label
import tkinter.messagebox as messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from globals.utils import Log
from processing import constants
from analysis.plotting import visualize_trial
import globals.constants as g_constants

df: pd.DataFrame = pd.read_pickle(constants.dataset_path) # get the saved dataset`
valid_trials: dict[str, list[float]] = {}
lines_for_current_trial: list[float] = []

# existing valid trials file -> load and use it
if os.path.exists(constants.manually_selected_valid_trials_path):
    Log.info(f"Found {constants.manually_selected_valid_trials_path}, skipping manual purge GUI.")
    with open(constants.manually_selected_valid_trials_path, "r") as f:
        valid_trials = json.load(f)

def trial_id(trial):
    return f"{trial.subject}.{trial.mvc_level}.{trial.trial_number}"

total_trials = len(df)
trials = iter(df.itertuples(index=True))

print(f"Total trials: {total_trials}, Valid trials: {len(valid_trials)}")

root = Tk()
root.title("Trial Sorter")

label = Label(root, text="")
label.pack(side="top", fill="x")  # Pack label first, fill horizontally

# setup plot scaffolding for visualization
fig, ax = plt.subplots(figsize=(18, 6))  # Smaller and wider
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

def show_trial():
    global last_trial, lines_for_current_trial
    ax.clear()
    lines_for_current_trial = [] # Reset lines for the new trial
    trial = next(trials, "complete")
    if trial == "complete":
        print("All trials processed.")
        label.config(text="Done! Close window to finish.")
        canvas.draw()
        return
    assert not isinstance(trial, str), "Expected trial to be a namedtuple, got string 'complete'."
    last_trial = trial  # Save for mark_as_valid
    label.config(text=f"{trial.subject} | MVC {trial.mvc_level} | Trial {trial.trial_number} ({cast(int, trial.Index) + 1}/{total_trials})")
    visualize_trial(neuron_data=cast(NDArray, trial.neuron_data), force_data=cast(NDArray, trial.force_data), ax=ax)

    canvas.draw()

def mark_as_valid():
    """Mark the current trial as valid and save it to the valid trials dictionary."""
    global last_trial, lines_for_current_trial
    tid = trial_id(last_trial)
    valid_trials[tid] = sorted(lines_for_current_trial) # store sorted x-coordinates
    with open(constants.manually_selected_valid_trials_path, "w") as f:
        json.dump(valid_trials, f, indent=2)

def key_pressed(event):
    if event.keysym == 'Return':  # mark the trial as valid
        mark_as_valid()  # mark current trial as valid
        show_trial()
    elif event.char == 'q':  # quit the GUI
        root.quit()
        return
    else:
        show_trial()

def onclick(event):
    global lines_for_current_trial
    if event.inaxes is None:
        return
    
    if len(lines_for_current_trial) >= 2:
        messagebox.showinfo("Limit Reached", "You can only draw two lines per trial.")
        return

    x = event.xdata
    lines_for_current_trial.append(x)
    # Draw on all axes in the figure
    for axis in event.inaxes.figure.axes:
        axis.axvline(x=x, color='red', linestyle='--')
    canvas.draw()
    print(f"Clicked at x = {x}, Lines: {lines_for_current_trial}")

canvas.mpl_connect("button_press_event", onclick)

root.bind('<Key>', key_pressed)
show_trial()
root.mainloop()

def postprocess_valid_trial(trial: pd.Series, marks: list[float]) -> pd.Series:
    """Post-process a valid trial by removing the segments outside the provided marks."""
    # no marks -> return the trial as is
    if len(marks) == 0:
        return trial
    
    if len(marks) > 2:
        raise ValueError(f"More than two marks provided for {trial.subject} | MVC {trial.mvc_level} | Trial {trial.trial_number}. ")

    trial_start, trial_end = 0, len(trial['force_data']) # get the start and end of the trial
    trial_middle_index = int(len(trial['force_data']) / 2)  # middle index of the force data
    marks = [int(mark * g_constants.sampling_frequency) for mark in marks] # convert to proper scale (from seconds to samples)
    marks = [trial_start if mark < trial_start else trial_end if mark > trial_end else mark for mark in marks] # reassign marks that are out of bounds

    marks.sort()

    if len(marks) == 1: # only one mark provided, use it to trim the trial

        # mark before middle index (on the left) -> set as trial start otherwise -> trial end
        if marks[0] < trial_middle_index:
            trial_start = marks[0]
        else:
            trial_end = marks[0]
    else: # two marks provided, use them to trim the trial
        trial_start = marks[0]
        trial_end = marks[1]

    # trim the trial data
    trial['force_data'] = trial['force_data'][trial_start:trial_end]
    trial['neuron_data'] = trial['neuron_data'][:, trial_start:trial_end]

    return trial

if messagebox.askyesno("Save", "Do you want to save the filtered dataset with only valid trials?"):
    trial_ids_series = df['subject'].astype(str) + "." + df['mvc_level'].astype(str) + "." + df['trial_number'].astype(str)
    df = df[trial_ids_series.isin(valid_trials.keys())]

    df = df.apply(lambda trial: postprocess_valid_trial(trial, valid_trials[trial_id(trial)]), axis=1)
    df.reset_index(drop=True, inplace=True)

    df.to_pickle(constants.dataset_path)
    messagebox.showinfo("Saved", "Filtered dataset saved.")
else:
    messagebox.showinfo("Not Saved", "Filtered dataset was not saved.")