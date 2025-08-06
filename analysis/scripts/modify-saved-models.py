# python -m analysis.scripts.modify-saved-models

from typing import cast
import inquirer
import os
import analysis.constants
import numpy as np
from processing.utils import get_dataframe
import json
import shutil

def get_candidates() -> dict[str, dict[str, np.floating | bool]]:
    registered_candidates = [c.replace(".json", "") for c in os.listdir(analysis.constants.views_candidates_dir)]
    saved_candidates = [c for c in os.listdir("prediction/out/candidates") if len(c) == 10]
    candidates = {k: {"registered": False, "saved": False, "score": 0.0} for k in set(registered_candidates) & set(saved_candidates)}

    #/ inefficient but as a wise man once said: "fuck it, we ball"
    for key in candidates:
        if key in registered_candidates:
            candidates[key]["registered"] = True
            metrics_df = get_dataframe(os.path.join("prediction/out/candidates", key, "metrics.pkl"))
            candidates[key]["score"] = round(metrics_df.iloc[0]["r2"], 4)
        if key in saved_candidates:
            candidates[key]["saved"] = True

    return candidates

def remove_candidate(candidate: str):
    saved_candidate_path = os.path.join("prediction/out/candidates", candidate)
    if os.path.exists(saved_candidate_path):
        shutil.rmtree(saved_candidate_path)
    candidate_view_identifier = candidate + ".json"
    registered_candidate_path = os.path.join(analysis.constants.views_candidates_dir, candidate_view_identifier)
    if os.path.exists(registered_candidate_path):
        os.remove(registered_candidate_path)

candidates = get_candidates()

questions = [
    inquirer.Checkbox(
        'candidates_to_remove',
        message="Choose candidates to remove",
        choices=[f"{k} | {'R' if v['registered'] else '-'},{'S' if v['saved'] else '-'} | {v['score']}%" for k, v in candidates.items()]
    ),
]

candidates_to_remove = [c[:10] for c in cast(list[str], cast(dict, inquirer.prompt(questions))["candidates_to_remove"])]

for candidate in candidates_to_remove:
    if candidate in candidates:
        remove_candidate(candidate)
        print(f"Removed candidate {candidate}")
    else:
        print(f"Candidate {candidate} not found, skipping...")

print("Done removing candidates.")
