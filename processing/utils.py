from typing import Optional
import os
import pandas as pd
from processing import constants

def get_dataframe(path: str = constants.dataset_path , subjects: Optional[list[str]] = None) -> pd.DataFrame:
    """Gets the saved dataframe dataset with processed entries filtered by subjects if provided."""
    
    # no dataset exists -> return empty dataframe
    if not os.path.exists(path):
        return pd.DataFrame() #/ should probably return None
    
    try:
        df: pd.DataFrame = pd.read_pickle(path)
        return df if subjects is None else df[df["subject"].isin(subjects)].reset_index(drop=True)
    except Exception:
        return pd.DataFrame()