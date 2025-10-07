import os
import pandas as pd

def read_csv_nonempty(path, usecols=None, expected_cols=None, **kwargs) -> pd.DataFrame:
    """
    Safe CSV reader: if file is missing or empty, return an empty DataFrame with expected columns.
    Prevents pandas.errors.EmptyDataError when upstream sources are absent.
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        cols = expected_cols or (usecols or [])
        cols = [str(c) for c in cols]
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, usecols=usecols, **kwargs)
