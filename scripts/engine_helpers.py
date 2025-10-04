# scripts/engine_helpers.py

from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = [
    "safe_divide",
    "make_team_last4_from_player_form",
]

def safe_divide(a, b, default: float = 0.0):
    """
    Elementwise safe division that returns `default` when denominator is 0/NaN.
    Works with scalars, numpy arrays, and pandas Series.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(a_arr, b_arr)
        # where b == 0 or NaN -> default
        mask = ~np.isfinite(out)  # inf or nan
        out[mask] = default

    # If the inputs were scalars, return a scalar
    if np.isscalar(a) and np.isscalar(b):
        return float(out)
    # If the inputs were pandas objects, preserve type
    if isinstance(a, (pd.Series, pd.Index)) or isinstance(b, (pd.Series, pd.Index)):
        return pd.Series(out, index=getattr(a, "index", None))
    return out


def make_team_last4_from_player_form(pform: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple 'last 4 weeks' team aggregate from player_form.
    - Robust to empty/missing data (returns empty DataFrame in that case).
    - Only uses numeric columns it can find; ignores id / text columns.

    Returns:
        DataFrame with one row per team and the mean of the last 4 weeks
        of any numeric columns present. Columns are suffixed with `_last4`.
    """
    # Guard: nothing to do
    if pform is None or len(pform) == 0:
        # empty result – engine will continue gracefully
        return pd.DataFrame()

    # Need at least these columns to group on
    required = {"team", "week"}
    if not required.issubset(set(pform.columns)):
        return pd.DataFrame()

    # Keep only numeric columns to aggregate
    numeric_cols = [
        c for c in pform.columns
        if c not in ["team", "week", "player", "player_name", "gsis_id", "recent_team", "position"]
        and pd.api.types.is_numeric_dtype(pform[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    # Sort so tail(4) is the latest four weeks
    pform = pform.sort_values(["team", "week"])

    # For each team, take the mean of the last 4 weeks of numeric columns
    def last4_mean(g: pd.DataFrame) -> pd.Series:
        tail4 = g.tail(4)
        return tail4[numeric_cols].mean(numeric_only=True)

    out = (
        pform.groupby("team", as_index=False)
        .apply(last4_mean)
        .reset_index(drop=True)
    )

    # Suffix to make it obvious these are last-4 aggregates
    out = out.add_suffix("_last4")
    out = out.rename(columns={"team_last4": "team"})  # keep a 'team' key
    return out
