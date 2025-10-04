# scripts/engine_helpers.py
from __future__ import annotations
import math
from typing import Union, Optional

import numpy as np
import pandas as pd

Number = Union[int, float, np.number]
ArrayLike = Union[pd.Series, pd.DataFrame, np.ndarray, list, tuple, Number]


def _as_series(x: ArrayLike) -> pd.Series:
    """Coerce scalars/arrays to a pandas Series for aligned vectorized math."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        # pick first column for safety; caller should pass a Series ideally
        return x.iloc[:, 0]
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return pd.Series([x.item()])
        return pd.Series(x)
    if isinstance(x, (list, tuple)):
        return pd.Series(x)
    # scalar
    return pd.Series([x])


def safe_divide(
    num: ArrayLike,
    den: ArrayLike,
    default: float = 0.0,
    *,
    clip_inf: bool = True,
    precision: Optional[int] = None,
) -> ArrayLike:
    """
    Division that never crashes on zero/NaN and preserves the input shape/type.

    - If denominator is 0 or NaN → returns `default` at that position.
    - Works for scalars or vector types (Series/ndarray/list).
    - Optionally rounds to `precision` decimal places.
    - If both inputs are scalars, returns a scalar float.

    Examples
    --------
    safe_divide(10, 2) -> 5.0
    safe_divide([1, 2, 3], [0, 2, 0], default=0) -> [0.0, 1.0, 0.0]
    """
    # Scalar–scalar fast path
    if isinstance(num, (int, float, np.number)) and isinstance(den, (int, float, np.number)):
        if den is None or (isinstance(den, float) and math.isnan(den)) or den == 0:
            out = float(default)
        else:
            out = float(num) / float(den)
        if precision is not None:
            out = round(out, precision)
        return out

    s_num = _as_series(num)
    s_den = _as_series(den)

    # Align lengths if different (pad the shorter with NaN)
    if len(s_num) != len(s_den):
        max_len = max(len(s_num), len(s_den))
        s_num = s_num.reindex(range(max_len))
        s_den = s_den.reindex(range(max_len))

    mask = (s_den != 0) & (~s_den.isna())
    out = pd.Series(np.full(len(s_den), default, dtype="float64"))

    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc[mask] = s_num.loc[mask].astype("float64") / s_den.loc[mask].astype("float64")

    if clip_inf:
        out.replace([np.inf, -np.inf], default, inplace=True)

    if precision is not None:
        out = out.round(precision)

    # Return in the same "shape kind" as the numerator
    if isinstance(num, pd.Series):
        out.index = num.index
        return out
    if isinstance(num, (list, tuple, np.ndarray)):
        return out.to_numpy()
    # if the original was scalar-like but we took vector path, return scalar if single value
    return out.iloc[0] if len(out) == 1 else out
