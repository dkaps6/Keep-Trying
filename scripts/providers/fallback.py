#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd
from typing import Callable, Optional

def use_if_empty(primary: Optional[pd.DataFrame],
                 *providers: Callable[[], Optional[pd.DataFrame]]) -> pd.DataFrame:
    """
    Return primary if it has rows; otherwise call providers in order.
    First provider that returns a non-empty DataFrame wins.
    If all empty/None, return primary (even if empty) to keep schema stable.
    """
    def nonempty(df: Optional[pd.DataFrame]) -> bool:
        try:
            return isinstance(df, pd.DataFrame) and not df.empty
        except Exception:
            return False

    if nonempty(primary):
        return primary

    for fn in providers:
        try:
            df = fn()
        except Exception:
            df = None
        if nonempty(df):
            return df
    return primary if primary is not None else pd.DataFrame()

