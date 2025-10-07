#!/usr/bin/env python3
from __future__ import annotations
import os
import pandas as pd

USER = os.getenv("NFLGSIS_USERNAME", "")
PASS = os.getenv("NFLGSIS_PASSWORD", "")

def injuries(season: int) -> pd.DataFrame | None:
    if not USER or not PASS:
        return None
    # TODO: implement your GSIS client here if your access allows.
    # return a DataFrame with columns: season, player, team, status, detail, update
    return None

