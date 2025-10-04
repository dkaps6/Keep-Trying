from __future__ import annotations
import pandas as pd
from typing import Protocol, Optional

WEEKLY_SCHEMA = [
    "gsis_id","week","player_name","recent_team","position",
    "rec_l4","rec_yds_l4","ra_l4","ry_l4","pass_att_l4","pass_yds_l4","rz_tgt_share_l4"
]

class WeeklyProvider(Protocol):
    name: str
    def fetch_weekly(self, season: int) -> pd.DataFrame: ...

def empty_weekly_df() -> pd.DataFrame:
    return pd.DataFrame(columns=WEEKLY_SCHEMA)

def coerce_weekly_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Keep known columns; missing ones will be added downstream
    keep = [c for c in WEEKLY_SCHEMA if c in df.columns]
    if not keep:
        return empty_weekly_df()
    out = df[keep].copy()
    for c in WEEKLY_SCHEMA:
        if c not in out.columns:
            out[c] = None
    return out[WEEKLY_SCHEMA]
