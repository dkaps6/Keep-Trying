# scripts/engine_helpers.py
from __future__ import annotations

import pandas as pd
import numpy as np


def _pick_team_col(df: pd.DataFrame) -> str | None:
    """
    Return the best-available team column name in a player_form frame.
    Our schema prefers 'recent_team', but we also accept 'team'.
    """
    for c in ("recent_team", "team"):
        if c in df.columns:
            return c
    return None


def make_team_last4_from_player_form(player_form: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player_form weekly rolling stats to team-level last-4.
    Works even if player_form is empty or lacks 'team' (uses 'recent_team').
    Returns a DataFrame with one row per team and safe defaults.

    Expected player_form columns (best-effort):
      gsis_id, week, player_name, recent_team|team, position,
      rec_l4, rec_yds_l4, ra_l4, ry_l4, pass_att_l4, pass_yds_l4, rz_tgt_share_l4
    """
    out_cols = [
        "team", "rec_l4", "rec_yds_l4", "ra_l4", "ry_l4",
        "pass_att_l4", "pass_yds_l4", "rz_tgt_share_l4"
    ]

    if player_form is None or player_form.empty:
        # Return an empty but schema-correct frame
        return pd.DataFrame(columns=out_cols)

    team_col = _pick_team_col(player_form)
    if team_col is None:
        # can't aggregate without any team column
        return pd.DataFrame(columns=out_cols)

    # Ensure numeric columns exist (fill with 0 if missing)
    needed = ["rec_l4", "rec_yds_l4", "ra_l4", "ry_l4",
              "pass_att_l4", "pass_yds_l4", "rz_tgt_share_l4"]
    pf = player_form.copy()
    for c in needed:
        if c not in pf.columns:
            pf[c] = 0.0

    pf = pf.rename(columns={team_col: "team"})

    # Our last-4 fields are already rolling per player; take a per-team mean for stability
    grp = pf.groupby("team", as_index=False)[needed].mean(numeric_only=True)

    # Ensure all columns are present / ordered
    grp = grp.reindex(columns=out_cols, fill_value=0.0)
    return grp
