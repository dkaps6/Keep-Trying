import os
import pandas as pd
from typing import Tuple, Dict, Optional

# ----- helpers ---------------------------------------------------------------

def _read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read CSV if it exists and is non-empty; else return None."""
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return None

def _safe_lower(x):
    try:
        return str(x).strip().lower()
    except Exception:
        return x

def _norm_team_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        # return a series of Nones to keep merges safe
        return pd.Series([None] * len(df), index=df.index)
    return df[col].astype(str).str.strip().str.upper()

# ----- public API ------------------------------------------------------------

def merge_external_features(
    df: pd.DataFrame,
    *,
    team_col: str = "team",
    opp_col: str = "opp_team",
    week_col: str = "week",
    team_form_csv: str = "metrics/team_form.csv",
    player_form_csv: str = "metrics/player_form.csv",
    id_map_csv: str = "metrics/id_map.csv",    # your fetch writes to metrics/, keep consistent
    weather_csv: str = "inputs/weather.csv",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Merge optional external features into df (props or games).
    Returns (augmented_df, notes). If inputs are missing/empty, it no-ops.
    """

    notes: Dict[str, str] = {}
    out = df.copy()

    # Normalize keys we might need
    out["_team_key_"] = _norm_team_col(out, team_col)
    out["_opp_key_"]  = _norm_team_col(out, opp_col)

    # --- load optional sources
    team_form   = _read_csv(team_form_csv)
    player_form = _read_csv(player_form_csv)
    id_map      = _read_csv(id_map_csv)
    weather     = _read_csv(weather_csv)

    # ---- TEAM FORM (team-level defensive/offensive environment) -------------
    if team_form is not None and not team_form.empty:
        # Expect a 'team' column in team_form (uppercase). If not, try to infer.
        if "team" in team_form.columns:
            tf = team_form.copy()
            tf["team"] = tf["team"].astype(str).str.strip().str.upper()

            # Prefix columns to avoid collisions
            tf_cols = [c for c in tf.columns if c not in ("team",)]
            tf = tf[["team"] + tf_cols]
            tf = tf.rename(columns={c: f"team_{c}" for c in tf_cols})

            # Merge for our team
            out = out.merge(tf, left_on="_team_key_", right_on="team", how="left")
            out = out.drop(columns=["team"], errors="ignore")

            # Also try merging opponent form as separate features (optional)
            tf_opp = team_form.copy()
            tf_opp["team"] = tf_opp["team"].astype(str).str.strip().str.upper()
            tf_opp_cols = [c for c in tf_opp.columns if c not in ("team",)]
            tf_opp = tf_opp[["team"] + tf_opp_cols]
            tf_opp = tf_opp.rename(columns={c: f"opp_{c}" for c in tf_opp_cols})

            out = out.merge(tf_opp, left_on="_opp_key_", right_on="team", how="left")
            out = out.drop(columns=["team"], errors="ignore")

            notes["team_form"] = "merged"
        else:
            notes["team_form"] = "missing 'team' column; skipped"
    else:
        notes["team_form"] = "unavailable; skipped"

    # ---- PLAYER FORM (per-player rolling or weekly features) -----------------
    if player_form is not None and not player_form.empty:
        # Try to find a player key to merge on. We support several common names.
        player_keys = [c for c in ("player", "player_name", "name") if c in out.columns]
        pf_keys     = [c for c in ("player", "player_name", "name") if c in player_form.columns]

        if player_keys and pf_keys:
            left_key  = player_keys[0]
            right_key = pf_keys[0]

            pf = player_form.copy()
            pf[right_key] = pf[right_key].astype(str).str.strip().str.lower()

            out[left_key] = out[left_key].astype(str).str.strip().str.lower()
            # Avoid collisions by prefixing pf columns
            keep_pf_cols = [c for c in pf.columns if c != right_key]
            pf = pf[[right_key] + keep_pf_cols]
            pf = pf.rename(columns={c: f"pf_{c}" for c in keep_pf_cols})
            out = out.merge(pf, left_on=left_key, right_on=right_key, how="left")
            out = out.drop(columns=[right_key], errors="ignore")
            notes["player_form"] = f"merged on {left_key}↔{right_key}"
        else:
            notes["player_form"] = "no common player key; skipped"
    else:
        notes["player_form"] = "unavailable; skipped"

    # ---- ID MAP (optional; helps map names to ids/teams/pos) -----------------
    if id_map is not None and not id_map.empty:
        # Heuristic merge: player name if present; otherwise no-op.
        if "player_name" in id_map.columns:
            idm = id_map.copy()
            idm["player_name"] = idm["player_name"].astype(str).str.strip().str.lower()
            if "player" in out.columns:
                out["player"] = out["player"].astype(str).str.strip().str.lower()
                keep_cols = [c for c in idm.columns if c != "player_name"]
                idm = idm[["player_name"] + keep_cols]
                idm = idm.rename(columns={c: f"idmap_{c}" for c in keep_cols})
                out = out.merge(idm, left_on="player", right_on="player_name", how="left")
                out = out.drop(columns=["player_name"], errors="ignore")
                notes["id_map"] = "merged on player"
            else:
                notes["id_map"] = "no player column; skipped"
        else:
            notes["id_map"] = "missing 'player_name' column; skipped"
    else:
        notes["id_map"] = "unavailable; skipped"

    # ---- WEATHER (optional; merge by team/week or by game identifiers) -------
    if weather is not None and not weather.empty:
        # Try team+week merge if those exist
        can_merge_by_team_week = (
            week_col in out.columns and "team" in weather.columns and "week" in weather.columns
        )
        if can_merge_by_team_week:
            wx = weather.copy()
            wx["team"] = wx["team"].astype(str).str.strip().str.upper()
            wx["week"] = wx["week"]

            # Prefix columns
            keep_cols = [c for c in wx.columns if c not in ("team", "week")]
            wx = wx[["team", "week"] + keep_cols]
            wx = wx.rename(columns={c: f"wx_{c}" for c in keep_cols})

            out = out.merge(
                wx, left_on=[ "_team_key_", week_col ], right_on=[ "team", "week" ], how="left"
            )
            out = out.drop(columns=["team", "week"], errors="ignore")
            notes["weather"] = "merged on team+week"
        else:
            notes["weather"] = "no team+week keys; skipped"
    else:
        notes["weather"] = "unavailable; skipped"

    # Housekeeping
    for col in ("_team_key_", "_opp_key_"):
        if col in out.columns:
            out = out.drop(columns=[col], errors="ignore")

    return out, notes

