# scripts/fetch_nfl_data.py
# ---------------------------------------------------------------------
# Builds:
#   - team_form.csv    (pressure, pass EPA allowed, PROE proxy)
#   - player_form.csv  (rolling last-4 form via provider chain)
#   - id_map.csv       (player_name -> gsis_id / team / position)
#
# Weekly provider chain for player_form (in order):
#   1) nflverse (nfl_data_py)
#   2) ESPN via sportsdataverse (if installed)
#   3) Manual PFR CSV in inputs/pfr_player_weekly_{season}.csv
#
# All builders are resilient: on failure they return a schema-correct
# (possibly empty) DataFrame so the pipeline never crashes.
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd

# Primary public source
import nfl_data_py as nfl

# ----- Provider chain (weekly) ------------------------------------------------
# These modules live in scripts/providers/*
from scripts.providers.base import empty_weekly_df, coerce_weekly_schema
from scripts.providers.nflverse_provider import NFLVerseProvider
from scripts.providers.espn_provider import ESPNProvider
from scripts.providers.pfr_provider import PFRProvider


# ===== Utils ==================================================================

def _parse_weeks(weeks: str) -> List[int]:
    """
    Turn "1-18" or "all" or "7" into a list of ints.
    """
    if isinstance(weeks, str):
        ws = weeks.strip().lower()
        if ws == "all":
            return list(range(1, 19))
        if "-" in ws:
            a, b = ws.split("-", 1)
            return list(range(int(a), int(b) + 1))
        return [int(ws)]
    if isinstance(weeks, (list, tuple)):
        return list(map(int, weeks))
    return [int(weeks)]


def _zscore(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd


# ===== Team Form ==============================================================

def build_team_form(season: int, weeks: str = "1-18") -> pd.DataFrame:
    """
    Build team-level matchup features:
      - Defensive pressure rate + Z
      - Defensive pass EPA allowed + Z
      - Neutral pass rate proxy (PROE-ish) vs league

    Uses nfl_data_py play-by-play. If a specific column is missing on a
    given season mirror, we fall back gracefully.
    """
    wks = _parse_weeks(weeks)

    # Pull PBP once
    print(f"Downloading pbp for season={season}, weeks={wks} ...")
    try:
        pbp = nfl.import_pbp_data([season])
    except Exception as e:
        print(f"⚠️  Failed to download pbp for {season}: {e}")
        return pd.DataFrame(columns=[
            "team", "pressure_rate", "pressure_z",
            "pass_epa_allowed", "pass_epa_z",
            "proe_proxy"
        ])

    if pbp is None or pbp.empty:
        print("⚠️  pbp empty; returning empty team_form.")
        return pd.DataFrame(columns=[
            "team", "pressure_rate", "pressure_z",
            "pass_epa_allowed", "pass_epa_z",
            "proe_proxy"
        ])

    # Filter to requested weeks if column exists
    if "week" in pbp.columns:
        pbp = pbp[pbp["week"].isin(wks)].copy()

    # ---- Defensive pressure rate
    # dropbacks ≈ pass_attempt + sacks + scrambles
    # pressure ≈ sacks + qb_hit (if qb_hit exists), else just sacks
    pa = pd.to_numeric(pbp.get("pass_attempt"), errors="coerce").fillna(0)
    sacks = pd.to_numeric(pbp.get("sack"), errors="coerce").fillna(0)
    scrambles = pd.to_numeric(pbp.get("qb_scramble"), errors="coerce").fillna(0)
    qb_hit = pd.to_numeric(pbp.get("qb_hit"), errors="coerce").fillna(0)

    dropbacks = pa + sacks + scrambles
    pressures = sacks + (qb_hit if "qb_hit" in pbp.columns else 0)

    defteam = pbp.get("defteam") if "defteam" in pbp.columns else pbp.get("defense_team")
    if defteam is None:
        # Absolute fallback: try offense team and just keep shape
        defteam = pbp.get("defteam_abbr", pd.Series(["UNK"] * len(pbp)))

    df_press = pd.DataFrame({
        "defteam": defteam,
        "pressures": pressures,
        "dropbacks": dropbacks
    })
    grp = df_press.groupby("defteam", as_index=False).sum(numeric_only=True)
    grp["pressure_rate"] = (grp["pressures"] / grp["dropbacks"].replace(0, np.nan)).fillna(0.0)

    # ---- Pass EPA allowed (defense)
    epa = pd.to_numeric(pbp.get("epa"), errors="coerce")
    is_pass = pd.to_numeric(pbp.get("pass"), errors="coerce").fillna(0).astype(int)
    if "defteam" not in pbp.columns:
        pbp["defteam"] = defteam

    pass_plays = pbp[(is_pass == 1) & epa.notna()]
    df_epa = pass_plays.groupby("defteam", as_index=False)["epa"].mean()
    df_epa = df_epa.rename(columns={"epa": "pass_epa_allowed"})

    # ---- Neutral pass rate proxy (PROE-ish)
    # Approximation: downs 1-2, win_prob between 0.2-0.8, yardline 20-80
    neutral = pbp.copy()
    if "down" in neutral.columns:
        neutral = neutral[neutral["down"].isin([1, 2])]
    if "wp" in neutral.columns:
        neutral = neutral[(neutral["wp"] >= 0.2) & (neutral["wp"] <= 0.8)]
    if "yardline_100" in neutral.columns:
        neutral = neutral[(neutral["yardline_100"] >= 20) & (neutral["yardline_100"] <= 80)]

    # Offense team column name differs by season mirror
    off = None
    for cand in ["posteam", "pos_team", "offense_team"]:
        if cand in neutral.columns:
            off = cand
            break
    if off is None:
        off = "posteam"
        neutral[off] = "UNK"

    neutral["is_pass"] = pd.to_numeric(neutral.get("pass"), errors="coerce").fillna(0).astype(int)
    team_pass = neutral.groupby(off, as_index=False)["is_pass"].mean()
    team_pass = team_pass.rename(columns={off: "team", "is_pass": "neutral_pass_rate"})
    league_neutral = float(team_pass["neutral_pass_rate"].mean()) if not team_pass.empty else 0.5
    team_pass["proe_proxy"] = team_pass["neutral_pass_rate"] - league_neutral
    team_pass = team_pass[["team", "proe_proxy"]]

    # ---- Merge all
    out = grp.merge(df_epa, how="left", left_on="defteam", right_on="defteam")
    out = out.rename(columns={"defteam": "team"})
    out = out.merge(team_pass, how="left", on="team")

    # Fill and z-scores
    out["pass_epa_allowed"] = out["pass_epa_allowed"].fillna(out["pass_epa_allowed"].mean())
    out["proe_proxy"] = out["proe_proxy"].fillna(0.0)

    out["pressure_z"] = _zscore(out["pressure_rate"])
    out["pass_epa_z"] = _zscore(out["pass_epa_allowed"])

    cols = ["team", "pressure_rate", "pressure_z",
            "pass_epa_allowed", "pass_epa_z", "proe_proxy"]
    out = out[cols].sort_values("team").reset_index(drop=True)
    return out


# ===== Player Form (weekly) ===================================================

def build_player_form(season: int, weeks: str = "1-18") -> pd.DataFrame:
    """
    Try weekly providers in order:
      1) nflverse (nfl_data_py)
      2) ESPN (sportsdataverse)
      3) PFR manual CSV (inputs/pfr_player_weekly_{season}.csv)

    Returns a schema-correct DataFrame (possibly empty) filtered to requested weeks.
    """
    providers = [NFLVerseProvider(), ESPNProvider(), PFRProvider()]
    for prov in providers:
        print(f"▶ Trying weekly provider: {prov.name}")
        df = prov.fetch_weekly(season)
        if not df.empty:
            print(f"  ✓ {prov.name}: {len(df)} rows")
            wks = _parse_weeks(weeks)
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            df = df[df["week"].isin(wks)].copy()
            return coerce_weekly_schema(df)
        else:
            print(f"  … {prov.name} returned empty; trying next.")
    print("⚠️  All weekly providers returned empty.")
    return empty_weekly_df()


# ===== ID Map =================================================================

def build_id_map(season: int) -> pd.DataFrame:
    """
    Basic player id map from rosters:
      player_name, gsis_id, recent_team, position
    """
    try:
        ros = nfl.import_rosters([season])
    except Exception as e:
        print(f"⚠️  import_rosters failed for {season}: {e}")
        return pd.DataFrame(columns=["player_name", "gsis_id", "recent_team", "position"])

    if ros is None or ros.empty:
        return pd.DataFrame(columns=["player_name", "gsis_id", "recent_team", "position"])

    # Normalize column names across seasons
    rename = {
        "player_name": "player_name",
        "team": "recent_team",
        "recent_team": "recent_team",
        "position": "position",
        "gsis_id": "gsis_id",
        "gsisid": "gsis_id",
        "player_id": "gsis_id",
    }
    cols = {c: rename.get(c, c) for c in ros.columns}
    ros = ros.rename(columns=cols)

    keep = ["player_name", "gsis_id", "recent_team", "position"]
    for c in keep:
        if c not in ros.columns:
            ros[c] = None

    out = ros[keep].drop_duplicates().dropna(subset=["player_name"])
    return out.reset_index(drop=True)
