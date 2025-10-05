#!/usr/bin/env python3
"""
fetch_all.py
-----------
Minimal, hardened external metrics fetcher (free sources) that:
- pulls PBP from nfl_data_py
- normalizes schema across version changes
- creates robust pass/rush flags
- builds team_week_form and team_form
- writes a per-run status summary

Usage:
  python -m scripts.fetch_all --season 2025
"""

import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import nfl_data_py as nfl
except Exception as e:
    nfl = None


# -------------------------
# Utility & hardening helpers
# -------------------------

def _ensure_dirs():
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def _ensure_columns(df: pd.DataFrame, cols_with_defaults: dict) -> pd.DataFrame:
    """Ensure columns exist; if missing, create with defaults."""
    for c, default in cols_with_defaults.items():
        if c not in df.columns:
            df[c] = default
    return df


def _bool_from_playtype(df: pd.DataFrame, playtype_col: str = "play_type") -> pd.DataFrame:
    """
    Create boolean pass_flag / rush_flag robustly across PBP schema variants.
    - Prefer 'play_type' string if present.
    - Fall back to 'pass' / 'rush' numeric columns if available.
    - Else create False defaults.
    """
    if playtype_col in df.columns:
        s = df[playtype_col].astype(str).str.lower()
        df["pass_flag"] = s.eq("pass")
        df["rush_flag"] = s.eq("run")
    else:
        # Old / variant columns
        if "pass" in df.columns:
            df["pass_flag"] = df["pass"].fillna(0).astype(bool)
        else:
            df["pass_flag"] = False

        if "rush" in df.columns:
            df["rush_flag"] = df["rush"].fillna(0).astype(bool)
        else:
            df["rush_flag"] = False

    df["pass_flag"] = df["pass_flag"].fillna(False).astype(bool)
    df["rush_flag"] = df["rush_flag"].fillna(False).astype(bool)
    return df


class RunStatus:
    """Collects per-artifact row counts + optional notes."""
    def __init__(self):
        self.items = []

    def add(self, name: str, rows: int, note: str = ""):
        self.items.append({"name": name, "rows": int(rows), "note": note})

    def write(self, path="metrics/fetch_status.json"):
        try:
            with open(path, "w") as f:
                json.dump(
                    {
                        "when": datetime.utcnow().isoformat() + "Z",
                        "artifacts": self.items,
                    },
                    f,
                    indent=2,
                )
            print(f"↳ Wrote {path}")
        except Exception as e:
            print(f"⚠️ Failed to write {path}: {e}")


# -------------------------
# Builders
# -------------------------

def build_team_week_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """Aggregate team form by week; robust to column differences."""
    df = pbp.copy()

    # Ensure basics; harmless if already present
    df = _ensure_columns(
        df,
        {
            "posteam": "",
            "defteam": "",
            "week": 0,
            "yards_gained": 0.0,
            "play_id": 0,
        },
    )
    df = _bool_from_playtype(df)

    # Keep only valid weeks (defensive)
    if "week" in df.columns:
        df = df[df["week"].notna()]
    else:
        df["week"] = 0

    grp = (
        df.groupby(["posteam", "week"], dropna=False)
        .agg(
            plays=("play_id", "count"),
            pass_plays=("pass_flag", "sum"),
            rush_plays=("rush_flag", "sum"),
            total_yards=("yards_gained", "sum"),
        )
        .reset_index()
    )
    grp["season"] = int(season)

    # Simple rate proxies
    grp["pass_rate"] = np.where(grp["plays"] > 0, grp["pass_plays"] / grp["plays"], 0.0)
    grp["rush_rate"] = np.where(grp["plays"] > 0, grp["rush_plays"] / grp["plays"], 0.0)
    return grp


def build_team_form(team_week: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date rollup per team (useful as a fallback)."""
    if team_week.empty:
        return pd.DataFrame(columns=["posteam", "season", "plays", "pass_rate", "rush_rate"])
    agg = (
        team_week.groupby(["posteam", "season"], dropna=False)
        .agg(
            plays=("plays", "sum"),
            pass_plays=("pass_plays", "sum"),
            rush_plays=("rush_plays", "sum"),
            total_yards=("total_yards", "sum"),
        )
        .reset_index()
    )
    agg["pass_rate"] = np.where(agg["plays"] > 0, agg["pass_plays"] / agg["plays"], 0.0)
    agg["rush_rate"] = np.where(agg["plays"] > 0, agg["rush_plays"] / agg["plays"], 0.0)
    return agg


# -------------------------
# Main flow
# -------------------------

def fetch_pbp(season: int) -> pd.DataFrame:
    """Fetch and normalize PBP for a season from nfl_data_py."""
    if nfl is None:
        print("⚠️ nfl_data_py not available; returning empty PBP")
        return pd.DataFrame()

    print(f"► Downloading play-by-play for season={season} …")
    try:
        pbp = nfl.import_pbp_data([season])  # already returns a DataFrame
    except Exception as e:
        print(f"⚠️ PBP fetch failed: {e}")
        return pd.DataFrame()

    print(f"✓ PBP rows: {len(pbp):,}")

    # Normalize schema across versions
    pbp = _ensure_columns(
        pbp,
        {
            "play_type": "",
            "yards_gained": 0.0,
            "drive_result": "Unknown",  # fixes earlier KeyError on drive_result
            "posteam": "",
            "defteam": "",
            "week": 0,
            "play_id": 0,
        },
    )
    pbp = _bool_from_playtype(pbp)
    return pbp


def main(season: int):
    _ensure_dirs()
    status = RunStatus()

    # 1) PBP
    pbp = fetch_pbp(season)

    # 2) team_week_form
    try:
        team_week = build_team_week_form(pbp, season)
        team_week.to_csv("metrics/team_week_form.csv", index=False)
        print(f"Wrote metrics/team_week_form.csv ({len(team_week)} rows)")
        status.add("team_week_form.csv", len(team_week))
    except Exception as e:
        print(f"⚠️ team_week_form failed: {e}")
        pd.DataFrame().to_csv("metrics/team_week_form.csv", index=False)
        status.add("team_week_form.csv", 0, note=str(e))

    # 3) team_form (season rollup)
    try:
        team_form = build_team_form(team_week if 'team_week' in locals() else pd.DataFrame())
        team_form.to_csv("metrics/team_form.csv", index=False)
        print(f"Wrote metrics/team_form.csv ({len(team_form)} rows)")
        status.add("team_form.csv", len(team_form))
    except Exception as e:
        print(f"⚠️ team_form failed: {e}")
        pd.DataFrame().to_csv("metrics/team_form.csv", index=False)
        status.add("team_form.csv", 0, note=str(e))

    # 4) Placeholder outputs (keep pipeline happy if not wired yet)
    #    If you already build these elsewhere, feel free to remove the placeholders.
    if not os.path.exists("metrics/player_form.csv"):
        pd.DataFrame().to_csv("metrics/player_form.csv", index=False)
        status.add("player_form.csv", 0, note="placeholder")
    if not os.path.exists("inputs/id_map.csv"):
        pd.DataFrame().to_csv("inputs/id_map.csv", index=False)
        status.add("id_map.csv", 0, note="placeholder")
    if not os.path.exists("inputs/weather.csv"):
        pd.DataFrame(columns=["game_id", "temp", "wind", "precip"]).to_csv("inputs/weather.csv", index=False)
        status.add("weather.csv", 0, note="placeholder")

    # 5) Status summary
    status.write("metrics/fetch_status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g., 2025")
    args = parser.parse_args()
    main(args.season)
