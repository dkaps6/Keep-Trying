# scripts/fetch_all.py
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


PFR_BASE = "https://www.pro-football-reference.com"


# --------------------
# I/O helpers
# --------------------
def _ensure_dirs() -> Dict[str, Path]:
    metrics = Path("metrics")
    inputs = Path("inputs")
    metrics.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    return {"metrics": metrics, "inputs": inputs}


def _save_csv_silent(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
        print(f"Wrote {path} {len(df)} rows")
    except Exception as e:
        print(f"Failed writing {path}: {e}")


# --------------------
# Web fetch helpers
# --------------------
def _read_html(url: str, table_contains: Optional[str] = None) -> List[pd.DataFrame]:
    """
    Robust HTML -> DataFrame fetcher using PFR.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    print(f"GET {url}")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    # PFR occasionally wraps tables in HTML comments; pandas handles many cases.
    dfs = pd.read_html(r.text)
    if table_contains:
        dfs = [d for d in dfs if any(table_contains.lower() in str(c).lower() for c in d.columns)]
    return dfs


def _sanitize_team_name(s: str) -> str:
    # Normalize common PFR team name quirks
    s = str(s).strip()
    # PFR sometimes has "*" on playoff teams
    s = s.replace("*", "")
    return s


# --------------------
# Team defensive form
# --------------------
@dataclass
class TeamFormConfig:
    season: int


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (s - mu) / sd


def _build_team_form_from_pfr(season: int) -> pd.DataFrame:
    """
    Build season-to-date team defensive proxies from PFR.
    We compute:
      - pressure_rate_proxy   -> z-score (higher is more pressure)
      - pass_ypa_allowed      -> used to create pass_epa_z (inverted z)
      - run_ypc_allowed       -> for funnels
      - pass_epa_z            -> derived from YPA allowed (lower YPA = better defense => higher z)
      - run_funnel/pass_funnel flags (simple thresholds vs league medians)
    """
    # Page with opponent offensive stats by team -> these correspond to defense allowed
    # Example: /years/2024/opp.htm
    url = f"{PFR_BASE}/years/{season}/opp.htm"
    try:
        dfs = _read_html(url)
    except Exception as e:
        print(f"PFR fetch failed for team opponents table: {e}")
        return pd.DataFrame(columns=["team", "week", "pressure_z", "pass_epa_z", "run_funnel", "pass_funnel"])

    # Heuristic: choose the widest table that has team rows
    df = max(dfs, key=lambda d: d.shape[1]) if dfs else pd.DataFrame()
    if df.empty:
        print("PFR returned no usable table for team defense.")
        return pd.DataFrame(columns=["team", "week", "pressure_z", "pass_epa_z", "run_funnel", "pass_funnel"])

    # Try to locate key columns with a few possible header variants
    # Columns we want (from the perspective of defense allowed):
    #   Team, Pass Att, Sacks, QBHits (if present), Sk%, Yds/Att (passing), Yds/A (rushing)
    # PFR column variants:
    def _pick(colnames: List[str]) -> Optional[str]:
        for c in df.columns:
            c_clean = str(c).strip().lower()
            for want in colnames:
                if want in c_clean:
                    return c
        return None

    team_col = _pick(["team"])
    pass_att_col = _pick(["pass att", "att"])  # opponent pass attempts
    sacks_col = _pick(["sk", "sacks"])
    qb_hits_col = _pick(["qb hits"])  # may not exist
    sk_rate_col = _pick(["sk%"])
    pass_ypa_col = _pick(["yds/att", "pass y/a", "yards per pass att"])
    rush_ypc_col = _pick(["yds/att rush", "rush y/a", "yards per rush att", "rush y/a"])

    # Minimal sanity check
    if team_col is None:
        print("Could not find 'Team' column on PFR; returning empty frame.")
        return pd.DataFrame(columns=["team", "week", "pressure_z", "pass_epa_z", "run_funnel", "pass_funnel"])

    # Keep rows that have a valid team name (exclude divisional summaries)
    df = df.copy()
    df[team_col] = df[team_col].apply(_sanitize_team_name)
    df = df[~df[team_col].isna() & (df[team_col] != "League Total")]

    # Convert numerics
    def to_num(col) -> pd.Series:
        if col is None or col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index)
        return pd.to_numeric(df[col], errors="coerce")

    att = to_num(pass_att_col)
    sacks = to_num(sacks_col)
    qb_hits = to_num(qb_hits_col)
    sk_rate_pct = to_num(sk_rate_col)  # usually already a percent
    pass_ypa = to_num(pass_ypa_col)
    rush_ypc = to_num(rush_ypc_col)

    # Pressure proxy:
    #   prefer Sk% if available; else (sacks + qb_hits)/att
    pressure_proxy = sk_rate_pct.copy()
    if pressure_proxy.isna().all():
        denom = att.replace(0, np.nan)
        pressure_proxy = (sacks.add(qb_hits.fillna(0), fill_value=0) / denom) * 100.0

    # Pass EPA proxy (invert YPA allowed: lower YPA allowed => higher “good” z-score)
    inv_pass_ypa = -pass_ypa

    # Build result
    out = pd.DataFrame({
        "team": df[team_col].astype(str).str.upper(),
        # week-less season aggregate; we still keep a 'week' col because engine expects it
        "week": 0,
        "pressure_z": _zscore(pressure_proxy).fillna(0.0),
        "pass_epa_z": _zscore(inv_pass_ypa).fillna(0.0),
        "pass_ypa_allowed": pass_ypa,
        "rush_ypc_allowed": rush_ypc,
    })

    # Simple funnel logic:
    #   team is a PASS funnel if (rush defense is good AND pass defense is weaker)
    #   team is a RUN funnel  if (pass defense is good AND rush defense is weaker)
    # This is intentionally simple; your rules file can use z-scores more richly.
    # Compute median to avoid extremes.
    p_med = np.nanmedian(out["pass_ypa_allowed"])
    r_med = np.nanmedian(out["rush_ypc_allowed"])

    out["pass_funnel"] = ((out["rush_ypc_allowed"] <= r_med) & (out["pass_ypa_allowed"] > p_med)).astype(int)
    out["run_funnel"] = ((out["pass_ypa_allowed"] <= p_med) & (out["rush_ypc_allowed"] > r_med)).astype(int)

    # Keep only the columns engine actually reads
    out = out[["team", "week", "pressure_z", "pass_epa_z", "run_funnel", "pass_funnel"]]
    return out


# --------------------
# Player form (stub)
# --------------------
def _empty_player_form() -> pd.DataFrame:
    """
    We keep a thin player_form.csv footprint so the engine/rules don’t break
    if you decide to extend player usage later.
    """
    return pd.DataFrame(columns=[
        "gsis_id", "week",
        "usage_routes", "usage_targets", "usage_carries",
        "mu_pred", "sigma_pred"
    ])


# --------------------
# ID map + weather (stubs)
# --------------------
def _empty_id_map() -> pd.DataFrame:
    return pd.DataFrame(columns=["player_name", "gsis_id", "recent_team", "position"])


def _optional_weather_shell() -> pd.DataFrame:
    return pd.DataFrame(columns=["game_id", "wind_mph", "temp_f"])


# --------------------
# Orchestrator
# --------------------
def build_all(season: int) -> None:
    dirs = _ensure_dirs()

    # TEAM FORM (PFR)
    try:
        print(f"Building team_form.csv from PFR for season={season} …")
        team_form = _build_team_form_from_pfr(season)
    except Exception as e:
        print(f"Failed building team_form from PFR: {e}")
        team_form = pd.DataFrame(columns=["team", "week", "pressure_z", "pass_epa_z", "run_funnel", "pass_funnel"])

    _save_csv_silent(team_form, dirs["metrics"] / "team_form.csv")

    # PLAYER FORM (stub for now)
    try:
        player_form = _empty_player_form()
    except Exception as e:
        print(f"Failed building player_form: {e}")
        player_form = _empty_player_form()

    _save_csv_silent(player_form, dirs["metrics"] / "player_form.csv")

    # ID MAP (stub)
    try:
        id_map = _empty_id_map()
    except Exception as e:
        print(f"Failed building id_map: {e}")
        id_map = _empty_id_map()
    _save_csv_silent(id_map, dirs["inputs"] / "id_map.csv")

    # WEATHER (optional stub)
    try:
        weather = _optional_weather_shell()
    except Exception as e:
        print(f"Failed building weather shell: {e}")
        weather = _optional_weather_shell()
    _save_csv_silent(weather, dirs["inputs"] / "weather.csv")


# --------------------
# CLI
# --------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch external metrics (PFR fallback).")
    parser.add_argument("--season", type=int, required=True, help="Season year, e.g., 2025")
    args = parser.parse_args(argv)

    try:
        build_all(args.season)
        return 0
    except Exception as e:
        print(f"Fatal error in fetch_all: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
