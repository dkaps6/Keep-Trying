# scripts/fetch_all.py
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

# ---- optional imports; we don't fail if missing ----
try:
    import nfl_data_py as nfl
except Exception:  # pragma: no cover
    nfl = None


def _ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


# =============================================================================
# TEAM FORM (free, from nfl_data_py pbp)
# =============================================================================
def build_team_form(season: int, weeks: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Builds a lightweight team_form from PBP aggregates.
    Fields: team, pressure_rate, pressure_z, pass_epa_allowed, pass_epa_z, proe_proxy
    """
    print(f"▶ Building team_form.csv ...")
    if nfl is None:
        print("⚠️  nfl_data_py not available; returning empty team_form")
        return pd.DataFrame(columns=[
            "team", "pressure_rate", "pressure_z", "pass_epa_allowed", "pass_epa_z", "proe_proxy"
        ])

    if weeks is None:
        weeks = list(range(1, 19))  # regular season weeks

    try:
        print(f"Downloading pbp for season={season}, weeks={weeks} ...")
        pbp = nfl.import_pbp_data([season], downcast=True, columns=None, cache=False, alt_path=None, weeks=weeks)
        print(f"{season} done.")
    except Exception as e:
        print(f"⚠️  import_pbp_data failed: {e}")
        return pd.DataFrame(columns=[
            "team", "pressure_rate", "pressure_z", "pass_epa_allowed", "pass_epa_z", "proe_proxy"
        ])

    # Defensive pressure proxy (simple)
    def_pressure = (pbp
        .query("pass==1 or qb_hit==1 or sack==1", engine="python")
        .groupby("defteam", as_index=False)[["qb_hit", "sack"]]
        .sum(numeric_only=True))
    def_pressure["dropbacks_approx"] = def_pressure["qb_hit"].fillna(0) + def_pressure["sack"].fillna(0)
    def_pressure["pressure_rate"] = (def_pressure["dropbacks_approx"] /
                                     def_pressure["dropbacks_approx"].replace(0, np.nan))
    def_pressure["pressure_rate"] = def_pressure["pressure_rate"].fillna(0.0)

    # Defensive pass EPA allowed
    try:
        epa_def = (pbp
                   .query("pass==1", engine="python")
                   .groupby("defteam", as_index=False)[["epa"]]
                   .mean(numeric_only=True)
                   .rename(columns={"epa": "pass_epa_allowed"}))
    except Exception:
        epa_def = pd.DataFrame({"defteam": [], "pass_epa_allowed": []})

    tf = pd.merge(def_pressure[["defteam", "pressure_rate"]],
                  epa_def, on="defteam", how="outer").rename(columns={"defteam": "team"})
    if tf.empty:
        return pd.DataFrame(columns=[
            "team", "pressure_rate", "pressure_z", "pass_epa_allowed", "pass_epa_z", "proe_proxy"
        ])

    # Standardize with z-scores
    tf["pressure_z"] = (tf["pressure_rate"] - tf["pressure_rate"].mean()) / (tf["pressure_rate"].std(ddof=0) or 1)
    tf["pass_epa_z"] = (tf["pass_epa_allowed"] - tf["pass_epa_allowed"].mean()) / (tf["pass_epa_allowed"].std(ddof=0) or 1)

    # PROE proxy (very rough; positive if pass EPA is strong and pressure low)
    tf["proe_proxy"] = -0.5 * tf["pass_epa_z"].fillna(0) + 0.25 * (-tf["pressure_z"].fillna(0))

    tf = _downcast(tf)
    return tf


# =============================================================================
# PLAYER FORM (providers chain with graceful fallbacks)
# =============================================================================
def _try_nflverse_weekly(season: int) -> pd.DataFrame:
    try:
        from providers import nflverse as prov
    except Exception:
        return pd.DataFrame()
    try:
        print("▶ Trying weekly provider: nflverse")
        return prov.weekly_player_form(season)
    except Exception as e:
        print(f"⚠️  nflverse weekly failed for {season}: {e}")
        return pd.DataFrame()


def _try_espn_weekly(season: int) -> pd.DataFrame:
    """ESPN via SportsDataverse; only if package importable."""
    try:
        import sportsdataverse  # noqa
    except Exception:
        print("ℹ️  sportsdataverse not installed/available; ESPN provider returns empty.")
        return pd.DataFrame()
    try:
        from providers import espn as prov
    except Exception:
        print("ℹ️  providers.espn module not present.")
        return pd.DataFrame()

    try:
        print("▶ Trying weekly provider: espn")
        return prov.weekly_player_form(season)
    except Exception as e:
        print(f"⚠️  espn weekly failed: {e}")
        return pd.DataFrame()


def _try_pfr_manual_weekly(season: int) -> pd.DataFrame:
    try:
        from providers import pfr_manual as prov
    except Exception:
        return pd.DataFrame()
    try:
        print("▶ Trying weekly provider: pfr_manual")
        return prov.weekly_player_form(season)
    except Exception as e:
        print(f"ℹ️  pfr_manual returned empty: {e}")
        return pd.DataFrame()


def build_player_form(season: int) -> pd.DataFrame:
    """Try providers in order: nflverse → espn → pfr_manual."""
    for fn in (_try_nflverse_weekly, _try_espn_weekly, _try_pfr_manual_weekly):
        df = fn(season)
        if df is not None and not df.empty:
            return _downcast(df)

    print("⚠️  All weekly providers returned empty.")
    return pd.DataFrame(columns=[
        "player_name", "gsis_id", "recent_team", "position",
        # add expected columns used by engine if any (filled later with 0s)
        "rec_l4", "rec_yds_l4", "ra_l4", "ry_l4", "pass_att_l4",
        "pass_yds_l4", "rz_tgt_share_l4"
    ])


# =============================================================================
# ID MAP (robust across nfl_data_py versions)
# =============================================================================
def build_id_map(season: int) -> pd.DataFrame:
    """
    Basic player id map from rosters; resilient to nfl_data_py versions.

    Tries:
      1) nfl.import_rosters([season])  (newer nfl_data_py)
      2) nfl.import_players()          (fallback; filter to active/nearby seasons)

    Returns: player_name, gsis_id, recent_team, position
    """
    print("▶ Building id_map.csv ...")
    if nfl is None:
        print("ℹ️  nfl_data_py not available; id_map will be empty.")
        return pd.DataFrame(columns=["player_name", "gsis_id", "recent_team", "position"])

    # Try modern rosters endpoint first
    roster_fn = getattr(nfl, "import_rosters", None)
    ros = pd.DataFrame()

    if callable(roster_fn):
        try:
            ros = roster_fn([season])
        except Exception as e:
            print(f"⚠️  import_rosters failed for {season}: {e}")

    # Fallback: import_players (wide table over many seasons)
    if ros is None or ros.empty:
        try:
            players = nfl.import_players()
        except Exception as e:
            print(f"⚠️  import_players failed: {e}")
            players = pd.DataFrame()

        if not players.empty:
            keep_cols = {
                "player_name": "player_name",
                "full_name": "player_name",
                "gsis_id": "gsis_id",
                "gsisid": "gsis_id",
                "team": "recent_team",
                "recent_team": "recent_team",
                "position": "position",
                "pos": "position",
                "last_season": "last_season",
                "first_season": "first_season",
            }
            m = {c: keep_cols[c] for c in players.columns if c in keep_cols}
            pl = players.rename(columns=m)

            # Filter to plausible actives around target season
            if "last_season" in pl.columns:
                pl = pl[pl["last_season"].fillna(0) >= (season - 2)]
            if "first_season" in pl.columns:
                pl = pl[pl["first_season"].fillna(season) <= season]

            for c in ["player_name", "gsis_id", "recent_team", "position"]:
                if c not in pl.columns:
                    pl[c] = None

            ros = (pl[["player_name", "gsis_id", "recent_team", "position"]]
                   .dropna(subset=["player_name"])
                   .drop_duplicates())

    if ros is None or ros.empty:
        print("ℹ️  id_map fallback produced empty output.")
        return pd.DataFrame(columns=["player_name", "gsis_id", "recent_team", "position"]).reset_index(drop=True)

    return (ros[["player_name", "gsis_id", "recent_team", "position"]]
            .drop_duplicates()
            .reset_index(drop=True))


# =============================================================================
# WEATHER (optional)
# =============================================================================
def build_weather(season: int) -> pd.DataFrame:
    try:
        from scripts.fetch_weather import build_weather_csv
        return build_weather_csv(season)  # writes file inside itself normally
    except Exception as e:
        print(f"ℹ️  weather fetch skipped: {e}")
        return pd.DataFrame(columns=["game_id", "wind", "temp", "precip"])


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    _ensure_dir("metrics")
    _ensure_dir("inputs")

    # TEAM FORM
    team_form = build_team_form(args.season)
    team_form = _downcast(team_form)
    team_form.to_csv("metrics/team_form.csv", index=False)
    print(f"✔ Wrote metrics/team_form.csv {len(team_form)} rows")

    # PLAYER FORM
    player_form = build_player_form(args.season)
    player_form = _downcast(player_form)
    player_form.to_csv("metrics/player_form.csv", index=False)
    print(f"✔ Wrote metrics/player_form.csv {len(player_form)} rows")

    # ID MAP
    id_map = build_id_map(args.season)
    id_map = _downcast(id_map)
    id_map.to_csv("metrics/id_map.csv", index=False)
    print(f"✔ Wrote metrics/id_map.csv {len(id_map)} rows")

    # WEATHER (optional)
    weather = build_weather(args.season)
    if weather is not None and not weather.empty:
        weather.to_csv("inputs/weather.csv", index=False)
        print(f"✔ Wrote inputs/weather.csv {len(weather)} rows")
    else:
        # write a tiny stub to keep downstream merges happy
        pd.DataFrame(columns=["game_id", "wind", "temp", "precip"]).to_csv("inputs/weather.csv", index=False)
        print("✔ Wrote inputs/weather.csv 0 rows (stub)")


if __name__ == "__main__":
    main()
