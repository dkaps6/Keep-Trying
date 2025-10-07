# scripts/fetch_nfl_data.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd

# Primaries (when available)
try:
    import nflreadpy as nrp
    HAS_NFLREADPY = True
except Exception:
    HAS_NFLREADPY = False
try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False

# ESPN fallback (lightweight)
from .build_player_form import _is_pass, _is_rush, _neutral_filter  # reuse helpers
from .build_player_form import _espn_players_table  # used for player shares if needed

# New sources
from .sources.apisports import season_team_player_tables as apisports_tables
from .sources.mysportsfeeds import season_team_player_tables as msf_tables

# -------------- nflverse loaders --------------
def _load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    seasons = list(seasons)
    # nflreadpy first
    if HAS_NFLREADPY:
        try:
            print(f"[fetch_nfl_data] USING PBP (nflreadpy) for {seasons}")
            df = nrp.load_pbp(seasons=seasons)
            if "season" not in df.columns and len(seasons) == 1:
                df["season"] = seasons[0]
            return df
        except Exception as e:
            warnings.warn(f"nflreadpy.load_pbp failed: {type(e).__name__}: {e}")
    # nfl_data_py fallback
    if HAS_NFL_DATA_PY:
        try:
            print(f"[fetch_nfl_data] FALLBACK PBP (nfl_data_py) for {seasons}")
            return nfl.import_pbp_data(seasons)
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_pbp_data failed: {type(e).__name__}: {e}")
    return pd.DataFrame()

# -------------- ESPN/alt team tables --------------
def _team_form_from_tables(team_df: pd.DataFrame) -> pd.DataFrame:
    if team_df is None or team_df.empty:
        return pd.DataFrame()
    team = (team_df.groupby("team")
            .agg(plays_est=("plays","mean"),
                 pass_att=("pass_att","sum"),
                 rush_att=("rush_att","sum"))
            .reset_index())
    team["pass_rate"] = team["pass_att"] / (team["pass_att"] + team["rush_att"]).replace({0: pd.NA})
    league_pass = float(team["pass_rate"].mean(skipna=True))
    team["proe"] = (team["pass_rate"] - league_pass).fillna(0.0)
    team["rz_rate"] = 0.20
    out = team[["team","plays_est","proe","rz_rate"]].copy()
    for c in ["def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z","pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z"]:
        out[c] = 0.0
    return out[[
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]]

def _espn_team_table(season: int) -> pd.DataFrame:
    # reuse ESPN player scrape to build team table if needed
    from .build_player_form import _espn_week_events, _espn_box_players, _parse_box_to_rows
    all_team = []
    for wk in range(1, 23):
        evts = _espn_week_events(season, wk)
        if not evts and wk > 3:
            break
        for eid in evts:
            tdf, _ = _parse_box_to_rows(eid, _espn_box_players(eid))
            if not tdf.empty:
                all_team.append(tdf)
    return pd.concat(all_team, ignore_index=True) if all_team else pd.DataFrame()

# -------------- main compute --------------
def compute_team_form(pbp_all: pd.DataFrame, current_season: int) -> pd.DataFrame:
    cols = [
        "team","opp_team","event_id",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]

    # If we have PBP for the current season, use it (best fidelity)
    cur = pd.DataFrame()
    if pbp_all is not None and not pbp_all.empty:
        cur = pbp_all.copy()
        if "season" in cur.columns:
            cur = cur[cur["season"] == current_season].copy()

    if not cur.empty:
        cur["is_pass"] = _is_pass(cur)
        cur["is_rush"] = _is_rush(cur)
        cur["is_rz"]   = (cur.get("yardline_100", 100) <= 20)

        off = (cur.groupby("posteam", as_index=False)
                  .agg(plays=("posteam","size"),
                       pass_plays=("is_pass","sum"),
                       rush_plays=("is_rush","sum"),
                       rz_rate=("is_rz","mean"))
                  .rename(columns={"posteam":"team"}))

        neutral_mask = _neutral_filter(cur)
        league_neutral = float(cur.loc[neutral_mask, "is_pass"].mean()) if neutral_mask.any() else 0.56
        team_neutral = (cur.loc[neutral_mask]
                          .groupby("posteam")["is_pass"].mean()
                          .rename("neutral_pass_rate")).reset_index().rename(columns={"posteam":"team"})
        off = off.merge(team_neutral, on="team", how="left")
        off["neutral_pass_rate"] = off["neutral_pass_rate"].fillna(league_neutral)
        off["proe"] = off["neutral_pass_rate"] - league_neutral

        d = cur.sort_values(["game_id","game_seconds_remaining"]).copy()
        d["dt"] = d.groupby("game_id")["game_seconds_remaining"].diff(-1).abs()
        pace = d.groupby("posteam")["dt"].median().rename("pace_sec_play").reset_index().rename(columns={"posteam":"team"})
        off = off.merge(pace, on="team", how="left")

        games = (cur[["game_id","posteam"]].drop_duplicates()
                    .groupby("posteam").size().rename("games")).reset_index().rename(columns={"posteam":"team"})
        off = off.merge(games, on="team", how="left")
        off["plays_est"] = (off["plays"] / off["games"].clip(lower=1)).fillna(off["plays"])

        g_pass = (cur[cur["is_pass"]].groupby("defteam", as_index=False)["epa"].mean().rename(columns={"epa":"def_pass_epa"}))
        g_rush = (cur[cur["is_rush"]].groupby("defteam", as_index=False)["epa"].mean().rename(columns={"epa":"def_rush_epa"}))
        g_cnt  = (cur.groupby("defteam", as_index=False).size().rename(columns={"size":"def_plays"}))
        opp = g_cnt.merge(g_pass, on="defteam", how="left").merge(g_rush, on="defteam", how="left").rename(columns={"defteam":"team"})
        opp["light_box_rate"] = 0.0; opp["heavy_box_rate"] = 0.0; opp["def_sack_rate"] = 0.0

        team = off.merge(opp, on="team", how="outer")

        def _z(name):
            s = team[name].astype(float)
            mu = s.mean(skipna=True); sd = s.std(ddof=0, skipna=True)
            if sd == 0 or pd.isna(sd): return s*0.0
            return (s - mu) / sd

        team["def_pass_epa_z"]    = _z("def_pass_epa")
        team["def_rush_epa_z"]    = _z("def_rush_epa")
        team["def_sack_rate_z"]   = _z("def_sack_rate")
        team["def_pressure_rate_z"]= 0.0
        team["pace_z"]            = _z("pace_sec_play")
        team["light_box_rate_z"]  = _z("light_box_rate")
        team["heavy_box_rate_z"]  = _z("heavy_box_rate")
        team["ay_per_att_z"]      = 0.0

        out = team[[
            "team",
            "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate"
        ]].copy()
        for c in ["plays_est","proe","rz_rate"]:
            out[c] = out[c].fillna(0.0)
        out["opp_team"] = pd.NA; out["event_id"] = pd.NA
        return out

    # No PBP: fallbacks ESPN -> API-SPORTS -> MSF
    espn_team = _espn_team_table(current_season)
    if not espn_team.empty:
        return _team_form_from_tables(espn_team)

    api_team, _ = apisports_tables(current_season)
    if not api_team.empty:
        return _team_form_from_tables(api_team)

    msf_team, _ = msf_tables(current_season)
    if not msf_team.empty:
        return _team_form_from_tables(msf_team)

    # last resort neutral
    return pd.DataFrame(columns=cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="2019-2024")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--write", default="data/team_form.csv")
    args = ap.parse_args()

    Path(args.write).parent.mkdir(parents=True, exist_ok=True)
    if "-" in args.history:
        lo, hi = [int(x) for x in args.history.split("-")]
        hist = list(range(lo, hi+1))
    else:
        hist = [int(x) for x in args.history.split(",")]

    pbp_hist = _load_pbp(hist)
    pbp_cur  = _load_pbp([args.season])
    pbp_all  = pd.concat([pbp_hist, pbp_cur], ignore_index=True, sort=False)

    df = compute_team_form(pbp_all, args.season)
    df.to_csv(args.write, index=False)
    print(f"[fetch_nfl_data] ✅ wrote {len(df)} rows → {args.write}")

if __name__ == "__main__":
    main()
