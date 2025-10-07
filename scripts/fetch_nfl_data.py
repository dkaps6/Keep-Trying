# scripts/fetch_nfl_data.py
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import pandas as pd

# Preferred: nflreadpy (nflverse releases)
try:
    import nflreadpy as nrp
    HAS_NFLREADPY = True
except Exception:
    HAS_NFLREADPY = False

# Fallback: nfl_data_py (legacy)
try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False

# ---------------- loaders ----------------

def load_pbp(seasons: list[int] | int) -> pd.DataFrame:
    if isinstance(seasons, int):
        seasons = [seasons]
    # 1) nflreadpy
    if HAS_NFLREADPY:
        try:
            print(f"[fetch_nfl_data] USING PBP (nflreadpy) for {seasons}")
            df = nrp.load_pbp(seasons=seasons, file_type="parquet")
            if "season" not in df.columns:
                df["season"] = seasons[0] if len(seasons) == 1 else pd.NA
            return df
        except Exception as e:
            warnings.warn(f"nflreadpy.load_pbp failed: {type(e).__name__}: {e}")
    # 2) nfl_data_py
    if HAS_NFL_DATA_PY:
        try:
            print(f"[fetch_nfl_data] FALLBACK PBP (nfl_data_py) for {seasons}")
            return nfl.import_pbp_data(seasons)
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_pbp_data failed: {type(e).__name__}: {e}")
    print("[fetch_nfl_data] ❌ PBP load failed; returning empty frame")
    return pd.DataFrame()

def _load_weekly(season: int) -> pd.DataFrame:
    df = pd.DataFrame()
    if HAS_NFLREADPY:
        try:
            print(f"[fetch_nfl_data] USING WEEKLY (nflreadpy) {season}")
            df = nrp.load_player_stats(seasons=[season], stat_type="weekly", file_type="parquet")
        except Exception as e:
            warnings.warn(f"nflreadpy.load_player_stats failed: {type(e).__name__}: {e}")
    if (df is None or df.empty) and HAS_NFL_DATA_PY:
        try:
            print(f"[fetch_nfl_data] FALLBACK WEEKLY (nfl_data_py) {season}")
            df = nfl.import_weekly_data([season])
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_weekly_data failed: {type(e).__name__}: {e}")
    return df if df is not None else pd.DataFrame()

# ---------------- helpers ----------------

def _is_pass(df: pd.DataFrame) -> pd.Series:
    return (df.get("pass", 0) == 1) if "pass" in df.columns else df.get("play_type","").astype(str).str.contains("pass", case=False, na=False)

def _is_rush(df: pd.DataFrame) -> pd.Series:
    return (df.get("rush", 0) == 1) if "rush" in df.columns else df.get("play_type","").astype(str).str.contains("rush", case=False, na=False)

def _neutral_filter(cur: pd.DataFrame) -> pd.Series:
    qtr = cur.get("qtr", pd.Series([0]*len(cur)))
    hs  = cur.get("half_seconds_remaining", pd.Series([0]*len(cur)))
    ytg = cur.get("ydstogo", pd.Series([10]*len(cur)))
    return (qtr.between(1,3)) | ((qtr==4) & (hs>300) & (ytg<=10))

# ---------------- compute team form ----------------

def _team_form_from_weekly(season: int) -> pd.DataFrame:
    df = _load_weekly(season)
    if df is None or df.empty:
        return pd.DataFrame()

    # normalize attempts/carries columns
    if "attempts" not in df.columns and "pass_attempts" in df.columns:
        df = df.rename(columns={"pass_attempts":"attempts"})
    if "carries" not in df.columns and "rush_attempts" in df.columns:
        df = df.rename(columns={"rush_attempts":"carries"})

    wk = (df.groupby(["team","week"], dropna=True)
            .agg(team_pass_att=("attempts","sum"),
                 team_rush_att=("carries","sum"))
            .reset_index())
    if wk.empty:
        return pd.DataFrame()

    wk["team_plays"] = wk["team_pass_att"].fillna(0) + wk["team_rush_att"].fillna(0)
    team = (wk.groupby("team")
                .agg(plays_est=("team_plays","mean"),
                     pass_att=("team_pass_att","sum"),
                     rush_att=("team_rush_att","sum"))
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

def compute_team_form(pbp_all: pd.DataFrame, current_season: int) -> pd.DataFrame:
    cols = [
        "team","opp_team","event_id",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]
    if pbp_all is None or pbp_all.empty:
        wk = _team_form_from_weekly(current_season)
        if not wk.empty:
            wk["opp_team"] = pd.NA; wk["event_id"] = pd.NA
            return wk[cols]
        return pd.DataFrame(columns=cols)

    cur = pbp_all.copy()
    if "season" in cur.columns:
        cur = cur[cur["season"] == current_season].copy()

    if cur.empty:
        wk = _team_form_from_weekly(current_season)
        if not wk.empty:
            wk["opp_team"] = pd.NA; wk["event_id"] = pd.NA
            return wk[cols]
        # neutral rows for all teams present in history:
        teams = sorted(set(pbp_all.get("posteam", pd.Series(dtype=str)).dropna()))
        out = pd.DataFrame({"team": teams})
        for c in cols[3:-3]: out[c] = 0.0
        out["plays_est"] = 60.0; out["proe"] = 0.0; out["rz_rate"] = 0.20
        out["opp_team"] = pd.NA; out["event_id"] = pd.NA
        return out[cols]

    cur["is_pass"] = _is_pass(cur)
    cur["is_rush"] = _is_rush(cur)
    cur["is_rz"]   = (cur.get("yardline_100", 100) <= 20)

    # OFFENSE
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
             .groupby("posteam").size().rename("games")
             ).reset_index().rename(columns={"posteam":"team"})
    off = off.merge(games, on="team", how="left")
    off["plays_est"] = (off["plays"] / off["games"].clip(lower=1)).fillna(off["plays"])

    # DEFENSE (no defteam collision)
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

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="2019-2024")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--write", default="data/team_form.csv")
    args = ap.parse_args()

    Path(args.write).parent.mkdir(parents=True, exist_ok=True)

    hist_years = [int(y) for y in args.history.split("-")]
    if len(hist_years) == 2:
        hist = list(range(hist_years[0], hist_years[1]+1))
    else:
        hist = [int(y) for y in args.history.split(",")]

    pbp_hist = load_pbp(hist)
    pbp_cur  = load_pbp(args.season)
    pbp_all  = pd.concat([pbp_hist, pbp_cur], ignore_index=True, sort=False)

    df = compute_team_form(pbp_all, args.season)
    df.to_csv(args.write, index=False)
    print(f"[fetch_nfl_data] ✅ wrote {len(df)} rows → {args.write}")

if __name__ == "__main__":
    main()
