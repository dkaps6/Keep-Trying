# === BEGIN scripts/fetch_nfl_data.py (patched full) ===
from __future__ import annotations
import argparse, warnings, time, requests
from pathlib import Path
from typing import Iterable
import pandas as pd

# ---- Primary (nflverse) sources ----
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

# ---- ESPN light fallback for team totals ----
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
ESPN_BOX = "https://site.api.espn.com/apis/v2/sports/football/nfl/boxscore"

def _http_json(url: str, params: dict | None = None, tries: int = 3, backoff: float = 0.7):
    for i in range(tries):
        try:
            r = requests.get(url, params=params or {}, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(backoff * (2**i))
    return None

def _espn_week_events(season: int, week: int) -> list[str]:
    data = _http_json(ESPN_SCOREBOARD, params={"week": week, "seasontype": 2, "dates": season})
    if not data:
        return []
    return [e.get("id") for e in data.get("events", []) if e.get("id")]

def _espn_box(event_id: str):
    return _http_json(ESPN_BOX, params={"event": event_id})

def _espn_team_table(season: int) -> pd.DataFrame:
    rows = []
    for wk in range(1, 23):
        evts = _espn_week_events(season, wk)
        if not evts and wk > 3:
            break
        for eid in evts:
            box = _espn_box(eid) or {}
            for t in box.get("teams", []):
                abbr = (t.get("team", {}) or {}).get("abbreviation") or (t.get("team", {}) or {}).get("displayName")
                pass_att = 0.0; rush_att = 0.0
                for grp in t.get("players", []):
                    for pl in grp.get("athletes", []):
                        for s in pl.get("stats", []):
                            nm = (s.get("name") or "").lower()
                            st = s.get("statistics") or {}
                            if nm == "passing":
                                pass_att += float(st.get("attempts") or 0)
                            elif nm == "rushing":
                                rush_att += float(st.get("attempts") or st.get("carries") or 0)
                rows.append({"team": abbr, "pass_att": pass_att, "rush_att": rush_att, "plays": pass_att + rush_att, "event_id": eid})
            time.sleep(0.05)
    return pd.DataFrame(rows)

# ---- API-SPORTS & MySportsFeeds fallbacks ----
from .sources.apisports import season_team_player_tables as apisports_tables
from .sources.mysportsfeeds import season_team_player_tables as msf_tables

# --- NFLGSIS import shim (works in GitHub Actions and local) ---
import sys, os
from pathlib import Path as _PathShim
_REPO_ROOT = _PathShim(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "sources"))
try:
    from scripts.sources.nflgsis import (
        login_session, list_games, team_player_tables as gsis_team_player_tables
    )
except Exception:
    try:
        from nflgsis import (  # type: ignore
            login_session, list_games, team_player_tables as gsis_team_player_tables
        )
    except Exception:
        login_session = list_games = gsis_team_player_tables = None
# --- end NFLGSIS shim ---

# ---------- Local helpers ----------
def _is_pass(df: pd.DataFrame) -> pd.Series:
    return (df.get("pass", 0) == 1) if "pass" in df.columns \
        else df.get("play_type","").astype(str).str.contains("pass", case=False, na=False)

def _is_rush(df: pd.DataFrame) -> pd.Series:
    return (df.get("rush", 0) == 1) if "rush" in df.columns \
        else df.get("play_type","").astype(str).str.contains("rush|run", case=False, na=False)

def _neutral_filter(cur: pd.DataFrame) -> pd.Series:
    qtr = cur.get("qtr", pd.Series([0]*len(cur)))
    hs  = cur.get("half_seconds_remaining", pd.Series([0]*len(cur)))
    ytg = cur.get("ydstogo", pd.Series([10]*len(cur)))
    return (qtr.between(1,3)) | ((qtr==4) & (hs>300) & (ytg<=10))

# ---------- nflverse loaders ----------
def _load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    seasons = list(seasons)
    if HAS_NFLREADPY:
        try:
            print(f"[fetch_nfl_data] USING PBP (nflreadpy) for {seasons}")
            df = nrp.load_pbp(seasons=seasons)
            if "season" not in df.columns and len(seasons) == 1:
                df["season"] = seasons[0]
            return df
        except Exception as e:
            warnings.warn(f"nflreadpy.load_pbp failed: {type(e).__name__}: {e}")
    if HAS_NFL_DATA_PY:
        try:
            print(f"[fetch_nfl_data] FALLBACK PBP (nfl_data_py) for {seasons}")
            return nfl.import_pbp_data(seasons)
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_pbp_data failed: {type(e).__name__}: {e}")
    return pd.DataFrame()

# ---------- team form builders ----------
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
    team["rz_rate"] = 0.20  # proxy when RZ not known
    out = team[["team","plays_est","proe","rz_rate"]].copy()
    # Fill required columns (zeros as neutral proxies)
    for c in ["def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
              "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z"]:
        out[c] = 0.0
    out["opp_team"] = pd.NA
    out["event_id"] = pd.NA
    return out[[
        "team","opp_team","event_id",
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

    # Prefer true PBP if available
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

        # Defensive z-proxies if EPA exists
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
            "team","def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate"
        ]].copy()
        for c in ["plays_est","proe","rz_rate"]:
            out[c] = out[c].fillna(0.0)
        out["opp_team"] = pd.NA; out["event_id"] = pd.NA
        return out[[
            "team","opp_team","event_id",
            "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate"
        ]]

    # No PBP → ESPN
    espn_team = _espn_team_table(current_season)
    if not espn_team.empty:
        print("[fetch_nfl_data] using ESPN team table")
        return _team_form_from_tables(espn_team)

    # API-SPORTS
    api_team, _ = apisports_tables(current_season)
    if not api_team.empty:
        print("[fetch_nfl_data] using API-SPORTS team table")
        return _team_form_from_tables(api_team)

    # MySportsFeeds
    msf_team, _ = msf_tables(current_season)
    if not msf_team.empty:
        print("[fetch_nfl_data] using MySportsFeeds team table")
        return _team_form_from_tables(msf_team)

    # NFLGSIS (authenticated)
    try:
        print("[fetch_nfl_data] trying NFLGSIS fallback …")
        s = login_session()
        games = list_games(s)
        if games:
            gids = [g["id"] for g in games]
            team_df, _ = gsis_team_player_tables(s, gids, limit=40)
            if not team_df.empty:
                print("[fetch_nfl_data] using NFLGSIS team table")
                return _team_form_from_tables(team_df)
    except Exception as e:
        warnings.warn(f"NFLGSIS fallback failed: {type(e).__name__}: {e}")

    # final: empty
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
# === END scripts/fetch_nfl_data.py ===
