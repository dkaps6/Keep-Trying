# scripts/fetch_nfl_data.py
from __future__ import annotations
import argparse, warnings, time
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple
import pandas as pd
import requests

# -------- Optional primary sources (nflverse) ----------
try:
    import nflreadpy as nrp  # preferred when available
    HAS_NFLREADPY = True
except Exception:
    HAS_NFLREADPY = False

try:
    import nfl_data_py as nfl  # legacy fallback
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False


# ========= ESPN FALLBACK (live 2025 weekly) =========
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
ESPN_EVENT_SUMMARY = "https://site.api.espn.com/apis/v2/sports/football/nfl/summary"
ESPN_BOX = "https://site.api.espn.com/apis/v2/sports/football/nfl/boxscore"

def _http_json(url: str, params: Dict[str, Any] | None = None, tries: int = 3, backoff: float = 0.7) -> Dict[str, Any] | None:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            # 404 → don’t retry forever
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(backoff * (2**i))
    return None

def _espn_week_events(season: int, week: int) -> List[str]:
    """Return ESPN event ids for a given season/week."""
    data = _http_json(ESPN_SCOREBOARD, params={"week": week, "seasontype": 2, "dates": season})
    if not data:
        return []
    evts = []
    for e in data.get("events", []):
        eid = e.get("id")
        if eid:
            evts.append(eid)
    return evts

def _espn_box_players(event_id: str) -> Dict[str, Any] | None:
    return _http_json(ESPN_BOX, params={"event": event_id})

def _espn_summary(event_id: str) -> Dict[str, Any] | None:
    return _http_json(ESPN_EVENT_SUMMARY, params={"event": event_id})

def _parse_box_to_rows(event_id: str, box: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (team_totals_df, player_totals_df) for one event.
    team_totals: team, pass_att, rush_att, plays
    player_totals: team, player, targets, receptions, carries, pass_yards, rush_yards, rec_yards, attempts
    """
    if not box:
        return pd.DataFrame(), pd.DataFrame()
    teams: Dict[str, Dict[str, float]] = {}
    players_rows: List[Dict[str, Any]] = []

    for t in box.get("teams", []):
        team_abbr = (t.get("team", {}) or {}).get("abbreviation")
        if not team_abbr:
            # try alternate path
            loc = (t.get("team", {}) or {}).get("location")
            nickname = (t.get("team", {}) or {}).get("name")
            team_abbr = (loc or "")[:1] + (nickname or "")[:1]
        teams[team_abbr] = {"pass_att": 0.0, "rush_att": 0.0}

        cats = t.get("statistics", []) or []
        # ESPN sometimes nests stats; safer path is iterate "statistics" in "players" below for totals
        for pcat in t.get("players", []):
            for cat in pcat.get("statistics", []):
                name = cat.get("name", "").lower()
                stats = cat.get("stats", [])
                # passing
                if name == "passing":
                    # format usually like ["C/ATT", "YDS", "TD", "INT", ...]
                    # But per-player: use "attempts" field inside stats dicts if present
                    # However, ESPN exposes "passingAttempts" inside "athletes" sometimes.
                    pass
                # we'll read per-player below

        # per-player lines for this team
        for grp in t.get("players", []):
            for pl in grp.get("athletes", []):
                name = pl.get("athlete", {}).get("displayName") or pl.get("athlete", {}).get("shortName")
                # categories for this player
                row = {"team": team_abbr, "player": name,
                       "targets": 0.0, "receptions": 0.0, "carries": 0.0,
                       "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": 0.0, "attempts": 0.0}
                for s in pl.get("stats", []):
                    # s like {"name":"passing","athlete":{"..."},"statistics":{"attempts":35,"completions":22,"yards":241,...}}
                    nm = (s.get("name") or "").lower()
                    statdict = s.get("statistics") or {}
                    if nm == "passing":
                        att = float(statdict.get("attempts") or 0)
                        yds = float(statdict.get("yards") or 0)
                        row["attempts"] += att
                        row["pass_yards"] += yds
                        teams[team_abbr]["pass_att"] += att
                    elif nm == "rushing":
                        att = float(statdict.get("attempts") or statdict.get("carries") or 0)
                        yds = float(statdict.get("yards") or 0)
                        row["carries"] += att
                        row["rush_yards"] += yds
                        teams[team_abbr]["rush_att"] += att
                    elif nm == "receiving":
                        tgt = float(statdict.get("targets") or 0)
                        rec = float(statdict.get("receptions") or 0)
                        yds = float(statdict.get("yards") or 0)
                        row["targets"] += tgt
                        row["receptions"] += rec
                        row["rec_yards"] += yds
                players_rows.append(row)

    team_rows = []
    for abbr, agg in teams.items():
        plays = float(agg.get("pass_att", 0.0)) + float(agg.get("rush_att", 0.0))
        team_rows.append({
            "team": abbr,
            "pass_att": float(agg.get("pass_att", 0.0)),
            "rush_att": float(agg.get("rush_att", 0.0)),
            "plays": plays,
            "event_id": event_id,
        })
    return pd.DataFrame(team_rows), pd.DataFrame(players_rows)

def _espn_season_tables(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """For a season-to-date, collect team totals and player totals from ESPN box scores."""
    all_team = []
    all_players = []
    # Weeks 1..22 (includes late-season+playoffs safeguard)
    for wk in range(1, 23):
        evts = _espn_week_events(season, wk)
        if not evts:
            # If we reached the first empty week after some data, break
            if wk > 3:  # small cushion
                break
            else:
                continue
        for eid in evts:
            box = _espn_box_players(eid)
            tdf, pdf = _parse_box_to_rows(eid, box or {})
            if not tdf.empty:
                tdf["week"] = wk
                all_team.append(tdf)
            if not pdf.empty:
                pdf["week"] = wk
                all_players.append(pdf)
            time.sleep(0.1)  # be gentle
    team_df = pd.concat(all_team, ignore_index=True) if all_team else pd.DataFrame()
    players_df = pd.concat(all_players, ignore_index=True) if all_players else pd.DataFrame()
    return team_df, players_df


# ========= nflverse loaders (when available) =========
def load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    seasons = list(seasons)
    # 1) nflreadpy
    if HAS_NFLREADPY:
        try:
            print(f"[fetch_nfl_data] USING PBP (nflreadpy) for {seasons}")
            df = nrp.load_pbp(seasons=seasons)  # v0.1.3 signature
            if "season" not in df.columns and len(seasons) == 1:
                df["season"] = seasons[0]
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
    print("[fetch_nfl_data] ❌ PBP load failed; returning empty")
    return pd.DataFrame()


# ========= compute team form (with ESPN fallback) =========
def _team_form_from_weekly_via_espn(season: int) -> pd.DataFrame:
    team_df, _ = _espn_season_tables(season)
    if team_df.empty:
        return pd.DataFrame()
    # aggregate to per-team means
    per_team = (team_df.groupby("team")
                .agg(plays_est=("plays", "mean"),
                     pass_att=("pass_att", "sum"),
                     rush_att=("rush_att", "sum"))
                .reset_index())
    per_team["pass_rate"] = per_team["pass_att"] / (per_team["pass_att"] + per_team["rush_att"]).replace({0: pd.NA})
    league_pass = float(per_team["pass_rate"].mean(skipna=True))
    per_team["proe"] = (per_team["pass_rate"] - league_pass).fillna(0.0)
    per_team["rz_rate"] = 0.20  # still a proxy

    out = per_team[["team","plays_est","proe","rz_rate"]].copy()
    # neutral defensive/pace z’s until PBP is live
    for c in ["def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
              "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z"]:
        out[c] = 0.0
    return out[[
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]]

def _is_pass(df: pd.DataFrame) -> pd.Series:
    return (df.get("pass", 0) == 1) if "pass" in df.columns else df.get("play_type","").astype(str).str.contains("pass", case=False, na=False)
def _is_rush(df: pd.DataFrame) -> pd.Series:
    return (df.get("rush", 0) == 1) if "rush" in df.columns else df.get("play_type","").astype(str).str.contains("rush", case=False, na=False)
def _neutral_filter(cur: pd.DataFrame) -> pd.Series:
    qtr = cur.get("qtr", pd.Series([0]*len(cur)))
    hs  = cur.get("half_seconds_remaining", pd.Series([0]*len(cur)))
    ytg = cur.get("ydstogo", pd.Series([10]*len(cur)))
    return (qtr.between(1,3)) | ((qtr==4) & (hs>300) & (ytg<=10))

def compute_team_form(pbp_all: pd.DataFrame, current_season: int) -> pd.DataFrame:
    cols = [
        "team","opp_team","event_id",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]
    # if no PBP at all → ESPN weekly
    if pbp_all is None or pbp_all.empty:
        wk = _team_form_from_weekly_via_espn(current_season)
        if not wk.empty:
            wk["opp_team"] = pd.NA; wk["event_id"] = pd.NA
            return wk[cols]
        return pd.DataFrame(columns=cols)

    cur = pbp_all.copy()
    if "season" in cur.columns:
        cur = cur[cur["season"] == current_season].copy()

    if cur.empty:
        wk = _team_form_from_weekly_via_espn(current_season)
        if not wk.empty:
            wk["opp_team"] = pd.NA; wk["event_id"] = pd.NA
            return wk[cols]
        # worst-case neutral
        teams = sorted(set(pbp_all.get("posteam", pd.Series(dtype=str)).dropna()))
        out = pd.DataFrame({"team": teams})
        for c in cols[3:-3]: out[c] = 0.0
        out["plays_est"] = 60.0; out["proe"] = 0.0; out["rz_rate"] = 0.20
        out["opp_team"] = pd.NA; out["event_id"] = pd.NA
        return out[cols]

    # PBP path (historic seasons)
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
    team_neutral = (cur.loc[neutral_mask].groupby("posteam")["is_pass"].mean()
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

    # DEFENSE
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


# ================= CLI =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="2019-2024")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--write", default="data/team_form.csv")
    args = ap.parse_args()

    Path(args.write).parent.mkdir(parents=True, exist_ok=True)

    # history for PBP (if available)
    hist_years = [int(y) for y in args.history.split("-")] if "-" in args.history else [int(y) for y in args.history.split(",")]
    if len(hist_years) == 2:
        hist = list(range(hist_years[0], hist_years[1] + 1))
    else:
        hist = hist_years

    pbp_hist = load_pbp(hist)
    pbp_cur  = load_pbp([args.season])
    pbp_all  = pd.concat([pbp_hist, pbp_cur], ignore_index=True, sort=False)

    df = compute_team_form(pbp_all, args.season)
    df.to_csv(args.write, index=False)
    print(f"[fetch_nfl_data] ✅ wrote {len(df)} rows → {args.write}")

if __name__ == "__main__":
    main()
