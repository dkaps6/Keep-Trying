# scripts/providers/espn_pbp.py
import pandas as pd, requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

def _safe_json(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def schedules(season: int) -> pd.DataFrame:
    data = _safe_json(f"{BASE}/scoreboard?year={season}")
    rows = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        week = (ev.get("week") or {}).get("number")
        gid = ev.get("id")
        comps = comp.get("competitors") or []
        home, away = None, None
        for c in comps:
            if c.get("homeAway") == "home":
                home = c.get("team",{}).get("abbreviation")
            if c.get("homeAway") == "away":
                away = c.get("team",{}).get("abbreviation")
        rows.append({"game_id": gid, "week": week, "home_team": home, "away_team": away})
    return pd.DataFrame(rows)

def injuries(season: int) -> pd.DataFrame:
    return pd.DataFrame()

def rosters(season: int) -> pd.DataFrame:
    return pd.DataFrame()

def depth_charts(season: int) -> pd.DataFrame:
    return pd.DataFrame()

def snap_counts(season: int) -> pd.DataFrame:
    return pd.DataFrame()

def team_stats_week(season: int) -> pd.DataFrame:
    sch = schedules(season)
    if sch.empty: return pd.DataFrame()
    sch["plays"] = 0
    return sch.rename(columns={"home_team":"team"})[["team","week","plays"]]

def player_stats_week(season: int) -> pd.DataFrame:
    return pd.DataFrame()

def pbp(season: int) -> pd.DataFrame:
    return schedules(season)
