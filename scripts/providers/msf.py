
#!/usr/bin/env python3
from __future__ import annotations
import os, base64, requests
import pandas as pd

MSF_KEY = os.getenv("MSF_KEY", "")
MSF_PASSWORD = os.getenv("MSF_PASSWORD", "")

BASE = "https://api.mysportsfeeds.com/v2.1/pull/nfl"

def _auth() -> dict | None:
    if not MSF_KEY or not MSF_PASSWORD:
        return None
    tok = base64.b64encode(f"{MSF_KEY}:{MSF_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {tok}"}

def _get_json(path: str, params: dict | None = None) -> dict | None:
    headers = _auth()
    if not headers:
        return None
    url = f"{BASE}/{path}"
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=45)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def _season_tag(season: int) -> str:
    return f"{season}-regular"

# ---------- Injuries ----------
def injuries(season: int) -> pd.DataFrame | None:
    st = _season_tag(season)
    data = _get_json(f"{st}/injuries.json")
    if not data: return None
    rows=[]
    for team in (data.get("injuries") or []):
        tname = (team.get("team") or {}).get("name")
        for p in (team.get("players") or []):
            info = p.get("player") or {}
            plist = p.get("injuries") or [{}]
            rec = plist[0] if plist else {}
            rows.append({
                "season": season,
                "player": (info.get("firstName","")+" "+info.get("lastName","")).strip(),
                "team": tname,
                "status": rec.get("status") or "",
                "detail": rec.get("note") or "",
                "update": rec.get("date") or "",
            })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

# ---------- Player game logs -> weekly player stats ----------
def player_stats_week(season: int) -> pd.DataFrame | None:
    st = _season_tag(season)
    data = _get_json(f"{st}/player_gamelogs.json", params={"limit": 100000})
    if not data: return None
    rows=[]
    for gl in (data.get("gamelogs") or []):
        team = (gl.get("team") or {}).get("name")
        game = (gl.get("game") or {})
        stats = (gl.get("stats") or {})
        pinfo = (gl.get("player") or {})
        week = game.get("week")
        def g(*path, default=0):
            cur = stats
            for k in path:
                cur = cur.get(k, {})
            if isinstance(cur, (int,float)): return cur
            if isinstance(cur, dict): return cur.get("amount", default)
            return default
        rows.append({
            "season": season,
            "week": week,
            "team": team,
            "player": (pinfo.get("firstName","")+" "+pinfo.get("lastName","")).strip(),
            "position": pinfo.get("position"),
            "rec":     g("receiving","receptions", default=0),
            "tgt":     g("receiving","targets", default=0),
            "rec_yds": g("receiving","yards", default=0),
            "rec_td":  g("receiving","touchdowns", default=0),
            "rush_att": g("rushing","attempts", default=0),
            "rush_yds": g("rushing","yards", default=0),
            "rush_td":  g("rushing","touchdowns", default=0),
            "pass_att": g("passing","attempts", default=0),
            "pass_yds": g("passing","yards", default=0),
            "pass_td":  g("passing","touchdowns", default=0),
            "sacks":    g("passing","sacks", default=0),
            "targets":  g("receiving","targets", default=0),
        })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

# ---------- Team game logs -> weekly team stats & pace proxy ----------
def team_stats_week(season: int) -> pd.DataFrame | None:
    st = _season_tag(season)
    data = _get_json(f"{st}/team_gamelogs.json", params={"limit": 100000})
    if not data: return None
    rows=[]
    for gl in (data.get("gamelogs") or []):
        team = (gl.get("team") or {}).get("name")
        game = (gl.get("game") or {})
        stats = (gl.get("stats") or {})
        week = game.get("week")
        def g(*path, default=0):
            cur = stats
            for k in path:
                cur = cur.get(k, {})
            if isinstance(cur, (int,float)): return cur
            if isinstance(cur, dict): return cur.get("amount", default)
            return default
        pass_att = g("passing","attempts", default=0)
        rush_att = g("rushing","attempts", default=0)
        sacks    = g("passing","sacks", default=0)
        plays    = pass_att + rush_att + sacks
        rows.append({
            "season": season,
            "week": week,
            "team": team,
            "plays": plays,
            "pass_att": pass_att,
            "rush_att": rush_att,
            "sacks": sacks,
            "def_pressure_rate": 0.0,
            "def_pass_epa": 0.0,
            "def_rush_epa": 0.0,
            "def_sack_rate": 0.0,
        })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

# ---------- Season aggregates from weekly ----------
def player_stats_reg(season: int) -> pd.DataFrame | None:
    wk = player_stats_week(season)
    if wk is None or wk.empty:
        return None
    grp = wk.groupby(["season","team","player","position"], as_index=False).sum(numeric_only=True)
    return grp

def team_stats_reg(season: int) -> pd.DataFrame | None:
    wk = team_stats_week(season)
    if wk is None or wk.empty:
        return None
    grp = wk.groupby(["season","team"], as_index=False).sum(numeric_only=True)
    return grp

# ---------- Schedules / Rosters ----------
def schedules(season: int) -> pd.DataFrame | None:
    st = _season_tag(season)
    data = _get_json(f"{st}/games.json", params={"limit": 100000})
    if not data: return None
    rows=[]
    for g in (data.get("games") or []):
        sched = g.get("schedule",{})
        teams = sched
        rows.append({
            "season": season,
            "game_id": sched.get("id"),
            "week": sched.get("week"),
            "commence_time": sched.get("startTime"),
            "home_team": (sched.get("homeTeam") or {}).get("abbreviation") or (sched.get("homeTeam") or {}).get("name"),
            "away_team": (sched.get("awayTeam") or {}).get("abbreviation") or (sched.get("awayTeam") or {}).get("name"),
        })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

def rosters(season: int) -> pd.DataFrame | None:
    st = _season_tag(season)
    data = _get_json(f"{st}/rosters.json", params={"limit": 100000})
    if not data: return None
    rows=[]
    for team in (data.get("rosters") or []):
        tname = (team.get("team") or {}).get("name")
        for p in (team.get("players") or []):
            info = p.get("player") or {}
            rows.append({
                "season": season,
                "team": tname,
                "player": (info.get("firstName","")+" "+info.get("lastName","")).strip(),
                "position": info.get("primaryPosition"),
                "jersey": info.get("jerseyNumber"),
            })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

def rosters_weekly(season: int): return None
def depth_charts(season: int): return None
def snap_counts(season: int): return None
def participation(season: int): return None
