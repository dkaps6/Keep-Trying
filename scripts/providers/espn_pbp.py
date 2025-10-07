# scripts/providers/espn_pbp.py
from __future__ import annotations
import math, time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import requests

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
# plays feed (JSON, rich participants + start/end spots)
ESPN_PLAYS_TMPL = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event_id}/competitions/{comp_id}/plays?limit=5000"

# ESPN teams are IDs; we’ll map to abbreviations used elsewhere
# Fallback: use ESPN displayAbbreviation when present
_TEAM_ABBR_OVERRIDES = {
    # add/override anything you need
    "LA": "LAR",  # ESPN sometimes shows LA for Rams
    "WSH": "WAS",
}

def _http_json(url: str, params: Optional[Dict]=None, retries: int=3, sleep: float=0.6) -> dict:
    last = None
    for _ in range(retries):
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        last = r
        time.sleep(sleep)
    if last is None:
        raise RuntimeError(f"ESPN request failed: {url}")
    raise RuntimeError(f"ESPN request failed: {url} -> {last.status_code}: {last.text[:200]}")

def _season_weeks(season: int) -> List[int]:
    # cover preseason through SB without caring which exact weeks exist
    return list(range(1, 23))  # adjust if you want preseason: range(0, 24)

def _scoreboard_events(season: int, week: int) -> List[dict]:
    js = _http_json(ESPN_SCOREBOARD, params={"dates": season, "week": week})
    return js.get("events", []) or []

def _resolve_competition_ids(evt: dict) -> Tuple[str,str]:
    event_id = evt.get("id") or ""
    comps = evt.get("competitions") or []
    comp_id = comps[0].get("id") if comps else event_id
    return str(event_id), str(comp_id)

def _abbr(team_obj: dict) -> str:
    abbr = (team_obj.get("abbreviation")
            or team_obj.get("displayAbbreviation")
            or team_obj.get("shortDisplayName")
            or team_obj.get("name")
            or "").upper()
    return _TEAM_ABBR_OVERRIDES.get(abbr, abbr)

def _participants_to_names(play: dict) -> Dict[str, Optional[str]]:
    names = {"passer": None, "receiver": None, "rusher": None}
    parts = play.get("participants") or []
    for p in parts:
        role = (p.get("type") or "").lower()
        ath  = p.get("athlete") or {}
        name = ath.get("displayName") or ath.get("shortName") or None
        if "passer" in role:
            names["passer"] = name
        elif "receiver" in role:
            names["receiver"] = name
        elif "rusher" in role or "runner" in role:
            names["rusher"] = name
    return names

def _yardline_100(play: dict, possession_abbr: str) -> Optional[float]:
    """
    Convert ESPN 'start' spot to yardline_100 from the offense perspective.
    ESPN 'start' and 'end' look like:
      {"yardLine": 32, "team": {"abbreviation":"NE"}, "down":2, ...}
    If team == possession team -> yardline_100 = 100 - yardLine
    Else -> yardline_100 = yardLine
    """
    start = play.get("start") or {}
    yl = start.get("yardLine")
    spot_team = _abbr((start.get("team") or {}))
    if yl is None:
        return None
    yl = float(yl)
    if not possession_abbr:
        return None
    if spot_team and spot_team != possession_abbr:
        return yl  # we are on defense’s side
    return 100.0 - yl

def _is_pass(play: dict) -> bool:
    t = (play.get("type") or {}).get("text", "").lower()
    abbr = (play.get("type") or {}).get("abbreviation", "").upper()
    # ESPN uses "Pass Incomplete", "Pass Complete", "Sack" etc
    return ("pass" in t) or (abbr in {"PASS", "SACK"})  # sack is a pass dropback

def _is_rush(play: dict) -> bool:
    t = (play.get("type") or {}).get("text", "").lower()
    abbr = (play.get("type") or {}).get("abbreviation", "").upper()
    return ("rush" in t) or (abbr in {"RUSH"})

def _yards_gained(play: dict) -> Optional[float]:
    # ESPN has a 'yards' field on play summary; if missing, derive from start/end
    if "yards" in play:
        try:
            return float(play["yards"])
        except Exception:
            pass
    start = play.get("start") or {}
    end   = play.get("end") or {}
    sy = start.get("yardLine")
    ey = end.get("yardLine")
    if sy is None or ey is None:
        return None
    # if same side, delta sign depends on offense direction; hard to know reliably → fallback to None
    try:
        return float(ey) - float(sy)
    except Exception:
        return None

def _down(play: dict) -> Optional[int]:
    d = (play.get("start") or {}).get("down")
    try:
        d = int(d)
        if d in (1,2,3,4):
            return d
    except Exception:
        pass
    return None

def _ydstogo(play: dict) -> Optional[int]:
    ytg = (play.get("start") or {}).get("distance")
    try:
        return int(ytg)
    except Exception:
        return None

def _play_rows(event_id: str, comp_id: str, week: int) -> List[dict]:
    js = _http_json(ESPN_PLAYS_TMPL.format(event_id=event_id, comp_id=comp_id))
    items = js.get("items") or []
    # team context lives on competition
    # we also need possession team per play (ESPN gives it on 'start.team')
    rows: List[dict] = []
    # Derive home/away abr for defensive team inference
    comp = js.get("competitions") or js.get("competition") or {}
    # fallback: we can get teams from event-level later if needed
    for play in items:
        start = play.get("start") or {}
        team_obj = start.get("team") or {}
        offense = _abbr(team_obj)
        # def team best-effort: ESPN doesn’t give explicit; many analytics infer it via matchup teams
        defense = None  # we can fill later if you need using event teams
        down = _down(play)
        ytg  = _ydstogo(play)
        yl100 = _yardline_100(play, offense)
        names = _participants_to_names(play)
        yg = _yards_gained(play)
        rows.append({
            "game_id": event_id,
            "week": week,
            "posteam": offense,
            "defteam": defense,
            "down": down,
            "ydstogo": ytg,
            "yardline_100": yl100,
            "receiver_player_name": names["receiver"],
            "rusher_player_name": names["rusher"],
            "passer_player_name": names["passer"],
            "yards_gained": yg,
            "pass": _is_pass(play),
            "rush": _is_rush(play),
        })
    return rows

def pbp(season: int) -> pd.DataFrame:
    """
    Build a PBP DataFrame with the columns our composers require.
    NOTE: EPA/air_yards aren’t provided by ESPN; our downstream code tolerates NaNs.
    """
    all_rows: List[dict] = []
    for wk in _season_weeks(season):
        try:
            events = _scoreboard_events(season, wk)
        except Exception:
            events = []
        for evt in events:
            try:
                event_id, comp_id = _resolve_competition_ids(evt)
                rows = _play_rows(event_id, comp_id, wk)
                all_rows.extend(rows)
            except Exception:
                # ignore broken games; continue
                continue
        # be polite to ESPN
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Minimal cleanup / types
    for c in ["week","down","ydstogo"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["yardline_100","yards_gained"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Ensure the exact names your code looks for also exist
    df = df.rename(columns={"yards_gained":"yards"})
    return df
