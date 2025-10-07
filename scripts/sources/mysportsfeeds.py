# scripts/sources/mysportsfeeds.py
from __future__ import annotations
import os, time, base64
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd

"""
MySportsFeeds v2.1 helper
Docs: https://www.mysportsfeeds.com/data-feeds/api-docs/
We pull game list + box/player logs to assemble minimal team & player totals.

Auth: Basic (username:key or token) with password (or blank). We support env:
  MSF_KEY, MSF_PASSWORD
"""

BASE = "https://api.mysportsfeeds.com/v2.1/pull/nfl"

def _auth_header() -> Dict[str, str]:
    user = os.getenv("MSF_KEY", "")
    pw = os.getenv("MSF_PASSWORD", "")
    creds = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}

def _get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
    try:
        r = requests.get(f"{BASE}/{path}", headers=_auth_header(), params=params or {}, timeout=25)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
    except Exception:
        pass
    return None

def _season_tag(season: int) -> str:
    # MySportsFeeds uses e.g. "2025-regular" or "2025-2026-regular".
    yr = str(season)
    return f"{yr}-regular"

def list_games(season: int) -> List[Dict[str, Any]]:
    js = _get(f"{_season_tag(season)}/games.json")
    games = []
    if js and js.get("games"):
        for g in js["games"]:
            gid = g.get("schedule", {}).get("id")
            if gid:
                games.append(g)
    return games

def game_box(event_id: str | int, season: int) -> Dict[str, Any] | None:
    return _get(f"{_season_tag(season)}/games/{event_id}/boxscore.json")

def season_team_player_tables(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    games = list_games(season)
    if not games:
        return pd.DataFrame(), pd.DataFrame()

    team_rows, player_rows = [], []
    for g in games:
        eid = g.get("schedule", {}).get("id")
        if not eid:
            continue
        box = game_box(eid, season)
        if not box:
            continue

        for side in box.get("stats", []):
            team_abbr = (side.get("team", {}) or {}).get("abbreviation")
            # team totals
            s = side.get("teamStats", {}) or {}
            pass_att = float(((s.get("passing", {}) or {}).get("passAttempts", {}) or {}).get("value") or 0)
            rush_att = float(((s.get("rushing", {}) or {}).get("rushAttempts", {}) or {}).get("value") or 0)
            team_rows.append({"team": team_abbr, "pass_att": pass_att, "rush_att": rush_att, "plays": pass_att + rush_att, "event_id": eid})

            # players
            for p in (side.get("players", []) or []):
                ath = p.get("player", {}) or {}
                pname = ath.get("firstName","") + " " + ath.get("lastName","")
                st = p.get("playerStats", {}) or {}
                row = {"team": team_abbr, "player": pname.strip(),
                       "targets": 0.0, "receptions": 0.0, "carries": 0.0,
                       "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": 0.0, "attempts": 0.0}
                # Passing
                row["attempts"] += float(((st.get("passing", {}) or {}).get("passAttempts", {}) or {}).get("value") or 0)
                row["pass_yards"] += float(((st.get("passing", {}) or {}).get("passYards", {}) or {}).get("value") or 0)
                # Rushing
                row["carries"] += float(((st.get("rushing", {}) or {}).get("rushAttempts", {}) or {}).get("value") or 0)
                row["rush_yards"] += float(((st.get("rushing", {}) or {}).get("rushYards", {}) or {}).get("value") or 0)
                # Receiving
                row["targets"] += float(((st.get("receiving", {}) or {}).get("targets", {}) or {}).get("value") or 0)
                row["receptions"] += float(((st.get("receiving", {}) or {}).get("receptions", {}) or {}).get("value") or 0)
                row["rec_yards"] += float(((st.get("receiving", {}) or {}).get("recYards", {}) or {}).get("value") or 0)
                player_rows.append(row)
        time.sleep(0.1)

    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)
