# scripts/sources/apisports.py
from __future__ import annotations
import os, time
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd

"""
API-SPORTS NFL (American Football) helper
Docs: https://v3.american-football.api-sports.io/
Free tier: ~100 req/day -> keep calls minimal and cache per run.
We extract TEAM totals and PLAYER totals sufficient for:
- team_form: plays_est, pass_rate -> proe proxy, rz_rate proxy (0.20)
- player_form: targets, carries, attempts, yards -> shares + efficiency
"""

BASE = "https://v3.american-football.api-sports.io"
HEADERS = lambda: {"x-apisports-key": os.getenv("APISPORTS_KEY", "")}

def _get(path: str, params: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        r = requests.get(f"{BASE}{path}", params=params, headers=HEADERS(), timeout=25)
        if r.status_code == 200:
            js = r.json()
            # API-SPORTS wraps payload as {response:[...]}
            return js
    except Exception:
        pass
    return None

def _league_id() -> int:
    # NFL league id is stable in v3; if they change, allow override via env
    return int(os.getenv("APISPORTS_LEAGUE_ID", "1"))

def _season_format(season: int) -> str:
    # API-Sports tends to use 'YYYY' for NFL seasons
    return str(season)

def list_games(season: int) -> List[Dict[str, Any]]:
    """Return list of games for season (regular+post), minimal fields."""
    lid = _league_id()
    js = _get("/games", {"league": lid, "season": _season_format(season)})
    if not js or not js.get("response"):
        return []
    return js["response"]

def game_stats(event_id: int | str) -> Dict[str, Any] | None:
    """Return game stats/box for a given game id."""
    lid = _league_id()
    js = _get("/games/statistics", {"id": event_id, "league": lid})
    return js if js and js.get("response") else None

def season_team_player_tables(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      team_df: team, pass_att, rush_att, plays, event_id
      player_df: team, player, targets, receptions, carries, pass_yards, rush_yards, rec_yards, attempts
    """
    games = list_games(season)
    if not games:
        return pd.DataFrame(), pd.DataFrame()

    team_rows, player_rows = [], []
    for g in games:
        gid = g.get("id") or g.get("game", {}).get("id")
        if not gid:
            continue
        stats = game_stats(gid)
        if not stats:
            continue

        # response is an array with [home, away] team statistics
        for side in stats.get("response", []):
            team = (side.get("team") or {}).get("name") or (side.get("team") or {}).get("nickname") or (side.get("team") or {}).get("code")
            abbr = (side.get("team") or {}).get("code") or team
            pass_att = 0.0; rush_att = 0.0
            # 'statistics' -> list of dicts per category; substructure varies slightly
            for cat in side.get("statistics", []):
                name = (cat.get("name") or "").lower()
                items = cat.get("statistics") or []
                # good implementations denote attempts in 'passing'/'rushing'
                if name == "passing":
                    for it in items:
                        if (it.get("name") or "").lower() in {"attempts","passingattempts","att"}:
                            try: pass_att = float(it.get("value", 0) or 0)
                            except Exception: pass
                if name == "rushing":
                    for it in items:
                        if (it.get("name") or "").lower() in {"attempts","carries","rushattempts","att"}:
                            try: rush_att = float(it.get("value", 0) or 0)
                            except Exception: pass
            team_rows.append({"team": abbr, "pass_att": pass_att, "rush_att": rush_att, "plays": pass_att + rush_att, "event_id": gid})

            # Players
            for pcat in side.get("players", []):
                athlete = pcat.get("player") or {}
                pname = athlete.get("name") or athlete.get("fullname") or athlete.get("id")
                row = {"team": abbr, "player": pname,
                       "targets": 0.0, "receptions": 0.0, "carries": 0.0,
                       "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": 0.0, "attempts": 0.0}
                for stat in pcat.get("statistics", []):
                    nm = (stat.get("title") or stat.get("name") or "").lower()
                    val = stat.get("value")
                    try:
                        valf = float(val)
                    except Exception:
                        continue
                    if "targets" in nm:
                        row["targets"] += valf
                    elif "reception" in nm and "percent" not in nm:
                        row["receptions"] += valf
                    elif "rushing attempts" in nm or "carries" in nm or (nm == "attempts" and "rushing" in (pcat.get("title","").lower())):
                        row["carries"] += valf
                    elif "passing attempts" in nm or (nm == "attempts" and "passing" in (pcat.get("title","").lower())):
                        row["attempts"] += valf
                    elif "passing yards" in nm:
                        row["pass_yards"] += valf
                    elif "rushing yards" in nm:
                        row["rush_yards"] += valf
                    elif "receiving yards" in nm:
                        row["rec_yards"] += valf
                player_rows.append(row)
        time.sleep(0.1)  # be polite

    team_df = pd.DataFrame(team_rows)
    player_df = pd.DataFrame(player_rows)
    return team_df, player_df
