# scripts/providers/espn_free.py
from __future__ import annotations
import time, requests, pandas as pd
from typing import Dict, List

ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ESPN_TEAM_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}?enable=roster"

HEADERS = {
    "User-Agent": "props-model/1.0 (+https://github.com/your-repo) Python-requests"
}

TEAM_ABBR_FIX = {
    # ESPN sometimes differs from book/nflverse abbreviations — patch here if needed
    "WSH": "WAS",
    "LV": "LVR",
    "LAR": "LA",
}

def _espn_get(url: str, params=None) -> Dict:
    for i in range(5):
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(0.75 * (2 ** i))
            continue
        r.raise_for_status()
    r.raise_for_status()

def _teams() -> List[Dict]:
    data = _espn_get(ESPN_TEAMS)
    # ESPN returns nested objects; extract teams
    teams = []
    for item in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        t = item.get("team", {})
        teams.append({
            "espn_team_id": t.get("id"),
            "team_abbr": t.get("abbreviation"),
            "team_name": t.get("displayName"),
        })
    return teams

def _roster_for_team(team_id: str) -> List[Dict]:
    j = _espn_get(ESPN_TEAM_ROSTER.format(team_id=team_id))
    entries = j.get("team", {}).get("athletes", [])  # grouped by position
    roster = []
    for group in entries:
        for a in group.get("items", []):
            roster.append({
                "espn_player_id": a.get("id"),
                "player": a.get("displayName"),
                "position": a.get("position", {}).get("abbreviation"),
                "status": (a.get("status", {}) or {}).get("name") or a.get("status") or "Active",
            })
    return roster

def build_id_map_from_espn() -> pd.DataFrame:
    teams = _teams()
    rows = []
    for t in teams:
        tid = t["espn_team_id"]
        abbr = TEAM_ABBR_FIX.get(t["team_abbr"], t["team_abbr"])
        try:
            roster = _roster_for_team(tid)
            for r in roster:
                rows.append({
                    "player": r["player"],
                    "team": abbr,
                    "player_id": f"espn:{r['espn_player_id']}",
                    "pos": r.get("position"),
                    "status": r.get("status"),
                    "espn_team_id": tid,
                })
            time.sleep(0.15)  # be polite
        except Exception as e:
            # don’t fail whole run if one team errors
            print(f"[espn_free] roster fetch failed for {abbr} ({tid}): {e}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["player","team"])
    return df
