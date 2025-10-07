#!/usr/bin/env python3
from __future__ import annotations
import os, requests
import pandas as pd

API = "https://v1.api-sports.io/nfl/v1"
KEY = os.getenv("APISPORTS_KEY", "")

def _get(path: str, params: dict) -> dict | None:
    if not KEY:
        return None
    headers = {"x-apisports-key": KEY}
    url = f"{API}{path}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=45)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def schedules(season: int) -> pd.DataFrame | None:
    data = _get("/games", {"season": season})
    if not data: return None
    rows=[]
    for g in data.get("response", []):
        t = g.get("teams") or {}
        rows.append({
            "season": season,
            "game_id": g.get("id"),
            "week": g.get("week"),
            "commence_time": g.get("date"),
            "home_team": (t.get("home") or {}).get("name"),
            "away_team": (t.get("away") or {}).get("name"),
        })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

def injuries(season: int) -> pd.DataFrame | None:
    data = _get("/injuries", {"season": season})
    if not data: return None
    rows=[]
    for it in data.get("response", []):
        p = it.get("player") or {}
        t = it.get("team") or {}
        rows.append({
            "season": season,
            "player": p.get("name"),
            "team": t.get("name"),
            "status": it.get("type") or it.get("status"),
            "detail": it.get("reason") or "",
            "update": it.get("date"),
        })
    df = pd.DataFrame(rows)
    return df if not df.empty else None
