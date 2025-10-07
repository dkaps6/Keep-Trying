#!/usr/bin/env python3
from __future__ import annotations
import os, time, requests
import pandas as pd

API = "https://v3.football.api-sports.io"
KEY = os.getenv("APISPORTS_KEY", "")

def _get(path: str, params: dict) -> dict | None:
    if not KEY:
        return None
    headers = {"x-apisports-key": KEY}
    r = requests.get(f"{API}{path}", headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data or "response" not in data:
        return None
    return data

def schedules(season: int) -> pd.DataFrame | None:
    # Regular season schedule
    data = _get("/games", {"league": 1, "season": season})
    if not data: return None
    rows = []
    for g in data["response"]:
        rows.append({
            "season": season,
            "game_id": g.get("id"),
            "week": g.get("week"),
            "commence_time": g.get("date"),
            "home_team": (g.get("teams") or {}).get("home", {}).get("name"),
            "away_team": (g.get("teams") or {}).get("away", {}).get("name"),
        })
    return pd.DataFrame(rows)

def injuries(season: int) -> pd.DataFrame | None:
    # APISports injury endpoint (limited but useful for “Out/Questionable/Probable” flags)
    data = _get("/injuries", {"league": 1, "season": season})
    if not data: return None
    rows = []
    for it in data["response"]:
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
    return pd.DataFrame(rows)

def standings(season: int) -> pd.DataFrame | None:
    data = _get("/standings", {"league": 1, "season": season})
    if not data: return None
    rows=[]
    for conf in data["response"]:
        for entry in conf or []:
            team = (entry.get("team") or {}).get("name")
            stats = entry.get("records") or {}
            rows.append({"season": season, "team": team, **{k: v for k,v in stats.items()}})
    return pd.DataFrame(rows)

