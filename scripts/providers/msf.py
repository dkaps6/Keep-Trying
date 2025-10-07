#!/usr/bin/env python3
from __future__ import annotations
import os, base64, requests
import pandas as pd

MSF_KEY = os.getenv("MSF_KEY", "")
MSF_PASSWORD = os.getenv("MSF_PASSWORD", "")

def _auth() -> dict | None:
    if not MSF_KEY or not MSF_PASSWORD: return None
    token = base64.b64encode(f"{MSF_KEY}:{MSF_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

def _get(url: str, params: dict | None = None) -> dict | None:
    headers = _auth()
    if not headers: return None
    r = requests.get(url, headers=headers, params=params or {}, timeout=30)
    if r.status_code != 200: return None
    try:
        return r.json()
    except Exception:
        return None

# EXAMPLE: injuries as a richer backup
def injuries(season: int) -> pd.DataFrame | None:
    # season format in MSF often "2025-2026-regular" or "2025-regular"
    season_tag = f"{season}-regular"
    url = f"https://api.mysportsfeeds.com/v2.1/pull/nfl/{season_tag}/injuries.json"
    data = _get(url)
    if not data: return None
    rows=[]
    for team in (data.get("injuries") or []):
        tname = (team.get("team") or {}).get("name")
        for p in (team.get("players") or []):
            info = p.get("player") or {}
            inj = p.get("injuries") or [{}]
            rec = inj[0] if inj else {}
            rows.append({
                "season": season,
                "player": info.get("firstName","")+" "+info.get("lastName",""),
                "team": tname,
                "status": rec.get("status"),
                "detail": rec.get("note") or "",
                "update": rec.get("date") or "",
            })
    return pd.DataFrame(rows)

# You can add more mirrors here (rosters_weekly, depth charts, etc.) as needed.

