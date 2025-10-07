#!/usr/bin/env python3
from __future__ import annotations
import requests
import pandas as pd
from xml.etree import ElementTree as ET

USER = os.getenv("NFLGSIS_USERNAME", "")
PASS = os.getenv("NFLGSIS_PASSWORD", "")

BASE = "https://www.nfl.info/nfldataexchange/dataexchange.asmx"

def _xml_get(method: str, params: dict) -> ET.Element | None:
    url = f"{BASE}/{method}"
    try:
        r = requests.get(url, params=params, timeout=45)
        if r.status_code != 200:
            return None
        return ET.fromstring(r.text)
    except Exception:
        return None

def schedules(season: int) -> pd.DataFrame | None:
    root = _xml_get("getSchedule", {"lseason": season, "lseasontype": "REG", "lclub": "", "lweek": ""})
    if root is None: return None
    rows=[]
    for g in root.findall(".//game"):
        rows.append({
            "season": season,
            "game_id": g.get("eid") or g.get("gsisId"),
            "week": g.get("week"),
            "commence_time": g.get("gdte") or g.get("date"),
            "home_team": g.get("home"),
            "away_team": g.get("visitor"),
        })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

def injuries(season: int) -> pd.DataFrame | None:
    root = _xml_get("getInjuryData", {"lseason": season, "lweek": "", "lseasontype": "REG"})
    if root is None: return None
    rows=[]
    for rec in root.findall(".//player"):
        rows.append({
            "season": season,
            "player": rec.get("name"),
            "team": rec.get("team"),
            "status": rec.get("status"),
            "detail": rec.get("injury"),
            "update": rec.get("reportDate"),
        })
    df = pd.DataFrame(rows)
    return df if not df.empty else None

def rosters(season: int): return None
def rosters_weekly(season: int): return None
def depth_charts(season: int): return None
def snap_counts(season: int): return None
def participation(season: int): return None
