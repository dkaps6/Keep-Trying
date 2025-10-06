# scripts/fetch_weather.py
# Compatible weather fetcher:
# - Prefers your existing inputs: outputs/game_lines.csv + data/stadiums.csv
# - Falls back to OddsAPI /events and an internal stadium map if needed
# - Uses Open-Meteo (free) to get hourly temp/wind/precip near kickoff
# - Writes data/weather.csv with expected columns for pricing weather multipliers

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional

import os
import pandas as pd
import requests
from dotenv import load_dotenv

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# ---- Fallback stadium map (lat, lon, altitude_ft, dome) ----
STADIUMS_FALLBACK: Dict[str, Dict[str, Any]] = {
    # AFC East
    "BUF": {"lat": 42.7738, "lon": -78.7868, "altitude_ft": 751, "dome": False},
    "MIA": {"lat": 25.9580, "lon": -80.2389, "altitude_ft": 7, "dome": False},
    "NE":  {"lat": 42.0909, "lon": -71.2643, "altitude_ft": 289, "dome": False},
    "NYJ": {"lat": 40.8136, "lon": -74.0745, "altitude_ft": 3, "dome": False},
    # AFC North
    "BAL": {"lat": 39.2780, "lon": -76.6227, "altitude_ft": 20, "dome": False},
    "CIN": {"lat": 39.0954, "lon": -84.5160, "altitude_ft": 482, "dome": False},
    "CLE": {"lat": 41.5061, "lon": -81.6995, "altitude_ft": 600, "dome": False},
    "PIT": {"lat": 40.4468, "lon": -80.0158, "altitude_ft": 732, "dome": False},
    # AFC South
    "HOU": {"lat": 29.6847, "lon": -95.4107, "altitude_ft": 50, "dome": True},
    "IND": {"lat": 39.7601, "lon": -86.1639, "altitude_ft": 715, "dome": True},
    "JAX": {"lat": 30.3239, "lon": -81.6373, "altitude_ft": 16, "dome": False},
    "TEN": {"lat": 36.1665, "lon": -86.7713, "altitude_ft": 476, "dome": False},
    # AFC West
    "DEN": {"lat": 39.7439, "lon": -105.0201, "altitude_ft": 5280, "dome": False},
    "KC":  {"lat": 39.0490, "lon": -94.4839, "altitude_ft": 910, "dome": False},
    "LV":  {"lat": 36.0908, "lon": -115.1830, "altitude_ft": 2200, "dome": True},
    "LAC": {"lat": 33.9535, "lon": -118.3391, "altitude_ft": 120, "dome": True},
    # NFC East
    "DAL": {"lat": 32.7473, "lon": -97.0945, "altitude_ft": 646, "dome": True},
    "NYG": {"lat": 40.8136, "lon": -74.0745, "altitude_ft": 3, "dome": False},
    "PHI": {"lat": 39.9008, "lon": -75.1675, "altitude_ft": 10, "dome": False},
    "WAS": {"lat": 38.9077, "lon": -76.8645, "altitude_ft": 50, "dome": False},
    # NFC North
    "CHI": {"lat": 41.8623, "lon": -87.6167, "altitude_ft": 594, "dome": False},
    "DET": {"lat": 42.3400, "lon": -83.0456, "altitude_ft": 604, "dome": True},
    "GB":  {"lat": 44.5013, "lon": -88.0622, "altitude_ft": 646, "dome": False},
    "MIN": {"lat": 44.9740, "lon": -93.2581, "altitude_ft": 840, "dome": True},
    # NFC South
    "ATL": {"lat": 33.7554, "lon": -84.4009, "altitude_ft": 1050, "dome": True},
    "CAR": {"lat": 35.2259, "lon": -80.8528, "altitude_ft": 748, "dome": False},
    "NO":  {"lat": 29.9510, "lon": -90.0812, "altitude_ft": 3, "dome": True},
    "TB":  {"lat": 27.9759, "lon": -82.5033, "altitude_ft": 25, "dome": False},
    # NFC West
    "ARI": {"lat": 33.5277, "lon": -112.2626, "altitude_ft": 1070, "dome": True},
    "LAR": {"lat": 33.9535, "lon": -118.3391, "altitude_ft": 120, "dome": True},
    "SF":  {"lat": 37.4030, "lon": -121.9700, "altitude_ft": 17, "dome": False},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "altitude_ft": 20, "dome": False},
}

# Simple alias map for team names coming from various sources
ALIASES = {
    "JAGUARS": "JAX", "JACKSONVILLE JAGUARS": "JAX", "JACKSONVILLE": "JAX",
    "COMMANDERS": "WAS", "WASHINGTON COMMANDERS": "WAS", "WASHINGTON": "WAS",
    "PACKERS": "GB", "GREEN BAY PACKERS": "GB", "GREEN BAY": "GB",
    "BUCCANEERS": "TB", "TAMPA BAY BUCCANEERS": "TB", "TAMPA BAY": "TB",
    "49ERS": "SF", "SAN FRANCISCO 49ERS": "SF",
    "CHARGERS": "LAC", "LOS ANGELES CHARGERS": "LAC",
    "RAMS": "LAR", "LOS ANGELES RAMS": "LAR",
    "PATRIOTS": "NE", "NEW ENGLAND PATRIOTS": "NE",
    "GIANTS": "NYG", "NEW YORK GIANTS": "NYG",
    "JETS": "NYJ", "NEW YORK JETS": "NYJ",
    "SAINTS": "NO", "NEW ORLEANS SAINTS": "NO",
}

def _nearest_hour_index(times: list[str], kickoff_iso: str) -> int:
    k = dt.datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
    deltas = [abs((dt.datetime.fromisoformat(t.replace("Z", "+00:00")) - k).total_seconds()) for t in times]
    return int(pd.Series(deltas).idxmin())

def _fetch_open_meteo(lat: float, lon: float, kickoff_iso: str) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    hourly = js.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {"temp_f": None, "wind_mph": None, "precip": "none"}
    idx = _nearest_hour_index(times, kickoff_iso)

    def _get(arr, i):
        return (arr[i] if i < len(arr) else None) if arr is not None else None

    temp_f = _get(hourly.get("temperature_2m"), idx)
    wind_mph = _get(hourly.get("wind_speed_10m"), idx)
    precip_amt = _get(hourly.get("precipitation"), idx)
    precip = "none"
    try:
        if precip_amt is not None and float(precip_amt) >= 0.05:
            precip = "rain"
        # Optional: if temp_f <= 32 and precip present, label "snow"
        if temp_f is not None and precip != "none" and float(temp_f) <= 32.0:
            precip = "snow"
    except Exception:
        pass
    return {"temp_f": temp_f, "wind_mph": wind_mph, "precip": precip}

def _resolve_team_key(team: str) -> Optional[str]:
    if not team:
        return None
    t = team.upper().strip()
    if t in STADIUMS_FALLBACK:
        return t
    return ALIASES.get(t)

def _load_events_from_game_lines(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    need = {"event_id", "home_team", "away_team", "commence_time"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    # Keep only columns we need
    return df[list(need)].drop_duplicates()

def _load_stadiums_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    need = {"team", "lat", "lon", "altitude_ft", "dome"}
    if not need.issubset(set(df.columns)):
        return {}
    out = {}
    for _, r in df.iterrows():
        key = str(r["team"]).upper().strip()
        out[key] = {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "altitude_ft": float(r["altitude_ft"]),
            "dome": bool(r["dome"]),
        }
    return out

def _fetch_events_from_odds_api(hours: int) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()
    now = dt.datetime.utcnow()
    commence_from = now.isoformat(timespec="seconds") + "Z"
    commence_to = (now + dt.timedelta(hours=hours)).isoformat(timespec="seconds") + "Z"
    url = f"{ODDS_BASE}/sports/{SPORT}/events"
    params = {"apiKey": api_key, "dateFormat": "iso",
              "commenceTimeFrom": commence_from, "commenceTimeTo": commence_to}
    r = requests.get(url, params=params, timeout=20)
    if not r.ok:
        return pd.DataFrame()
    items = r.json() or []
    # Normalize to expected columns
    rows = []
    for e in items:
        rows.append({
            "event_id": e.get("id"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "commence_time": e.get("commence_time"),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Fetch per-event weather → data/weather.csv (with fallbacks)")
    ap.add_argument("--hours", type=int, default=168, help="Lookahead window (default 7 days)")
    ap.add_argument("--game-lines", default="outputs/game_lines.csv", help="Preferred source of events")
    ap.add_argument("--stadiums", default="data/stadiums.csv", help="Preferred stadium metadata")
    ap.add_argument("--out", default="data/weather.csv")
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)

    # 1) Events: prefer your game_lines.csv; fallback to Odds API /events
    events = _load_events_from_game_lines(Path(args.game_lines))
    if events.empty:
        events = _fetch_events_from_odds_api(args.hours)
    if events.empty:
        # Nothing to do → write empty file for strict pipeline awareness
        pd.DataFrame(columns=[
            "event_id","home_team","away_team","commence_time",
            "wind_mph","temp_f","precip","altitude_ft","dome"
        ]).to_csv(args.out, index=False)
        print("[fetch_weather] No events found. Wrote empty data/weather.csv")
        return

    # 2) Stadiums: prefer your CSV; fallback to internal map
    stadiums = _load_stadiums_csv(Path(args.stadiums))
    if not stadiums:
        stadiums = STADIUMS_FALLBACK

    rows = []
    for _, e in events.iterrows():
        event_id = e.get("event_id")
        home_team = e.get("home_team")
        away_team = e.get("away_team")
        commence_time = e.get("commence_time")

        key = _resolve_team_key(str(home_team))
        st = stadiums.get(key or "", None)

        if not st:
            rows.append({
                "event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
                "wind_mph": None,
                "temp_f": None,
                "precip": "none",
                "altitude_ft": None,
                "dome": None,
            })
            continue

        lat = st["lat"]; lon = st["lon"]
        altitude = st.get("altitude_ft")
        dome = bool(st.get("dome", False))

        wx = _fetch_open_meteo(lat=lat, lon=lon, kickoff_iso=str(commence_time))

        wind = 0.0 if dome else wx.get("wind_mph")
        precip = "none" if dome else wx.get("precip")

        rows.append({
            "event_id": event_id,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "wind_mph": wind,
            "temp_f": wx.get("temp_f"),
            "precip": precip,
            "altitude_ft": altitude,
            "dome": dome,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[fetch_weather] wrote {len(df):,} rows → {args.out}")

if __name__ == "__main__":
    main()
