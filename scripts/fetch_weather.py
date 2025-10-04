# scripts/fetch_weather.py
from __future__ import annotations
import pandas as pd
import requests
from datetime import datetime
import os

STADIUMS_CSV = "data/stadiums.csv"

def _load_stadiums() -> pd.DataFrame:
    if os.path.exists(STADIUMS_CSV):
        return pd.read_csv(STADIUMS_CSV)
    # minimal fallback: you can expand this file over time
    df = pd.DataFrame([
        {"team":"BUF","stadium":"Highmark Stadium","lat":42.7738,"lon":-78.7868},
        {"team":"KC","stadium":"GEHA Field at Arrowhead","lat":39.049,"lon":-94.4843},
        {"team":"DAL","stadium":"AT&T Stadium","lat":32.7473,"lon":-97.0945},
        {"team":"PHI","stadium":"Lincoln Financial Field","lat":39.9008,"lon":-75.1675},
    ])
    os.makedirs("data", exist_ok=True)
    df.to_csv(STADIUMS_CSV, index=False)
    return df

def _open_meteo(lat: float, lon: float, t: pd.Timestamp) -> dict:
    # hourly forecast around kickoff time (UTC assumed; open-meteo takes ISO date with tz offset)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,precipitation_probability,windspeed_10m"
        f"&start_date={t.date()}&end_date={t.date()}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def build_weather_csv(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl
    sched = nfl.import_schedules([season])
    # Keep regular-season games with kickoffs
    sched = sched[~sched["gameday"].isna()].copy()
    sched["kickoff"] = pd.to_datetime(sched["gameday"] + " " + sched["gametime"], errors="coerce", utc=True)

    stad = _load_stadiums()
    stad_map = {row["team"]: (row["lat"], row["lon"]) for _, row in stad.iterrows()}

    rows = []
    for _, g in sched.iterrows():
        home = g.get("home_team")
        latlon = stad_map.get(home)
        if not latlon: 
            continue
        lat, lon = latlon
        k = g["kickoff"]
        try:
            data = _open_meteo(lat, lon, k)
            # find hour == kickoff hour
            hourly = pd.DataFrame(data.get("hourly", {}))
            if hourly.empty or "time" not in hourly: 
                continue
            hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
            hrow = hourly.iloc[(hourly["time"]-k).abs().argsort()[:1]]
            rec = {
                "season": season,
                "week": int(g.get("week", 0) or 0),
                "home_team": home,
                "away_team": g.get("away_team"),
                "kickoff_utc": k.isoformat(),
                "wind_mph": float(hrow["windspeed_10m"].iloc[0]) if "windspeed_10m" in hrow else None,
                "precip": "rain" if ("precipitation" in hrow and hrow["precipitation"].iloc[0] and float(hrow["precipitation"].iloc[0])>0) else "none",
                "temperature_f": float(hrow["temperature_2m"].iloc[0]) * 9/5 + 32 if "temperature_2m" in hrow else None,
            }
            rows.append(rec)
        except Exception as e:
            # non-fatal
            continue
    return pd.DataFrame(rows)
