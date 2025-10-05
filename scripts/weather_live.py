# scripts/weather_live.py
# Pulls live/near-term weather for NFL stadium cities using Open-Meteo (no API key).
# Reads inputs/stadiums.csv, geocodes with Open-Meteo's free geocoder,
# fetches hourly forecast for the requested date, and writes inputs/weather_live.csv.
#
# Usage:
#   python -m scripts.weather_live --date 2025-10-05
# Optional:
#   --kickoff "13:00"         (local time; default 12:00)
#   --buffer-hours 2          (averages weather +/- buffer hours around kickoff)
#
# Output columns:
#   team, abbr, date, kickoff_local, lat, lon, temp_f, wind_mph, precip_mm, precip_prob, is_dome
#
# Notes:
# - If is_dome==TRUE in stadiums.csv, we still write a row but set wind=0, precip=0 and leave temp at 70F.

from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta
from typing import Tuple, Optional

import pandas as pd
import requests

STADIUMS_PATH = os.path.join("inputs", "stadiums.csv")
OUT_PATH = os.path.join("inputs", "weather_live.csv")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def _read_stadiums() -> pd.DataFrame:
    if not os.path.exists(STADIUMS_PATH):
        raise FileNotFoundError(f"{STADIUMS_PATH} not found")
    df = pd.read_csv(STADIUMS_PATH)
    # sanity columns
    required = {"team","abbr","city","state","is_dome"}
    for c in required:
        if c not in df.columns:
            raise ValueError(f"inputs/stadiums.csv missing required column: {c}")
    return df

def _geocode_city(city: str, state: str) -> Optional[Tuple[float, float]]:
    q = f"{city}, {state}"
    try:
        r = requests.get(GEOCODE_URL, params={"name": q, "count": 1, "language": "en"}, timeout=20)
        if r.status_code != 200:
            return None
        j = r.json()
        results = j.get("results") or []
        if not results:
            return None
        lat = float(results[0]["latitude"])
        lon = float(results[0]["longitude"])
        return lat, lon
    except Exception:
        return None

def _to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0

def _to_mph(ms: float) -> float:
    return ms * 2.23693629

def _fetch_hourly(lat: float, lon: float, date: str) -> Optional[pd.DataFrame]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,precipitation_probability,precipitation",
        "start_date": date,
        "end_date": date,
        "timezone": "auto",
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=25)
        if r.status_code != 200:
            return None
        j = r.json()
        hourly = j.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return None
        df = pd.DataFrame({
            "time": pd.to_datetime(times),
            "temp_c": hourly.get("temperature_2m", []),
            "wind_ms": hourly.get("wind_speed_10m", []),
            "precip_prob": hourly.get("precipitation_probability", []),
            "precip_mm": hourly.get("precipitation", []),
        })
        df["temp_f"] = df["temp_c"].astype(float).map(_to_f)
        df["wind_mph"] = df["wind_ms"].astype(float).map(_to_mph)
        return df
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Game date YYYY-MM-DD")
    ap.add_argument("--kickoff", default="12:00", help="Local kickoff time HH:MM (default 12:00)")
    ap.add_argument("--buffer-hours", type=int, default=2, help="Average weather +/- this many hours around kickoff")
    args = ap.parse_args()

    date = args.date
    kickoff_h, kickoff_m = [int(x) for x in args.kickoff.split(":")]
    buffer_h = max(0, int(args.buffer_hours))

    stad = _read_stadiums()
    out_rows = []

    for _, row in stad.iterrows():
        team = str(row["team"])
        abbr = str(row["abbr"])
        city = str(row["city"])
        state = str(row["state"])
        is_dome = str(row["is_dome"]).upper() in ("TRUE","1","YES","Y")

        # Domes: deterministic
        if is_dome:
            out_rows.append({
                "team": team, "abbr": abbr, "date": date, "kickoff_local": args.kickoff,
                "lat": None, "lon": None,
                "temp_f": 70.0, "wind_mph": 0.0, "precip_mm": 0.0, "precip_prob": 0.0,
                "is_dome": True,
            })
            continue

        # Outdoor: geocode + hourly forecast
        geo = _geocode_city(city, state)
        if not geo:
            # Unknown city → write row with NaNs so downstream logic can skip safely
            out_rows.append({
                "team": team, "abbr": abbr, "date": date, "kickoff_local": args.kickoff,
                "lat": None, "lon": None,
                "temp_f": None, "wind_mph": None, "precip_mm": None, "precip_prob": None,
                "is_dome": False,
            })
            continue

        lat, lon = geo
        hourly = _fetch_hourly(lat, lon, date)
        if hourly is None or hourly.empty:
            out_rows.append({
                "team": team, "abbr": abbr, "date": date, "kickoff_local": args.kickoff,
                "lat": lat, "lon": lon,
                "temp_f": None, "wind_mph": None, "precip_mm": None, "precip_prob": None,
                "is_dome": False,
            })
            continue

        # Find time window around kickoff
        # We don't know local tz offset explicitly; Open-Meteo returns 'time' already in local tz.
        # Coerce to naive time match by HH:MM string.
        target_hm = f"{kickoff_h:02d}:{kickoff_m:02d}"
        hourly["hm"] = hourly["time"].dt.strftime("%H:%M")
        center = hourly[hourly["hm"] == target_hm]
        if center.empty:
            # pick nearest hour
            hourly["abs_hour_diff"] = (hourly["time"].dt.hour - kickoff_h).abs()
            idx = hourly["abs_hour_diff"].idxmin()
            center_idx = int(idx)
        else:
            center_idx = int(center.index[0])

        start_idx = max(0, center_idx - buffer_h)
        end_idx = min(len(hourly) - 1, center_idx + buffer_h)

        window = hourly.iloc[start_idx:end_idx+1]
        out_rows.append({
            "team": team, "abbr": abbr, "date": date, "kickoff_local": args.kickoff,
            "lat": lat, "lon": lon,
            "temp_f": float(window["temp_f"].mean()),
            "wind_mph": float(window["wind_mph"].mean()),
            "precip_mm": float(window["precip_mm"].mean()),
            "precip_prob": float(window["precip_prob"].mean()),
            "is_dome": False,
        })

    out_df = pd.DataFrame(out_rows)
    os.makedirs("inputs", exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out_df)} rows → {OUT_PATH}")

if __name__ == "__main__":
    main()
