# scripts/fetch_weather.py
# ------------------------------------------------------------------------
# Fetch game weather data with stadium coordinate fallbacks, UTC-safe time handling
# ------------------------------------------------------------------------

from __future__ import annotations
import argparse
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

# ------------------------------------------------------------------------
# 🏟️ Stadium fallback table (32 stadiums + aliases)
#  - You may also provide data/stadiums.csv with columns: stadium,lat,lon,dome
# ------------------------------------------------------------------------
STADIUM_FALLBACKS: Dict[str, Dict[str, Any]] = {
    # AFC
    "Highmark Stadium": {"lat": 42.7738, "lon": -78.7869, "dome": False},  # Bills
    "Hard Rock Stadium": {"lat": 25.9579, "lon": -80.2389, "dome": False},  # Dolphins
    "Gillette Stadium": {"lat": 42.0909, "lon": -71.2643, "dome": False},   # Patriots
    "MetLife Stadium": {"lat": 40.8136, "lon": -74.0744, "dome": False},    # Jets (also Giants)

    "Paycor Stadium": {"lat": 39.0954, "lon": -84.5161, "dome": False},     # Bengals
    "Cleveland Browns Stadium": {"lat": 41.5061, "lon": -81.6995, "dome": False},
    "Acrisure Stadium": {"lat": 40.4468, "lon": -80.0158, "dome": False},    # Steelers
    "M&T Bank Stadium": {"lat": 39.2779, "lon": -76.6227, "dome": False},    # Ravens

    "NRG Stadium": {"lat": 29.6847, "lon": -95.4107, "dome": True},          # Texans (retractable/indoor)
    "Lucas Oil Stadium": {"lat": 39.7601, "lon": -86.1639, "dome": True},    # Colts
    "EverBank Stadium": {"lat": 30.3239, "lon": -81.6373, "dome": False},    # Jaguars (ex TIAA Bank Field)
    "Nissan Stadium": {"lat": 36.1664, "lon": -86.7713, "dome": False},      # Titans

    "Empower Field at Mile High": {"lat": 39.7439, "lon": -105.0201, "dome": False},  # Broncos
    "Arrowhead Stadium": {"lat": 39.0490, "lon": -94.4839, "dome": False},             # Chiefs (GEHA Field)
    "Allegiant Stadium": {"lat": 36.0908, "lon": -115.1830, "dome": True},             # Raiders
    "SoFi Stadium": {"lat": 33.9535, "lon": -118.3387, "dome": True},                  # Chargers (also Rams)

    # NFC
    "Soldier Field": {"lat": 41.8623, "lon": -87.6167, "dome": False},       # Bears
    "Ford Field": {"lat": 42.3399, "lon": -83.0456, "dome": True},           # Lions
    "Lambeau Field": {"lat": 44.5013, "lon": -88.0622, "dome": False},       # Packers
    "U.S. Bank Stadium": {"lat": 44.9737, "lon": -93.2577, "dome": True},    # Vikings

    "AT&T Stadium": {"lat": 32.7473, "lon": -97.0945, "dome": True},         # Cowboys (retractable → treat as dome)
    "Lincoln Financial Field": {"lat": 39.9008, "lon": -75.1675, "dome": False},  # Eagles
    "MetLife Stadium (Giants)": {"lat": 40.8136, "lon": -74.0744, "dome": False},  # Giants alias
    "FedEx Field": {"lat": 38.9077, "lon": -76.8645, "dome": False},         # Commanders

    "Mercedes-Benz Stadium": {"lat": 33.7554, "lon": -84.4009, "dome": True},# Falcons
    "Bank of America Stadium": {"lat": 35.2251, "lon": -80.8529, "dome": False},  # Panthers
    "Caesars Superdome": {"lat": 29.9508, "lon": -90.0815, "dome": True},    # Saints
    "Raymond James Stadium": {"lat": 27.9759, "lon": -82.5033, "dome": False},# Buccaneers

    "Levi's Stadium": {"lat": 37.4030, "lon": -121.9700, "dome": False},     # 49ers
    "Lumen Field": {"lat": 47.5952, "lon": -122.3316, "dome": False},        # Seahawks
    "State Farm Stadium": {"lat": 33.5276, "lon": -112.2626, "dome": True},  # Cardinals (retractable → dome)
    "SoFi Stadium (Rams)": {"lat": 33.9535, "lon": -118.3387, "dome": True}, # Rams alias
}

# Common aliases → canonical name
STADIUM_ALIASES: Dict[str, str] = {
    "geha field at arrowhead stadium": "Arrowhead Stadium",
    "firstenergy stadium": "Cleveland Browns Stadium",
    "tiaa bank field": "EverBank Stadium",
    "new orleans superdome": "Caesars Superdome",
    "mercedes-benz superdome": "Caesars Superdome",
    "metlife stadium": "MetLife Stadium",            # Jets
    "metlife stadium (ny giants)": "MetLife Stadium (Giants)",  # Giants
    "giants stadium": "MetLife Stadium (Giants)",
}

# ------------------------------------------------------------------------
# 🧭 Timezone normalization (fixes offset-naive/aware bug)
# ------------------------------------------------------------------------
def _to_aware_utc(x) -> Optional[datetime]:
    if isinstance(x, datetime):
        return x.replace(tzinfo=timezone.utc) if x.tzinfo is None else x.astimezone(timezone.utc)
    s = str(x).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    try:
        dt_parsed = datetime.fromisoformat(s)
    except Exception:
        try:
            dt_parsed = pd.to_datetime(s, utc=True).to_pydatetime()  # type: ignore
        except Exception:
            return None
    if isinstance(dt_parsed, pd.Timestamp):
        dt_parsed = dt_parsed.to_pydatetime()  # type: ignore
    if dt_parsed.tzinfo is None:
        dt_parsed = dt_parsed.replace(tzinfo=timezone.utc)
    else:
        dt_parsed = dt_parsed.astimezone(timezone.utc)
    return dt_parsed


def _nearest_hour_index(times: list[str], kickoff_iso: str) -> int:
    k = _to_aware_utc(kickoff_iso)
    if k is None:
        return 0
    deltas = []
    for t in times:
        dt_t = _to_aware_utc(t)
        deltas.append(float("inf") if dt_t is None else abs((dt_t - k).total_seconds()))
    return int(deltas.index(min(deltas))) if deltas else 0


# ------------------------------------------------------------------------
# 🧭 Stadium lookup (fallbacks + optional CSV)
# ------------------------------------------------------------------------
def _lookup_stadium_coords(stadium: str, lat, lon, dome) -> tuple[Optional[float], Optional[float], bool]:
    """
    If lat/lon are missing, try:
      1) exact match in STADIUM_FALLBACKS
      2) alias mapping (case-insensitive)
      3) case-insensitive key match
      4) data/stadiums.csv (if present)
    Returns (lat, lon, dome_flag)
    """
    # already set → keep as-is
    try:
        lat_ok = float(lat)
        lon_ok = float(lon)
        if not pd.isna(lat_ok) and not pd.isna(lon_ok):
            return lat_ok, lon_ok, bool(dome)
    except Exception:
        pass

    name = (stadium or "").strip()

    # 1) exact
    if name in STADIUM_FALLBACKS:
        fb = STADIUM_FALLBACKS[name]
        return fb["lat"], fb["lon"], bool(fb.get("dome", False))

    # 2) alias
    alias_key = STADIUM_ALIASES.get(name.lower())
    if alias_key and alias_key in STADIUM_FALLBACKS:
        fb = STADIUM_FALLBACKS[alias_key]
        return fb["lat"], fb["lon"], bool(fb.get("dome", False))

    # 3) case-insensitive key match
    for k in STADIUM_FALLBACKS.keys():
        if k.lower() == name.lower():
            fb = STADIUM_FALLBACKS[k]
            return fb["lat"], fb["lon"], bool(fb.get("dome", False))

    # 4) CSV fallback
    csv_path = Path("data/stadiums.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if {"stadium", "lat", "lon"}.issubset(df.columns):
                m = df[df["stadium"].astype(str).str.lower() == name.lower()]
                if not m.empty:
                    row = m.iloc[0]
                    return float(row["lat"]), float(row["lon"]), bool(row.get("dome", False))
        except Exception:
            pass

    return None, None, bool(dome)


# ------------------------------------------------------------------------
# 🌦 Weather fetchers
# ------------------------------------------------------------------------
def _fetch_open_meteo(lat: float, lon: float, kickoff_iso: str) -> Dict[str, Any] | None:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "forecast_days": 2,
        "timezone": "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return None

    idx = _nearest_hour_index(times, kickoff_iso)
    out = {
        "temp_f": (hourly.get("temperature_2m", [None])[idx] * 9/5 + 32) if "temperature_2m" in hourly else None,
        "precip": hourly.get("precipitation", [None])[idx],
        "wind_mph": (hourly.get("wind_speed_10m", [None])[idx] * 0.621371) if "wind_speed_10m" in hourly else None,
        "source": "open_meteo",
    }
    return out


def _fetch_openweather(lat: float, lon: float, api_key: str) -> Dict[str, Any] | None:
    if not api_key:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        return {
            "temp_f": j.get("main", {}).get("temp"),
            "precip": (j.get("rain", {}) or {}).get("1h", 0.0),
            "wind_mph": (j.get("wind", {}) or {}).get("speed"),
            "source": "openweather",
        }
    except Exception:
        return None


# ------------------------------------------------------------------------
# 🔁 Pipeline
# ------------------------------------------------------------------------
def fetch_weather_for_games(df_games: pd.DataFrame, api_key: str | None = None) -> pd.DataFrame:
    rows = []

    for _, row in df_games.iterrows():
        stadium = str(row.get("stadium", "")).strip()
        lat = row.get("lat")
        lon = row.get("lon")
        dome = bool(row.get("dome", False))

        # 🏟️ Coordinate fallback
        lat, lon, dome_fb = _lookup_stadium_coords(stadium, lat, lon, dome)
        dome = bool(dome or dome_fb)

        # ⛔ Domes: synthetic neutral conditions
        if dome:
            wx = {"temp_f": 72.0, "precip": 0.0, "wind_mph": 0.0, "source": "dome"}
        else:
            kickoff = str(row.get("commence_time", ""))
            if lat is None or lon is None:
                wx = {"temp_f": None, "precip": None, "wind_mph": None, "source": "missing_coords"}
            else:
                wx = _fetch_open_meteo(lat=float(lat), lon=float(lon), kickoff_iso=kickoff)
                if wx is None and api_key:
                    wx = _fetch_openweather(float(lat), float(lon), api_key)
                wx = wx or {"temp_f": None, "precip": None, "wind_mph": None, "source": None}

        wx["event_id"] = row.get("event_id")
        wx["home_team"] = row.get("home_team")
        wx["away_team"] = row.get("away_team")
        wx["stadium"] = stadium
        wx["commence_time"] = row.get("commence_time")
        wx["dome"] = dome
        rows.append(wx)
        time.sleep(0.25)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------
# 🚀 CLI entry
# ------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games.csv")
    ap.add_argument("--write", default="data/weather.csv")
    ap.add_argument("--owm-key", default="")
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)

    try:
        games = pd.read_csv(args.games)
    except Exception:
        print("[fetch_weather] No games found → writing empty CSV")
        pd.DataFrame(columns=[
            "event_id", "home_team", "away_team", "stadium", "commence_time",
            "wind_mph", "temp_f", "precip", "dome", "source"
        ]).to_csv(args.write, index=False)
        return

    df = fetch_weather_for_games(games, api_key=args.owm_key)
    df.to_csv(args.write, index=False)
    print(f"[fetch_weather] ✅ wrote {len(df)} rows → {args.write}")


if __name__ == "__main__":
    main()
