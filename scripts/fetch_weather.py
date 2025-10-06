# scripts/fetch_weather.py
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import pandas as pd
import requests

UA = {"User-Agent": "Keep-Trying/1.0 (github actions; contact: none)"}

def _read_stadiums() -> pd.DataFrame:
    p = Path("data/stadiums.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"team","lat","lon"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    return df[["team","lat","lon"]].copy()

def _read_events() -> pd.DataFrame:
    p = Path("outputs/game_lines.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # expected columns: event_id, home_team, commence_time (ISO8601 UTC)
    return df[["event_id","home_team","commence_time"]].dropna()

def _iso_to_dt_utc(s: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(dt.timezone.utc)
    except Exception:
        return None

def _round_to_hour(t: dt.datetime) -> dt.datetime:
    return t.replace(minute=0, second=0, microsecond=0)

def _precip_from_forecast(text: str) -> str:
    s = (text or "").lower()
    if any(w in s for w in ["snow","flurries","blowing snow"]): return "snow"
    if any(w in s for w in ["rain","showers","thunderstorm","drizzle"]): return "rain"
    return "none"

def _nws_hourly(lat: float, lon: float) -> pd.DataFrame:
    try:
        r = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=UA, timeout=20)
        r.raise_for_status()
        hourly_url = r.json()["properties"]["forecastHourly"]
        r2 = requests.get(hourly_url, headers=UA, timeout=20)
        r2.raise_for_status()
        periods = r2.json()["properties"]["periods"]
        df = pd.json_normalize(periods)
        # columns: startTime, temperature, windSpeed, shortForecast
        return df
    except Exception:
        return pd.DataFrame()

def fetch_weather() -> pd.DataFrame:
    stad = _read_stadiums()
    ev   = _read_events()
    if stad.empty or ev.empty:
        return pd.DataFrame(columns=["event_id","wind_mph","temp_f","precip"])

    ev = ev.merge(stad.rename(columns={"team":"home_team"}), on="home_team", how="left")
    out_rows = []
    for _, row in ev.iterrows():
        event_id = row["event_id"]
        lat, lon = row["lat"], row["lon"]
        kickoff  = _iso_to_dt_utc(str(row["commence_time"]))
        if kickoff is None or pd.isna(lat) or pd.isna(lon):
            continue
        kickoff_hr = _round_to_hour(kickoff)

        hourly = _nws_hourly(float(lat), float(lon))
        if hourly.empty or "startTime" not in hourly.columns:
            continue

        hourly["t"] = pd.to_datetime(hourly["startTime"], utc=True)
        # nearest period at/after kickoff hour
        h = hourly.loc[hourly["t"] >= pd.Timestamp(kickoff_hr)].head(1)
        if h.empty:
            h = hourly.tail(1)

        temp_f = h["temperature"].iloc[0] if "temperature" in h.columns else None
        wind_s = h["windSpeed"].iloc[0] if "windSpeed" in h.columns else ""
        wind_mph = None
        try:
            wind_mph = int(str(wind_s).split()[0])
        except Exception:
            wind_mph = None
        short = h["shortForecast"].iloc[0] if "shortForecast" in h.columns else ""
        precip = _precip_from_forecast(short)

        out_rows.append({"event_id": event_id, "wind_mph": wind_mph, "temp_f": temp_f, "precip": precip})

    return pd.DataFrame(out_rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="data/weather.csv")
    args = ap.parse_args()
    df = fetch_weather()
    Path(args.write).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.write, index=False)
    print(f"[weather] wrote {len(df)} rows to {args.write}")

if __name__ == "__main__":
    main()
