# scripts/providers/espn_pbp.py
import pandas as pd, requests, io

def pbp(season: int):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    games = []
    for ev in data.get("events", []):
        gid = ev.get("id")
        home = ev.get("competitions", [{}])[0].get("competitors", [{}])[0].get("team", {}).get("abbreviation")
        away = ev.get("competitions", [{}])[0].get("competitors", [{}])[1].get("team", {}).get("abbreviation")
        games.append({"game_id": gid, "week": ev.get("week", {}).get("number"), "home_team": home, "away_team": away})
    return pd.DataFrame(games)
