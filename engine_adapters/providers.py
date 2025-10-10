from __future__ import annotations
import os, json, requests, pandas as pd
from pathlib import Path

OUT = Path("external"); OUT.mkdir(exist_ok=True)
NFLV_OUT = OUT/"nflverse_bundle"/"outputs"; NFLV_OUT.mkdir(parents=True, exist_ok=True)

def _ok(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def run_nflverse(season: int, date: str|None):
    # nfl_data_py is our primary
    try:
        import nfl_data_py as nfl
        pbp = nfl.import_pbp_data([season])
        sched = nfl.import_schedules([season])
        # Write mirrors for builders if you want to inspect
        (NFLV_OUT/"pbp.parquet").parent.mkdir(parents=True, exist_ok=True)
        pbp.to_parquet(NFLV_OUT/"pbp.parquet", index=False)
        sched.to_csv(NFLV_OUT/"schedules.csv", index=False)
        return {"ok": True, "source":"nflverse", "notes":["pbp+sched from nfl_data_py"], "rows":{"pbp":len(pbp),"sched":len(sched)}}
    except Exception as e:
        return {"ok": False, "source":"nflverse", "notes":[f"nfl_data_py failed: {e}"]}

def run_espn(season: int, date: str|None):
    # use public site.api endpoints; cookie optional but improves reliability
    try:
        cookie = os.getenv("ESPN_COOKIE","")
        headers = {"Cookie": cookie} if cookie else {}
        # scoreboards for season (coarse); this is enough for builders to align event_ids if needed
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?year={season}"
        r = requests.get(url, headers=headers, timeout=20); r.raise_for_status()
        data = r.json()
        pd.DataFrame(data.get("events",[])).to_json(NFLV_OUT/"espn_scoreboard.json", orient="records")
        return {"ok": True, "source":"ESPN", "rows":{"events":len(data.get('events',[]))}, "notes":["scoreboard pulled"]}
    except Exception as e:
        return {"ok": False, "source":"ESPN", "notes":[str(e)]}

def run_nflgsis(season: int, date: str|None):
    # Placeholder: many GSIS feeds require licensed endpoints. We just prove auth is present.
    if not (os.getenv("NFLGSIS_USERNAME") and os.getenv("NFLGSIS_PASSWORD")):
        return {"ok": False, "source":"NFLGSIS", "notes":["missing creds"]}
    return {"ok": False, "source":"NFLGSIS", "notes":["skipped (no public endpoint here)"]}

def run_msf(season: int, date: str|None):
    if not os.getenv("MSF_KEY"):
        return {"ok": False, "source":"MySportsFeeds", "notes":["missing key"]}
    # you can add real pulls here; we only mark as present
    return {"ok": False, "source":"MySportsFeeds", "notes":["stub (wire your paid endpoint here)"]}

def run_apisports(season: int, date: str|None):
    if not os.getenv("APISPORTS_KEY"):
        return {"ok": False, "source":"API-Sports", "notes":["missing key"]}
    return {"ok": False, "source":"API-Sports", "notes":["stub (used only as last fallback)"]}
