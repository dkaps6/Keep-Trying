from __future__ import annotations
import os, subprocess, shlex
from pathlib import Path
from engine_adapters.providers import (
    run_nflverse, run_espn, run_nflgsis, run_apisports, run_msf
)

REPO = Path(__file__).resolve().parent

def _run(cmd: str) -> None:
    print(f"[engine] $ {cmd}", flush=True)
    proc = subprocess.run(shlex.split(cmd), cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def _provider_chain(season: int, date: str|None):
    print("[engine] Provider order: nflverse -> ESPN -> NFLGSIS -> API-Sports -> MSF")
    for runner in (run_nflverse, run_espn, run_nflgsis, run_apisports, run_msf):
        res = runner(season, date)
        print(f"[engine] {res['source']} wrote={res.get('wrote',{})} rows={res.get('rows',{})} notes={'; '.join(res.get('notes', []))}")
        if res.get("ok"):
            print(f"[engine] ✅ using {res['source']} outputs")
            os.environ["PROVIDER_USED"] = res["source"]
            return
    print("[engine] ⚠️ No provider produced usable data — metrics may be empty")

def run_pipeline(date: str = "", season: str = "", **kwargs):
    if season:
        try:
            _provider_chain(int(season), date or None)
        except Exception as e:
            print(f"[engine] provider chain error: {e}")
    else:
        print("[engine] WARN: season missing; skipping provider chain")

    for p in ("data", "outputs", "logs"):
        Path(p).mkdir(parents=True, exist_ok=True)

    if season:
        _run(f"python scripts/make_team_form.py --season {season}")
        _run(f"python scripts/make_player_form.py --season {season}")
        _run(f"python scripts/make_metrics.py --season {season}")
    else:
        print("[engine] WARN: season not provided; metrics skipped")

    books = kwargs.get("books") or ["draftkings","fanduel","betmgm","caesars"]
    markets = kwargs.get("markets") or []
    mk = ",".join(markets)
    bk = ",".join(books)
    _run(f"python scripts/fetch_props_oddsapi.py --books {bk} --markets {mk} --date {date} --out outputs/props_raw.csv")

    _run(f"python -m scripts.models.run_predictors --season {season}")

    print("[engine] pipeline complete.")
    return 0

def main(**kwargs):
    return run_pipeline(**kwargs)
