# engine.py
from __future__ import annotations
import os, subprocess, shlex, sys
from pathlib import Path
from engine_adapters.providers import (
    run_nflverse, run_espn, run_nflgsis, run_apisports, run_msf
)

def _run(cmd: str):
    print(f"[engine] ▶ {cmd}", flush=True)
    proc = subprocess.run(shlex.split(cmd))
    if proc.returncode != 0:
        print(f"[engine] ✖ step failed (exit {proc.returncode})", flush=True)
        sys.exit(proc.returncode)

def _provider_chain(season: int, date: str | None):
    print("[engine] Provider order: nflverse -> ESPN -> NFLGSIS -> API-Sports -> MSF")
    for runner in (run_nflverse, run_espn, run_nflgsis, run_apisports, run_msf):
        res = runner(season, date)
        print(f"[engine] {res['source']} wrote={res.get('wrote',{})} "
              f"rows={res.get('rows',{})} notes={'; '.join(res.get('notes', []))}")
        if res.get("ok"):
            print(f"[engine] ✅ using {res['source']} outputs")
            os.environ["PROVIDER_USED"] = res["source"]
            return
    print("[engine] ⚠ No provider produced usable data — continuing with builders")

def run_pipeline(*, season: str, date: str = "", books=None, markets=None):
    # Ensure dirs exist
    for p in ("data", "outputs", "logs", "outputs/metrics"):
        Path(p).mkdir(parents=True, exist_ok=True)

    # 1) Provider chain (best-effort)
    try:
        _provider_chain(int(season), date or None)
    except Exception as e:
        print(f"[engine] provider chain error (non-fatal): {e}")

    # 2) Build team + player form + metrics
    _run(f"python scripts/make_team_form.py --season {season}")
    _run(f"python scripts/make_player_form.py --season {season}")
    _run(f"python scripts/make_metrics.py --season {season}")

    # 3) Props (Odds API)
    _books = books or ["draftkings", "fanduel", "betmgm", "caesars"]
    _mkts  = markets or []
    mk = ",".join(_mkts)
    _run(f"python scripts/fetch_props_oddsapi.py --books {','.join(_books)} "
         f"{'--markets '+mk if mk else ''} --date {date} --out outputs/props_raw.csv")

    # 4) Pricing and predictors
    _run("python scripts/pricing.py --props outputs/props_raw.csv")
    _run(f"python -m scripts.models.run_predictors --season {season}")

    print("[engine] ✅ pipeline complete.")
    return 0

def main(**kwargs):
    return run_pipeline(**kwargs)
