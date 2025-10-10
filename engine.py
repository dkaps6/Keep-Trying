# engine.py
from __future__ import annotations
import os, sys, shlex, subprocess
from pathlib import Path
from datetime import datetime

# if you already have these adapters, they’re used as-is
try:
    from engine_adapters.providers import run_nflverse, run_espn, run_nflgsis, run_apisports, run_msf
except Exception:
    # soft stubs so pipeline still runs if adapters aren’t present
    def _stub(name):
        def _run(season:int, date:str|None):
            return {"ok": False, "source": name, "rows": {}, "wrote": {}, "notes": [f"{name} adapter missing"]}
        return _run
    run_nflverse=_stub("nflverse"); run_espn=_stub("espn")
    run_nflgsis=_stub("nflgsis"); run_apisports=_stub("apisports"); run_msf=_stub("msf")

def _run(cmd: str):
    print(f"[engine] ▶ {cmd}", flush=True)
    rc = subprocess.call(shlex.split(cmd))
    if rc != 0:
        print(f"[engine] ✖ step failed (exit {rc})", flush=True)
        sys.exit(rc)

def _size(p: str) -> str:
    fp = Path(p)
    return f"{fp.stat().st_size}B" if fp.exists() else "MISSING"

def _provider_chain(season: int, date: str | None):
    print("[engine] Provider order: nflverse → ESPN → NFLGSIS → API-Sports → MSF")
    for runner in (run_nflverse, run_espn, run_nflgsis, run_apisports, run_msf):
        try:
            res = runner(season, date)
        except Exception as e:
            res = {"ok": False, "source": runner.__name__, "notes": [str(e)]}
        print(f"[engine] provider={res.get('source')} ok={res.get('ok')} notes={'; '.join(res.get('notes', []))}")
        if res.get("ok"):
            os.environ["PROVIDER_USED"] = res.get("source", "")
            return
    print("[engine] ⚠ no external provider succeeded; will rely on builders")

def run_pipeline(*, season: str, date: str = "", books=None, markets=None):
    # make dirs
    for d in ("data", "outputs", "outputs/metrics", "logs"):
        Path(d).mkdir(parents=True, exist_ok=True)

    # show keys presence (masked)
    print(f"[engine] ODDS_API_KEY={'set' if os.getenv('ODDS_API_KEY') else 'missing'}  ESPN_COOKIE={'set' if os.getenv('ESPN_COOKIE') else 'missing'}")

    # 1) upstream data providers (best-effort)
    try:
        _provider_chain(int(season), date or None)
    except Exception as e:
        print(f"[engine] provider chain error (non-fatal): {e}")

    # 2) build engineered tables
    _run(f"python scripts/make_team_form.py --season {season}")
    print(f"[engine]   data/team_form.csv → {_size('data/team_form.csv')}")
    _run(f"python scripts/make_player_form.py --season {season}")
    print(f"[engine]   data/player_form.csv → {_size('data/player_form.csv')}")
    _run(f"python scripts/make_metrics.py --season {season}")
    print(f"[engine]   data/metrics_ready.csv → {_size('data/metrics_ready.csv')}")

    # 3) props (Odds API)
    b = ",".join(books or ["draftkings","fanduel","betmgm","caesars"])
    m = ",".join(markets or [])
    _run(f"python scripts/fetch_props_oddsapi.py --books {b} {'--markets '+m if m else ''} --date {date} --out outputs/props_raw.csv")
    print(f"[engine]   outputs/props_raw.csv → {_size('outputs/props_raw.csv')}")
    print(f"[engine]   outputs/odds_game.csv → {_size('outputs/odds_game.csv')}")

    # 4) pricing + predictors
    _run("python scripts/pricing.py --props outputs/props_raw.csv")
    _run(f"python -m scripts.models.run_predictors --season {season}")
    print(f"[engine]   outputs/master_model_predictions.csv → {_size('outputs/master_model_predictions.csv')}")

    rid = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    print(f"[engine] ✅ complete (run_id={rid})")
    return 0

# optional CLI so you can run `python engine.py --season 2025 --date 2025-10-12`
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--date", default="")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    a = ap.parse_args()
    run_pipeline(
        season=a.season,
        date=a.date,
        books=[b.strip() for b in a.books.split(",") if b.strip()],
        markets=[m.strip() for m in a.markets.split(",") if m.strip()] or None,
    )
