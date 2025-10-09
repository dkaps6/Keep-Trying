# engine.py
from __future__ import annotations
import subprocess, shlex
from pathlib import Path

REPO = Path(__file__).resolve().parent

def _run(cmd: str) -> None:
    print(f"[engine] $ {cmd}", flush=True)
    proc = subprocess.run(shlex.split(cmd), cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

def run_pipeline(
    date: str = "",
    season: str = "",
    hours: int = 0,
    cap: int = 0,
    books=None,
    markets=None,
    order=None,
    teams=None,
    selection=None,
    write_dir: str = "outputs",
    basename: str | None = None,
    odds_game_df=None,
    odds_props_df=None,
):
    # Optional: persist consensus odds snapshots if given
    try:
        import pandas as pd
        if getattr(odds_game_df, "empty", True) is False:
            p = REPO / "data" / "odds_game_consensus.csv"; p.parent.mkdir(parents=True, exist_ok=True)
            odds_game_df.to_csv(p, index=False); print(f"[engine] wrote {p} ({len(odds_game_df)})")
        if getattr(odds_props_df, "empty", True) is False:
            p = REPO / "data" / "odds_props_consensus.csv"; p.parent.mkdir(parents=True, exist_ok=True)
            odds_props_df.to_csv(p, index=False); print(f"[engine] wrote {p} ({len(odds_props_df)})")
    except Exception as e:
        print(f"[engine] consensus write skipped: {e}")

    # 1) Metrics
    if season:
        _run(f"python scripts/make_metrics.py --season {shlex.quote(str(season))}")
    else:
        print("[engine] WARN: season not provided; metrics step skipped")

    # 2) Props
    books_arg = ",".join(books) if books else "draftkings,fanduel,betmgm,caesars"
    mk_arg = ",".join(markets) if markets else ""
    date_arg = str(date) if date else ""
    _run(f"python scripts/fetch_props_oddsapi.py --books {shlex.quote(books_arg)} --markets {shlex.quote(mk_arg)} --date {shlex.quote(date_arg)} --out outputs/props_raw.csv")

    # 3) Pricing
    if season:
        _run(f"python scripts/pricing.py --season {shlex.quote(str(season))}")
    else:
        _run("python scripts/pricing.py")

    print("[engine] pipeline complete.")
    return 0

def main(**kwargs):
    return run_pipeline(**kwargs)
