# engine.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

# 1) Player props (Odds API / DK)
from scripts.odds_api import fetch_props

# 2) External metrics (team_form via nfl_data_py, weather via NWS)
#    NOTE: this main() accepts only season (no date)
from scripts.fetch_all import main as fetch_metrics

# 3) Pricing/model + writer (compat helper)
from scripts.pricing import price_props, write_outputs


def run_pipeline(
    target_date: str,
    season: int,
    out_dir: str = "outputs",
    teams: list[str] | None = None,
    events: list[str] | None = None,
    markets: list[str] | None = None,
    provider_order: str | None = None,
) -> pd.DataFrame:
    """
    End-to-end:
      - fetch player props
      - fetch external metrics (team_form + weather)
      - price with model
      - write outputs and return priced DataFrame
    """
    print(f"[engine] fetching props… date={target_date} season={season}")

    # Only pass filters that were provided
    props_kwargs = {}
    if teams:           props_kwargs["teams"] = teams
    if events:          props_kwargs["events"] = events
    if markets:         props_kwargs["markets"] = markets
    if provider_order:  props_kwargs["provider_order"] = provider_order

    props = fetch_props(date=target_date, season=season, **props_kwargs)
    if props is None or props.empty:
        raise SystemExit("[engine] No props available to price (props fetch returned 0 rows).")

    print(f"[engine] fetching metrics (team_form + weather)… season={season}")
    # IMPORTANT: no 'date' here
    fetch_metrics(season=season)

    print("[engine] pricing…")
    priced_clean = price_props(props)  # also writes standard outputs internally if you left that logic

    # Ensure the “clean” table is written the way the old engine expected
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_outputs(priced_clean, out_dir=out_dir, basename="props_priced_clean")

    # Optional summary (best-effort)
    try:
        from scripts.validate_outputs import summarize
        summarize(f"{out_dir}/props_priced_clean.csv", top_k=15, min_edge_mark=0.01, write_md=f"{out_dir}/SUMMARY.md")
    except Exception as e:
        print(f"[engine] summary skipped: {e}")

    return priced_clean
