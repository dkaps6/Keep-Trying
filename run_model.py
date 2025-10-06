# run_model.py
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from typing import Dict, Any


def _parse_csv_list(val: str | None) -> list[str]:
    if not val:
        return []
    return [x.strip() for x in str(val).split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end pricing pipeline")
    ap.add_argument("--date", default="today", help="target date (YYYY-MM-DD or 'today')")
    ap.add_argument("--season", type=int, required=True, help="season year, e.g. 2025")
    ap.add_argument(
        "--write",
        default="outputs",
        help="directory to write outputs (CSV/XLSX/SUMMARY.md)",
    )

    # Optional filters (nargs='?' allows the flag to appear with no value)
    ap.add_argument("--teams", nargs="?", default="", help="comma-separated team names to filter (e.g. Chiefs,Jaguars)")
    ap.add_argument("--events", nargs="?", default="", help="comma-separated event IDs to filter (Odds API event ids)")
    ap.add_argument("--markets", nargs="?", default="", help="comma-separated markets (e.g. receiving_yards,player_receptions)")
    ap.add_argument("--provider-order", nargs="?", default="", help="override provider order, e.g. 'odds,dk' or 'dk,odds'")

    args = ap.parse_args()

    # Import engine dynamically
    try:
        engine = importlib.import_module("engine")
    except Exception as e:
        print(f"[run_model] failed to import engine: {e}", file=sys.stderr)
        raise
    try:
        print(f"[run_model] Loaded engine module from: {engine.__file__}")
    except Exception:
        pass

    # Build optional passthrough kwargs from CLI
    requested: Dict[str, Any] = {
        "teams": _parse_csv_list(args.teams),
        "events": _parse_csv_list(args.events),
        "markets": _parse_csv_list(args.markets),
        "provider_order": args.provider_order or "",
    }
    # Drop empties
    requested = {k: v for k, v in requested.items() if (v if not isinstance(v, str) else v.strip())}

    # Only pass kwargs the current engine.run_pipeline actually accepts
    try:
        sig = inspect.signature(engine.run_pipeline)
        allowed = {p.name for p in sig.parameters.values()}
        passthrough = {k: v for k, v in requested.items() if k in allowed}
    except Exception:
        # If introspection fails, don't pass any optional kwargs
        passthrough = {}

    print("[run_model] starting pipeline…")
    try:
        df = engine.run_pipeline(
            target_date=args.date,
            season=args.season,
            out_dir=args.write,
            **passthrough,
        )
        n = 0 if df is None else len(df)
        print(f"[run_model] pipeline completed. Wrote outputs to: {args.write}  (rows={n})")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[run_model] pipeline crashed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

