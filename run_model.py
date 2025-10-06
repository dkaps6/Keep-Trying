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

    # Optional filters (passed only if engine.run_pipeline accepts them)
    ap.add_argument("--teams", nargs="?", default="", help="comma-separated team names (e.g. Chiefs,Jaguars)")
    ap.add_argument("--events", nargs="?", default="", help="comma-separated event IDs")
    ap.add_argument("--markets", nargs="?", default="", help="comma-separated markets (e.g. receiving_yards,player_receptions)")
    ap.add_argument("--provider-order", nargs="?", default="", help="override provider order, e.g. 'odds,dk' or 'dk,odds'")

    args = ap.parse_args()

    # Import engine
    try:
        engine = importlib.import_module("engine")
        print(f"[run_model] Loaded engine module from: {engine.__file__}")
    except Exception as e:
        print(f"[run_model] failed to import engine: {e}", file=sys.stderr)
        raise

    # Build optional passthroughs from CLI
    requested: Dict[str, Any] = {
        "teams": _parse_csv_list(args.teams),
        "events": _parse_csv_list(args.events),
        "markets": _parse_csv_list(args.markets),
        "provider_order": (args.provider_order or "").strip(),
    }
    requested = {k: v for k, v in requested.items() if (v if not isinstance(v, str) else v)}

    # Introspect engine.run_pipeline and adapt argument names
    try:
        sig = inspect.signature(engine.run_pipeline)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    kwargs: Dict[str, Any] = {}

    # Date param name
    if "target_date" in params:
        kwargs["target_date"] = args.date
    elif "date" in params:
        kwargs["date"] = args.date
    else:
        # if engine doesn't take date, just omit it
        pass

    # Season (most engines require this)
    if "season" in params:
        kwargs["season"] = args.season

    # Output directory param name
    if "out_dir" in params:
        kwargs["out_dir"] = args.write
    elif "write_outputs" in params:
        kwargs["write_outputs"] = args.write
    elif "write" in params:
        kwargs["write"] = args.write
    # else: engine handles writing internally; omit

    # Add optional filters only if the engine accepts them
    for k, v in requested.items():
        if k in params:
            kwargs[k] = v

    print("[run_model] starting pipeline…")
    try:
        df = engine.run_pipeline(**kwargs)
        n = 0 if df is None else len(df)
        print(f"[run_model] pipeline completed. Wrote outputs to: {args.write}  (rows={n})")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[run_model] pipeline crashed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
