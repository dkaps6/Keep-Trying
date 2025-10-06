# run_model.py
from __future__ import annotations

import argparse
import importlib
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end pricing pipeline")
    ap.add_argument("--date", default="today", help="target date (ISO YYYY-MM-DD or 'today')")
    ap.add_argument("--season", type=int, required=True, help="season year, e.g. 2025")
    ap.add_argument(
        "--write",
        default="outputs",
        help="directory to write outputs (CSV/XLSX/SUMMARY.md)",
    )
    args = ap.parse_args()

    # Load engine dynamically so this file stays tiny and the engine can evolve.
    try:
        engine = importlib.import_module("engine")
    except Exception as e:
        print(f"[run_model] failed to import engine: {e}", file=sys.stderr)
        raise

    try:
        print(f"[run_model] Loaded engine module from: {engine.__file__}")
    except Exception:
        pass

    # Execute the pipeline
    try:
        print("[run_model] starting pipeline…")
        df = engine.run_pipeline(target_date=args.date, season=args.season, out_dir=args.write)
        n = 0 if df is None else len(df)
        print(f"[run_model] pipeline completed. Wrote outputs to: {args.write}  (rows={n})")
    except SystemExit as e:
        # let explicit SystemExit (e.g., no props available) propagate as nonzero
        raise
    except Exception as e:
        print(f"[run_model] pipeline crashed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
