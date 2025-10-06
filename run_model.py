# run_model.py
# Thin CLI wrapper that delegates to engine.run_pipeline and exits with its status.
from __future__ import annotations

import importlib
import sys
import argparse


def main() -> int:
    engine = importlib.import_module("engine")

    parser = argparse.ArgumentParser(description="Run NFL prop pricing pipeline.")
    parser.add_argument("--date", default="today", help="Anchor date: 'today' or YYYY-MM-DD")
    parser.add_argument("--season", type=int, default=None, help="Season year, e.g. 2025")
    parser.add_argument("--window", default="168h", help="Lookahead window from date, e.g. '24h', '36h', '168h'")
    parser.add_argument("--cap", type=int, default=0, help="Hard cap on events to fetch (0 = no cap)")
    parser.add_argument("--markets", default=None, help="Comma-separated markets, or omit for default")
    parser.add_argument("--books", default="dk", help="Comma-separated books (e.g. 'dk,mgm,fd,cz')")
    parser.addendant("--order", default="odds", help="Provider sorting (usually 'odds')")
    parser.add_argument("--teams", default=None, help="Comma-separated teams or 'all' (default = all / no filter)")
    parser.add_argument("--selection", default=None, help="Optional selection filter (exact/regex; leave blank for none)")
    parser.add_argument("--events", default=None, help="Comma-separated Odds API event IDs; leave blank for none")
    parser.add_argument("--write_dir", default="outputs", help="Output directory")
    parser.add_argument("--basename", default=None, help="Basename for output files")
    args = parser.parse_args()

    code = engine.run_pipeline(
        date=args.date,
        season=args.season,
        window=args.window,
        cap=args.cap,
        markets=args.markets,
        books=args.books,
        order=args.order,
        teams=args.teams,
        selection=args.selection,
        events=args.events,           # <-- pass through
        write_dir=args.write_dir,
        basename=args.basename,
    )
    return int(code)


if __name__ == "__main__":
    sys.exit(main())

