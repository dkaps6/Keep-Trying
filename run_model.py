#!/usr/bin/env python3
"""
Entry point for the slate pipeline.

Adds optional filters so you can run the full slate or only specific games:
  --teams  "Jets,Cowboys"      (keeps events where either team name matches)
  --events "6e206c...,ff8222"  (keeps only these Odds API event IDs)

We pass filters via env vars so any module that fetches odds (scripts/odds_api.py)
will see them even if called indirectly.
"""
import os
import importlib
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="today", help="logical date (usually 'today')")
    p.add_argument("--season", default="2025", help="season tag used in outputs/metrics")
    p.add_argument("--write", default="outputs", help="directory to write outputs to")

    # NEW: selection filters
    p.add_argument("--teams", default="", help="Comma-separated team substrings to include (e.g., 'Jets,Cowboys')")
    p.add_argument("--events", default="", help="Comma-separated Odds API event IDs to include")

    args = p.parse_args()

    if args.teams:
        os.environ["ODDS_API_INCLUDE_TEAMS"] = args.teams
    if args.events:
        os.environ["ODDS_API_INCLUDE_EVENTS"] = args.events

    # Allow both the new and old engine signatures
    engine = importlib.import_module("engine")
    try:
        engine.run_pipeline(target_date=args.date, season=args.season, write_outputs=args.write)
    except TypeError:
        engine.run_pipeline(args.date, args.season, args.write)


if __name__ == "__main__":
    main()
