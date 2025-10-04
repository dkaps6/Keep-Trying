import argparse
from datetime import datetime
from engine import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="today", help="target date (YYYY-MM-DD or 'today')")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--write", default="outputs", help="output directory")
    args = parser.parse_args()

    target_date = datetime.utcnow().strftime("%Y-%m-%d") if args.date == "today" else args.date
    res = run_pipeline(target_date=target_date, season=args.season, out_dir=args.write)
    print(f"✅ Wrote CSVs to {args.write}: game_lines.csv, props_priced.csv")

if __name__ == "__main__":
    main()
