# run_model.py
import argparse
from engine import run_pipeline

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="today")
    p.add_argument("--season", default="2025")
    p.add_argument("--write", default="outputs")
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    run_pipeline(target_date=args.date, season=args.season, write=True)
