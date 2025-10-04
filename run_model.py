# run_model.py
import argparse
import importlib
import sys
import traceback

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="today")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--write", default="outputs")
    return p.parse_args()

def main():
    args = parse_args()

    # Import the engine module safely
    try:
        engine = importlib.import_module("engine")
    except Exception as e:
        print("❌ Failed to import engine module:", e)
        traceback.print_exc()
        sys.exit(1)

    # Ensure run_pipeline exists
    if not hasattr(engine, "run_pipeline"):
        print("❌ engine.run_pipeline not found. Module exports:",
              [k for k in dir(engine) if not k.startswith("_")])
        sys.exit(1)

    # Run the pipeline
    result = engine.run_pipeline(
        target_date=args.date,
        season=args.season,
        out_dir=args.write,
    )
    print("✅ Pipeline completed. Wrote outputs to:", args.write)

if __name__ == "__main__":
    main()
