#!/usr/bin/env python3
import argparse, sys, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    args = ap.parse_args()
    season = int(args.season)
    try:
        import nfl_data_py as nfl
        cap = min(season, 2024)
        part = nfl.import_participation(years=[cap])
        if part is None or part.empty:
            raise RuntimeError("import_participation returned empty")
        out = part.groupby(["season","team"], as_index=False).agg(offense_snaps=("offense_pct","mean"))
    except Exception as e:
        print(f"[aggregate_box_counts] skipped: {e}", file=sys.stderr)
        out = pd.DataFrame(columns=["season","team","offense_snaps"])
    out.to_csv("outputs/proe/box_counts.csv", index=False)
    print("wrote outputs/proe/box_counts.csv rows=", len(out))
if __name__ == "__main__":
    main()
