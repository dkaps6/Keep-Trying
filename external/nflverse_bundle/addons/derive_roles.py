#!/usr/bin/env python3
import argparse, sys, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    args = ap.parse_args()
    try:
        import nfl_data_py as nfl
        years=[int(args.season)]
        wk = nfl.import_weekly_data(years)
        if wk.empty:
            raise RuntimeError("import_weekly_data empty")
        roles = wk.groupby(["season","team","player_display_name"], as_index=False).agg(
            targets=("targets","sum"),
            rush_att=("rushing_attempts","sum"),
            rec=("receptions","sum")
        )
    except Exception as e:
        print(f"[derive_roles] skipped: {e}", file=sys.stderr)
        roles = pd.DataFrame(columns=["season","team","player_display_name","targets","rush_att","rec"])
    roles.to_csv("outputs/proe/roles.csv", index=False)
    print("wrote outputs/proe/roles.csv rows=", len(roles))
if __name__ == "__main__":
    main()
