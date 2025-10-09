#!/usr/bin/env python3
import argparse, sys, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=False)
    args = ap.parse_args()
    pd.DataFrame(columns=["team","player","status"]).to_csv("data/injuries.csv", index=False)
    print("wrote data/injuries.csv rows=0 (stub)")
if __name__ == "__main__":
    main()
