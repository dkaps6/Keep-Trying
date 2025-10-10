from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--season", type=int, required=True); a=ap.parse_args()
    Path("data").mkdir(exist_ok=True)

    tf = pd.read_csv("data/team_form.csv") if Path("data/team_form.csv").exists() else pd.DataFrame()
    pf = pd.read_csv("data/player_form.csv") if Path("data/player_form.csv").exists() else pd.DataFrame()
    if tf.empty or pf.empty:
        (Path("data")/"metrics_ready.csv").write_text("")
        print("[metrics] missing inputs; wrote empty metrics_ready.csv"); return

    # normalize keys
    for df in (tf,pf):
        if "team" in df.columns: df["team"]=df["team"].astype(str)

    mf = pf.merge(tf, on="team", how="left", suffixes=("","_team"))
    mf.to_csv("data/metrics_ready.csv", index=False)
    print(f"[metrics] rows={len(mf)} → data/metrics_ready.csv")

if __name__=="__main__":
    main()
