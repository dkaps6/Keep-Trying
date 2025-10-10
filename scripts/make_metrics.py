from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path
def main(season: int):
    Path('data').mkdir(parents=True, exist_ok=True)
    pf = pd.read_csv('data/player_form.csv') if Path('data/player_form.csv').exists() else pd.DataFrame()
    tf = pd.read_csv('data/team_form.csv') if Path('data/team_form.csv').exists() else pd.DataFrame()
    out = pf.merge(tf, on='team', how='left') if not pf.empty else tf
    out.to_csv('data/metrics_ready.csv', index=False)
    print(f"[make_metrics] rows={len(out)} → data/metrics_ready.csv")
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--season', type=int, required=True); a=ap.parse_args(); main(a.season)
