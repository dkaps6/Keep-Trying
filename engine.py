# engine.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from scripts.fetch_all import main as fetch_main
from scripts.normalize_props import normalize
from scripts.pricing import price_props, write_outputs

ROOT = Path(".")
OUT = ROOT / "outputs"
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

def run_pipeline(target_date: str="today", season: str="2025", write: bool=True) -> pd.DataFrame:
    print("[engine] fetching props...")
    fetch_main(date=target_date, season=season)  # will fail-fast if no props

    print("[engine] normalizing...")
    raw = pd.read_csv(OUT/"props_raw.csv")
    norm = normalize(raw)

    # optional id map join (do not drop unmatched)
    id_map_path = DATA/"id_map.csv"
    if id_map_path.exists():
        try:
            mp = pd.read_csv(id_map_path)
            if set(["player_name","player_id"]).issubset(mp.columns):
                norm = norm.merge(mp.rename(columns={"player_name":"player"}), on="player", how="left")
                norm["needs_mapping"] = norm["player_id"].isna()
        except Exception as e:
            print("[engine] id_map read failed:", e)

    print(f"[engine] pricing {len(norm)} consolidated rows...")
    priced = price_props(norm)

    if write:
        write_outputs(priced, OUT)

    print("[engine] done. wrote to:", OUT)
    return priced

