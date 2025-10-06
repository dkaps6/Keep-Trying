# scripts/validate_outputs.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ONE_LINE_FMT = (
    "props={props}  avg_edge={avg:.2%}  >=1%:{ge1:.1%}  >=4%:{ge4:.1%}  >=6%:{ge6:.1%}  "
    "greens={greens}  ambers={ambers}  reds={reds}"
)

KEEP_TOP = [
    "event_id","player","team","defense_team","market","line",
    "bet_side","edge_abs","color","bookmaker",
    "over_odds","under_odds","p_market_over","p_over_blend","fair_over_odds",
]

def _safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def summarize(clean_path: str, top_k: int, min_edge_mark: float, write_md: str) -> int:
    p = Path(clean_path)
    if not p.exists():
        print(f"[validate] file not found: {clean_path}")
        return 0

    df = pd.read_csv(p)
    if df.empty:
        print("[validate] no rows in props_priced_clean.csv")
        return 0

    # normalize
    for c in ("edge_abs","p_market_over","p_over_blend"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["market"] = df["market"].astype(str)
    if "color" not in df.columns:
        df["color"] = "RED"

    props = len(df)
    avg_edge = float(df["edge_abs"].mean(skipna=True)) if "edge_abs" in df else 0.0
    ge1 = float((df["edge_abs"] >= 0.01).mean()) if "edge_abs" in df else 0.0
    ge4 = float((df["edge_abs"] >= 0.04).mean()) if "edge_abs" in df else 0.0
    ge6 = float((df["edge_abs"] >= 0.06).mean()) if "edge_abs" in df else 0.0

    greens = int((df.get("color","RED") == "GREEN").sum())
    ambers = int((df.get("color","RED") == "AMBER").sum())
    reds   = int((df.get("color","RED") == "RED").sum())

    # one-liner for logs
    print("[summary]", ONE_LINE_FMT.format(
        props=props, avg=avg_edge, ge1=ge1, ge4=ge4, ge6=ge6,
        greens=greens, ambers=ambers, reds=reds
    ))

    # by market
    by_market = (df.groupby("market")["edge_abs"]
                   .agg(props="count", avg_edge="mean", p95=lambda s: s.quantile(0.95))
                   .reset_index().sort_values(["avg_edge","props"], ascending=[False, False]))
    # top edges table
    top = (df.sort_values("edge_abs", ascending=False)
             .loc[df["edge_abs"].ge(min_edge_mark)]
             .head(top_k)
             .reindex(columns=[c for c in KEEP_TOP if c in df.columns]))
    top_path = Path("outputs/top_edges.csv")
    top_path.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(top_path, index=False)

    # write a small markdown
    md = [
        "# Run Summary",
        "",
        f"- **Props priced:** {props}",
        f"- **Average edge:** {avg_edge:.2%}",
        f"- **≥1%:** {ge1:.1%} &nbsp;&nbsp; **≥4%:** {ge4:.1%} &nbsp;&nbsp; **≥6%:** {ge6:.1%}",
        f"- **GREEN/AMBER/RED:** {greens}/{ambers}/{reds}",
        "",
        "## By market",
        "",
        by_market.to_markdown(index=False),
        "",
        f"## Top edges (min edge {min_edge_mark:.0%}, top {top_k})",
        "",
        top.to_markdown(index=False) if not top.empty else "_none_",
        "",
        f"_CSV copy written to `{top_path}`_",
    ]
    md_path = Path(write_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[summary] wrote {md_path} and {top_path}")
    return props

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="outputs/props_priced_clean.csv")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--min-edge", type=float, default=0.01,
                    help="only show top rows with absolute edge >= this (default 0.01 = 1%)")
    ap.add_argument("--write-md", default="outputs/SUMMARY.md")
    args = ap.parse_args()

    summarize(args.clean, args.top_k, args.min_edge, args.write_md)

if __name__ == "__main__":
    main()
