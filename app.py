# app.py
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

OUT_DIR = Path("outputs")

def run_cmd(cmd: list[str], env: dict | None = None) -> int:
    st.write("```bash\n" + " ".join(cmd) + "\n```")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        st.text(line.rstrip())
    return proc.wait()

@st.cache_data(ttl=60)
def load_clean() -> pd.DataFrame:
    p = OUT_DIR / "props_priced_clean.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # normalize
    for c in ("edge_abs", "p_market_over", "p_over_blend"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def colorize(df: pd.DataFrame) -> pd.DataFrame:
    if "color" not in df.columns:
        return df
    def _row_color(row):
        c = str(row.get("color", "RED")).upper()
        if c == "GREEN":  return "background-color: #e6ffed"
        if c == "AMBER":  return "background-color: #fff5e6"
        return ""
    return df.style.apply(lambda r: [_row_color(r)]*len(r), axis=1)

st.set_page_config(page_title="NFL Props Model", layout="wide")

st.title("NFL Props Model — Web UI")

with st.sidebar:
    st.header("Controls")
    season     = st.number_input("Season", min_value=2018, max_value=2030, value=2025, step=1)
    date_str   = st.text_input("Date (ISO or 'today')", value="today")
    provider   = st.selectbox("Provider order", ["odds,dq", "odds,dk", "dk,odds", "odds"])  # label only
    win_hours  = st.number_input("Credit guard: Only events starting in next N hours", min_value=1, max_value=168, value=48)
    max_events = st.number_input("Credit guard: Max events to fetch", min_value=1, max_value=64, value=24)
    run_fetch  = st.button("🔁 Run full pipeline (fetch + price)")
    reload_btn = st.button("🔄 Refresh table")

    st.caption("Set `ODDS_API_KEY` in your environment / Streamlit secrets.")

# Run pipeline on demand (safe: uses your existing CLI entry points)
if run_fetch:
    env = dict(os.environ)
    # pass credit guards through env (scripts use these if present)
    env["ODDS_API_EVENT_WINDOW_HOURS"] = str(int(win_hours))
    env["ODDS_API_MAX_EVENTS"] = str(int(max_events))
    # ODDS_API_KEY should already be in env (locally) or in Streamlit secrets
    if "ODDS_API_KEY" not in env and "ODDS_API_KEY" in st.secrets:
        env["ODDS_API_KEY"] = st.secrets["ODDS_API_KEY"]

    st.subheader("Fetch metrics (nfl_data_py + NWS)")
    code = run_cmd([sys.executable, "-m", "scripts.fetch_all", "--season", str(season)], env=env)
    if code != 0:
        st.error("fetch_all failed")
    else:
        st.success("metrics/weather fetched")

    st.subheader("Run pricing model")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    code = run_cmd([sys.executable, "run_model.py", "--date", date_str, "--season", str(season), "--write", "outputs"], env=env)
    if code != 0:
        st.error("run_model failed")
    else:
        st.success("pricing completed")

# Data view
df = load_clean() if not reload_btn else load_clean.clear() or load_clean()

st.subheader("Top Edges")
if df.empty:
    st.info("No outputs yet. Click **Run full pipeline**.")
else:
    # quick summary
    avg_edge = float(df["edge_abs"].mean(skipna=True)) if "edge_abs" in df else 0.0
    greens = int((df.get("color","RED") == "GREEN").sum())
    ambers = int((df.get("color","RED") == "AMBER").sum())
    reds   = int((df.get("color","RED") == "RED").sum())
    st.write(f"Props: **{len(df)}** | Avg edge: **{avg_edge:.2%}** | GREEN/AMBER/RED: **{greens}/{ambers}/{reds}**")

    # filters
    cols = st.columns(3)
    with cols[0]:
        teams = sorted(pd.unique(df[["team","defense_team"]].values.ravel("K")))
        team_filter = st.multiselect("Team filter (either side)", options=[t for t in teams if isinstance(t, str)])
    with cols[1]:
        markets = sorted(df["market"].dropna().unique().tolist())
        market_filter = st.multiselect("Markets", options=markets, default=markets)
    with cols[2]:
        min_edge = st.slider("Min edge %", 0.0, 10.0, 1.0, 0.5) / 100.0

    view = df.copy()
    if team_filter:
        view = view[(view["team"].isin(team_filter)) | (view["defense_team"].isin(team_filter))]
    if market_filter:
        view = view[view["market"].isin(market_filter)]
    if "edge_abs" in view.columns:
        view = view[view["edge_abs"].ge(min_edge)]

    # order by edge
    if "edge_abs" in view.columns:
        view = view.sort_values("edge_abs", ascending=False)

    # select standard columns
    keep = [
        "event_id","player","team","defense_team","market","line",
        "bet_side","edge_abs","color","bookmaker",
        "over_odds","under_odds","p_market_over","p_over_blend","fair_over_odds",
        "mu_model","wind_mph","temp_f","precip"
    ]
    keep = [c for c in keep if c in view.columns]
    styled = colorize(view[keep])
    st.dataframe(styled, use_container_width=True)

    # download buttons
    if (OUT_DIR / "props_priced_clean.csv").exists():
        st.download_button("Download CSV", data=(OUT_DIR / "props_priced_clean.csv").read_bytes(),
                           file_name="props_priced_clean.csv", mime="text/csv")
    if (OUT_DIR / "SUMMARY.md").exists():
        st.download_button("Download SUMMARY.md", data=(OUT_DIR / "SUMMARY.md").read_bytes(),
                           file_name="SUMMARY.md", mime="text/markdown")
