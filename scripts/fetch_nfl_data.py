# scripts/fetch_nfl_data.py
from __future__ import annotations
import argparse, warnings, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

# Optional sources for historic/current PBP & weekly
try:
    import nflreadpy as nrp
    HAS_NFLREADPY = True
except Exception:
    HAS_NFLREADPY = False

try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False


# -----------------------------
# Small IO helpers
# -----------------------------
def _to_pandas(obj):
    """Accept pandas or polars; return pandas.DataFrame."""
    try:
        import polars as pl  # type: ignore
        if isinstance(obj, pl.DataFrame):
            return obj.to_pandas()
    except Exception:
        pass
    return obj


def _safe_read_csv(path: Path, **kw) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            warnings.warn(f"Failed reading {path}, returning empty.")
    return pd.DataFrame()


def _ensure_cols(df: pd.DataFrame, cols: List[str], fill=0.0) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


# -----------------------------
# nflverse loaders (optional)
# -----------------------------
def _load_pbp_year(year: int) -> pd.DataFrame:
    """Try nflreadpy, then nfl_data_py; return pandas df (may be empty)."""
    if HAS_NFLREADPY:
        try:
            print(f"[fetch_nfl_data] USING PBP (nflreadpy) for {year}")
            df = nrp.load_pbp(seasons=[year])
            df = _to_pandas(df)
            if "season" not in df.columns:
                df["season"] = year
            return df
        except Exception as e:
            warnings.warn(f"nflreadpy.load_pbp({year}) failed: {type(e).__name__}: {e}")
    if HAS_NFL_DATA_PY:
        try:
            print(f"[fetch_nfl_data] FALLBACK PBP (nfl_data_py) for {year}")
            df = nfl.import_pbp_data([year])
            return _to_pandas(df)
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_pbp_data({year}) failed: {type(e).__name__}: {e}")
    return pd.DataFrame()


def _load_weekly_year(year: int) -> pd.DataFrame:
    """Weekly player stats – used as extra context if needed."""
    if HAS_NFLREADPY:
        try:
            print(f"[fetch_nfl_data] USING WEEKLY (nflreadpy) {year}")
            df = nrp.load_player_stats(seasons=[year])
            return _to_pandas(df)
        except Exception as e:
            warnings.warn(f"nflreadpy.load_player_stats({year}) failed: {type(e).__name__}: {e}")
    if HAS_NFL_DATA_PY:
        try:
            print(f"[fetch_nfl_data] FALLBACK WEEKLY (nfl_data_py) {year}")
            df = nfl.import_weekly_data([year])
            return _to_pandas(df)
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_weekly_data({year}) failed: {type(e).__name__}: {e}")
    return pd.DataFrame()


# -----------------------------
# Team form build
# -----------------------------
BUNDLE_TEAM_FORM = Path("outputs/metrics/team_form.csv")
PFR_ADVANCED = Path("outputs/metrics/pfr_advanced_team.csv")      # optional
NGS_WEEKLY   = Path("outputs/metrics/ngs_weekly_team.csv")        # optional
BOX_COUNTS   = Path("outputs/metrics/box_counts_team.csv")        # optional


def _maybe_bundle_team() -> pd.DataFrame:
    """Prefer team_form from bundle if present."""
    df = _safe_read_csv(BUNDLE_TEAM_FORM)
    if df.empty:
        return df
    # Required columns you use downstream (z-scored later in pricing/features):
    req = [
        "team", "def_pass_epa", "def_rush_epa", "def_sack_rate",
        "pace", "proe", "light_box_rate", "heavy_box_rate",
    ]
    df = _ensure_cols(df, req, fill=0.0)
    # Normalize team code
    if "team" not in df.columns and "posteam" in df.columns:
        df = df.rename(columns={"posteam":"team"})
    df["team"] = df["team"].astype(str).str.upper()
    return df


def _augment_with_optionals(team: pd.DataFrame) -> pd.DataFrame:
    """Optionally merge PFR advanced, NGS weekly, and aggregated box counts."""
    out = team.copy()

    pfr = _safe_read_csv(PFR_ADVANCED)
    if not pfr.empty:
        # Expect columns like: team, def_pressure_rate, def_adj_sack_rate, def_pass_yds_allowed_per_att, ...
        pfr["team"] = pfr["team"].astype(str).str.upper()
        out = out.merge(pfr.drop_duplicates(subset=["team"]), on="team", how="left")

    ngs = _safe_read_csv(NGS_WEEKLY)
    if not ngs.empty:
        # Expect columns: team, pressure_rate, time_to_throw, etc.
        ngs["team"] = ngs["team"].astype(str).str.upper()
        out = out.merge(ngs.drop_duplicates(subset=["team"]), on="team", how="left")

    boxes = _safe_read_csv(BOX_COUNTS)
    if not boxes.empty:
        # Expect columns: team, light_box_rate, heavy_box_rate
        boxes["team"] = boxes["team"].astype(str).str.upper()
        for c in ["light_box_rate", "heavy_box_rate"]:
            if c not in boxes.columns:
                boxes[c] = 0.0
        out = out.drop(columns=[c for c in ["light_box_rate","heavy_box_rate"] if c in out.columns], errors="ignore")
        out = out.merge(boxes[["team","light_box_rate","heavy_box_rate"]], on="team", how="left")

    # Fill missing numeric with 0; keep strings as-is
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(0.0)
    return out


def compute_team_form_from_pbp(seasons: List[int]) -> pd.DataFrame:
    """Minimal team-level aggregates from PBP if bundle not present."""
    frames = []
    for y in seasons:
        df = _load_pbp_year(y)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[
            "team","def_pass_epa","def_rush_epa","def_sack_rate","pace","proe",
            "light_box_rate","heavy_box_rate"
        ])

    pbp = pd.concat(frames, ignore_index=True, sort=False)

    # Basic defensive EPA splits (opp perspective)
    # Where available in nflverse tables:
    #   def_pass_epa: EPA on opponent pass plays vs this defense
    #   def_rush_epa: EPA on opponent rush plays vs this defense
    pbp["is_pass"] = (pbp.get("pass", 0) == 1) if "pass" in pbp.columns else pbp.get("play_type","").astype(str).str.contains("pass", case=False, na=False)
    pbp["is_rush"] = (pbp.get("rush", 0) == 1) if "rush" in pbp.columns else pbp.get("play_type","").astype(str).str.contains("rush", case=False, na=False)
    pbp["defteam"] = pbp.get("defteam", pbp.get("def_team", ""))

    grp = pbp.groupby("defteam", dropna=True)
    def_pass = grp.apply(lambda g: g.loc[g["is_pass"], "epa"].mean() if "epa" in g else 0.0).rename("def_pass_epa")
    def_rush = grp.apply(lambda g: g.loc[g["is_rush"], "epa"].mean() if "epa" in g else 0.0).rename("def_rush_epa")

    # Proxy sack rate (from play-by-play when available)
    if "sack" in pbp.columns:
        def_sack = grp["sack"].mean().rename("def_sack_rate")
    else:
        def_sack = pd.Series(0.0, index=grp.size().index, name="def_sack_rate")

    # Pace proxy: seconds per snap inverse (if not in pbp, set 0)
    pace = pd.Series(0.0, index=grp.size().index, name="pace")
    # PROE needs xPass model; if bundle missing, keep 0 and let downstream tilt with market anchor
    proe = pd.Series(0.0, index=grp.size().index, name="proe")

    out = pd.concat([def_pass, def_rush, def_sack, pace, proe], axis=1).reset_index().rename(columns={"defteam":"team"})
    out["light_box_rate"] = 0.0
    out["heavy_box_rate"] = 0.0
    out["team"] = out["team"].astype(str).str.upper()
    return out


def build_team_form(season: int, history: str) -> pd.DataFrame:
    """
    Prefer: outputs/metrics/team_form.csv (bundle)
    Else:   compute a minimal team_form from PBP across history (and current if available).
    Then:   augment with optional PFR advanced / NGS weekly / box aggregates.
    """
    df = _maybe_bundle_team()
    if df.empty:
        years = []
        if history:
            for y in str(history).split(","):
                y = y.strip()
                if "-" in y:
                    a, b = y.split("-")
                    years += list(range(int(a), int(b) + 1))
                else:
                    years.append(int(y))
        if season not in years:
            years.append(season)
        df = compute_team_form_from_pbp(sorted(set(years)))
    df = _augment_with_optionals(df)

    # z-scores prepared downstream; ensure minimal columns exist
    df = _ensure_cols(df, [
        "team","def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","light_box_rate","heavy_box_rate"
    ], fill=0.0)
    return df


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--history", default="2019-2024")
    ap.add_argument("--write", default="data/team_form.csv")
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    df = build_team_form(args.season, args.history)
    df.to_csv(args.write, index=False)
    print(f"[fetch_nfl_data] ✅ wrote {len(df)} rows → {args.write}")


if __name__ == "__main__":
    main()
