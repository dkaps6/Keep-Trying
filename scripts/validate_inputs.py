# scripts/validate_inputs.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

@dataclass
class CheckSpec:
    path: str
    required_cols: List[str] = field(default_factory=list)
    allow_empty: bool = False
    description: str = ""

REQUIRED: List[CheckSpec] = [
    CheckSpec(
        path="data/team_form.csv",
        required_cols=[
            "team",
            "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate",
        ],
        description="Team context (pace/pressure/EPA/boxes/air-yards z-scores + plays_est/PROE)."
    ),
    CheckSpec(
        path="data/player_form.csv",
        required_cols=[
            "player","team","position",
            "target_share","rush_share","rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","ypt","qb_ypa"
        ],
        description="Per-player shares & efficiency."
    ),
    CheckSpec(
        path="data/weather.csv",
        required_cols=["event_id","wind_mph","temp_f","precip"],
        allow_empty=False,
        description="Per-event weather (nearest to kickoff)."
    ),
    CheckSpec(
        path="data/roles.csv",
        required_cols=["player","team","role"],
        description="Player roles."
    ),
    CheckSpec(
        path="data/injuries.csv",
        required_cols=["player","team","status"],
        description="Injury statuses."
    ),
    CheckSpec(
        path="data/coverage.csv",
        required_cols=["defense_team","tag"],
        description="Coverage tags."
    ),
    CheckSpec(
        path="data/cb_assignments.csv",
        required_cols=["defense_team","receiver"],
        description="CB matchups."
    ),
    CheckSpec(
        path="outputs/game_lines.csv",
        required_cols=["event_id","home_team","away_team","home_wp","away_wp","commence_time"],
        description="Game win probabilities (script escalators)."
    ),
    CheckSpec(
        path="outputs/props_raw.csv",
        required_cols=[
            "id","commence_time","home_team","away_team",
            "bookmaker_key","bookmaker_title",
            "market_api","market_internal","player","side","line","price"
        ],
        description="Raw props rows."
    ),
]

SOFT_CHECKS = {
    "data/team_form.csv": {"def_pass_epa_z": 0.4, "def_rush_epa_z": 0.4},
    "data/player_form.csv": {"target_share": 0.6, "rush_share": 0.6},
    "data/weather.csv": {"wind_mph": 0.8, "temp_f": 0.8},
    "outputs/props_raw.csv": {"line": 0.15, "price": 0.15},
}

def fail_or_warn(msg: str, strict: bool, errors: List[str], warnings: List[str]) -> None:
    if strict:
        errors.append(msg)
        print(f"❌ {msg}")
    else:
        warnings.append(msg)
        print(f"⚠️  {msg}")

def check_file(spec: CheckSpec, strict: bool, errors: List[str], warnings: List[str]) -> Optional[pd.DataFrame]:
    p = Path(spec.path)
    if not p.exists():
        fail_or_warn(f"Missing file: {spec.path} — {spec.description}", strict, errors, warnings)
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        fail_or_warn(f"Unreadable CSV: {spec.path} ({e})", strict, errors, warnings)
        return None
    if df.empty and not spec.allow_empty:
        fail_or_warn(f"Empty CSV: {spec.path} — expected rows. {spec.description}", strict, errors, warnings)
        return df
    missing = [c for c in spec.required_cols if c not in df.columns]
    if missing:
        fail_or_warn(f"Missing column(s) in {spec.path}: {missing} — {spec.description}", strict, errors, warnings)
    # simple dup check
    key_sets = [["event_id","player","market_internal","line"], ["player","team","role"], ["player","team","status"]]
    for keys in key_sets:
        if all(k in df.columns for k in keys):
            dups = int(df.duplicated(subset=keys).sum())
            if dups > 0:
                fail_or_warn(f"{spec.path}: {dups} duplicate rows on keys {keys}", strict, errors, warnings)
            break
    # soft NaN checks
    for col, max_ratio in SOFT_CHECKS.get(spec.path, {}).items():
        if col in df.columns and len(df) > 0:
            r = float(df[col].isna().mean())
            if r > max_ratio:
                fail_or_warn(f"{spec.path}: column '{col}' NaN ratio {r:.2%} > {max_ratio:.2%}", strict, errors, warnings)
    return df

def main():
    ap = argparse.ArgumentParser(description="Validate required inputs before pricing.")
    ap.add_argument("--warn", action="store_true", help="Warnings instead of failures")
    ap.add_argument("--json", default=None, help="Optional path for JSON report")
    args = ap.parse_args()

    strict = not args.warn
    errors, warnings = [], []
    report: Dict[str, Dict[str, any]] = {}

    for spec in REQUIRED:
        print(f"▶ Checking {spec.path} ...")
        df = check_file(spec, strict, errors, warnings)
        report[spec.path] = {
            "exists": Path(spec.path).exists(),
            "rows": (0 if df is None else len(df)),
            "required_cols": spec.required_cols,
            "description": spec.description,
        }

    status = "PASS" if not errors else ("WARN" if not strict else "FAIL")
    print("\n=== Validation Summary ===")
    print(f"Status: {status}")
    if errors:
        print("\nErrors:")
        for e in errors: print(f" - {e}")
    if warnings:
        print("\nWarnings:")
        for w in warnings: print(f" - {w}")

    if args.json:
        out = {"status": status, "errors": errors, "warnings": warnings, "files": report}
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON report → {args.json}")

    if errors and strict: sys.exit(2)
    sys.exit(0)

if __name__ == "__main__":
    main()
