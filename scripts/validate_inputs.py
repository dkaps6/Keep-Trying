# scripts/validate_inputs.py
# Preflight validator for your pipeline inputs.
# - Verifies required CSVs exist, are non-empty, and contain required columns
# - Optional "soft" checks (NaN ratios, duplicate rows)
# - Fails fast with actionable messages (strict by default; --warn to downgrade to warnings)

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
    # Team / environment context
    CheckSpec(
        path="data/team_form.csv",
        required_cols=[
            # keep liberal — add more if your pricing/volume rely on them
            "team", "opp_team", "event_id",
            "def_pressure_rate_z", "def_pass_epa_z", "def_rush_epa_z", "def_sack_rate_z",
            "pace_z", "light_box_rate_z", "heavy_box_rate_z", "ay_per_att_z",
            # If your volume path uses them, they should be present:
            # "plays_est", "proe", "rz_rate"
        ],
        description="Team context (pace/pressure/EPA/boxes/air-yards z-scores)."
    ),

    # Weather (joined by event_id)
    CheckSpec(
        path="data/weather.csv",
        required_cols=["event_id", "wind_mph", "temp_f", "precip"],
        allow_empty=False,
        description="Per-event weather (nearest to kickoff)."
    ),

    # Roles & injuries for redistribution rules
    CheckSpec(
        path="data/roles.csv",
        required_cols=["player", "team", "role"],
        description="Player roles (WR1/WR2/SLOT/RB1/TE1/...)."
    ),
    CheckSpec(
        path="data/injuries.csv",
        required_cols=["player", "team", "status"],
        description="Injury statuses (Out/Doubtful/Questionable/Limited/Probable)."
    ),

    # Coverage / CB assignment
    CheckSpec(
        path="data/coverage.csv",
        required_cols=["defense_team", "tag"],
        description="Coverage tags (top_shadow/heavy_man/heavy_zone)."
    ),
    CheckSpec(
        path="data/cb_assignments.csv",
        required_cols=["defense_team", "receiver"],
        description="Optional: CB matchups; include 'cb' + 'quality' or 'penalty' if available."
    ),

    # Identity map (for name normalization)
    CheckSpec(
        path="data/id_map.csv",
        required_cols=["player_name", "team"],
        description="Player name → (team, position, role) mapping."
    ),

    # Game lines (if you use win prob/script escalators)
    CheckSpec(
        path="outputs/game_lines.csv",
        required_cols=["event_id", "home_team", "away_team", "home_wp", "away_wp", "commence_time"],
        description="H2H-derived win probability for script escalators."
    ),

    # Props ingestion (raw dump from your fetch step)
    CheckSpec(
        path="outputs/props_raw.csv",
        required_cols=[
            "id", "commence_time", "home_team", "away_team",
            "bookmaker_key", "bookmaker_title",
            "market_api", "market_internal", "player", "side", "line", "price"
        ],
        description="Raw props rows (book × market × player × side × line)."
    ),
]


SOFT_CHECKS = {
    # file -> {col: max_nan_ratio}
    "data/team_form.csv": {"def_pressure_rate_z": 0.3, "def_pass_epa_z": 0.3, "def_rush_epa_z": 0.3},
    "data/weather.csv": {"wind_mph": 0.75, "temp_f": 0.75},
    "outputs/props_raw.csv": {"line": 0.1, "price": 0.1},
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
        fail_or_warn(
            f"Missing column(s) in {spec.path}: {missing} — {spec.description}",
            strict, errors, warnings
        )

    # Duplicate detection on typical keys (best-effort without schema coupling)
    key_candidates = [
        ["event_id", "home_team", "away_team"],
        ["event_id", "player", "market_internal", "line"],
        ["player", "team", "role"],
        ["player", "team", "status"],
    ]
    for keys in key_candidates:
        if all(k in df.columns for k in keys):
            dups = df.duplicated(subset=keys).sum()
            if dups > 0:
                fail_or_warn(f"{spec.path}: {dups} duplicate rows on keys {keys}", strict, errors, warnings)
            break

    # Soft NaN checks
    soft = SOFT_CHECKS.get(spec.path, {})
    for col, ratio in soft.items():
        if col in df.columns and len(df) > 0:
            nanr = float(df[col].isna().mean())
            if nanr > ratio:
                fail_or_warn(
                    f"{spec.path}: column '{col}' has NaN ratio {nanr:.2%} > {ratio:.2%}",
                    strict, errors, warnings
                )

    return df


def main():
    ap = argparse.ArgumentParser(description="Validate required inputs before pricing runs.")
    ap.add_argument("--warn", action="store_true", help="Downgrade failures to warnings (exit 0)")
    ap.add_argument("--json", default=None, help="Optional: write a JSON report to this path")
    args = ap.parse_args()

    strict = not args.warn
    errors: List[str] = []
    warnings: List[str] = []
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
        for e in errors:
            print(f" - {e}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f" - {w}")

    if args.json:
        out = {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "files": report,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON report → {args.json}")

    # Exit codes: 0 pass/warn, 2 fail (strict)
    if errors and strict:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
