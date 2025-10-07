#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make_all.py (patched version with safe resolver)

import os, sys, pandas as pd
from pathlib import Path

OUT_BUNDLE = Path("external/nflverse_bundle/outputs")
OUT_METRICS = Path("outputs/metrics")
DATA_MIRROR = Path("data")

# ---------- dirs ----------
def _mkdirs():
    for p in (OUT_BUNDLE, OUT_METRICS, DATA_MIRROR):
        p.mkdir(parents=True, exist_ok=True)

# ---------- dynamic imports for providers ----------
def _import_or_none(modname: str):
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return None

nflverse = _import_or_none("scripts.providers.nflverse")
msf = _import_or_none("scripts.providers.msf")
apis = _import_or_none("scripts.providers.apisports")
gsis = _import_or_none("scripts.providers.nflgsis")

# ---------- io helpers ----------
def _ok(df: pd.DataFrame | None) -> bool:
    try:
        return isinstance(df, pd.DataFrame) and not df.empty
    except Exception:
        return False

# ---------- safe resolver (fixes ambiguous DataFrame truth) ----------
def _get_or_resolve(name: str, season, cache):
    """
    Return cached DataFrame if present and non-empty; otherwise resolve and cache it.
    Never uses DataFrame truthiness to avoid 'truth value is ambiguous' errors.
    """
    df = cache.get(name, None)
    if df is None or (hasattr(df, "empty") and df.empty):
        df = resolve_table(name, season, cache)
    cache[name] = df
    return df

# ---------- example resolver + composer ----------
def resolve_table(name, season, cache):
    print(f"[resolve_table] Loading {name} for {season}")
    return pd.DataFrame()

def compose_team_form(season, cache):
    pbp = _get_or_resolve("pbp", season, cache)
    print(f"[compose_team_form] Using pbp: {pbp.shape if _ok(pbp) else 'empty'}")
    return pd.DataFrame(), pd.DataFrame()

def main():
    season = 2025
    cache = {}
    _mkdirs()
    team_form, proe_week = compose_team_form(season, cache)
    print("[make_all] Done.")

if __name__ == "__main__":
    main()
