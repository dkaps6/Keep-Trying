# engine.py  — minimal wiring to the new fetcher & normalizer
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.props_hybrid import get_props as fetch_props
from scripts.normalize_props import normalize_props

# (your existing price_props / write_outputs live elsewhere)
from scripts.pricing import price_props, write_outputs  # adjust if needed


def _to_list_if_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if not s or s.lower() in {"all", "none", "null"}:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _none_if_blank(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return None if s == "" else s


def run_pipeline(
    *,
    date: str = "today",
    season: Optional[int] = None,
    window: Optional[str | int] = "168h",
    cap: int = 0,
    markets: Optional[str] = None,
    books: str = "draftkings,fanduel",
    order: str = "odds",
    teams: Optional[str] = None,
    selection: Optional[str] = None,
    events: Optional[str] = None,
    write_dir: str = "outputs",
    basename: Optional[str] = None,
) -> int:
       # --- alias legacy CLI arg "hours" -> "window" ---
    import inspect
    frame_locals = inspect.currentframe().f_locals
    if "hours" in frame_locals and "window" not in frame_locals:
        window = frame_locals["hours"]

   """Orchestrates the end-to-end pricing run."""
    try:
        team_filter = _to_list_if_csv(teams)
        event_ids = _to_list_if_csv(events)

        print(f"[engine] fetching props… date={date} season={season}")
        print(f"[engine] markets={markets or 'default'} order={order} books={books}")
        print(f"[engine] selection={_none_if_blank(selection)} window={window} cap={cap}")

        df_raw = fetch_props(
            date=date,
            season=season,
            window=window,
            cap=cap,
            markets=markets,
            books=books,
            order=order,
            team_filter=team_filter,
            selection=_none_if_blank(selection),
            event_ids=event_ids,
            regions="us",
            use_probe=True,     # probe per-event for offered markets
            sleep=0.15,
            timeout=15,
        )

        if df_raw is None or df_raw.empty:
            print("[engine] No props available to price (props fetch returned empty DataFrame).")
            return 1

        print(f"[engine] fetched {len(df_raw)} raw rows")
        df_norm = normalize_props(df_raw)
        if df_norm is None or df_norm.empty:
            print("[engine] Normalization produced 0 rows; check market mapping.")
            return 1

        df_priced = price_props(df_norm)
        if df_priced is None or df_priced.empty:
            print("[engine] Pricing produced 0 rows.")
            return 1

        Path(write_dir).mkdir(parents=True, exist_ok=True)
        if not basename:
            basename = f"props_priced_{date}"
        write_outputs(df_priced, write_dir, basename)
        print("[engine] pipeline complete.")
        return 0

    except Exception as e:
        print(f"[engine] EXCEPTION: {e}")
        traceback.print_exc()
        return 1
