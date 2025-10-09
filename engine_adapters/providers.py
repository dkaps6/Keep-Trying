# engine_adapters/providers.py
from __future__ import annotations
from typing import Dict, List, TypedDict, Optional
from pathlib import Path
import importlib, inspect
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT_EXT = REPO / "external" / "nflverse_bundle" / "outputs"
DATA.mkdir(parents=True, exist_ok=True)
OUT_EXT.mkdir(parents=True, exist_ok=True)

class ProviderResult(TypedDict, total=False):
    ok: bool
    source: str
    wrote: Dict[str, str]   # logical_name -> path
    rows: Dict[str, int]    # logical_name -> rowcount
    notes: List[str]

def _df_rows(path: Path) -> int:
    try:
        if path.suffix == ".parquet":
            return len(pd.read_parquet(path))
        return len(pd.read_csv(path))
    except Exception:
        return 0

def _mirror(src: Path, dest: Path) -> int:
    if not src.exists():
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return _df_rows(dest)

def _best_existing(*candidates: Path) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None

def _result(source: str) -> ProviderResult:
    return {"ok": False, "source": source, "wrote": {}, "rows": {}, "notes": []}

def _try_module(source: str, module_name: str, call_hints: List[str], season: int, date: Optional[str]) -> ProviderResult:
    res = _result(source)
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        res["notes"].append(f"import failed: {e}")
        return res

    called = False
    for name in call_hints:
        fn = getattr(mod, name, None)
        if fn and inspect.isfunction(fn):
            try:
                sig = inspect.signature(fn)
                kwargs = {}
                if "season" in sig.parameters:
                    kwargs["season"] = season
                if "date" in sig.parameters:
                    kwargs["date"] = date
                if "year" in sig.parameters and "season" not in kwargs:
                    kwargs["year"] = season
                if "yyyy" in sig.parameters and "season" not in kwargs:
                    kwargs["yyyy"] = season
                fn(**kwargs)  # run it
                called = True
                res["notes"].append(f"called {module_name}.{name}{kwargs}")
                break
            except Exception as e:
                res["notes"].append(f"{name} failed: {e}")
    if not called:
        # try a module-level main() if present
        try:
            if hasattr(mod, "main") and inspect.isfunction(mod.main):
                mod.main()
                res["notes"].append(f"called {module_name}.main()")
                called = True
        except Exception as e:
            res["notes"].append(f"main failed: {e}")

    # After running, collect outputs we care about (any provider may produce them).
    wrote = {}
    rows = {}

    # PBP
    pbp_src = _best_existing(OUT_EXT / "pbp" / f"pbp_{season}.parquet",
                             OUT_EXT / "pbp" / f"pbp_{season}.csv",
                             DATA / f"pbp_{season}.parquet",
                             DATA / f"pbp_{season}.csv")
    if pbp_src:
        target = DATA / f"pbp_{season}.parquet" if pbp_src.suffix == ".parquet" else DATA / f"pbp_{season}.csv"
        r = _mirror(pbp_src, target)
        wrote["pbp"] = str(target); rows["pbp"] = r

    # Weekly player stats
    weekly_src = _best_existing(OUT_EXT / "player_stats" / f"player_stats_week_{season}.csv",
                                DATA / "player_stats_week.csv")
    if weekly_src:
        r = _mirror(weekly_src, DATA / "player_stats_week.csv")
        wrote["player_stats_week"] = str(DATA / "player_stats_week.csv"); rows["player_stats_week"] = r

    # Rosters
    rost_src = _best_existing(OUT_EXT / "rosters" / f"rosters_{season}.csv",
                              DATA / "rosters.csv")
    if rost_src:
        r = _mirror(rost_src, DATA / "rosters.csv")
        wrote["rosters"] = str(DATA / "rosters.csv"); rows["rosters"] = r

    # Injuries (if any provider wrote it)
    inj_src = _best_existing(DATA / "injuries.csv", OUT_EXT / "injuries.csv")
    if inj_src:
        r = _mirror(inj_src, DATA / "injuries.csv")
        wrote["injuries"] = str(DATA / "injuries.csv"); rows["injuries"] = r

    res["wrote"] = wrote
    res["rows"] = rows
    # OK if we have at least weekly or pbp
    res["ok"] = bool(rows.get("player_stats_week", 0) or rows.get("pbp", 0))
    return res

def run_espn(season: int, date: Optional[str]) -> ProviderResult:
    return _try_module("ESPN", "espn_pbp", ["fetch", "run", "download", "build"], season, date)

def run_nflgsis(season: int, date: Optional[str]) -> ProviderResult:
    # module name user provided: nflgsis.py
    return _try_module("NFLGSIS", "nflgsis", ["fetch", "run", "download", "build"], season, date)

def run_msf(season: int, date: Optional[str]) -> ProviderResult:
    return _try_module("MySportsFeeds", "msf", ["fetch", "run", "download", "build"], season, date)

def run_apisports(season: int, date: Optional[str]) -> ProviderResult:
    return _try_module("API-Sports", "apisports", ["fetch", "run", "download", "build"], season, date)
