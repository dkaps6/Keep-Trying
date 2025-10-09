# engine_adapters/providers.py — nflverse-first + robust imports/collection
from __future__ import annotations
from typing import Dict, List, TypedDict, Optional
from pathlib import Path
import importlib, inspect, sys
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = REPO / "data"
OUT_EXT = REPO / "external" / "nflverse_bundle" / "outputs"
DATA.mkdir(parents=True, exist_ok=True)

class ProviderResult(TypedDict, total=False):
    ok: bool
    source: str
    wrote: Dict[str, str]
    rows: Dict[str, int]
    notes: List[str]

def _rows(path: Path) -> int:
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
    return _rows(dest)

def _first_exists(*cands: Path) -> Optional[Path]:
    for p in cands:
        if p.exists():
            return p
    return None

def _result(source: str) -> ProviderResult:
    return {"ok": False, "source": source, "wrote": {}, "rows": {}, "notes": []}

def _import_first(module_names: List[str], notes: List[str]):
    for m in module_names:
        try:
            return importlib.import_module(m)
        except Exception as e:
            notes.append(f"import failed for {m}: {e}")
    return None

def _try_module(source: str, module_names: List[str], call_hints: List[str],
                season: int, date: Optional[str]) -> ProviderResult:
    res = _result(source)
    mod = _import_first(module_names, res["notes"])
    if mod is None:
        return res

    called = False
    for name in call_hints:
        fn = getattr(mod, name, None)
        if fn and inspect.isfunction(fn):
            try:
                sig = inspect.signature(fn)
                kwargs = {}
                if "season" in sig.parameters: kwargs["season"] = season
                if "date" in sig.parameters: kwargs["date"] = date
                fn(**kwargs)
                called = True
                res["notes"].append(f"called {mod.__name__}.{name}{kwargs}")
                break
            except Exception as e:
                res["notes"].append(f"{name} failed: {e}")
    if not called and hasattr(mod, "main") and inspect.isfunction(mod.main):
        try:
            mod.main()
            res["notes"].append(f"called {mod.__name__}.main()")
            called = True
        except Exception as e:
            res["notes"].append(f"main failed: {e}")

    # Collect usable outputs from either external bundle or data/
    wrote, rows = {}, {}

    # PBP
    pbp_src = _first_exists(
        OUT_EXT / "pbp" / f"pbp_{season}.parquet",
        OUT_EXT / "pbp" / f"pbp_{season}.csv",
        DATA / f"pbp_{season}.parquet",
        DATA / f"pbp_{season}.csv",
    )
    if pbp_src:
        target = DATA / pbp_src.name
        r = _mirror(pbp_src, target)
        wrote["pbp"] = str(target); rows["pbp"] = r

    # Weekly player stats
    wk_src = _first_exists(
        OUT_EXT / "player_stats" / f"player_stats_week_{season}.csv",
        DATA / "player_stats_week.csv",
    )
    if wk_src:
        r = _mirror(wk_src, DATA / "player_stats_week.csv")
        wrote["player_stats_week"] = str(DATA / "player_stats_week.csv"); rows["player_stats_week"] = r

    # Rosters
    rost_src = _first_exists(
        OUT_EXT / "rosters" / f"rosters_{season}.csv",
        DATA / "rosters.csv",
    )
    if rost_src:
        r = _mirror(rost_src, DATA / "rosters.csv")
        wrote["rosters"] = str(DATA / "rosters.csv"); rows["rosters"] = r

    # Injuries (optional)
    inj_src = _first_exists(DATA / "injuries.csv", OUT_EXT / "injuries.csv")
    if inj_src:
        r = _mirror(inj_src, DATA / "injuries.csv")
        wrote["injuries"] = str(DATA / "injuries.csv"); rows["injuries"] = r

    res["wrote"] = wrote
    res["rows"] = rows
    res["ok"] = bool(rows.get("player_stats_week", 0) or rows.get("pbp", 0))
    return res

# ---- provider runners ----
def run_nflverse(season: int, date: Optional[str]) -> ProviderResult:
    # Try your external bundle first, then a pure-python entry (below) if present
    return _try_module(
        "nflverse",
        ["external.nflverse_bundle.fetch_all", "scripts.providers.nflverse_entry"],
        ["fetch_all", "fetch", "run", "build", "main"],
        season, date
    )

def run_espn(season: int, date: Optional[str]) -> ProviderResult:
    return _try_module(
        "ESPN",
        ["scripts.providers.espn_entry", "scripts.providers.espn_pbp", "espn_pbp"],
        ["fetch", "run", "download", "build"],
        season, date
    )

def run_nflgsis(season: int, date: Optional[str]) -> ProviderResult:
    return _try_module(
        "NFLGSIS",
        ["nflgsis", "scripts.providers.nflgsis"],
        ["fetch", "run", "download", "build", "main"],
        season, date
    )

def run_msf(season: int, date: Optional[str]) -> ProviderResult:
    return _try_module(
        "MySportsFeeds",
        ["msf", "scripts.providers.msf"],
        ["fetch", "run", "download", "build", "main"],
        season, date
    )

def run_apisports(season: int, date: Optional[str]) -> ProviderResult:
    return _try_module(
        "API-Sports",
        ["apisports", "scripts.providers.apisports"],
        ["fetch", "run", "download", "build", "main"],
        season, date
    )
