from __future__ import annotations
from importlib import import_module
def fetch(season: int, date: str | None = None) -> None:
    try: real = import_module("msf")
    except Exception: return
    for name in ("fetch_all","fetch","run","main"):
        fn = getattr(real, name, None)
        if callable(fn):
            try:
                if "season" in fn.__code__.co_varnames and "date" in fn.__code__.co_varnames: fn(season=season, date=date)
                elif "season" in fn.__code__.co_varnames: fn(season=season)
                else: fn()
            except Exception: pass
            break
