import pandas as pd, re

CACHE_PATH = "inputs/player_id_cache.csv"

def _norm_name(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z\s\.\'-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def load_cache() -> pd.DataFrame:
    try:
        cache = pd.read_csv(CACHE_PATH, dtype=str)
        return cache[["player_name_raw","gsis_id"]].dropna().drop_duplicates()
    except Exception:
        return pd.DataFrame(columns=["player_name_raw","gsis_id"])

def map_players(props_df: pd.DataFrame, ids_df: pd.DataFrame) -> pd.DataFrame:
    cache = load_cache()
    df = props_df.copy()
    df["name_norm"] = df["player_name_raw"].fillna("").map(_norm_name)

    ids = ids_df.copy()
    ids["player_name_norm"] = ids["player_name"].fillna("").map(_norm_name)

    out = df.merge(ids[["player_name_norm","gsis_id","position","recent_team"]],
                   left_on="name_norm", right_on="player_name_norm", how="left")
    out.drop(columns=["player_name_norm"], inplace=True)

    if not cache.empty:
        out = out.merge(cache, on="player_name_raw", how="left", suffixes=("","_cache"))
        out["gsis_id"] = out["gsis_id_cache"].fillna(out["gsis_id"])
        out.drop(columns=[c for c in out.columns if c.endswith("_cache")], inplace=True)
    return out

