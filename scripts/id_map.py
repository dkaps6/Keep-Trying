import pandas as pd
import re

CACHE_PATH = "inputs/player_id_cache.csv"

def _norm_name(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z\s\.']", " ", s)
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
    props_df["name_norm"] = props_df["player_name_raw"].fillna("").map(_norm_name)
    ids_df = ids_df.copy()
    ids_df["player_name_norm"] = ids_df["player_name"].fillna("").map(_norm_name)

    # direct join on normalized name
    m1 = props_df.merge(ids_df[["player_name_norm","gsis_id"]],
                        left_on="name_norm", right_on="player_name_norm", how="left")
    m1.drop(columns=["player_name_norm"], inplace=True)

    # apply cache overrides
    if not cache.empty:
        m1 = m1.merge(cache, on="player_name_raw", how="left", suffixes=("","_cache"))
        m1["gsis_id"] = m1["gsis_id_cache"].fillna(m1["gsis_id"])
        m1 = m1.drop(columns=[c for c in m1.columns if c.endswith("_cache")])

    return m1
