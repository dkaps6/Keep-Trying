# scripts/robust_merges.py
from __future__ import annotations
import pandas as pd

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def safe_merge_id_map(props: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join props to id_map so missing mappings don't nuke the slate.
    Writes metrics/needs_mapping.csv with (player, team) pairs that need attention.
    Expected keys: ['player','team'] or ['player_name'].
    Ensures 'player_id' column exists even if id_map is empty.
    """
    props = props.copy()

    # Ensure expected text keys are normalized for matching
    for c in ("player", "team"):
        if c in props.columns:
            props[c] = _norm_str(props[c])

    # If id_map missing or empty, keep rows and flag missing mappings
    if id_map is None or len(id_map) == 0:
        if "player_id" not in props.columns:
            props["player_id"] = pd.NA
        _write_needs_mapping(props, on_cols=[c for c in ("player","team") if c in props.columns])
        print("[warn] id_map empty; preserved props and wrote metrics/needs_mapping.csv")
        return props

    id_map = id_map.copy()
    for c in ("player", "team", "player_name"):
        if c in id_map.columns:
            id_map[c] = _norm_str(id_map[c])

    # Choose a join key
    if all(c in id_map.columns for c in ("player","team")) and all(c in props.columns for c in ("player","team")):
        on = ["player", "team"]
    elif "player_name" in id_map.columns and "player" in props.columns:
        id_map = id_map.rename(columns={"player_name": "player"})
        on = ["player"]
    else:
        # No usable keys—create empty player_id and report
        if "player_id" not in props.columns:
            props["player_id"] = pd.NA
        _write_needs_mapping(props, on_cols=[c for c in ("player","team") if c in props.columns])
        print("[warn] id_map lacks expected key columns; attached player_id=None")
        return props

    merged = props.merge(id_map, on=on, how="left", validate="m:1")
    if "player_id" not in merged.columns:
        merged["player_id"] = pd.NA

    missing = merged["player_id"].isna().sum()
    if missing:
        _write_needs_mapping(merged[merged["player_id"].isna()], on_cols=on)
        print(f"[warn] missing player_id for {missing} rows → metrics/needs_mapping.csv")
    return merged

def safe_merge_weather(df: pd.DataFrame, weather: pd.DataFrame, key_cols=("game_id",)) -> pd.DataFrame:
    """
    Left-join weather if available. If empty/missing, skip without dropping rows.
    """
    if weather is None or len(weather) == 0:
        print("[info] weather empty; skipping weather join")
        return df
    keys = [k for k in key_cols if (k in df.columns and k in weather.columns)]
    if not keys:
        print("[info] weather provided but no shared keys; skipping")
        return df
    return df.merge(weather, on=keys, how="left")

def _write_needs_mapping(df: pd.DataFrame, on_cols: list[str]) -> None:
    cols = [c for c in on_cols if c in df.columns]
    if not cols:
        return
    path = "metrics/needs_mapping.csv"
    df[cols].drop_duplicates().to_csv(path, index=False)
