# scripts/models/__init__.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Leg:
    player_id: str
    player: str
    team: str
    market: str  # e.g., "rec_yards"
    line: float
    features: Dict[str, Any]  # merged row from metrics + matchup + book

@dataclass
class LegResult:
    p_model: float        # raw model probability (before market blend)
    mu: float | None      # optional
    sigma: float | None   # optional
    notes: str = ""

