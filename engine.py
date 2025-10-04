# engine.py
import os
import re
import pandas as pd

from scripts.odds_api import fetch_game_lines, fetch_props_all_events
from scripts.features_external import build_external
from scripts.id_map import map_players
from scripts.pricing import (
    american_to_prob, devig_two_way, blend, prob_to_american,
    edge_pct, kelly_fraction, tier
)
from scripts.model_core import (
    base_sigma, over_prob_normal,
    estimate_team_rates, player_shares,
    mu_receptions, mu_rec_yards, mu_rush_atts, mu_rush_yards, mu_pass_yards
)
from scripts.elite_rules import (
    funnel_multiplier, volatility_widen, pace_smoothing, sack_to_attempts
)
from scripts.engine_helpers import make_team_last4_from_player_form
from scripts.rules_engine import apply_rules
from scripts.volume import consensus_spread_total, team_volume_estimates


# ---------- helpers ----------
def _norm(val) -> str:
    """
    Normalize any input to a lowercase ascii-ish token string.
    Safe for None/NaN/numbers.
    """
    if val is None:
        return ""
    s = str(val)
    if s.strip().lower() in {"nan", "none", "null"}:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------- main pipeline ----------
def run_pipeline(target_date: str, season: int, out_dir: str = "outputs"):
    """
    Orchestrates the pipeline:
      1) Fetch game lines and player props from The Odds API
      2) Build external features (ids, team/player rolling form, injuries, depth)
      3) Map sportsbook player names -> gsis_id
      4) Compute μ/σ with volume scaffolding + elite rules
      5) Price at book lines (model prob -> 65/35 blend -> fair odds, edge, kelly, tier)
      6) Write tidy CSVs into outputs/
    """

    # --- 1) Odds
    game_df  = fetch_game_lines()
    props_df = fetch_props_all_events()

    # Consensus spread/total (for volume + win prob)
    cons = consensus_spread_total(game_df)  # cols: event_id, home_spread, total

    # --- 2) External features
    ext = build_external(season)
    ids = ext["ids"]
    team_form = ext["team_form"]
    pform = ext["player_form"]

    # Quick team context map (L4 totals for shares)
    team_l4_map = make_team_last4_from_player_form(pform)

    # --- 3) Map player names → ids/position/team
    props_df = map_players(props_df, ids)

    # Helper: team L4 context (fallbacks if unknown)
    def get_team_context(team):
        return team_l4_map.get(team, {"tgt_team_l4": 30.0, "rush_att_team_l4": 25.0})

    rows = []
    for _, r in props_df.iterrows():
        market = r["market"]
        if market not in {
            "player_reception_yds",      # receiving yards
            "player_receptions",
            "player_rush_yds",
            "player_rush_attempts",
            "player_pass_yds",
            "player_pass_tds",           # placeholder (rough)
            "player_anytime_td",         # placeholder (rough)
        }:
            continue

        price = r["price"]
        line  = r["point"]
        side  = r["outcome"]

        # Convert American odds -> implied raw prob; de-vig two-way if we have both sides
        p_raw = american_to_prob(price)
        p_over_raw  = p_raw if side in ("Over", "Yes") else None
        p_under_raw = p_raw if side in ("Under", "No") else None
        p_over_fair, p_under_fair = devig_two_way(p_over_raw, p_under_raw)
        market_p_fair = p_over_fair if side in ("Over", "Yes") else p_under_fair

        # ---------- Volume model from consensus lines ----------
        ev = cons[cons["event_id"].eq(r["event_id"])].head(1)
        ev_row = ev.iloc[0] if not ev.empty else {}

        team_code = r.get("recent_team") or ""
        home_name = r.get("home") or ""
        away_name = r.get("away") or ""
        nteam = _norm(team_code)
        is_home = bool(nteam) and (nteam in _norm(home_name))

        # Optional: we could pull per-team recent pass/rush from team_form here
        tf_row = None

        plays, pass_rate, rush_rate, win_prob = team_volume_estimates(ev_row, is_home, tf_row)

        # ---------- Player shares/efficiency from last-4 ----------
        team_ctx = get_team_context(r.get("recent_team"))
        pid = r.get("gsis_id")
        pr = pform[pform["gsis_id"] == pid]
        if not pr.empty:
            pr = pr.sort_values("week").tail(1).iloc[0].to_dict()
        else:
            pr = {}

        tgt_share, rush_share, catch_rate, ypr, ypc = player_shares(pr, team_ctx)

        # mild funnel multipliers (neutral for now; wire z-scores later)
        pass_mult = funnel_multiplier(True,  def_rush_epa_z=0.0, def_pass_epa_z=0.0)
        rush_mult = funnel_multiplier(False, def_rush_epa_z=0.0, def_pass_epa_z=0.0)

        plays = pace_smoothing(plays, 0.0, 0.0)

        # ---------- market-specific μ / σ ----------
        sigma = base_sigma(market)

        if market == "player_receptions":
            mu_rec = mu_receptions(plays, pass_rate, tgt_share, catch_rate) * pass_mult
            mu = mu_rec

        elif market == "player_reception_yds":  # receiving yards
            mu_rec = mu_receptions(plays, pass_rate, tgt_share, catch_rate) * pass_mult
            mu = mu_rec_yards(mu_rec, ypr)

        elif market == "player_rush_attempts":
            atts = mu_rush_atts(plays, rush_rate, rush_share) * rush_mult
            atts = sack_to_attempts(atts, sack_rate_above_avg=0.0)  # hook for live sack rate
            mu = atts

        elif market == "player_rush_yds":
            atts = mu_rush_atts(plays, rush_rate, rush_share) * rush_mult
            mu = mu_rush_yards(atts, ypc)

        elif market == "player_pass_yds":
            qb_ypa = (pr.get("pass_yds_l4", 0) / max(pr.get("pass_att_l4", 1), 1)) if pr else 6.8
            dropbacks = plays * pass_rate
            mu = mu_pass_yards(dropbacks, qb_ypa, z_opp_pressure=0.0, z_opp_pass_epa=0.0)
            sigma = volatility_widen(sigma, pressure_mismatch=False, qb_inconsistent=False)

        elif market == "player_pass_tds":
            # Rough mapping until we swap to Bernoulli/Poisson via team totals
            qb_ypa = (pr.get("pass_yds_l4", 0) / max(pr.get("pass_att_l4", 1), 1)) if pr else 6.8
            dropbacks = plays * pass_rate
            pass_yds_mu = mu_pass_yards(dropbacks, qb_ypa, z_opp_pressure=0.0, z_opp_pass_epa=0.0)
            mu = max(0.5, pass_yds_mu / 150.0)
            sigma = max(0.6, sigma * 0.04)

        elif market == "player_anytime_td":
            mu = 0.0
            sigma = 1.0

        else:
            mu = 0.0

        # ---------- Elite rules: adjust μ / σ & capture notes ----------
        mu, sigma, notes = apply_rules(
            market=market,
            side=side,
            mu=mu,
            sigma=sigma,
            player=pr,
            team_ctx=team_ctx,
            opp_pressure_z=0.0,     # fill with real z-scores later
            opp_pass_epa_z=0.0,
            run_funnel=False,
            pass_funnel=False,
            alpha_limited=False,     # can be set from ext["inj"]
            tough_shadow=False,
            heavy_man=False,
            heavy_zone=False,
            team_ay_att_z=0.0,
            light_box_share=None,
            heavy_box_share=None,
            win_prob=win_prob,       # from volume model
            qb_inconsistent=False,
            pressure_mismatch=False,
        )

        # ---------- price at the quoted line ----------
        if side in ("Over", "Yes"):
            model_p = over_prob_normal(line, mu, sigma)
        else:
            model_p = 1 - over_prob_normal(line, mu, sigma)

        p_blend = blend(model_p, market_p_fair, w_model=0.65)
        fair_american = prob_to_american(p_blend)
        edge = edge_pct(p_blend, market_p_fair)
        kelly = kelly_fraction(p_blend, price, cap=0.05)

        rows.append({
            **r.to_dict(),
            "model_mu": mu, "model_sigma": sigma,
            "model_prob": model_p, "market_prob_fair": market_p_fair,
            "blend_prob": p_blend, "blend_fair_odds": fair_american,
            "edge_pct": edge, "kelly_cap": kelly, "tier": tier(edge),
            "notes": notes,
            "plays_est": plays, "pass_rate_est": pass_rate, "rush_rate_est": rush_rate,
            "win_prob_est": win_prob,
        })

    out = pd.DataFrame(rows)

    # --- ensure output dir exists BEFORE writing ---
    out_dir = out_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Write outputs
    out.to_csv(f"{out_dir}/props_priced.csv", index=False)
    game_df.to_csv(f"{out_dir}/game_lines.csv", index=False)

    return {"game_lines": game_df, "props_priced": out}
