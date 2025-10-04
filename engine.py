import pandas as pd, numpy as np
from scripts.odds_api import fetch_game_lines, fetch_props_all_events
from scripts.features_external import build_external
from scripts.id_map import map_players
from scripts.pricing import (american_to_prob, devig_two_way, blend, prob_to_american,
                             edge_pct, kelly_fraction, tier)
from scripts.model_core import (base_sigma, over_prob_normal,
                                estimate_team_rates, player_shares,
                                mu_receptions, mu_rec_yards, mu_rush_atts, mu_rush_yards, mu_pass_yards)
from scripts.elite_rules import (funnel_multiplier, volatility_widen, pace_smoothing, sack_to_attempts)
from scripts.engine_helpers import make_team_last4_from_player_form

def run_pipeline(target_date: str, season: int, out_dir: str = "outputs"):
    # --- 1) Odds
    game_df  = fetch_game_lines()
    props_df = fetch_props_all_events()

    # --- 2) External features
    ext = build_external(season)
    ids = ext["ids"]; team_form = ext["team_form"]; pform = ext["player_form"]

    # Quick team context map (L4 totals for shares)
    team_l4_map = make_team_last4_from_player_form(pform)

    # --- 3) Map player names → ids/position/team
    props_df = map_players(props_df, ids)

    # Helper: team L4 row by team string (fallback defaults)
    def get_team_context(team):
        return team_l4_map.get(team, {"tgt_team_l4": 30.0, "rush_att_team_l4": 25.0})

    # --- 4) Price each prop
    rows = []
    for _, r in props_df.iterrows():
        market = r["market"]
        if market not in {
            "player_rec_yds","player_receptions",
            "player_rush_yds","player_rush_attempts",
            "player_pass_yds","player_pass_tds",  # TDs priced later if you want
            "player_anytime_td"
        }:
            continue

        price = r["price"]; line = r["point"]; side = r["outcome"]
        p_raw = american_to_prob(price)
        p_over_raw = p_raw if side in ("Over","Yes") else None
        p_under_raw = p_raw if side in ("Under","No") else None
        p_over_fair, p_under_fair = devig_two_way(p_over_raw, p_under_raw)
        market_p_fair = p_over_fair if side in ("Over","Yes") else p_under_fair

        # --- Volume baseline
        team = r.get("recent_team")
        team_ctx = get_team_context(team)
        # team rates from team_form: just use league-average if not found
        plays, pass_rate, rush_rate = 60.0, 0.55, 0.45
        # Optional: if you want per-team exact rates, you can join by team/week

        # Player shares & efficiencies from player_form (latest)
        pid = r.get("gsis_id")
        pr = pform[pform["gsis_id"] == pid]
        if not pr.empty:
            pr = pr.sort_values("week").tail(1).iloc[0].to_dict()
        else:
            pr = {}
        tgt_share, rush_share, catch_rate, ypr, ypc = player_shares(pr, team_ctx)

        # apply funnel multipliers to efficiency (simple 4%)
        pass_mult = funnel_multiplier(True,  def_rush_epa_z=0.0, def_pass_epa_z=0.0)
        rush_mult = funnel_multiplier(False, def_rush_epa_z=0.0, def_pass_epa_z=0.0)

        # pace smoothing (z-scores unknown here → no-op; leaving hook)
        plays = pace_smoothing(plays, 0.0, 0.0)

        # ---- market-specific μ/σ
        sigma = base_sigma(market)
        volatility_flag = False

        if market == "player_receptions":
            mu_rec = mu_receptions(plays, pass_rate, tgt_share, catch_rate) * pass_mult
            mu = mu_rec
        elif market == "player_rec_yds":
            mu_rec = mu_receptions(plays, pass_rate, tgt_share, catch_rate) * pass_mult
            mu = mu_rec_yards(mu_rec, ypr)
        elif market == "player_rush_attempts":
            atts = mu_rush_atts(plays, rush_rate, rush_share) * rush_mult
            atts = sack_to_attempts(atts, sack_rate_above_avg=0.0)
            mu = atts
        elif market == "player_rush_yds":
            atts = mu_rush_atts(plays, rush_rate, rush_share) * rush_mult
            ypc_eff = ypc
            mu = mu_rush_yards(atts, ypc_eff)
        elif market == "player_pass_yds":
            # approximate team dropbacks from plays*pass_rate; qb ypa from player_form
            qb_ypa = (pr.get("pass_yds_l4", 0)/max(pr.get("pass_att_l4", 1), 1)) if pr else 6.8
            dropbacks = plays * pass_rate
            mu = mu_pass_yards(dropbacks, qb_ypa, z_opp_pressure=0.0, z_opp_pass_epa=0.0)
            # volatility widening hook
            sigma = volatility_widen(sigma, pressure_mismatch=False, qb_inconsistent=False)
        else:
            # simple placeholder for TDs; you can improve later
            mu = 0.0

        # ---- convert μ/σ to model probability at quoted line
        if side in ("Over","Yes"):
            model_p = over_prob_normal(line, mu, sigma)
        else:
            model_p = 1 - over_prob_normal(line, mu, sigma)

        # Blend with market
        p_blend = blend(model_p, market_p_fair, w_model=0.65)
        fair_american = prob_to_american(p_blend)
        edge = edge_pct(p_blend, market_p_fair)
        kelly = kelly_fraction(p_blend, price, cap=0.05)

        rows.append({
            **r.to_dict(),
            "model_mu": mu, "model_sigma": sigma,
            "model_prob": model_p, "market_prob_fair": market_p_fair,
            "blend_prob": p_blend, "blend_fair_odds": fair_american,
            "edge_pct": edge, "kelly_cap": kelly, "tier": tier(edge)
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"{out_dir}/props_priced.csv", index=False)
    game_df.to_csv(f"{out_dir}/game_lines.csv", index=False)
    return {"game_lines": game_df, "props_priced": out}
