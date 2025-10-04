import pandas as pd
from scripts.odds_api import fetch_game_lines, fetch_props_all_events
from scripts.features_external import build_external
from scripts.id_map import map_players
from scripts.model_core import (volume_mean, base_sigma, pressure_qb_adjust,
                                apply_funnel, widen_sigma, over_prob_normal)
from scripts.pricing import (american_to_prob, devig_two_way, blend, prob_to_american,
                             edge_pct, kelly_fraction, tier)

def run_pipeline(target_date: str, season: int, out_dir: str = "outputs"):
    # 1) Odds
    game_df = fetch_game_lines()
    props_df = fetch_props_all_events()

    # 2) External features
    ext = build_external(season)
    ids = ext["ids"]; team_form = ext["team_form"]; pform = ext["player_form"]

    # 2a) Map player names → ids (cache-aware)
    props_df = map_players(props_df, ids)

    # TODO: Join features here using gsis_id/team/week once you add scheduling logic for target_date
    # For now we price with scaffold μ/σ so pipeline runs end-to-end.

    # 3) Pricing
    rows = []
    for _, r in props_df.iterrows():
        market = r["market"]
        if market not in {
            "player_rec_yds","player_receptions",
            "player_rush_yds","player_rush_attempts",
            "player_pass_yds","player_pass_tds",
            "player_anytime_td"
        }:
            continue

        price = r["price"]; line = r["point"]; side = r["outcome"]
        p_raw = american_to_prob(price)

        p_over_raw = p_raw if side in ("Over","Yes") else None
        p_under_raw = p_raw if side in ("Under","No") else None
        p_over_fair, p_under_fair = devig_two_way(p_over_raw, p_under_raw)
        market_p_fair = p_over_fair if side in ("Over","Yes") else p_under_fair

        # ---- MODEL μ, σ (scaffold) ----
        mu = volume_mean({"mu_base": line if line is not None else 0.0})
        if market == "player_pass_yds":
            mu = pressure_qb_adjust(mu, z_opp_pressure=0.0, z_opp_epa_pass=0.0)
        mu = apply_funnel(mu, is_run_funnel=False, is_pass_funnel=False)
        sigma = widen_sigma(base_sigma(market), volatility_flag=False)

        # Over/Under prob at quoted line
        if side in ("Over","Yes"):
            model_p = over_prob_normal(line, mu, sigma)
        else:
            model_p = 1 - over_prob_normal(line, mu, sigma)

        # Blend 65/35
        p_blend = blend(model_p, market_p_fair, w_model=0.65)
        fair_american = prob_to_american(p_blend)
        edge = edge_pct(p_blend, market_p_fair)
        kelly = kelly_fraction(p_blend, price, cap=0.05)

        rows.append({
            **r.to_dict(),
            "gsis_id": r.get("gsis_id"),
            "model_mu": mu,
            "model_sigma": sigma,
            "model_prob": model_p,
            "market_prob_fair": market_p_fair,
            "blend_prob": p_blend,
            "blend_fair_odds": fair_american,
            "edge_pct": edge,
            "kelly_cap": kelly,
            "tier": tier(edge)
        })

    out = pd.DataFrame(rows)

    # 4) Write outputs
    out_dir = out_dir or "outputs"
    out.to_csv(f"{out_dir}/props_priced.csv", index=False)
    game_df.to_csv(f"{out_dir}/game_lines.csv", index=False)
    return {"game_lines": game_df, "props_priced": out}
