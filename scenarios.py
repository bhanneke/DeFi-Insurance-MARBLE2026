#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the stress-test and pool-return sensitivity results of Section 6.

Scenarios (all inherit the Table-2 baseline of defi_insurance_simulation.py,
seeded and deterministic):

  baseline   verification metrics for the baseline itself
  stress     4x hack arrival intensity (incident_scale = 4.0):
             claims ~$132.0M (256 bps/yr), prudential cap binds on ~30% of
             days (vs ~7% baseline), NO insolvencies (payouts respect the
             enforced cap), median LP APY ~ -4.3%
  rpool05    pool return lowered to r_market (5%): median LP APY ~5.1%,
             ~40 bps below the participation threshold r_market + rho_LP;
             average utilization rises to ~11.0
  rpool06    pool return 6%: median LP APY ~6.3%, above the threshold

Usage:
  python scenarios.py stress            # full 1,000-run scenario (~minutes)
  python scenarios.py stress --runs 50  # quick pass
Metrics are printed as JSON and written to outputs/scenarios/<name>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import defi_insurance_simulation as sim

SCENARIOS = {
    "baseline": {},
    "stress": {"incident_scale": 4.0},
    "rpool05": {"r_pool": 0.05},
    "rpool06": {"r_pool": 0.06},
}


def metrics_from_runs(runs, params):
    out = {}
    U = np.stack([r["U"] for r in runs])
    Umax = np.stack([r["Umax"] for r in runs])
    graw = np.stack([r["gamma_raw"] for r in runs])
    gsim = np.stack([r["gamma"] for r in runs])
    out["avg_U"] = float(U.mean())
    out["avg_Umax"] = float(Umax.mean())
    out["avg_gamma_sim"] = float(gsim.mean())
    # coverage_eff = min(raw, Umax*CLP), so the cap binds iff U == Umax
    out["cap_binding_share_days"] = float(np.isclose(U, Umax, rtol=1e-9).mean())
    out["gamma_raw_unit_clip_share_days"] = float(((graw >= 1.0) | (graw <= 0.0)).mean())
    # collateral-to-coverage (t = 0)
    ratios = np.concatenate([r["CC0"] / np.maximum(r["coverage0"], 1e-12) for r in runs])
    out["median_CC_over_coverage"] = float(np.median(ratios))
    out["aggregate_CC_over_coverage"] = float(np.mean(
        [r["CC0"].sum() / r["coverage0"].sum() for r in runs]))
    out["mean_c_max"] = float(np.mean(
        [r["coverage0"].max() / r["coverage0"].sum() for r in runs]))
    # solvency
    sf = np.array([float(r["daily_shortfall"].sum()) for r in runs])
    out["share_runs_with_shortfall"] = float((sf > 0).mean())
    out["mean_shortfall_per_run_M"] = float(sf.mean())
    # claims and pricing
    cov_years = np.array([float(np.sum(r["coverage_eff"]) / 365.0) for r in runs])
    claims = np.array([float(r["cum_payouts"][-1]) for r in runs])
    out["mean_covered_years_M"] = float(cov_years.mean())
    out["mean_claims_M"] = float(claims.mean())
    out["loss_rate_bps"] = float(1e4 * claims.mean() / cov_years.mean())
    out["mean_prot_rev_M"] = float(np.mean([r["cum_protocol_share"][-1] for r in runs]))
    # net protocol cost = forfeited collateral + opportunity cost - yield share
    years = len(runs[0]["U"]) / 365.0
    burns = np.array([float(r["cum_burn_CC"][-1]) for r in runs])
    opp = np.array([params.r_market * float(np.mean(r["sumCC"])) * years for r in runs])
    out["net_cost_bps"] = float(
        1e4 * (burns.mean() + opp.mean() - out["mean_prot_rev_M"]) / cov_years.mean())
    # run-level APYs (same accounting as gather_runlevel_metrics)
    df = sim.gather_runlevel_metrics(runs, params)
    for col in ("lp_apy", "prot_apy"):
        out[f"median_{col}"] = float(df[col].median())
        out[f"p5_{col}"] = float(np.percentile(df[col], 5))
        out[f"p95_{col}"] = float(np.percentile(df[col], 95))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--runs", type=int, default=None,
                    help="override n_mc_runs (paper baseline: 1000)")
    args = ap.parse_args()

    overrides = dict(SCENARIOS[args.scenario])
    if args.runs:
        overrides["n_mc_runs"] = args.runs
    params = sim.SimulationParams(**overrides)
    runs = sim.run_monte_carlo(params)

    out = metrics_from_runs(runs, params)
    out["scenario"] = args.scenario
    out["n_mc_runs"] = params.n_mc_runs

    dest = Path("outputs/scenarios")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{args.scenario}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
