#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeFi Insurance Simulation — Paper-Aligned (v6)
- Correct split of pool revenue (LP/Protocol/Operator via γ and φ)
- Spec fees on TOTAL CAPITAL; added to pool revenue and split
- Track realized daily returns for LPs and Protocols
- Compute covered-years, empirical loss rate, net cost to protocols
- Print a concise market summary after MC
All currency units are $M (millions).
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict
import pandas as pd
from pathlib import Path
import pickle

# ---------------------------- Parameters ----------------------------

@dataclass
class SimulationParams:
    # Horizon
    n_days: int = 365 * 2
    n_protocols: int = 500
    n_mc_runs: int = 1000  # paper baseline; reduce (e.g. 200) for a quick run
    seed: int = 1234

    # Eq. (2): coverage = mu * CC^theta * (1 + xi_i)   (all in $M)
    mu: float = 3.0
    theta: float = 0.50

    # Operator fee
    phi: float = 0.01

    # Prudential cap U_max (Eq. 5)
    U_min: float = 1.0
    U_max_hi: float = 30.0
    kappa_Ucap: float = 100.0
    use_fixed_Umax: bool = False

    # Hack arrival multiplier relative to market-implied rates (stress: 4.0)
    incident_scale: float = 1.0

    # Pricing mix for gamma (Eq. 8)
    alpha: float = 0.6
    beta: float = 1.
    delta: float = 0.7
    omega: Tuple[float, float, float, float] = (0.40, 0.30, 0.20, 0.10)

    # Targets
    U_target: float = 15.0

    # Market returns / premia (annual)
    r_market: float = 0.05
    r_pool: float = 0.1
    rho_LP: float = 0.005
    kappa_LP: float = 2   # LP capital adjustment speed

    # Speculator fees (annualized yields applied to total capital via pool split)
    fee_base_annual: float = 0.03   # base fees
    fee_hack_jump: float = 0.1     # extra on hack day (annualized, one-day)

    # Initial pool capital ($M)
    CLP0: float = 250.0

    # Optional reinvestment of protocol share (False = paid out)
    retain_protocol_share: bool = False

# ---------------------------- Helpers ----------------------------

def infer_lambda_from_term_structure(P_hack_vec, T_vec=(0.25, 0.5, 0.75, 1.0)):
    P = np.array(P_hack_vec, dtype=float)
    lam_grid = np.linspace(0.0, 0.05, 4001)
    diffs = (1.0 - np.exp(-np.outer(lam_grid, T_vec))) - P[None, :]
    errs = np.square(diffs).sum(axis=1)
    lam_hat = lam_grid[int(np.argmin(errs))]
    return lam_hat

def compute_p_anchor_p_risk(P_hack_vec, omega):
    lam = infer_lambda_from_term_structure(P_hack_vec)
    p_anchor = 1.0 - np.exp(-lam)
    w = np.array(omega, dtype=float); w = w / w.sum()
    p_risk = float((w * np.array(P_hack_vec)).sum())
    return lam, p_anchor, p_risk

def Umax_from_paper(p_anchor, params: SimulationParams):
    if params.use_fixed_Umax or params.kappa_Ucap == 0.0:
        return params.U_max_hi
    return params.U_min + (params.U_max_hi - params.U_min) / (1.0 + params.kappa_Ucap * p_anchor)

def gamma_from_paper(U, U_target, p_risk, p_anchor, alpha, beta, delta):
    rU = (U / max(U_target, 1e-12)) ** beta
    rP = (p_risk / max(p_anchor, 1e-12)) ** delta
    g = alpha * rU + (1.0 - alpha) * rP
    return float(np.clip(g, 0.0, 1.0))

def protocol_target_CC(mu, theta, xi_i, rhoP_i, p_anchor, r_market, TVL_i):
    # FOC of Eq. (9) incl. expected forfeiture -p*E[min(CC, Loss)]: in the
    # relevant region (CC < covered loss) the marginal collateral cost is
    # r_market + p_anchor, not r_market alone.
    denom = (1.0 + rhoP_i) * max(p_anchor, 1e-12) * mu * theta * (1.0 + xi_i)
    CC_foc = ((r_market + p_anchor) / denom) ** (1.0 / (theta - 1.0))
    CC_cap = (TVL_i / (mu * (1.0 + xi_i) + 1e-12)) ** (1.0 / theta)
    return float(max(0.0, min(CC_foc, CC_cap)))

# ---------------------------- Population ----------------------------

def init_protocol_population(rng: np.random.Generator, params: SimulationParams) -> Dict[str, np.ndarray]:
    n = params.n_protocols

    # Heavy-tailed TVL up to $10B ($10,000M)
    TVL = 5.0 * (1.0 + rng.pareto(a=1.2, size=n))
    TVL = np.clip(TVL, 2.0, 10_000.0)

    # --- Protocol risk aversion ρP: lognormal, median 1.5, capped ---
    rho_med = 1.5          # median of ρP
    sigma_ln = 0.5        # lognormal σ: bigger = heavier right tail
    rhoP = rng.lognormal(mean=np.log(rho_med), sigma=sigma_ln, size=n)
    rhoP = np.clip(rhoP, 0.0, 3.0)   # cap extreme risk aversion (e.g., 3 or 5)


    # Security multiplier skewed left & increasing with size
    base = 0.6 + 1.0 * rng.beta(5.0, 2.0, size=n)
    rank = np.argsort(np.argsort(TVL)) / (n - 1 + 1e-9)
    xi = base * (1.0 + 0.4 * rank)
    xi = np.clip(xi, 0.6, 3.0)

    # Mean loss fraction
    Lbar = np.clip(rng.normal(0.5, 0.2, size=n), 0.1, 0.9)

    return dict(rhoP=rhoP, xi=xi, TVL=TVL, Lbar=Lbar)

def sample_hack_term_structure(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a HACK term structure consistent with a one-factor hazard model.
    """
    lam = rng.gamma(shape=2.0, scale=0.002)  # mean = 2 * 0.002 = 0.004 per protocol-year
    T_vec = np.array([0.25, 0.5, 0.75, 1.0])
    P = 1.0 - np.exp(-lam * T_vec)
    noise = rng.normal(0.0, 0.0015, size=4)
    P = np.clip(P + np.sort(noise), 0.0, 0.10)
    return P

# ---------------------------- Single-run simulation ----------------------------

def run_single_simulation(params: SimulationParams, run_seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(run_seed)
    pop = init_protocol_population(rng, params)

    nT = params.n_days
    T_years = np.arange(nT) / 365.0

    # State
    CLP = params.CLP0  # $M
    CC = np.zeros(params.n_protocols, dtype=float)  # $M

    # Outputs
    U_path        = np.zeros(nT, dtype=float)
    Umax_path     = np.zeros(nT, dtype=float)
    gamma_path    = np.zeros(nT, dtype=float)  # actual LP share
    gamma_fair_path = np.zeros(nT, dtype=float)
    gamma_raw_path  = np.zeros(nT, dtype=float)

    CLP_path       = np.zeros(nT, dtype=float)
    pool_total_path = np.zeros(nT, dtype=float)
    sumCC_path     = np.zeros(nT, dtype=float)

    daily_hacks      = np.zeros(nT, dtype=int)
    daily_payouts    = np.zeros(nT, dtype=float)  # total payouts (CC + LP)
    cum_payouts      = np.zeros(nT, dtype=float)
    daily_payouts_LP = np.zeros(nT, dtype=float)  # from CLP
    cum_payouts_LP   = np.zeros(nT, dtype=float)

    daily_burn_CC = np.zeros(nT, dtype=float)
    cum_burn_CC   = np.zeros(nT, dtype=float)
    cum_burn = 0.0
    daily_shortfall = np.zeros(nT, dtype=float)  # unpaid claims (insolvency events)
    CC0 = None; coverage0 = None

    coverage_eff_path = np.zeros(nT, float)
    lp_take_series    = np.zeros(nT, float)
    prot_take_series  = np.zeros(nT, float)
    op_take_series    = np.zeros(nT, float)
    lp_real_ret       = np.zeros(nT, float)
    prot_real_ret     = np.zeros(nT, float)

    cum_spec_fees     = np.zeros(nT, float)
    cum_lp_share      = np.zeros(nT, float)
    cum_protocol_share = np.zeros(nT, float)
    cum_operator_share = np.zeros(nT, float)

    proto_snapshot = {
        "protocol_id": np.arange(params.n_protocols),
        "rhoP":  pop["rhoP"],
        "xi":    pop["xi"],
        "TVL":   pop["TVL"],
        "Lbar":  pop["Lbar"],
    }

    cum_p   = 0.0
    cum_fee = 0.0
    cum_lp  = 0.0
    cum_pr  = 0.0
    cum_op  = 0.0
    cum_p_LP = 0.0

    for t in range(nT):
        # 1) term structure -> lambda_hat, p_anchor, p_risk
        P_hack = sample_hack_term_structure(rng)
        lam_hat, p_anchor, p_risk = compute_p_anchor_p_risk(P_hack, params.omega)

        # 2) U_max
        Umax_today = Umax_from_paper(p_anchor, params)

        # 3) Protocol best response: choose CC_i
        for i in range(params.n_protocols):
            CC[i] = protocol_target_CC(params.mu, params.theta,
                                       pop["xi"][i], pop["rhoP"][i],
                                       p_anchor, params.r_market,
                                       pop["TVL"][i])
        coverage_i  = params.mu * (CC ** params.theta) * (1.0 + pop["xi"])  # $M
        coverage_i  = np.minimum(coverage_i, pop["TVL"])  # coverage cannot exceed insured assets
        coverage_raw = coverage_i.sum()
        if t == 0:
            CC0 = CC.copy(); coverage0 = coverage_i.copy()

        # 4) Cap and U
        coverage_eff        = min(coverage_raw, Umax_today * CLP)
        U                   = coverage_eff / max(CLP, 1e-12)
        coverage_eff_path[t] = coverage_eff
        # pro-rata rationing when the cap binds: payouts are made on
        # effective (issued) coverage, consistent with covered-dollar-years
        cov_scale           = coverage_eff / max(coverage_raw, 1e-12)

        # 5) hack event & payout (system-wide Poisson)
        lambda_day_per_proto = lam_hat / 365.0
        incident_scale       = getattr(params, "incident_scale", 1.0)
        lambda_day_system    = lambda_day_per_proto * params.n_protocols * incident_scale

        k            = rng.poisson(lambda_day_system)
        payout_total = 0.0
        payout_LP    = 0.0
        hack_today   = False

        if k > 0:
            weights = pop["TVL"] / pop["TVL"].sum()
            for _ in range(k):
                idx     = rng.choice(params.n_protocols, p=weights)
                loss_i  = pop["Lbar"][idx] * pop["TVL"][idx]
                cov_i   = float(min(params.mu * (CC[idx] ** params.theta) * (1.0 + pop["xi"][idx]), pop["TVL"][idx])) * cov_scale
                pay_i   = min(loss_i, cov_i)
                if pay_i <= 0.0:
                    continue

                # CC burns first
                from_CC = min(CC[idx], pay_i)
                CC[idx] -= from_CC
                remaining = pay_i - from_CC
                daily_burn_CC[t] += from_CC

                # Remaining from LP pool
                from_CLP = min(remaining, CLP)
                CLP -= from_CLP
                daily_shortfall[t] += remaining - from_CLP

                payout_total += (from_CC + from_CLP)
                payout_LP    += from_CLP

            CLP      = max(CLP, 1e-9)
            hack_today = payout_total > 0.0

        daily_hacks[t]      = min(int(k), 9)
        daily_payouts[t]    = payout_total
        daily_payouts_LP[t] = payout_LP
        cum_p              += payout_total
        cum_payouts[t]      = cum_p
        cum_p_LP           += payout_LP
        cum_payouts_LP[t]   = cum_p_LP

        cum_burn           += daily_burn_CC[t]
        cum_burn_CC[t]      = cum_burn

        # 6) Pool revenues and revenue split (with fair vs raw gamma)
        CLP_prev    = CLP
        sumCC_prev  = float(np.sum(CC))
        total_cap   = CLP_prev + sumCC_prev

        base_yield_today = (params.r_pool / 365.0) * total_cap
        fee_yield_day    = params.fee_base_annual / 365.0 + (
            params.fee_hack_jump / 365.0 if hack_today else 0.0
        )
        fees_today   = fee_yield_day * total_cap
        gross_pool_rev = base_yield_today + fees_today  # <-- defined before use

        # raw gamma from Eq.(8)
        gamma_raw  = gamma_from_paper(
            U, params.U_target, p_risk, p_anchor,
            params.alpha, params.beta, params.delta
        )
        # capital-proportional "fair" share
        gamma_fair = CLP_prev / max(total_cap, 1e-12)

        # blend raw risk-based and fair capital-based share
        share_LP = np.clip(0.5 * gamma_raw + 0.5 * gamma_fair, 0.0, 1.0)

        # store paths
        gamma_path[t]      = share_LP
        gamma_fair_path[t] = gamma_fair
        gamma_raw_path[t]  = gamma_raw

        # actual split of gross_pool_rev
        operator_take = params.phi * gross_pool_rev
        splitable     = (1.0 - params.phi) * gross_pool_rev

        lp_take   = share_LP * splitable
        prot_take = (1.0 - share_LP) * splitable

        # update states
        CLP += lp_take
        if params.retain_protocol_share:
            w = np.maximum(CC, 1e-9)
            w = w / w.sum()
            CC += prot_take * w

        # realized daily returns
        lp_income_today  = lp_take - payout_LP
        lp_real_ret[t]   = lp_income_today / max(CLP_prev, 1e-12)
        prot_real_ret[t] = prot_take      / max(sumCC_prev, 1e-12)

        # cumulative trackers
        cum_fee += fees_today
        cum_op  += operator_take
        cum_lp  += lp_take
        cum_pr  += prot_take

        cum_spec_fees[t]       = cum_fee
        cum_lp_share[t]        = cum_lp
        cum_protocol_share[t]  = cum_pr
        cum_operator_share[t]  = cum_op

        lp_take_series[t]   = lp_take
        prot_take_series[t] = prot_take
        op_take_series[t]   = operator_take

        # 7) LP capital flow / adjustment
        Lbar_w            = np.average(pop["Lbar"], weights=np.maximum(coverage_i, 1e-12))
        expected_loss_rate = p_anchor * Lbar_w  # your current choice
        r_fee             = params.fee_base_annual + (params.fee_hack_jump if hack_today else 0.0)
        r_income_to_LP    = share_LP * (1.0 - params.phi) * (params.r_pool + r_fee)
        r_LP_exp          = r_income_to_LP - expected_loss_rate

        # 8) LP capital flow (use REALIZED return instead of expected)
        r_LP_realized_today = lp_real_ret[t] * 365.0  # approximate annualized realized return

        target = params.r_market + params.rho_LP  # outside option

        CLP += (params.kappa_LP / 365.0) * (r_LP_realized_today - target) * CLP
        CLP = max(CLP, 1e-6)

        # 9) record state paths
        U_path[t]        = U
        Umax_path[t]     = Umax_today
        CLP_path[t]      = CLP
        sumCC_path[t]    = float(np.sum(CC))
        pool_total_path[t] = CLP + sumCC_path[t]

    return dict(
        U=U_path,
        Umax=Umax_path,
        gamma=gamma_path,
        gamma_fair=gamma_fair_path,
        gamma_raw=gamma_raw_path,
        CLP=CLP_path,
        pool_total=pool_total_path,
        sumCC=sumCC_path,
        daily_payouts=daily_payouts,
        cum_payouts=cum_payouts,
        daily_payouts_LP=daily_payouts_LP,
        cum_payouts_LP=cum_payouts_LP,
        daily_burn_CC=daily_burn_CC,
        cum_burn_CC=cum_burn_CC,
        daily_hacks=daily_hacks,
        cum_spec_fees=cum_spec_fees,
        cum_lp_share=cum_lp_share,
        cum_protocol_share=cum_protocol_share,
        cum_operator_share=cum_operator_share,
        coverage_eff=coverage_eff_path,
        lp_take=lp_take_series,
        prot_take=prot_take_series,
        op_take=op_take_series,
        lp_real_ret=lp_real_ret,
        prot_real_ret=prot_real_ret,
        proto_snapshot=proto_snapshot,
        T_years=T_years,
        daily_shortfall=daily_shortfall,
        CC0=CC0,
        coverage0=coverage0,
    )

# ---------------------------- MC, summary, plots ----------------------------

def run_monte_carlo(params: SimulationParams) -> List[Dict[str, np.ndarray]]:
    runs = []
    base_seed = params.seed
    print(f"Running {params.n_mc_runs} Monte Carlo simulations...")
    for k in range(1, params.n_mc_runs + 1):
        runs.append(run_single_simulation(params, run_seed=base_seed + k))
        if k % 50 == 0 or k == params.n_mc_runs:
            print(f"  Completed {k}/{params.n_mc_runs} runs")
    print("✓ Completed all runs\n")
    return runs

def summarize_runs(all_runs, params):
    import numpy as np
    keys = ["U","Umax","gamma","gamma_fair","gamma_raw","CLP","pool_total","sumCC","daily_payouts","cum_payouts",
            "daily_payouts_LP","cum_payouts_LP", 
            "cum_spec_fees","cum_lp_share","cum_protocol_share","cum_operator_share",
            "coverage_eff","lp_take","prot_take","op_take","lp_real_ret","prot_real_ret"]
    summary = {}
    for k in keys:
        mat = np.stack([r[k] for r in all_runs], axis=0)
        summary[k + "_mean"] = mat.mean(axis=0)
        summary[k + "_lo"]   = np.percentile(mat, 5, axis=0)
        summary[k + "_hi"]   = np.percentile(mat, 95, axis=0)

    # Scalars
    T = all_runs[0]["U"].shape[0]
    years = T / 365.0
    dt = 1/365.0

    # Covered-years (denominator for pricing)
    cov_years_runs = [float(np.sum(r["coverage_eff"]) * dt) for r in all_runs]  # $M·years
    claims_runs    = [float(r["cum_payouts"][-1]) for r in all_runs]
    lp_rev_runs    = [float(r["cum_lp_share"][-1]) for r in all_runs]
    prot_rev_runs  = [float(r["cum_protocol_share"][-1]) for r in all_runs]
    op_rev_runs    = [float(r["cum_operator_share"][-1]) for r in all_runs]

    # Realized APY (income over average capital) per run
    lp_apy_runs = []
    prot_apy_runs = []
    for r in all_runs:
        total_burn = float(r["cum_burn_CC"][-1])       # or np.sum(r["daily_burn_CC"])
        lp_income = float(np.sum(r["lp_take"]) - np.sum(r["daily_payouts_LP"]))
        avg_clp   = float(np.mean(r["CLP"]))
        prot_income  = float(np.sum(r["prot_take"]) - total_burn)   # <-- NEW
        avg_sumcc   = float(np.mean(r["sumCC"]))

        prot_rev     = float(np.sum(r["prot_take"]))
        avg_sumcc    = float(np.mean(r["sumCC"]))

        opp_cost     = params.r_market * avg_sumcc * years

        prot_income  = prot_rev - total_burn - opp_cost

        # guard against divide-by-zero
        if avg_clp <= 1e-9:
            lp_apy = 0.0
        else:
            lp_apy = lp_income / (avg_clp * years)

        if avg_sumcc <= 1e-9:
            prot_apy = 0.0
        else:
            prot_apy = prot_income / (avg_sumcc * years)

        lp_apy_runs.append(lp_apy)
        prot_apy_runs.append(prot_apy)
        
    summary["covered_years_$M_mean"] = float(np.mean(cov_years_runs))
    summary["claims_mean"]           = float(np.mean(claims_runs))
    summary["lp_rev_mean"]           = float(np.mean(lp_rev_runs))
    summary["prot_rev_mean"]         = float(np.mean(prot_rev_runs))
    summary["op_rev_mean"]           = float(np.mean(op_rev_runs))
    burn_mat = np.stack([r["cum_burn_CC"] for r in all_runs], axis=0)
    summary["cum_burn_CC_mean"] = burn_mat.mean(axis=0)
    summary["cum_burn_CC_lo"]   = np.percentile(burn_mat, 5, axis=0)
    summary["cum_burn_CC_hi"]   = np.percentile(burn_mat, 95, axis=0)

    # Pricing metrics
    summary["loss_rate_per_$Myear"] = summary["claims_mean"] / max(summary["covered_years_$M_mean"], 1e-12)
    # net cost of coverage to protocols (see gather_runlevel_metrics)
    burn_runs = [float(r["cum_burn_CC"][-1]) for r in all_runs]
    opp_runs  = [params.r_market * float(np.mean(r["sumCC"])) * years for r in all_runs]
    summary["net_cost_to_protocols_per_$Myear"] = (
        float(np.mean(burn_runs)) + float(np.mean(opp_runs)) - summary["prot_rev_mean"]
    ) / max(summary["covered_years_$M_mean"], 1e-12)

    # Realized APYs (MC mean + 5–95% band for context)
    summary["LP_realized_APY"]        = float(np.mean(lp_apy_runs))
    summary["LP_realized_APY_lo"]     = float(np.percentile(lp_apy_runs, 5))
    summary["LP_realized_APY_hi"]     = float(np.percentile(lp_apy_runs, 95))
    summary["Protocol_realized_APY"]  = float(np.mean(prot_apy_runs))
    summary["Protocol_realized_APY_lo"] = float(np.percentile(prot_apy_runs, 5))
    summary["Protocol_realized_APY_hi"] = float(np.percentile(prot_apy_runs, 95))

    summary["T_years"] = all_runs[0]["T_years"]
    return summary

def plot_results(summary: Dict[str, np.ndarray], params: SimulationParams, save_path: str = None):
    t = summary["T_years"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # A: Utilization & Umax
    ax = axs[0][0]
    ax.fill_between(t, summary["U_lo"], summary["U_hi"], alpha=0.25, label="U 90% CI")
    lU, = ax.plot(t, summary["U_mean"], lw=2, label="U (mean)")
    lUmx, = ax.plot(t, summary["Umax_mean"], lw=1, ls="--", label="U_max (mean)")
    ax.set_title("Utilization & Prudential Cap"); ax.set_xlabel("Years"); ax.set_ylabel("U")
    ax.legend(loc="upper left")

    # B: Gamma
    ax = axs[0][1]
    ax.fill_between(t, summary["gamma_lo"], summary["gamma_hi"], alpha=0.25, label="γ 90% CI")
    lg, = ax.plot(t, summary["gamma_mean"], lw=2, label="γ (mean)")
    ax.set_title("Revenue Share γ(t)"); ax.set_xlabel("Years"); ax.set_ylabel("γ")
    ax.legend(loc="upper left")

    # --- C: Capital composition (CLP + Collateral), both with 90% CI ---
    ax = axs[1][0]

    # CLP CI band
    ax.fill_between(
        t,
        summary["CLP_lo"],
        summary["CLP_hi"],
        alpha=0.20,
        color="tab:blue",
        label="CLP 90% CI"
    )

    # CLP mean
    l1, = ax.plot(
        t,
        summary["CLP_mean"],
        lw=2,
        color="tab:blue",
        label="CLP (mean)"
    )

    # Collateral CI
    ax.fill_between(
        t,
        summary["sumCC_lo"],
        summary["sumCC_hi"],
        alpha=0.15,
        color="tab:orange",
        label="Σ Collateral 90% CI"
    )

    # Collateral mean
    l2, = ax.plot(
        t,
        summary["sumCC_mean"],
        lw=1.8,
        ls="--",
        color="tab:orange",
        label="Σ Collateral (mean)"
    )

    ax.set_title("Capital Pool Composition ($M)")
    ax.set_xlabel("Years")
    ax.set_ylabel("$M")

    # Clean legend
    ax.legend(loc="upper left")

    # D: Cumulative payouts
    ax = axs[1][1]
    ax.fill_between(t, summary["cum_payouts_lo"], summary["cum_payouts_hi"], alpha=0.25, label="90% CI")
    lp, = ax.plot(t, summary["cum_payouts_mean"], lw=2, label="Cumulative payouts")
    ax.set_title("Cumulative Hack Payouts ($M)"); ax.set_xlabel("Years"); ax.set_ylabel("$M")
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        out = Path(save_path); out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        print(f"✓ Figure saved to: {out.resolve()}")
    else:
        plt.show()

def plot_protocol_distribution(snapshot: Dict[str, np.ndarray], path: str = "outputs/protocol_distribution_hist.png"):
    import matplotlib.pyplot as plt, numpy as np, os
    os.makedirs(Path(path).parent, exist_ok=True)
    TVL = snapshot["TVL"]; rhoP = snapshot["rhoP"]; xi = snapshot["xi"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].hist(TVL, bins=60);  axs[0].set_title("Protocol TVL ($M)")
    axs[1].hist(rhoP, bins=40); axs[1].set_title("Risk Aversion ρ_P")
    axs[2].hist(xi, bins=40);   axs[2].set_title("Security Multiplier ξ")
    for ax in axs: ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"✓ Protocol distribution figure saved to: {Path(path).resolve()}")

def plot_return_decomposition(summary: Dict[str, np.ndarray],
                              path: str = "outputs/returns_decomposition.png",
                              show_pct_stack: bool = False):
    """
    Figure A: cumulative $ flows (mean across MC runs)
      - LP gross, LP net (after payouts), Protocol share, Operator share
    Figure B (optional): % share of cumulative revenue (stacked area)
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    t = summary["T_years"]

    # mean paths pulled from summary (already averaged across MC)
    lp_gross   = summary["cum_lp_share_mean"]
    prot_gross = summary["cum_protocol_share_mean"]
    op_gross   = summary["cum_operator_share_mean"]

    payouts_total = summary["cum_payouts_mean"]         # still used for context if needed
    payouts_LP    = summary["cum_payouts_LP_mean"]      # NEW: LP-funded payouts only
    burns_CC   = summary["cum_burn_CC_mean"]          # NEW
    prot_net = prot_gross - burns_CC                  # NEW
    lp_net = lp_gross - payouts_LP

    if show_pct_stack:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"wspace": 0.25})
    else:
        fig, axs = plt.subplots(1, 1, figsize=(8.5, 5))
        axs = np.array([axs])

    # --- Panel A: cumulative $ ---
    ax = axs[0]
    ax.plot(t, lp_gross,   lw=2, label="LP gross ($M)")
    ax.plot(t, lp_net,     lw=2, ls="--", label="LP net ($M)")
    ax.plot(t, prot_gross, lw=1.8, label="Protocol share ($M)")
    ax.plot(t, prot_net,   lw=1.8, ls="--", label="Protocol net ($M)")   # NEW
    ax.plot(t, op_gross,   lw=1.5, label="Operator share ($M)")
    ax.set_title("Cumulative Revenue & Net to LPs")
    ax.set_xlabel("Years"); ax.set_ylabel("$M")
    ax.legend(loc="upper left")

    # --- Panel B (optional): stacked % of cumulative revenue (ex payouts) ---
    if show_pct_stack:
        ax = axs[1]
        total_rev = lp_gross + prot_gross + op_gross
        total_rev = np.maximum(total_rev, 1e-9)
        lp_pct   = lp_gross   / total_rev
        prot_pct = prot_gross / total_rev
        op_pct   = op_gross   / total_rev
        ax.stackplot(t, lp_pct, prot_pct, op_pct, labels=["LP", "Protocol", "Operator"])
        ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1.0])
        ax.set_title("Share of Cumulative Revenue (ex payouts)")
        ax.set_xlabel("Years"); ax.set_ylabel("Share")
        ax.legend(loc="lower right")

    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"✓ Return decomposition figure saved to: {out.resolve()}")


def gather_runlevel_metrics(all_runs, params):
    rows = []
    import numpy as np, pandas as pd
    for r in all_runs:
        T     = len(r["U"])
        years = T / 365.0

        covered_years_M = float(np.sum(r["coverage_eff"]) / 365.0)
        claims_M        = float(r["cum_payouts"][-1])

        # Revenues
        lp_rev_M   = float(np.sum(r["lp_take"]))
        prot_rev_M = float(np.sum(r["prot_take"]))

        # Losses & burns
        lp_losses_M = float(np.sum(r["daily_payouts_LP"]))
        total_burn  = float(r["cum_burn_CC"][-1])

        avg_clp   = float(np.mean(r["CLP"]))
        avg_sumcc = float(np.mean(r["sumCC"]))

        # Opportunity cost for protocols
        opp_cost  = params.r_market * avg_sumcc * years

        # Net incomes
        lp_income   = lp_rev_M   - lp_losses_M
        prot_income = prot_rev_M - total_burn - opp_cost

        lp_apy   = 0.0 if avg_clp   <= 1e-9 else lp_income   / (avg_clp   * years)
        prot_apy = 0.0 if avg_sumcc <= 1e-9 else prot_income / (avg_sumcc * years)

        loss_rate = claims_M / max(covered_years_M, 1e-12)
        # net cost of coverage to protocols: forfeited collateral plus
        # collateral opportunity cost minus yield share (claims received are
        # the insurance benefit, not a cost) = -prot_income per covered-year
        net_cost  = (total_burn + opp_cost - prot_rev_M) / max(covered_years_M, 1e-12)

        rows.append(dict(
            lp_apy=lp_apy,
            prot_apy=prot_apy,
            loss_rate_per_Myear=loss_rate,
            net_cost_per_Myear=net_cost,
            cum_payouts_M=claims_M
        ))
    return pd.DataFrame(rows)

def plot_boxplots(df, path="outputs/boxplots_across_runs.png"):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np

    # Convert to paper-friendly units
    lp_pct   = 100.0 * df["lp_apy"]
    prot_pct = 100.0 * df["prot_apy"]
    loss_bps = 1e4   * df["loss_rate_per_Myear"]      # $/($M·yr) → bps/yr
    nc_bps   = 1e4   * df["net_cost_per_Myear"]
    payouts  = df["cum_payouts_M"]

    metrics = [
        ("LP APY (%)", lp_pct),
        ("Protocol APY (%)", prot_pct),
        ("Loss rate (bps/yr)", loss_bps),
        ("Net cost (bps/yr)", nc_bps),
        ("Cum payouts ($M)", payouts),
    ]

    fig, axs = plt.subplots(1, len(metrics), figsize=(14, 4))
    for i, (title, data) in enumerate(metrics):
        ax = axs[i]
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor="lightsteelblue", alpha=0.6),
                        medianprops=dict(color="darkblue", lw=1.6))
        ax.set_title(title)
        ax.set_xticks([]); ax.grid(axis="y", alpha=0.25)

        # Summary stats
        q5, med, q95 = np.percentile(data, [5, 50, 95])
        # Draw text under axis (slightly below the plot area)
        ax.text(1, ax.get_ylim()[0] - 0.05*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                f"5%={q5:.2f}\nMed={med:.2f}\n95%={q95:.2f}",
                ha="center", va="top", fontsize=8, linespacing=0.9)

    plt.tight_layout()
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"✓ Box plots with labeled stats saved to: {out.resolve()}")

def print_lp_apy_decomposition(summary, params):
    tot = float(np.mean(summary["pool_total_mean"]))
    clp = float(np.mean(summary["CLP_mean"]))
    fees_base = params.fee_base_annual
    # avg γ and hack fee impact (approx): use mean gamma and spec fees share from cum paths
    gamma_bar = float(np.mean(summary["gamma_mean"]))
    phi = params.phi
    # estimate average annual spec fees from cum series slope:
    T = len(summary["gamma_mean"])
    years = T/365.0
    spec_rev = (summary["cum_spec_fees_mean"][-1]) / years  # $M/yr (mean run)
    base_rev = params.r_pool * tot                           # $M/yr
    gross = base_rev + spec_rev
    lp_rev = (1-phi)*gamma_bar * gross                      # $M/yr to LP
    payouts = (summary["cum_payouts_LP_mean"][-1]) / years     # $M/yr
    lp_net = lp_rev - payouts
    apy_est = lp_net / max(clp,1e-9)
    print(f"[LP APY decomposition]")
    print(f"  total_cap ~ ${tot:.0f}M, CLP ~ ${clp:.0f}M")
    print(f"  base_rev ~ ${base_rev:.1f}M/yr, spec_rev ~ ${spec_rev:.1f}M/yr, gross ~ ${gross:.1f}M/yr")
    print(f"  gamma ~ {gamma_bar:.2f}, phi ~ {phi:.2f} → LP_rev ~ ${lp_rev:.1f}M/yr")
    print(f"  payouts ~ ${payouts:.1f}M/yr → LP_net ~ ${lp_net:.1f}M/yr")
    print(f"  implied LP APY ~ {100*apy_est:.1f}%")

def summarize_hacks(all_runs):
    import numpy as np

    all_event_sizes = []
    events_per_run = []

    for r in all_runs:
        daily = r["daily_payouts"]       # total hack-paid amount per day
        # count events as >0 payouts
        events = daily[daily > 0]
        events_per_run.append(len(events))
        all_event_sizes.extend(events.tolist())

    all_event_sizes = np.array(all_event_sizes)

    if len(all_event_sizes) == 0:
        print("\n[Hack Summary] No hack events recorded.\n")
        return
    print("\n======================  HACK EVENT SUMMARY  ======================")
    print(f"Total hack events across all runs: {len(all_event_sizes)}")
    print(f"Average events per run: {np.mean(events_per_run):.2f}")
    print(f"Median events per run:  {np.median(events_per_run):.2f}")
    print(f"Run-level event count 5–95%ile: "
          f"{np.percentile(events_per_run,5):.0f} – {np.percentile(events_per_run,95):.0f}")

    print("\n--- Hack Size Distribution ($M) ---")
    print(f"Mean hack size:    {np.mean(all_event_sizes):.3f} M")
    print(f"Median hack size:  {np.median(all_event_sizes):.3f} M")
    print(f"5th–95th pct:      {np.percentile(all_event_sizes,5):.3f} – {np.percentile(all_event_sizes,95):.3f} M")
    print(f"Max hack size:     {np.max(all_event_sizes):.3f} M")
    print("==================================================================\n")


def summarize_protocol_TVL(all_runs):
    import numpy as np

    # use the population snapshot from the FIRST run (identical distribution each run)
    snap = all_runs[0]["proto_snapshot"]
    TVL = np.array(snap["TVL"])

    print("\n======================  PROTOCOL TVL SUMMARY  ======================")
    print(f"Number of protocols: {len(TVL)}")

    print(f"\n--- TVL Distribution ($M) ---")
    print(f"Mean TVL:        {np.mean(TVL):.2f} M")
    print(f"Median TVL:      {np.median(TVL):.2f} M")
    print(f"5th–95th pct:    {np.percentile(TVL,5):.2f} – {np.percentile(TVL,95):.2f} M")
    print(f"Max TVL:         {np.max(TVL):.2f} M")

    # concentration
    sorted_TVL = np.sort(TVL)[::-1]
    total_TVL = np.sum(TVL)
    top10 = np.sum(sorted_TVL[:50])    # top 10% of 500 = 50 protocols
    top1  = np.sum(sorted_TVL[:5])     # top 1% = 5 protocols

    print("\n--- Concentration ---")
    print(f"Total TVL:       {total_TVL:.2f} M")
    print(f"Top 10% share:   {100 * top10 / total_TVL:.2f}%")
    print(f"Top 1% share:    {100 * top1  / total_TVL:.2f}%")

    # optional: hack-weighted TVL
    # (Are hacks hitting larger protocols more often?)
    all_event_TVL = []
    for r in all_runs:
        daily = r["daily_payouts"]
        # find events
        idxs = np.where(daily > 0)[0]
        for t in idxs:
            # sample_hack_term_structure picks based on weights = TVL
            # we can approximate by weighted mean TVL
            all_event_TVL.append(np.average(TVL, weights=TVL))

    if len(all_event_TVL) > 0:
        print(f"\nApprox hack-weighted TVL: {np.mean(all_event_TVL):.2f} M")
    print("==================================================================\n")

# ---------------------------- Summary Printer ----------------------------

def print_summary(summary, params):
    print("\n================  PAPER-ALIGNED MARKET SUMMARY  ================\n")
    print(f"Horizon: {params.n_days/365:.1f} years, MC runs: {params.n_mc_runs}, Protocols: {params.n_protocols}")
    print(f"Avg Utilization U: {np.mean(summary['U_mean']):.2f}")
    print(f"Avg Revenue Share γ: {np.mean(summary['gamma_mean']):.2f}")
    print(f"Covered-years (mean): ${summary['covered_years_$M_mean']:.2f}M·years")
    print(f"Total claims (mean):  ${summary['claims_mean']:.2f}M")
    print(f"Empirical loss rate:  ${summary['loss_rate_per_$Myear']:.4f} per $M·year")
    print(f"Net cost to protocols per covered-year: ${summary['net_cost_to_protocols_per_$Myear']:.4f} per $M·year")
    print(f"Cum LP gross revenue (split) mean:     ${summary['lp_rev_mean']:.2f}M")
    print(f"Cum Protocol gross revenue (split) mean:${summary['prot_rev_mean']:.2f}M")
    print(f"Cum Operator revenue mean:              ${summary['op_rev_mean']:.2f}M")

    print("\nRealized LP APY (net of payouts): "
      f"{summary['LP_realized_APY']:.2%} "
      f"[{summary['LP_realized_APY_lo']:.2%}, {summary['LP_realized_APY_hi']:.2%}]")

    print("Realized Protocol APY (net of CC burns): "
      f"{summary['Protocol_realized_APY']:.2%} "
      f"[{summary['Protocol_realized_APY_lo']:.2%}, {summary['Protocol_realized_APY_hi']:.2%}]")
    print("\n===============================================================\n")
    loss_bps = 1e4 * summary['loss_rate_per_$Myear']      # 10000 = 100 * 100 (to bps)
    netcost_bps = 1e4 * summary['net_cost_to_protocols_per_$Myear']
    print(f"Empirical loss rate:  {summary['loss_rate_per_$Myear']:.4f} $/($M·yr)  [{loss_bps:.2f} bps/yr]")
    print(f"Net cost to protocols per covered-year: {summary['net_cost_to_protocols_per_$Myear']:.4f} $/($M·yr)  [{netcost_bps:.2f} bps/yr]")



# ---------------------------- Main ----------------------------

def main():
    params = SimulationParams()

    outdir = Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- run sims ---
    runs = run_monte_carlo(params)

    # --- save runs + summary so we can replot later without rerunning MC ---
    with open(outdir / "runs.pkl", "wb") as f:
        pickle.dump(runs, f)

    summary = summarize_runs(runs, params)
    with open(outdir / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    # boxplots
    df_runs = gather_runlevel_metrics(runs, params)
    df_runs.to_csv(outdir / "runlevel_metrics.csv", index=False)
    plot_boxplots(df_runs, path=str(outdir / "boxplots_across_runs.png"))

    # protocol distribution
    plot_protocol_distribution(
        runs[0]["proto_snapshot"],
        path=str(outdir / "protocol_distribution_hist.png")
    )
    pd.DataFrame(runs[0]["proto_snapshot"]).to_csv(outdir / "protocol_distribution.csv", index=False)

    # main dashboard + returns plot
    print_summary(summary, params)
    plot_results(summary, params, save_path=str(outdir / "simulation_results.png"))
    plot_return_decomposition(summary, path=str(outdir / "returns_decomposition.png"), show_pct_stack=False)
    summarize_hacks(runs)
    summarize_protocol_TVL(runs)
if __name__ == "__main__":
    main()