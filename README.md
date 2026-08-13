# DeFi Insurance — MARBLE 2026 replication package

Simulation code, symbolic verification, and reference outputs for:

> Hanneke, B.: *Decentralized Finance: A Market Mechanism for Cybersecurity
> Risk Insurance.* In: Proceedings of the 7th International Conference on
> Mathematical Research for Blockchain Economy (MARBLE 2026), Springer.
> To appear.

The paper proposes a two-layer market mechanism for on-chain cybersecurity
risk transfer: binary HACK/NOHACK prediction markets provide continuous,
market-implied hack probabilities (insurance pricing layer), while protocols
post forfeitable collateral to access coverage from a liquidity-provider
capital pool whose yield share adjusts dynamically to utilization and
market-priced risk (insurance provision layer).

This repository contains exactly the material behind the paper — nothing else.

## Contents

| Path | Description |
|---|---|
| `defi_insurance_simulation.py` | Monte-Carlo simulation behind Section 6 ("Stylized Simulation"): Table-2 baseline (θ=0.5, μ=3, U_target=15, dynamic prudential cap κ_U=100), insolvency-shortfall tracking, `incident_scale` stress multiplier |
| `scenarios.py` | Reproduces the stress test (4× hack intensity) and the pool-return sensitivity of Section 6 |
| `analytical/` | SymPy verification of the paper's analytical results (Theorem 1 primitives, Propositions 2–3 and corollaries) and their correspondence to the simulation — see `analytical/README.md`; all 29 checks pass (`verification_log.txt`) |
| `outputs/` | Reference outputs: Figs. 2–4 of the paper, per-run metrics, protocol population |
| `nexus_benchmark/` | Methodology behind the Nexus Mutual premium benchmark of Section 6 (median 2.6%/yr across 10,893 covers) |

## Reproducing the paper

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python defi_insurance_simulation.py      # baseline: Figs. 2-4 + headline metrics
python scenarios.py stress               # stress test numbers
python scenarios.py rpool05              # pool-return sensitivity (also: rpool06)
python analytical/verify_analytical_results.py   # symbolic verification
```

The baseline runs 1,000 seeded Monte Carlo runs (base seed 1234, per-run
offsets) over 500 heterogeneous protocols and a 2-year daily horizon, so all
results are deterministic: the figures in `outputs/` regenerate
**byte-identically**, and the printed headline metrics match Section 6 exactly
(average utilization 9.94, yield share 0.55, $6,155M covered-dollar-years,
loss rate 76.13 bps/yr, net protocol cost −70.35 bps/yr, LP median APY 11.23%).
A full baseline takes a few minutes; use `n_mc_runs = 200` in
`SimulationParams` (or `scenarios.py --runs 200`) for a quick pass.

Figure mapping: `boxplots_across_runs.png` = Fig. 2, `simulation_results.png`
= Fig. 3, `protocol_distribution_hist.png` = Fig. 4 (Fig. 1 is a hand-drawn
mechanism diagram).

## Code-to-paper mapping

| Paper | Code |
|---|---|
| Eq. (2) coverage function | `protocol_target_CC`, coverage computation in `run_single_simulation` step 3 |
| Eq. (3) utilization | `run_single_simulation` step 4 |
| Eq. (4) hazard-rate estimation from the HACK term structure | `infer_lambda_from_term_structure` |
| Eq. (5) prudential cap U_max(t) | `Umax_from_paper` (dynamic cap, `kappa_Ucap = 100`) |
| Eqs. (6)–(7) risk index and anchor | `compute_p_anchor_p_risk` |
| Eq. (8) yield-share γ | `gamma_from_paper` |
| γ_fair blending (η = 0.5, full-version Appendix B) | `run_single_simulation` step 6 |
| Prop. 2(ii) LP capital adjustment | `run_single_simulation` steps 7–8 |
| Table 2 baseline parameters | `SimulationParams` |
| Protocol population (Pareto TVL, risk aversion, security multipliers) | `init_protocol_population` |
| Hack arrivals and payout waterfall (collateral burns first, then LP pool) | `run_single_simulation` step 5 |
| Theorem 1 primitives, Props. 2–3 and corollaries (symbolic) | `analytical/verify_analytical_results.py` |
| Section 6 stress test and sensitivity | `scenarios.py` |
| Section 6 Nexus Mutual benchmark | `nexus_benchmark/nexus_premium_analysis.py` |

## Requirements

Python ≥ 3.9 with `numpy`, `pandas`, `matplotlib`; `sympy` for `analytical/`
(see `requirements.txt`).
