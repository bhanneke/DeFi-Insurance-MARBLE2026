# Symbolic verification of the analytical results

`verify_analytical_results.py` uses [SymPy](https://www.sympy.org) to derive
or verify, step by step, the analytical results of the MARBLE 2026 paper
*Decentralized Finance: A Market Mechanism for Cybersecurity Risk Insurance*,
and to certify that the closed forms match the implementation in
`../defi_insurance_simulation.py`.

## What is verified

| # | Paper result | What the script does |
|---|---|---|
| 1 | Eq. (2), Theorem 1 (A1) protocol side | Derives the protocol FOC, solves it for the closed-form collateral choice `CC*`, asserts it equals `protocol_target_CC` in the simulation, and shows the payoff is strictly concave in collateral for θ ∈ (0,1) |
| 2 | Eq. (4) hazard fit, Eq. (7) anchor | Least-squares fit recovers the true hazard on an exponential term structure (unique minimum); the anchor is a probability in (0,1) |
| 3 | Eq. (5) prudential cap | U_max(t) strictly decreases in the market-implied annual hack probability, with limits U_max^hi (p→0) and U_min (p→∞) |
| 4 | Eqs. (6)–(7) | Lemma: under a flat hazard, the short-maturity-weighted risk index never exceeds the annual anchor, so the risk ratio in Eq. (8) is ≤ 1 |
| 5 | Eq. (8), premise of Theorem 1 | The yield share is strictly increasing in utilization and in the risk index |
| 6 | Prop. 2(i), Eq. (14) + Appendix C corollary | Solving r_LP ≥ r_market + ρ_LP for γ reproduces the participation bound exactly; solving for r_pool reproduces the minimum-pool-yield corollary |
| 7 | Capital-expansion corollary | ∂γ_min/∂C_S < 0 unconditionally; for C_C the exact sign condition is derived and certified at the Table-2 calibration |
| 8 | Prop. 2(ii), Eq. (15) | The local stability condition dr_LP/dC_LP < 0 is derived in closed form and certified at the Table-2 calibration; the (clipped) LP payoff is numerically unimodal in C_LP (assumption (A1), LP side) |
| 9 | Prop. 3, Eq. (17) | The solvency bound U ≤ 1/(ψ·c_max) and its k-hack generalization; at the simulated concentration (c_max ≈ 0.5%) the bound (~208) far exceeds the average dynamic cap (~21) |
| 10 | Code ↔ paper | `protocol_target_CC`, `Umax_from_paper`, and `gamma_from_paper` agree with the symbolic Eqs. (2), (5), (8) on 200 random parameter draws each |

## What is not (and cannot be) mechanized

- **Proposition 1** (truthful risk assessment) is a free-entry no-arbitrage
  argument; there is no algebra to verify beyond the fair-value identity
  V = p·DF of Eq. (12).
- **Theorem 1 existence** invokes Berge's maximum theorem and Kakutani's
  fixed-point theorem on compact strategy sets. The script verifies the
  primitives those arguments consume (continuity/concavity of the protocol
  payoff, monotonicity and boundedness of γ) and certifies quasi-concavity
  of the LP payoff numerically at the baseline calibration — consistent with
  the paper's statement that (A1) holds when the yield-share feedback is
  weak relative to coverage concavity.

## Run

```bash
pip install sympy numpy
python analytical/verify_analytical_results.py
```

Exit code 0 and `All checks passed.` on success. `verification_log.txt`
contains the output of the run committed alongside this file.
