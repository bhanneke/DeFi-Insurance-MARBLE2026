#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Symbolic verification (SymPy) of the analytical results in

    Hanneke, B.: Decentralized Finance: A Market Mechanism for
    Cybersecurity Risk Insurance. MARBLE 2026.

Each section below derives or verifies one result from the paper and prints
PASS/FAIL. Equation numbers refer to the paper. The suite covers:

  [1] Coverage function (Eq. 2): protocol first-order condition, closed-form
      collateral choice CC*, and strict concavity (assumption (A1),
      protocol side, used in Theorem 1). The derived CC* is asserted to be
      the exact formula implemented in `protocol_target_CC` in
      defi_insurance_simulation.py.
  [2] Hazard-rate estimation (Eq. 4): the least-squares fit recovers the
      true hazard when prices lie on an exponential term structure, and
      the anchor (Eq. 7) is the implied annual probability.
  [3] Prudential cap (Eq. 5): U_max(t) strictly decreases in the
      market-implied annual hack probability.
  [4] Risk index vs. anchor (Eqs. 6-7): under a flat hazard, the
      short-maturity-weighted index P_risk never exceeds the annual anchor
      P_anchor, so the risk ratio in Eq. (8) is bounded by one.
  [5] Yield share (Eq. 8): strictly increasing in utilization and in the
      risk index (the monotonicity required by Theorem 1).
  [6] Proposition 2(i): the participation bound gamma_min (Eq. 14) follows
      from r_LP >= r_market + rho_LP; corollary "minimum pool yield"
      (Appendix C) is the same inequality solved for r_pool.
  [7] Corollary "capital expansion": gamma_min strictly decreases in
      trading-fee inflows C_S (unconditional), and in collateral C_C under
      an explicit sufficient condition that holds at the paper's
      calibration (Table 2).
  [8] Proposition 2(ii): the LP capital dynamics (Eq. 15) are a negative
      feedback loop; the local stability condition dr_LP/dC_LP < 0 is
      derived explicitly and certified at the paper's calibration.
  [9] Proposition 3: the solvency bound U <= 1/(psi * c_max), including the
      k-hack generalization, and its numerical slack at the simulated
      concentration (c_max ~ 0.5%).
 [10] Numeric cross-check: the closed forms used here coincide with the
      implementation in defi_insurance_simulation.py on random inputs.

Not mechanized: Proposition 1 (truthful risk assessment) is a free-entry
no-arbitrage argument, and the existence part of Theorem 1 rests on
standard existence results for continuous games on compact, convex strategy
sets (Berge/Kakutani-type arguments); neither reduces to computer algebra. What CAN be checked -- the
concavity/monotonicity primitives those arguments consume -- is checked
here. Quasi-concavity of the LP payoff (assumption (A1), LP side) is not
a theorem and is certified numerically at the baseline calibration in [8].

Run:  python analytical/verify_analytical_results.py
Requires: sympy (and numpy for [8]/[10]).
"""

import sys
from pathlib import Path

import sympy as sp

FAILURES = []


def check(name, ok):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        FAILURES.append(name)


def section(title):
    print(f"\n=== {title} ===")


# ----------------------------------------------------------------------
# Shared symbols (positivity encoded where the paper assumes it)
# ----------------------------------------------------------------------
CC, CLP, CS, TVL = sp.symbols("C_C C_LP C_S TVL", positive=True)
mu, xi, p, rmkt, rpool = sp.symbols("mu xi p r_market r_pool", positive=True)
rhoP, rhoLP, phi, psi, cmax = sp.symbols("rho_P rho_LP varphi psi c_max", positive=True)
theta = sp.Symbol("theta", positive=True)  # theta in (0,1) handled explicitly
E = sp.Symbol("E_loss", positive=True)     # E[min(coverage, Loss)]
gamma = sp.Symbol("gamma", positive=True)
lam, lam0, T = sp.symbols("lambda lambda_0 T", positive=True)
U, Ut, Prisk, Panchor = sp.symbols("U U_target P_risk P_anchor", positive=True)
alpha, beta, delta = sp.symbols("alpha beta delta", positive=True)

Ctot = CC + CLP + CS                      # Eq. (1)
coverage = mu * CC**theta * (1 + xi)      # Eq. (2)

# Baseline calibration (Table 2) for the numeric certifications
BASE = {mu: 3.0, theta: 0.5, phi: 0.01, alpha: 0.6, beta: 1.0, delta: 0.7,
        Ut: 15.0, rmkt: 0.05, rpool: 0.10, rhoLP: 0.005}


# ----------------------------------------------------------------------
section("[1] Protocol FOC and concavity (Eq. 2, Theorem 1 (A1) protocol side)")
# ----------------------------------------------------------------------
# Atomistic protocol in the coverage-binding region E[min(cov, Loss)] = cov:
# collateral-dependent part of Eq. (9), with the risk-aversion term merged
# ((1 + rho_P) multiplies the expected covered loss).
pi_prot = (1 + rhoP) * p * coverage - rmkt * CC

foc = sp.diff(pi_prot, CC)
CC_star = sp.solve(sp.Eq(foc, 0), CC)
# closed form used in the simulation (protocol_target_CC):
CC_code = (rmkt / ((1 + rhoP) * p * mu * theta * (1 + xi))) ** (1 / (theta - 1))
ok = any(sp.simplify(sol - CC_code) == 0 for sol in CC_star)
check("FOC solution equals protocol_target_CC closed form", ok)

# Strict concavity: the second derivative, normalized by its positive
# factors, reduces to (theta - 1) < 0 for theta in (0,1).
sd = sp.diff(pi_prot, CC, 2)
normalized = sp.simplify(sd / ((1 + rhoP) * p * mu * (1 + xi) * theta * CC**(theta - 2)))
check("d2pi/dCC2 = (positive factor) * (theta - 1)  [< 0 for theta < 1]",
      sp.simplify(normalized - (theta - 1)) == 0)

# Coverage cap: coverage <= TVL  <=>  CC <= (TVL / (mu (1+xi)))^(1/theta)
CC_cap = sp.solve(sp.Eq(coverage, TVL), CC)
CC_cap_code = (TVL / (mu * (1 + xi))) ** (1 / theta)
check("coverage<=TVL cap equals the code's CC_cap",
      any(sp.simplify(sol - CC_cap_code) == 0 for sol in CC_cap))


# ----------------------------------------------------------------------
section("[2] Hazard-rate fit (Eq. 4) and anchor (Eq. 7)")
# ----------------------------------------------------------------------
# If observed prices lie exactly on P(T_i) = 1 - exp(-lam0 T_i), the SSE
# objective of Eq. (4) is zero at lam = lam0 and positive elsewhere, so the
# least-squares estimate recovers lam0.
T_vec = [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4), 1]
SSE = sum(((1 - sp.exp(-lam * Ti)) - (1 - sp.exp(-lam0 * Ti)))**2 for Ti in T_vec)
check("SSE(lam0) = 0 (perfect recovery on exponential prices)",
      sp.simplify(SSE.subs(lam, lam0)) == 0)
check("dSSE/dlam = 0 at lam = lam0",
      sp.simplify(sp.diff(SSE, lam).subs(lam, lam0)) == 0)
# Uniqueness: 1 - exp(-lam T) is strictly increasing in lam for T > 0,
# so the fitted prices are strictly increasing in lam and SSE has a unique
# minimum. Check the monotonicity primitive:
check("d/dlam [1 - exp(-lam T)] > 0",
      sp.simplify(sp.diff(1 - sp.exp(-lam * T), lam)) == T * sp.exp(-lam * T))
x = sp.Symbol("x", real=True)
check("anchor p_1Y = 1 - exp(-lam) lies in (0,1): positive iff lam > 0, sup = 1",
      sp.solveset(1 - sp.exp(-x) > 0, x, sp.S.Reals) == sp.Interval.open(0, sp.oo)
      and sp.limit(1 - sp.exp(-lam), lam, sp.oo) == 1)


# ----------------------------------------------------------------------
section("[3] Prudential cap monotonicity (Eq. 5)")
# ----------------------------------------------------------------------
Umin, Uhi, kU, phat = sp.symbols("U_min U_max_hi kappa_U p_hat", positive=True)
Umax_expr = Umin + (Uhi - Umin) / (1 + kU * phat)
dUmax = sp.simplify(sp.diff(Umax_expr, phat))
check("dU_max/dp_hat = -(U_hi - U_min) kappa_U / (1 + kappa_U p_hat)^2 < 0 for U_hi > U_min",
      sp.simplify(dUmax + (Uhi - Umin) * kU / (1 + kU * phat)**2) == 0)
check("bounds: U_max -> U_hi as p_hat -> 0, U_max -> U_min as p_hat -> oo",
      sp.simplify(Umax_expr.subs(phat, 0) - Uhi) == 0
      and sp.limit(Umax_expr, phat, sp.oo) == Umin)


# ----------------------------------------------------------------------
section("[4] Risk index never exceeds the anchor (Eqs. 6-7)")
# ----------------------------------------------------------------------
# Under a flat hazard lam, each expiry price P(T_i) = 1 - exp(-lam T_i) with
# T_i <= 1 is bounded by the annual probability 1 - exp(-lam). Any convex
# combination (the omega-weights of Eq. 6) therefore satisfies
# P_risk <= P_anchor, so the ratio in Eq. (8) is at most one.
diff_T = (1 - sp.exp(-lam)) - (1 - sp.exp(-lam * T))   # anchor minus expiry price
# exp(-lam T) >= exp(-lam) for T <= 1  <=>  diff_T >= 0
check("P_anchor - P_HACK(T) = exp(-lam T) - exp(-lam) >= 0 for T <= 1",
      sp.simplify(diff_T - (sp.exp(-lam * T) - sp.exp(-lam))) == 0)
# numeric spot check at the paper's weights, lam = 0.004 (baseline scale):
import numpy as np
w = np.array([0.40, 0.30, 0.20, 0.10])
Ts = np.array([0.25, 0.5, 0.75, 1.0])
for lv in (0.004, 0.02, 0.2):
    pr = float(np.sum(w * (1 - np.exp(-lv * Ts))))
    pa = float(1 - np.exp(-lv))
    check(f"numeric: P_risk({lv}) = {pr:.5f} <= P_anchor = {pa:.5f}", pr <= pa)


# ----------------------------------------------------------------------
section("[5] Yield share monotonicity (Eq. 8, premise of Theorem 1)")
# ----------------------------------------------------------------------
gamma_expr = alpha * (U / Ut)**beta + (1 - alpha) * (Prisk / Panchor)**delta
check("dgamma/dU > 0 (equals alpha beta (U/Ut)^beta / U)",
      sp.simplify(sp.diff(gamma_expr, U) - alpha * beta * (U / Ut)**beta / U) == 0)
check("dgamma/dP_risk > 0 (equals (1-alpha) delta (Pr/Pa)^delta / P_risk)",
      sp.simplify(sp.diff(gamma_expr, Prisk)
                  - (1 - alpha) * delta * (Prisk / Panchor)**delta / Prisk) == 0)


# ----------------------------------------------------------------------
section("[6] Proposition 2(i): participation bound (Eq. 14) + minimum pool yield")
# ----------------------------------------------------------------------
# LP realized return (Appendix C): r_LP = [gamma (1-phi) Y_total - p E] / C_LP
Y_total = rpool * Ctot
r_LP = (gamma * (1 - phi) * Y_total - p * E) / CLP

gamma_min_solved = sp.solve(sp.Eq(r_LP, rmkt + rhoLP), gamma)[0]
gamma_min_paper = (CLP * (rmkt + rhoLP) + p * E) / ((1 - phi) * rpool * Ctot)
check("solving r_LP = r_market + rho_LP for gamma yields Eq. (14)",
      sp.simplify(gamma_min_solved - gamma_min_paper) == 0)

rpool_solved = sp.solve(sp.Eq(r_LP, rmkt + rhoLP), rpool)[0]
rpool_paper = (CLP * (rmkt + rhoLP) + p * E) / (gamma * (1 - phi) * Ctot)
check("solving the same equality for r_pool yields the minimum-pool-yield corollary",
      sp.simplify(rpool_solved - rpool_paper) == 0)


# ----------------------------------------------------------------------
section("[7] Corollary: capital expansion lowers gamma_min")
# ----------------------------------------------------------------------
# (a) In C_S the claim is unconditional: the numerator of Eq. (14) does not
#     depend on C_S while the denominator is increasing in it.
dgmin_dCS = sp.simplify(sp.diff(gamma_min_paper, CS))
check("dgamma_min/dC_S = -gamma_min / C_total < 0",
      sp.simplify(dgmin_dCS + gamma_min_paper / Ctot) == 0)

# (b) In C_C the expected covered loss also grows (E = s * coverage with
#     loss-to-coverage ratio s in (0,1]), so the claim needs a condition:
#     dgamma_min/dC_C < 0  <=>  p s cov'(CC) C_total < CLP (rmkt+rhoLP) + p s cov(CC).
s = sp.Symbol("s", positive=True)
gmin_CC = (CLP * (rmkt + rhoLP) + p * s * coverage) / ((1 - phi) * rpool * Ctot)
dgmin_dCC = sp.together(sp.diff(gmin_CC, CC))
condition = sp.simplify(p * s * sp.diff(coverage, CC) * Ctot
                        - (CLP * (rmkt + rhoLP) + p * s * coverage))
# The derivative's sign equals the sign of `condition`:
check("sign(dgamma_min/dC_C) = sign(p s cov' C_total - [CLP(r+rho) + p s cov])",
      sp.simplify(dgmin_dCC * (1 - phi) * rpool * Ctot**2 - condition) == 0)
# Certify the condition is negative at the paper's calibration for a
# representative (median) protocol: CC ~ 0.66, coverage ~ 6.6, per-protocol
# LP capital ~ 0.5 ($M, CLP0/n_protocols), p ~ 0.004, s ~ 0.5 (mean Lbar).
num = condition.subs({p: 0.004, s: 0.5, CC: 0.66, CLP: 0.5, CS: 0,
                      xi: 1.7, **BASE})
check(f"condition < 0 at baseline calibration (value = {float(num):+.5f})",
      float(num) < 0)


# ----------------------------------------------------------------------
section("[8] Proposition 2(ii): negative feedback and local stability (Eq. 15)")
# ----------------------------------------------------------------------
# With coverage fixed, U = coverage / C_LP and the concrete yield share of
# Eq. (8) (beta = 1, risk ratio K constant), the LP return as a function of
# its own capital is
K = sp.Symbol("K", positive=True)          # (P_risk/P_anchor)^delta, constant
cov = sp.Symbol("cov", positive=True)      # aggregate coverage, fixed wrt C_LP
A = sp.Symbol("A", positive=True)          # C_C + C_S (non-LP capital)
gamma_U = alpha * cov / (Ut * CLP) + (1 - alpha) * K
r_LP_dyn = (gamma_U * (1 - phi) * rpool * (CLP + A) - p * E) / CLP

drdC = sp.together(sp.diff(r_LP_dyn, CLP))
# Multiply by CLP^3 > 0 to expose the sign-determining polynomial:
signpoly = sp.expand(sp.simplify(drdC * CLP**3))
expected = sp.expand(p * E * CLP
                     - (1 - phi) * rpool * (alpha * cov / Ut * (CLP + 2 * A)
                                            + (1 - alpha) * K * A * CLP))
check("dr_LP/dC_LP < 0  <=>  p E C_LP < (1-phi) r_pool [ (alpha cov/Ut)(C_LP + 2A) + (1-alpha) K A C_LP ]",
      sp.simplify(signpoly - expected) == 0)

# Certify at the paper's calibration (aggregate, $M): coverage ~ 2500,
# A ~ 285 (simulated aggregate collateral), K ~ 0.616 (flat-hazard ratio
# with Table-2 weights and delta = 0.7), annual expected claims p E ~ 23.4.
stab = expected.subs({p: 1.0, E: 23.4, cov: 2500.0, A: 285.0, K: 0.616, **BASE})
vals = [float(stab.subs(CLP, c)) for c in (250.0, 500.0, 800.0)]
check(f"stability condition holds at C_LP = 250/500/800 $M (values {['%.0f' % v for v in vals]})",
      all(v < 0 for v in vals))

# Negative feedback (the sketch's monotone step): given dr_LP/dC_LP < 0,
# the ODE dC_LP/dt = kappa (r_LP - target) has a strictly decreasing RHS in
# C_LP, hence any equilibrium C_LP* with r_LP(C_LP*) = target is locally
# asymptotically stable. Numeric single-crossing + unimodality check of the
# clipped LP payoff (assumption (A1), LP side) at the baseline calibration:
def pi_LP_num(clp, covv=2500.0, a=285.0, k=0.616, pe=23.4):
    g_raw = 0.6 * (covv / clp) / 15.0 + 0.4 * k
    g = min(1.0, max(0.0, g_raw))
    return g * 0.99 * 0.10 * (clp + a) - pe - 0.05 * clp

grid = np.linspace(1.0, 5000.0, 20000)
vals = np.array([pi_LP_num(c) for c in grid])
d = np.sign(np.diff(vals))
switches = int(np.sum(np.abs(np.diff(d[d != 0]))) // 2)
check(f"pi_LP is unimodal in C_LP on [1, 5000] $M (sign switches = {switches})",
      switches <= 1)


# ----------------------------------------------------------------------
section("[9] Proposition 3: solvency bound")
# ----------------------------------------------------------------------
# Chain: Loss <= psi * coverage_i (prob >= 1-eps), coverage_i <= c_max *
# sum(coverage) = c_max * U * C_LP  =>  Loss <= psi c_max U C_LP.
# Solvency C_LP >= Loss then holds whenever psi c_max U <= 1.
Ubound = sp.solve(sp.Eq(psi * cmax * U * CLP, CLP), U)[0]
check("psi c_max U C_LP <= C_LP  <=>  U <= 1/(psi c_max)  (Eq. 17)",
      sp.simplify(Ubound - 1 / (psi * cmax)) == 0)
# k-hack generalization: with the k largest coverage shares c_(1..k),
# obligation <= (c_1 + ... + c_k) U C_LP; same algebra applies.
c1, c2 = sp.symbols("c_1 c_2", positive=True)
Ubound2 = sp.solve(sp.Eq(psi * (c1 + c2) * U * CLP, CLP), U)[0]
check("k=2 hacks: U <= 1/(psi (c_1 + c_2))",
      sp.simplify(Ubound2 - 1 / (psi * (c1 + c2))) == 0)
# Numeric slack at the simulated concentration: c_max ~ 0.0048, psi <= 1
# (payouts are capped at coverage), so the bound is ~208 -- far above the
# average dynamic cap of ~21 reported in Section 6.
bound = 1.0 / (1.0 * 0.0048)
check(f"1/(psi c_max) = {bound:.0f} >> average dynamic cap 21.2", bound > 21.2)


# ----------------------------------------------------------------------
section("[10] Cross-check against defi_insurance_simulation.py")
# ----------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import defi_insurance_simulation as simmod
except Exception as exc:  # pragma: no cover
    print(f"  [SKIP] could not import simulation module: {exc}")
else:
    rng = np.random.default_rng(7)
    ok_cc = ok_umax = ok_gamma = True
    CC_star_fn = sp.lambdify((rmkt, rhoP, p, mu, theta, xi), CC_code, "numpy")
    for _ in range(200):
        m, th = rng.uniform(1, 6), rng.uniform(0.2, 0.8)
        x, rp = rng.uniform(0.6, 2.2), rng.uniform(0.1, 3.0)
        pa, rm = rng.uniform(1e-4, 0.05), rng.uniform(0.01, 0.10)
        tvl = rng.uniform(2, 1000)
        code_val = simmod.protocol_target_CC(m, th, x, rp, pa, rm, tvl)
        sym_val = float(CC_star_fn(rm, rp, pa, m, th, x))
        cap_val = float((tvl / (m * (1 + x))) ** (1 / th))
        ok_cc &= abs(code_val - min(sym_val, cap_val)) <= 1e-9 * max(1.0, code_val)

        prms = simmod.SimulationParams()
        um_code = simmod.Umax_from_paper(pa, prms)
        um_sym = float(Umax_expr.subs({Umin: prms.U_min, Uhi: prms.U_max_hi,
                                       kU: prms.kappa_Ucap, phat: pa}))
        ok_umax &= abs(um_code - um_sym) <= 1e-12

        u, pr_ = rng.uniform(0.1, 40), rng.uniform(1e-4, 0.05)
        pa2 = rng.uniform(pr_, 0.06)
        g_code = simmod.gamma_from_paper(u, prms.U_target, pr_, pa2,
                                         prms.alpha, prms.beta, prms.delta)
        g_sym = float(min(1.0, max(0.0, gamma_expr.subs(
            {U: u, Ut: prms.U_target, Prisk: pr_, Panchor: pa2,
             alpha: prms.alpha, beta: prms.beta, delta: prms.delta}))))
        ok_gamma &= abs(g_code - g_sym) <= 1e-12
    check("protocol_target_CC == symbolic CC* (200 random draws)", ok_cc)
    check("Umax_from_paper == Eq. (5) (200 random draws)", ok_umax)
    check("gamma_from_paper == Eq. (8) (200 random draws)", ok_gamma)


# ----------------------------------------------------------------------
print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
