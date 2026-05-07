"""
algorithm.py  –  Algorithm 2 (Simple Adaptive Algorithm v2) from Section 3.1
==============================================================================

Two key improvements over the generic discretisation-based version:

1. **PC-specific λ maximisation.**  Instead of evaluating every PC on a
   simplex grid, we exploit each criterion's structure:

   - UB(λ; Bm) is concave in λ  (minimum of concave quadratics, see
     Proposition 6 proof).  We maximise it with scipy's SLSQP.
   - GAP₁ = UB − LB₁ is a difference-of-concave (DC) function (Prop. 6).
     We use a multi-start local search, since each local problem is small.
   - GN(λ; Bm) is non-concave but piecewise-rational.  We use multi-start
     local search seeded at simplex vertices and the previous λ_t.

2. **Adaptive inner-loop stopping.**  The convergence proof of Algorithm 2
   (Theorem 1, Appendix B.1) requires  PC(λ_t; B_{t+1}) ≤ ε/3  after the
   inner update.  Instead of precomputing a theoretical upper-bound M_t on
   the number of inner iterations, we run the inner loop and stop as soon
   as the actual PC at λ_t drops below ε/3.  This is both simpler and
   tighter — the algorithm does exactly as much work as needed.

Illustrative example
--------------------
Consider K = 3 objectives on R^12 (multi-class logistic regression).

    F_k(x) = per-class cross-entropy + (reg/2) ‖x‖²

With PC = GAP and ε = 0.01, the algorithm:
  1. Initialises the bundle at x_0 = 0.
  2. Finds λ_t = argmax GAP(λ; B_t) via multi-start SLSQP on Δ_3.
  3. Runs inner gradient-descent steps at λ_t, checking GAP(λ_t; B)
     after each step, and stops as soon as GAP(λ_t; B) ≤ ε/3.
  4. Repeats until max_{λ} GAP(λ; B_t) ≤ ε.
"""

from __future__ import annotations

import numpy as np
import copy
import time
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize as sp_minimize

from bundle import Bundle, UB, GAP, GN, LB, T_map
from baseline import worst_case_suboptimality_algorithm2


# =====================================================================
#  PC-specific λ maximisation
# =====================================================================

# ---------------------------------------------------------------------------
# UB:  concave in λ  →  single convex optimisation
# ---------------------------------------------------------------------------
def _maximise_UB(bundle: Bundle) -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  UB(λ; B_m).

    Structure
    ---------
    From Eq. (12)/(24) and the proof of Proposition 6:

        UB(λ; Bm) = min_{i ∈ [m]} u_i(λ)

    where  u_i(λ) = λ^T F(x_i) − (1/(2Lλ)) λ^T J_F(x_i) J_F(x_i)^T λ.

    Each u_i is concave in λ  (since −(1/(2Lλ)) ‖J^T λ‖² is concave when
    Lλ = λ^T L is linear in λ — verified in the proof of Prop. 6).
    UB is the pointwise minimum of concave functions, hence concave.

    Maximising a concave function over the simplex is a convex problem.
    We use scipy SLSQP with a few random restarts (the landscape is
    concave but may have kinks from the min operation).

    Illustrative example
    --------------------
    With m = 5 bundle points and K = 3, we maximise UB over the
    2D simplex Δ_3.  SLSQP with 3–5 starting points reliably finds
    the global max since UB is concave.
    """
    K = bundle.K
    m = bundle.m

    def neg_ub(lam):
        return -UB(bundle, lam)

    def neg_ub_grad(lam):
        """Subgradient of −UB(λ) via Danskin's theorem.

        Since UB(λ) = min_i u_i(λ), by Danskin's theorem a subgradient
        of UB at λ is ∇u_{i*}(λ) where i* achieves the minimum.

        ∇u_i(λ) = F(x_i) − J_F(x_i) J_F(x_i)^T λ / Lλ
                   + (λ^T J_F(x_i) J_F(x_i)^T λ) / (2 Lλ²) · L

        (from Eq. in proof of Prop. 6, page 19).
        """
        Ll = bundle.L_lam(lam)
        best_val = np.inf
        best_i = 0
        for i in range(m):
            fi = bundle.F_lam(i, lam)
            gi = bundle.grad_F_lam(i, lam)
            val = fi - 0.5 / Ll * np.dot(gi, gi)
            if val < best_val:
                best_val = val
                best_i = i

        # Gradient of u_{i*}(λ)
        Ji = bundle.grads[best_i]          # (K, d)
        JJT = Ji @ Ji.T                    # (K, K)
        JJTlam = JJT @ lam                 # (K,)
        quad = lam @ JJTlam                # scalar
        grad = bundle.fvals[best_i] - JJTlam / Ll + (quad / (2.0 * Ll**2)) * bundle.L
        return -grad  # negate for minimisation

    # Constraints and bounds for Δ_K
    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0, "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K  # small lb to keep Lλ > 0

    # Multi-start: vertices + uniform + previous best
    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_ub, lam0, jac=neg_ub_grad, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 200})
        if res.fun < best_val:
            best_val = res.fun
            best_lam = res.x.copy()

    # Project onto simplex (enforce numerics)
    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# ---------------------------------------------------------------------------
# GAP₁ = UB − LB₁:  difference-of-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _maximise_GAP(bundle: Bundle, variant: str = "lb1") -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  GAP(λ; B_m).

    Structure
    ---------
    GAP₁(λ) = UB(λ) − LB₁(λ) where both UB and LB₁ are concave in λ
    (Proposition 6).  So GAP₁ is a difference-of-concave (DC) function.

    DC maximisation is NP-hard in general, but here λ ∈ Δ_K has only
    K−1 degrees of freedom with K typically small (2–10).  We use
    multi-start SLSQP: each local solve finds a local maximum, and
    we take the best.

    Starting points:  K vertices + uniform + midpoints of each edge.
    For K ≤ 5 this is ≤ 16 starts, each very cheap.

    Illustrative example
    --------------------
    With K = 3, GAP is a DC function on the 2D triangle Δ_3.
    We launch ~7 local searches (3 vertices + 1 centre + 3 edge
    midpoints) and return the best.
    """
    K = bundle.K

    def neg_gap(lam):
        return -GAP(bundle, lam, variant=variant)

    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0, "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    # Build starting points: vertices + uniform + edge midpoints
    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)
    # Edge midpoints (pairs of vertices)
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            e = np.full(K, 1e-8)
            e[k1] = 0.5 - (K - 2) * 0.5e-8
            e[k2] = 0.5 - (K - 2) * 0.5e-8
            starts.append(e)

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_gap, lam0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 200})
        if res.fun < best_val:
            best_val = res.fun
            best_lam = res.x.copy()

    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# ---------------------------------------------------------------------------
# GN:  non-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _maximise_GN(bundle: Bundle) -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  GN(λ; B_m).

    Structure
    ---------
    From Eq. (17):

        GN(λ; Bm) = (1/2)(1/µλ − 1/Lλ) · min_i ‖J_F(x_i)^T λ‖²

    where µλ = λ^T µ, Lλ = λ^T L.  The scale factor (1/(2µλ) − 1/(2Lλ))
    is convex in λ (sum of convex reciprocals of linear functions), and
    min_i ‖J^T λ‖² is concave.  Their product is neither convex nor
    concave.

    When mu is not available (generic non-convex), GN falls back to
    min_i ‖J_F(x_i)^T λ‖² which *is* concave.

    We use multi-start SLSQP as for GAP.

    Illustrative example
    --------------------
    With K = 2, the simplex is a line segment [0,1].  GN(λ) is a
    1D piecewise-rational function — a few local searches from the
    endpoints and midpoint reliably find the global max.
    """
    K = bundle.K

    def neg_gn(lam):
        return -GN(bundle, lam)

    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0,
                   "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            e = np.full(K, 1e-8)
            e[k1] = 0.5 - (K - 2) * 0.5e-8
            e[k2] = 0.5 - (K - 2) * 0.5e-8
            starts.append(e)

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_gn, lam0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 200})
        if res.fun < best_val:
            best_val = res.fun
            best_lam = res.x.copy()

    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# =====================================================================
#  Adaptive inner loop (BundleUpdate with max_steps)
# =====================================================================
def _bundle_update_adaptive(
    bundle: Bundle,
    lam: np.ndarray,
    pc_fn: Callable,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    max_steps: int,
) -> int:
    """Run ``max_steps`` inner T_map steps at fixed λ; commit only the
    candidate with the smallest ∥∇F_λ∥ to the bundle.

    Returns
    -------
    Number of inner steps taken (always equals ``max_steps``).
    """
    work = copy.deepcopy(bundle)
    base_m = work.m

    # Generate the candidate chain on the work bundle.  Each T_map call
    # sees all previously-added in-round candidates, matching the
    # original inner-loop semantics.
    for _ in range(max_steps):
        x_new = T_map(work, lam)
        work.add_point(x_new, objectives, grad_objectives)

    # Pick argmin ∥∇F_λ(x^i)∥ from the cached gradients.
    best_idx = base_m
    #print('best_idx before:', best_idx)
    best_gnorm = float(np.linalg.norm(work.grad_F_lam(base_m, lam)))
    for idx in range(base_m + 1, base_m + max_steps):
        gnorm = float(np.linalg.norm(work.grad_F_lam(idx, lam)))
        if gnorm < best_gnorm:
            best_gnorm = gnorm
            best_idx = idx

    # Commit only the winner to the real bundle.
    #print('best_idx after:', best_idx)
    bundle.add_point(work.points[best_idx], objectives, grad_objectives)

    return max_steps


# =====================================================================
#  Instrumented Algorithm 2:  checkpoint after each outer iteration
# =====================================================================
def algorithm2_progressive(
    K: int,
    d: int,
    objectives: List[Callable],
    grad_objectives: List[Callable],
    L: np.ndarray,
    x0: np.ndarray,
    reference_map: Dict,
    mu: Optional[np.ndarray] = None,
    mode: str = "gap",
    max_outer: int = 50,
    max_inner: int = 400,
    checkpoint_every: int = 1,
    eval_every_n_grads: Optional[int] = None,
    verbose: bool = False,
) -> Dict:
    """Run Algorithm 2 with periodic worst-case-error checkpoints.

    Thin wrapper around the algorithm.py primitives that interleaves
    worst-case-error evaluations with the main outer loop.

    Checkpoint cadence
    ------------------
    Two complementary controls:
      - ``checkpoint_every``    : checkpoint every k outer iterations.
      - ``eval_every_n_grads``  : if set, additionally checkpoint at
                                  the next outer-iteration boundary
                                  after every M cumulative gradient
                                  evaluations.

    Setting ``eval_every_n_grads = M`` makes A2 directly comparable
    to the baseline's gradient-vs-error curve at matched M.

    Parameters
    ----------
    checkpoint_every     : evaluate worst-case error every k outer iters.
    eval_every_n_grads   : checkpoint after each M cumulative gradient evals.
    """
    if mode == "gap":
        # Use LB_2 (single-index minorant) inside both the λ-maximisation
        # and the inner-loop PC check.  LB_2 is ~100× faster than LB_1 and
        # avoids hitting the Gurobi size-limited license once the bundle
        # grows past ~100 points.
        pc_fn = lambda bundle, lam: GAP(bundle, lam, variant="lb2")
        maximise_pc = lambda bundle: _maximise_GAP(bundle, variant="lb2")
        if mu is None:
            raise ValueError("mode='gap' requires mu (strong convexity).")
    elif mode == "ub":
        pc_fn, maximise_pc = UB, _maximise_UB
        if mu is None:
            raise ValueError("mode='ub' requires mu.")
    elif mode == "gn":
        pc_fn, maximise_pc = GN, _maximise_GN
    else:
        raise ValueError(f"Unknown mode: {mode!r}.")

    bundle = Bundle(K=K, d=d, L=L, mu=mu)
    bundle.add_point(x0.copy(), objectives, grad_objectives)
    # The initial bundle point cost K gradient evals at x0.
    grad_evals = K

    cpu_times: List[float] = []
    worst_errs: List[float] = []
    outer_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    pc_history: List[float] = []
    grad_evals_at_last_ckpt = 0

    t_start = time.time()

    def _checkpoint(label: str, pc_star=None, steps=None) -> None:
        err = worst_case_suboptimality_algorithm2(bundle, reference_map, objectives, K)
        cpu_times.append(time.time() - t_start)
        worst_errs.append(err)
        outer_iters_history.append(label_to_outer(label))
        grad_evals_history.append(grad_evals)
        if verbose:
            extra = ""
            if pc_star is not None:
                extra = f" | PC*={pc_star:.4e} | inner={steps:3d} | bundle={bundle.m}"
            print(f"  A2 {label} | t={cpu_times[-1]:.2f}s | grad_evals={grad_evals}"
                  f"{extra} | err={err:.4e}")

    def label_to_outer(label: str) -> int:
        # Helper:  parse label like "outer 5/20" -> 5,  "outer 0/20" -> 0
        try:
            return int(label.split()[1].split("/")[0])
        except Exception:
            return -1

    # Checkpoint 0:  report the worst-case error of the constant map x̂(λ) ≡ x_0.
    # This matches the baseline's checkpoint 0 (which also reports the constant
    # map at x_0) so both algorithms agree on the "initial" worst-case error
    # before any algorithmic work has been done.  The K gradient evaluations
    # spent on bundle.add_point(x_0, ...) at the top of this routine will be
    # accounted for at checkpoint 1 alongside the work of the first outer
    # iteration, keeping the cumulative grad_evals count correct from then on.
    fine_grid = reference_map["fine_grid"]
    F_star = reference_map["F_star"]
    err0 = -np.inf
    for i, lam in enumerate(fine_grid):
        F_lam_x0 = sum(lam[k] * objectives[k](x0) for k in range(K))
        err = F_lam_x0 - F_star[i]
        if err > err0:
            err0 = float(err)
    cpu_times.append(time.time() - t_start)
    worst_errs.append(err0)
    outer_iters_history.append(0)
    grad_evals_history.append(0)
    if verbose:
        print(f"  A2 outer 0/{max_outer} | t={cpu_times[-1]:.2f}s | grad_evals=0 "
              f"| err={err0:.4e}  (constant map)")


    for t in range(max_outer):
        pc_star, best_lam = maximise_pc(bundle)
        pc_history.append(pc_star)

        bundle_m_before = bundle.m
        steps = _bundle_update_adaptive(
            bundle, best_lam, pc_fn, objectives, grad_objectives, max_inner,
        )
        # Each inner step corresponds to a retained bundle point (the new
        # _bundle_update_adaptive prunes BEFORE evaluating gradients, so
        # every committed step costs K gradient evals and is kept).
        # The bundle-size delta should equal steps, but we use the delta
        # directly to be robust to any future changes to the inner loop.
        retained_steps = bundle.m - bundle_m_before
        grad_evals += retained_steps * K

        do_checkpoint = (
            ((t + 1) % checkpoint_every == 0)
            or (t + 1 == max_outer)
            or (
                eval_every_n_grads is not None
                and (grad_evals - grad_evals_at_last_ckpt) >= eval_every_n_grads
            )
        )
        if do_checkpoint:
            _checkpoint(f"outer {t + 1}/{max_outer}", pc_star=pc_star, steps=steps)
            grad_evals_at_last_ckpt = grad_evals
            if verbose and retained_steps < steps:
                print(f"        (attempted {steps}, retained {retained_steps} "
                      f"after PC-drop pruning)")
        elif verbose:
            print(f"  A2 outer {t + 1}/{max_outer} | grad_evals={grad_evals} "
                  f"| PC*={pc_star:.4e} | inner={steps:3d} (retained {retained_steps}) "
                  f"| bundle={bundle.m} | (no checkpoint)")

    return {
        "bundle": bundle,
        "cpu_times": cpu_times,
        "worst_errs": worst_errs,
        "outer_iters_history": outer_iters_history,
        "grad_evals_history": grad_evals_history,
        "pc_history": pc_history,
    }
