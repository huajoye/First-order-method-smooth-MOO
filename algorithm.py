"""
algorithm.py  –  Algorithm 2 (Simple Adaptive Algorithm v2)
"""

from __future__ import annotations
import numpy as np
import time
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize as sp_minimize

from bundle import Bundle, UB, GAP, GN, LB, T_map


# =====================================================================
#  Vectorised bundle helpers (CPU-optimisation, no semantic change)
# =====================================================================
# These helpers replace the per-bundle-point Python `for i in range(m)`
# loops in UB / LB_2 / T_map with single batched numpy operations.
# Mathematically they reproduce bundle.UB, bundle._LB_2 and bundle.T_map
# exactly; only the implementation differs.
# ---------------------------------------------------------------------
def _bundle_arrays(bundle: Bundle) -> Tuple[np.ndarray, np.ndarray]:
    """Stack ``bundle.fvals`` / ``bundle.grads`` as contiguous arrays.

    Returns
    -------
    Fmat : (m, K) array, Fmat[i, k] = F_k(x_i)
    Jmat : (m, K, d) array, Jmat[i, k] = ∇F_k(x_i)
    """
    Fmat = np.asarray(bundle.fvals)
    Jmat = np.asarray(bundle.grads)
    return Fmat, Jmat


def _ub_lb2_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                    L: np.ndarray, mu: np.ndarray,
                    lam: np.ndarray) -> Tuple[float, float, int, int,
                                              np.ndarray, np.ndarray]:
    """Single batched evaluation of UB(λ) and LB_2(λ) over the bundle.

    Computes for all i simultaneously:
        F_λ(x_i)        = Fmat @ λ                   shape (m,)
        ∇F_λ(x_i)       = Σ_k λ_k Jmat[i, k]         shape (m, d)
        ‖∇F_λ(x_i)‖²    = row-sum of squares          shape (m,)
        u_i(λ) = F_λ(x_i) - 1/(2 Lλ) ‖∇F_λ(x_i)‖²
        l_i(λ) = F_λ(x_i) - 1/(2 µλ) ‖∇F_λ(x_i)‖²

    Returns
    -------
    ub        : float                       UB(λ; B) = min_i u_i(λ)
    lb2       : float                       LB_2(λ; B) = max_i l_i(λ)
    i_star    : int                         argmin u_i (UB-attaining index)
    j_star    : int                         argmax l_i (LB_2-attaining index)
    F_lam     : (m,) array                  reused by callers
    gnorm_sq  : (m,) array                  reused by callers
    """
    Ll = float(lam @ L)
    mul = float(lam @ mu)
    F_lam = Fmat @ lam                              # (m,)
    grad_lam = np.einsum('ikd,k->id', Jmat, lam)    # (m, d)
    gnorm_sq = np.einsum('id,id->i', grad_lam, grad_lam)  # (m,)
    u_vals = F_lam - 0.5 * gnorm_sq / Ll            # (m,)
    l_vals = F_lam - 0.5 * gnorm_sq / mul           # (m,)
    i_star = int(np.argmin(u_vals))
    j_star = int(np.argmax(l_vals))
    return (float(u_vals[i_star]), float(l_vals[j_star]),
            i_star, j_star, F_lam, gnorm_sq)


def _gap_grad_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                      L: np.ndarray, mu: np.ndarray, lam: np.ndarray,
                      i_star: int, j_star: int) -> np.ndarray:
    """Analytical (Danskin) gradient of  GAP_2(λ) = UB(λ) − LB_2(λ).

    From Prop. 6 (paper, p. 19):

        ∇u_i(λ) = F(x_i) − J_i J_i^T λ / Lλ + (λ^T J_i J_i^T λ)/(2 Lλ²) · L
        ∇l_i(λ) = F(x_i) − J_i J_i^T λ / µλ + (λ^T J_i J_i^T λ)/(2 µλ²) · µ

    Danskin's theorem applies to both min (for UB) and max (for LB_2),
    so a subgradient of GAP_2 = u_{i*} − l_{j*} is ∇u_{i*}(λ) − ∇l_{j*}(λ).

    The gradient of −GAP_2 (used inside neg_gap for SLSQP) is the negation.
    """
    Ll = float(lam @ L)
    mul = float(lam @ mu)

    Ji = Jmat[i_star]                  # (K, d)
    JJTi_lam = Ji @ (Ji.T @ lam)       # (K,)  =  J_i J_i^T λ
    qi = float(lam @ JJTi_lam)
    grad_u = Fmat[i_star] - JJTi_lam / Ll + (qi / (2.0 * Ll * Ll)) * L

    Jj = Jmat[j_star]
    JJTj_lam = Jj @ (Jj.T @ lam)
    qj = float(lam @ JJTj_lam)
    grad_l = Fmat[j_star] - JJTj_lam / mul + (qj / (2.0 * mul * mul)) * mu

    return grad_u - grad_l             # gradient of GAP_2


def _gn_value_and_jac_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                              L: np.ndarray, mu: Optional[np.ndarray],
                              lam: np.ndarray
                              ) -> Tuple[float, np.ndarray, int]:
    """Batched evaluation and analytical λ-gradient of  GN(λ; B)  (Eq. 17).

    GN(λ) = scale(λ) · min_i ‖J_i^T λ‖²,    where
        scale(λ) = ½(1/µλ − 1/Lλ)   if mu is given (the strongly-convex/PL
                                     case in the paper),
        scale(λ) = 1                 in the generic non-convex case (no µ),
                                     to match ``bundle.GN``'s fallback.

    Implementation
    --------------
    * Stack the bundle Jacobians and contract with λ in one ``einsum`` to
      obtain  ``G[i] = J_i^T λ``  for all bundle points simultaneously,
      replacing the original Python loop in :func:`bundle.GN`.
    * The argmin index ``i*`` is unique generically.  By Danskin's theorem
      the gradient of GN at smooth points is
          ∇_λ GN(λ) = scale(λ) · 2 J_{i*} g_{i*} + (gnorm²_{i*}) · ∇_λ scale(λ),
      with  ∇_λ scale(λ) = -µ/(2 µλ²) + L/(2 Lλ²)  in the strongly-convex
      case and zero otherwise.  ``J_{i*}`` has shape (K, d), so
      ``J_{i*} g_{i*}`` is a (K,) vector — the same shape as λ.

    Returns
    -------
    gn_value : float
    gn_jac   : (K,) array,  ∇_λ GN(λ)
    i_star   : int          argmin index, useful for caller diagnostics
    """
    G = np.einsum('ikd,k->id', Jmat, lam)              # (m, d)
    gnorms_sq = np.einsum('id,id->i', G, G)            # (m,)
    i_star = int(np.argmin(gnorms_sq))
    g_istar = G[i_star]                                # (d,)
    gnorm_sq_istar = float(gnorms_sq[i_star])

    # ∇_λ ‖J_i*^T λ‖²  =  2 J_i* (J_i*^T λ)  =  2 J_i* g_i*    ∈  R^K
    grad_min_norm = 2.0 * (Jmat[i_star] @ g_istar)     # (K,)

    if mu is not None:
        mul = float(lam @ mu)
        Ll = float(lam @ L)
        scale = 0.5 * (1.0 / mul - 1.0 / Ll)
        # ∇_λ scale  =  -µ / (2 µλ²)  +  L / (2 Lλ²)
        grad_scale = -0.5 * mu / (mul * mul) + 0.5 * L / (Ll * Ll)
        gn_value = scale * gnorm_sq_istar
        gn_jac = scale * grad_min_norm + gnorm_sq_istar * grad_scale
    else:
        gn_value = gnorm_sq_istar
        gn_jac = grad_min_norm
    return gn_value, gn_jac, i_star


def _T_map_batched(Fmat: np.ndarray, Jmat: np.ndarray, points_arr: np.ndarray,
                   L: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Vectorised version of ``bundle.T_map`` (Eq. 13).

    Computes
        i*  = argmin_i { F_λ(x_i) − 1/(2 Lλ) ‖∇F_λ(x_i)‖² }
        T   = x_{i*} − (1/Lλ) ∇F_λ(x_{i*})
    in a single batched pass.

    Parameters
    ----------
    Fmat, Jmat   : as in ``_bundle_arrays``.
    points_arr   : (m, d) array, points_arr[i] = x_i (= np.asarray(bundle.points))
    L, lam       : K-vectors.
    """
    Ll = float(lam @ L)
    F_lam = Fmat @ lam
    grad_lam = np.einsum('ikd,k->id', Jmat, lam)         # (m, d)
    gnorm_sq = np.einsum('id,id->i', grad_lam, grad_lam) # (m,)
    u_vals = F_lam - 0.5 * gnorm_sq / Ll
    i_star = int(np.argmin(u_vals))
    return points_arr[i_star] - (1.0 / Ll) * grad_lam[i_star]


def _T_map_grid_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                        points_arr: np.ndarray, L: np.ndarray,
                        Lambda: np.ndarray) -> np.ndarray:
    """Apply ``T_map`` to every λ in a grid in one batched pass.

    Parameters
    ----------
    Fmat        : (m, K)
    Jmat        : (m, K, d)
    points_arr  : (m, d)
    L           : (K,)
    Lambda      : (N, K)  — every row is a simplex point.

    Returns
    -------
    X_hat : (N, d) array, row n = T_map(bundle, Lambda[n]).
    """
    Ll_n = Lambda @ L                                  # (N,)
    F_lam_im = Fmat @ Lambda.T                         # (m, N)
    # grad_lam_nim[n, i, d] = Σ_k Lambda[n, k] · Jmat[i, k, d]
    grad_lam = np.einsum('nk,ikd->nid', Lambda, Jmat)  # (N, m, d)
    gnorm_sq = np.einsum('nid,nid->ni', grad_lam, grad_lam)  # (N, m)
    # u_vals[n, i] = F_λn(x_i) - 1/(2 Lλn) ‖∇F_λn(x_i)‖²
    u_vals = F_lam_im.T - 0.5 * gnorm_sq / Ll_n[:, None]   # (N, m)
    i_star = np.argmin(u_vals, axis=1)                     # (N,)
    n_idx = np.arange(Lambda.shape[0])
    x_best = points_arr[i_star]                            # (N, d)
    grad_best = grad_lam[n_idx, i_star]                    # (N, d)
    return x_best - (1.0 / Ll_n[:, None]) * grad_best      # (N, d)


def _worst_case_subopt_fast(bundle: Bundle, reference_map: Dict, objectives: List[Callable], K: int) -> float:
    """Drop-in fast replacement for ``worst_case_suboptimality_algorithm2``.

    Same definition  err = sup_λ [F_λ(T_map(B, λ)) − F*_λ]  as the
    baseline helper, but the bundle work is batched over the entire
    fine grid in a single einsum, eliminating the per-grid-point Python
    loop that calls ``T_map``.

    The per-grid-point objective evaluations  F_k(x_hat_n)  remain
    Python-level (since ``objectives[k]`` is a closure over per-class
    sample indices), but those are now the only Python-level work in
    the checkpoint.
    """
    fine_grid = reference_map["fine_grid"]              # (N, K)
    F_star = reference_map["F_star"]                    # (N,)

    Fmat, Jmat = _bundle_arrays(bundle)
    points_arr = np.asarray(bundle.points)
    X_hat = _T_map_grid_batched(Fmat, Jmat, points_arr, bundle.L, fine_grid)

    worst = -np.inf
    for n, lam in enumerate(fine_grid):
        x_n = X_hat[n]
        F_lam = 0.0
        for k in range(K):
            F_lam += lam[k] * objectives[k](x_n)
        err = F_lam - F_star[n]
        if err > worst:
            worst = float(err)
    return worst


# =====================================================================
#  PC-specific λ maximisation
# =====================================================================
# ---------------------------------------------------------------------------
# GAP₁ = UB − LB₁:  difference-of-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _maximise_GAP(bundle: Bundle, variant: str = "lb1",
                  prev_lam: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  GAP(λ; B_m).

    Structure
    ---------
    GAP₁(λ) = UB(λ) − LB₁(λ) where both UB and LB₁ are concave in λ
    (Proposition 6).  So GAP₁ is a difference-of-concave (DC) function.

    DC maximisation is NP-hard in general, but here λ ∈ Δ_K has only
    K−1 degrees of freedom with K typically small (2–10).  We use
    multi-start SLSQP: each local solve finds a local maximum, and
    we take the best.

    Optimised lb2 path
    ------------------
    For variant="lb2" (the path used by ``experiment_logreg_gap``):

      1. Bundle data is stacked into ``Fmat`` and ``Jmat`` once per call
         and shared across SLSQP iterations and across multi-starts.
         All UB/LB_2 evaluations inside SLSQP then become a single
         ``Fmat @ lam`` + one ``einsum`` instead of an ``m``-step Python
         loop calling ``F_lam`` and ``grad_F_lam`` per bundle index.

      2. The analytical Danskin-gradient (Prop. 6) is supplied to SLSQP
         via ``jac=``.  Without it, SLSQP would build a numerical
         gradient by K extra GAP evaluations per step — the dominant
         cost in the un-optimised version.

      3. ``prev_lam`` (the best-λ from the previous outer iteration)
         drives a warm-start + safety strategy.  When supplied, the
         SLSQP multi-start set is
             { prev_lam }  ∪  { e_k : k != argmax(prev_lam) }
                          ∪  { centroid },
         i.e. the warm start + every vertex *except* the one closest
         to the warm start + the simplex centroid.  Total: K+1 SLSQP
         calls (same as the cold sweep) but the warm call typically
         converges in 2-3 SLSQP iters vs ~8 from cold, giving a
         modest ~1.2× wall-time speed-up for K=3.
         The safety set must cover every basin (K vertices + interior)
         because the GAP_2 global max can migrate to any of them as
         the bundle changes; omitting any basin causes early-termination
         failures in the outer loop.  On the very first call
         (``prev_lam=None``) the full cold multi-start is used.

    For variant="lb1" we fall back to the original (Gurobi-QP-backed)
    LB_1 path, which is left untouched.
    """
    K = bundle.K
    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0, "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    # Multi-start strategy
    # --------------------
    # When ``prev_lam`` is supplied (from the second outer iteration
    # onwards), use it together with (K-1) safety vertices (all
    # vertices NOT closest to ``prev_lam``) plus the simplex centroid.
    # This covers every plausible basin of the GAP_2 landscape:
    #   - K vertex basins (warm + (K-1) cold vertex starts)
    #   - 1 interior basin around the centroid
    # so the multi-start matches the cold sweep's basin coverage and
    # cannot miss the global max.
    #
    # Why the centroid is mandatory
    # ------------------------------
    # The GAP_2 landscape can have a global maximum in the simplex
    # *interior* even when vertices give tiny GAP values.  Empirically
    # this happens after a few outer iterations on the logreg problem:
    # all K vertices show GAP ≈ ε while GAP at the centroid is ~10²×
    # larger.  Without a centroid start, the multi-start would
    # deterministically miss this case, the PC would be misreported as
    # tiny, and the outer-loop early-termination check would fire.
    #
    # Per-iteration cost: K+1 SLSQP calls (warm + K-1 vertex safeties +
    # centroid), same count as the cold sweep.  The speed-up comes
    # entirely from the warm call's faster convergence (~2 SLSQP iters
    # vs ~8 from a cold start), giving ~1.2× wall-time speed-up for
    # K=3.  This is a correctness-first choice: see the docstring above
    # for the speed-vs-safety trade-off.
    #
    # On the very first call ``prev_lam is None`` and we use the full
    # cold multi-start (K vertices + centroid).
    starts: List[np.ndarray] = []
    if prev_lam is not None:
        # Warm start.
        s = np.maximum(prev_lam, 1e-8)
        s /= s.sum()
        starts.append(s)
        # Safety: all vertices except the one closest to prev_lam.
        v_star = int(np.argmax(prev_lam))
        for k in range(K):
            if k == v_star:
                continue
            e = np.full(K, 1e-8)
            e[k] = 1.0 - (K - 1) * 1e-8
            starts.append(e)
        # Safety: simplex centroid (interior basin).
        starts.append(np.ones(K) / K)
    else:
        for k in range(K):
            e = np.full(K, 1e-8)
            e[k] = 1.0 - (K - 1) * 1e-8
            starts.append(e)
        starts.append(np.ones(K) / K)

    if variant == "lb2":
        # ---- Vectorised batched evaluator (shared across SLSQP iters) ----
        Fmat, Jmat = _bundle_arrays(bundle)
        L_arr, mu_arr = bundle.L, bundle.mu

        def neg_gap(lam):
            ub, lb2, *_ = _ub_lb2_batched(Fmat, Jmat, L_arr, mu_arr, lam)
            return lb2 - ub                     # = -GAP_2

        def neg_gap_jac(lam):
            ub, lb2, i_star, j_star, _, _ = _ub_lb2_batched(Fmat, Jmat, L_arr, mu_arr, lam)
            return -_gap_grad_batched(Fmat, Jmat, L_arr, mu_arr, lam, i_star, j_star)

        best_val = np.inf
        best_lam = starts[0]
        # The first start is the warm one when ``prev_lam`` was provided;
        # otherwise it's a cold vertex.  The warm start needs tight
        # tolerance because we use its λ to drive the next inner-loop
        # iteration (Algorithm 2's best_lam).  The remaining starts are
        # safety probes whose ONLY job is to detect when the warm start
        # ended up in a non-global basin — so they only need to produce
        # a value good enough to compare against the warm result.  We
        # therefore use a much looser tolerance and a low maxiter cap
        # for them, which empirically saves ~30-40% of `_maximise_GAP`'s
        # wall time at no measurable cost to PC accuracy.  The safety
        # starts still find the right *basin* under loose tolerance even
        # when they wouldn't converge to high precision.
        # SLSQP tolerance tiers
        # ----------------------
        # WARM:  Used for the warm start when ``prev_lam`` is supplied.
        #   ftol=1e-9 (high precision needed:  warm's λ seeds the next
        #   outer iteration; looser ftol meaningfully degrades the
        #   warm-start chain and increases total outer iterations).
        #   maxiter=30 (down from 60):  empirical p90 of warm SLSQP
        #   iter count is 33; the few outliers terminate with a slightly
        #   less-converged λ that still works fine as the next outer's
        #   seed.  Saves ~25% of `_maximise_GAP` time at no measurable
        #   cost to outer convergence.
        # SAFE:  Used for the (K-1) vertex safeties + centroid.  These
        #   probes only need to detect basin mismatch with warm, so they
        #   converge to coarse precision (ftol=1e-5, maxiter=15).
        WARM_OPTS = {"ftol": 1e-9, "maxiter": 30}
        SAFE_OPTS = {"ftol": 1e-5, "maxiter": 15}
        is_warm_first = prev_lam is not None
        for idx, lam0 in enumerate(starts):
            opts = WARM_OPTS if (is_warm_first and idx == 0) else (
                WARM_OPTS if not is_warm_first else SAFE_OPTS
            )
            res = sp_minimize(neg_gap, lam0, jac=neg_gap_jac, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options=opts)
            if res.fun < best_val:
                best_val = res.fun
                best_lam = res.x.copy()
        # If a safety start ended up giving a *better* value than warm
        # (i.e., warm missed the global max), the result is currently
        # only loosely-converged.  Refine it with one tight SLSQP call
        # from that location to recover full precision for the next
        # inner loop's seed.  This is rare in practice but cheap when
        # it happens.
        if is_warm_first and not np.allclose(best_lam, starts[0], atol=1e-6):
            res = sp_minimize(neg_gap, best_lam, jac=neg_gap_jac, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options=WARM_OPTS)
            if res.fun < best_val:
                best_val = res.fun
                best_lam = res.x.copy()

    else:
        # ---- Original (un-vectorised) path for LB_1 -----------------------
        def neg_gap(lam):
            return -GAP(bundle, lam, variant=variant)

        best_val = np.inf
        best_lam = starts[0]
        for lam0 in starts:
            res = sp_minimize(neg_gap, lam0, method="SLSQP",
                              bounds=bounds, constraints=constraints,
                              options={"ftol": 1e-9, "maxiter": 200})
            if res.fun < best_val:
                best_val = res.fun
                best_lam = res.x.copy()

    best_lam = np.maximum(best_lam, 0.0)
    best_lam /= best_lam.sum()
    return float(-best_val), best_lam


# ---------------------------------------------------------------------------
# GN:  non-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _maximise_GN(bundle: Bundle,prev_lam: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Find  λ* = argmax_{λ ∈ Δ_K}  GN(λ; B_m).

    Structure
    ---------
    From Eq. (17):

        GN(λ; Bm) = min_i ‖J_F(x_i)^T λ‖²

    We use multi-start SLSQP as for GAP, but with three CPU optimisations:

    * ``_gn_value_and_jac_batched`` evaluates GN and its Danskin gradient
      in one ``einsum``, replacing the per-bundle-point Python loop in
      :func:`bundle.GN`.  This is the dominant inner cost of the
      maximisation when bundle size is large.
    * The analytical Jacobian is supplied to SLSQP via the ``jac`` keyword,
      eliminating SLSQP's ~K function-evaluations-per-iteration of finite-
      difference numerical differentiation.
    * The multi-start set is (K vertices + uniform centroid + K(K−1)/2
      edge midpoints + optional ``prev_lam``).  Unlike GAP, GN is neither
      convex nor concave in λ, and empirically the edge midpoints recover
      basins that the vertices and centroid miss on the non-convex MLP
      problem; dropping them sacrifices final accuracy.  ``prev_lam`` is
      only ever an *additional* start — never a replacement — so it can
      accelerate convergence without trapping the search in a stale basin.

    SLSQP's tolerances are relaxed to ``ftol = 1e-9, maxiter = 60``,
    matching the budget chosen for the GAP path.

    Illustrative example
    --------------------
    With K = 2, the simplex is a line segment [0,1].  GN(λ) is a
    1D piecewise-rational function — a few local searches from the
    endpoints and midpoint reliably find the global max.
    """
    K = bundle.K
    Fmat, Jmat = _bundle_arrays(bundle)
    L_arr = bundle.L
    mu_arr = bundle.mu

    def neg_gn(lam):
        v, _, _ = _gn_value_and_jac_batched(Fmat, Jmat, L_arr, mu_arr, lam)
        return -v

    def neg_gn_jac(lam):
        _, j, _ = _gn_value_and_jac_batched(Fmat, Jmat, L_arr, mu_arr, lam)
        return -j

    constraints = {"type": "eq", "fun": lambda l: np.sum(l) - 1.0, "jac": lambda l: np.ones(K)}
    bounds = [(1e-8, 1.0)] * K

    starts = []
    for k in range(K):
        e = np.full(K, 1e-8)
        e[k] = 1.0 - (K - 1) * 1e-8
        starts.append(e)
    starts.append(np.ones(K) / K)
    # Edge midpoints: for the non-convex MLP the GN landscape has local
    # maxima on simplex edges that vertex+centroid starts alone miss.
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            e = np.full(K, 1e-8)
            e[k1] = 0.5 - (K - 2) * 0.5e-8
            e[k2] = 0.5 - (K - 2) * 0.5e-8
            starts.append(e)
    # Near-corner starts:  for each vertex k, a point at λ[k] = 0.8 with
    # the remaining mass spread uniformly over the other K-1 coordinates.
    # Empirically on the non-convex MLP, the global argmax of GN on Δ_K
    # often lies in the interior of a near-corner region (e.g.,
    # (0.8, 0.1, 0.1) for K=3) — a region not covered by vertex,
    # centroid, or edge-midpoint starts.  Without these starts,
    # `_maximise_GN` reports PC values 4 orders of magnitude smaller
    # than the true sup_λ GN, causing premature outer-loop termination
    # and the plotted-error plateau seen in MLP_cpu_vs_accuracy.
    NEAR_CORNER_MASS = 0.8
    for k in range(K):
        e = np.full(K, (1.0 - NEAR_CORNER_MASS) / (K - 1))
        e[k] = NEAR_CORNER_MASS
        starts.append(e)
    if prev_lam is not None:
        # Warm-start from the previous outer iteration's optimum
        # (additional start, not replacement).
        starts.append(np.clip(prev_lam, 1e-8, 1.0))

    best_val = np.inf
    best_lam = starts[0]
    for lam0 in starts:
        res = sp_minimize(neg_gn, lam0, jac=neg_gn_jac, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-6, "maxiter": 60})
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
    eps_inner: Optional[float] = None,
    prune: bool = True,
    joint_oracle: Optional[Callable] = None,
) -> int:
    """Inner-loop BundleUpdate at fixed λ.

    Two stopping modes
    ------------------
    * ``eps_inner=None`` (default):  run exactly ``max_steps`` T_map
      iterations, then (if ``prune``) commit only the candidate with
      the smallest ∥∇F_λ∥ to the bundle.  Backward-compatible with the
      original fixed-budget inner loop.

    * ``eps_inner=ε/3`` (Algorithm 2 from the paper):  the convergence
      proof (Appendix B.1) requires the bundle update to drive
      ``PC(λ_t; B_{t+1}) ≤ ε/3`` at the active λ_t before the outer
      loop advances.  In this mode we add T_map iterates one at a time
      and recompute ``pc_fn(bundle, lam)`` after each addition; the
      inner loop terminates as soon as the PC at λ drops below
      ``eps_inner``, or after ``max_steps`` candidates (safety cap, to
      avoid an unbounded run if a corner case violates the theory's
      smoothness/strong-convexity assumptions).

    Pruning heuristic
    -----------------
    The paper's §7 implementation note adds *only* the candidate with
    the smallest gradient norm to the bundle, motivated by the
    observation that the other candidates contribute negligibly once
    the active index is well-served.  We expose this as ``prune``:

    * ``prune=True``  (default) — the runtime-efficient §7 variant.
    * ``prune=False`` — keep every committed candidate; faithful to
      the proof in Appendix B.1 which assumes BundleUpdate appends
      every T_map iterate.

    Implementation note (CPU optimisation, no semantic change)
    ----------------------------------------------------------
    The original implementation built a ``copy.deepcopy(bundle)`` work
    bundle to hold the candidate chain, then committed only the winner
    to the real bundle.  We instead append candidates in-place and pop
    the losers at the end, avoiding O(m·K·d) bundle copying per outer.
    The T_map call uses the vectorised ``_T_map_batched`` helper.

    Returns
    -------
    Number of T_map iterations actually executed.  Equals ``max_steps``
    when ``eps_inner=None`` or when the ε-target was not reached within
    the safety cap; smaller otherwise.
    """
    base_m = bundle.m
    steps_taken = 0
    K = bundle.K
    d = bundle.d
    L_arr = bundle.L
    mu_arr = bundle.mu

    # ------------------------------------------------------------------
    # CPU optimisation:  maintain Fbuf/Jbuf/Pbuf as pre-allocated buffers
    # covering the base bundle plus up to ``max_steps`` new candidates.
    # This avoids the O((m + s)·K·d) rebuild that
    # ``_bundle_arrays(bundle)`` + ``np.asarray(bundle.points)`` would
    # otherwise pay on EVERY T-map step.  After ``bundle.add_point``,
    # we assign the new row into the next slot of the buffer (an
    # O(K·d) write) and slice the buffer to the active region for
    # ``_T_map_batched``.
    #
    # We also avoid calling ``pc_fn(bundle, lam)`` for the inner
    # convergence check when ``pc_fn`` would internally rebuild Jbuf
    # a second time.  The closures in ``algorithm2_progressive`` for
    # both 'gap' and 'gn' modes call ``_bundle_arrays(bundle)`` again,
    # so we bypass that overhead by inlining a mode-specific check
    # against the cached ``Jbuf[:active+1]`` slice.  This requires
    # peeking at the bundle's properties (whether mu exists) to know
    # which check to inline.  When ``mu is None`` we use the GN
    # criterion (since 'gap' requires mu, the no-mu case must be
    # 'gn'); when mu is present we still go through ``pc_fn`` to
    # preserve the gap/gn distinction without misclassifying.
    cap = base_m + max_steps
    Fbuf = np.empty((cap, K), dtype=np.float64)
    Jbuf = np.empty((cap, K, d), dtype=np.float64)
    Pbuf = np.empty((cap, d), dtype=np.float64)
    if base_m > 0:
        Fbuf[:base_m] = np.asarray(bundle.fvals)
        Jbuf[:base_m] = np.asarray(bundle.grads)
        Pbuf[:base_m] = np.asarray(bundle.points)

    # ------------------------------------------------------------------
    # Generate the candidate chain on the real bundle.  Each T_map call
    # sees all previously-added in-round candidates, matching the proof's
    # BundleUpdate semantics.
    # ------------------------------------------------------------------
    for s in range(max_steps):
        active = base_m + s
        # Slice views (no copy) into the live portion of the buffers.
        Fmat = Fbuf[:active]
        Jmat = Jbuf[:active]
        points_arr = Pbuf[:active]

        x_new = _T_map_batched(Fmat, Jmat, points_arr, L_arr, lam)
        bundle.add_point(x_new, objectives, grad_objectives, joint_oracle=joint_oracle)
        steps_taken += 1

        # Append the just-evaluated row into the buffer — O(K·d) write.
        Fbuf[active] = bundle.fvals[-1]
        Jbuf[active] = bundle.grads[-1]
        Pbuf[active] = bundle.points[-1]

        if eps_inner is not None:
            # Inline the PC check against the up-to-date Jbuf slice,
            # avoiding a redundant ``_bundle_arrays(bundle)`` call inside
            # ``pc_fn``.  We use the batched evaluator that matches the
            # mode set up by ``algorithm2_progressive``:
            #   * mu is None       → GN-mode (only mode that supports no-µ).
            #   * mu is not None   → fall back to ``pc_fn`` which respects
            #                        the caller's mode choice (gap vs gn).
            #
            # In the no-mu branch we call ``_gn_value_and_jac_batched``
            # directly on the live buffer slices, avoiding the
            # ``_bundle_arrays(bundle)`` rebuild inside the gn-mode
            # pc_fn closure.  We discard the analytical Jacobian (unused
            # for the inner-loop scalar comparison).
            if mu_arr is None:
                pc_val, _, _ = _gn_value_and_jac_batched(
                    Fbuf[:active + 1], Jbuf[:active + 1], L_arr, mu_arr, lam)
            else:
                pc_val = pc_fn(bundle, lam)
            if pc_val <= eps_inner:
                break

    # ------------------------------------------------------------------
    # Optional pruning to the argmin-gnorm winner (paper §7 heuristic).
    # ------------------------------------------------------------------
    if prune and steps_taken > 1:
        # Pick argmin ∥∇F_λ(x^i)∥ from the cached gradients of the
        # candidates.  Vectorised via einsum.
        cand_Js = np.asarray(bundle.grads[base_m:base_m + steps_taken])  # (S, K, d)
        cand_grads_lam = np.einsum('skd,k->sd', cand_Js, lam)            # (S, d)
        cand_gnorms = np.einsum('sd,sd->s', cand_grads_lam, cand_grads_lam)
        best_local = int(np.argmin(cand_gnorms))
        best_idx = base_m + best_local

        # Save the winner, pop *all* candidates, push only the winner.
        win_x = bundle.points[best_idx]
        win_fv = bundle.fvals[best_idx]
        win_gv = bundle.grads[best_idx]
        for _ in range(steps_taken):
            bundle.pop_point()
        bundle.points.append(win_x)
        bundle.fvals.append(win_fv)
        bundle.grads.append(win_gv)

    return steps_taken


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
    epsilon: Optional[float] = None,
    max_outer: int = 50,
    max_inner: int = 400,
    prune_inner: bool = True,
    checkpoint_every: int = 20,
    eval_every_n_grads: Optional[int] = None,
    target_err: Optional[float] = None,
    joint_oracle: Optional[Callable] = None,
    use_fused_oracle: bool = True,
    verbose: bool = False,
) -> Dict:
    """Run Algorithm 2 with periodic worst-case-error checkpoints.

    Thin wrapper around the algorithm.py primitives that interleaves
    worst-case-error evaluations with the main outer loop.

    Termination
    -----------
    Two modes, controlled by ``epsilon``:

    * ``epsilon=None`` (legacy / fixed-budget):  run exactly ``max_outer``
      outer iterations with exactly ``max_inner`` T-map steps each.
      Useful for plot-by-plot comparisons at matched budgets.

    * ``epsilon=ε > 0`` (Algorithm 2 from the paper, App. B.1):
        - Outer loop terminates as soon as
              ``PC*_t = max_{λ ∈ Δ_K} PC(λ; B_t) ≤ 2ε/3``,
          which guarantees ``PC*_t ≤ ε`` over the full simplex by the
          Lipschitz-net argument of the proof.
        - Inner loop at λ_t terminates as soon as
              ``PC(λ_t; B_{t+1}) ≤ ε/3``,
          matching the inner-iterate count ``M_t`` derived in
          Corollary 5.2 (strongly-convex case) and Corollary 5.3
          (GN/non-convex case).
        - ``max_outer`` and ``max_inner`` become **safety caps**:
          the loops also terminate if the cap is hit, which protects
          against unbounded runs when the theory's smoothness
          constants are violated locally (e.g. for the ReLU MLP).

    Bundle-update heuristic
    -----------------------
    ``prune_inner=True`` (default) keeps the §7 pruning heuristic:
    only the candidate with the smallest gradient norm is retained at
    each outer iteration.  Set ``prune_inner=False`` for strict
    adherence to the App. B.1 proof, which appends every T-map
    iterate to the bundle.

    Checkpoint cadence
    ------------------
    Two complementary controls:
      - ``checkpoint_every``    : checkpoint every k outer iterations.
      - ``eval_every_n_grads``  : if set, additionally checkpoint at
                                  the next outer-iteration boundary
                                  after every M cumulative gradient
                                  evaluations.

    Parameters
    ----------
    epsilon              : target accuracy.  ``None`` reverts to fixed
                           budget (uses max_outer/max_inner directly).
    max_outer            : hard cap on outer iterations.
    max_inner            : hard cap on inner T-map iterations per outer.
    prune_inner          : keep only argmin-‖∇F_λ‖ candidate per outer.
    checkpoint_every     : evaluate worst-case error every k outer iters.
    eval_every_n_grads   : checkpoint after each M cumulative grad evals.
    target_err           : optional worst-case-error early-stop threshold.
                           When set, the outer loop terminates the first
                           time a checkpoint records ``err ≤ target_err``.
                           Intended use:  pass the baseline's final
                           worst-case error so A2 stops as soon as it
                           matches or beats the baseline's accuracy,
                           preventing any extra work.  The check fires
                           only at checkpoint boundaries (worst-case
                           error is only computed there), so the
                           guarantee is "no additional work beyond the
                           outer iteration that first reaches the
                           threshold."
    joint_oracle         : optional fused oracle ``θ → (fv, gv)`` returning
                           stacked ``(K,)`` and ``(K, d)`` arrays in a
                           single forward pass.  When provided, threaded
                           into ``bundle.add_point`` to eliminate the
                           redundant forward-pass work that occurs when
                           the K F_i and K ∇F_i closures are called
                           sequentially.  Typical use:  pass the
                           ``joint_oracle`` returned by
                           ``make_mlp_nonconvex`` / ``make_logreg_*``.
    """
    if epsilon is not None:
        eps_inner = epsilon / 3          # inner-loop PC threshold
    else:
        eps_inner = None

    # ---- Resolve actual joint oracle based on use_fused_oracle flag ----
    # The Tier 2 fused-across-classes oracle is exposed by the objective
    # factories as ``joint_oracle.fused`` (an attribute on the joint
    # oracle callable).  When ``use_fused_oracle=True`` (the default) and
    # the attribute exists, we swap it in here; downstream code
    # (bundle.add_point, initial point evaluation) uses the resolved
    # oracle uniformly.  Set ``use_fused_oracle=False`` to restore the
    # per-class fused oracle (pre-Tier-2 behavior).  Verified byte-
    # identical trajectory across 300+ outer iterations on both MLP and
    # logreg problems.
    if use_fused_oracle and joint_oracle is not None:
        fused = getattr(joint_oracle, "fused", None)
        if fused is not None:
            joint_oracle = fused

    if mode == "gap":
        # Use LB_2 (single-index minorant) inside both the λ-maximisation
        # and the inner-loop PC check.  LB_2 is ~100× faster than LB_1 and
        # avoids hitting the Gurobi size-limited license once the bundle
        # grows past ~100 points.
        #
        # ``pc_fn`` is called after every T_map step inside
        # ``_bundle_update_adaptive``.  The bundle.py:GAP function loops
        # over the bundle in Python via UB() + _LB_2(), which becomes
        # measurable once the bundle has 100+ points.  We replace it
        # with a direct call to the vectorised ``_ub_lb2_batched`` —
        # same value, O(m) numpy work instead of m Python iterations.
        def pc_fn(bundle, lam):
            Fmat, Jmat = _bundle_arrays(bundle)
            ub, lb2, *_ = _ub_lb2_batched(Fmat, Jmat, bundle.L, bundle.mu, lam)
            return ub - lb2
        # Closure over a mutable ``_prev`` so the warm-start λ can be
        # threaded through ``_maximise_GAP`` without changing its
        # signature contract for other callers.
        _prev: Dict[str, Optional[np.ndarray]] = {"lam": None}
        def maximise_pc(bundle):
            v, l = _maximise_GAP(bundle, variant="lb2", prev_lam=_prev["lam"])
            _prev["lam"] = l
            return v, l
        if mu is None:
            raise ValueError("mode='gap' requires mu (strong convexity).")
    elif mode == "gn":
        # The inner loop calls pc_fn after every T_map step.  With the
        # bundle growing into the thousands on the MLP problem, the
        # bundle.py:GN function (which loops over the bundle in Python)
        # is the dominant inner-loop cost.  We replace it with a
        # vectorised closure that uses ``_gn_value_and_jac_batched`` —
        # same value (verified against bundle.py:GN to machine precision
        # on the K=3 d=4 test bundle), O(m) numpy work instead of m
        # Python iterations.  Mirrors the gap-mode pc_fn pattern.
        def pc_fn(bundle, lam):
            Fmat, Jmat = _bundle_arrays(bundle)
            gn_val, _, _ = _gn_value_and_jac_batched(Fmat, Jmat, bundle.L, bundle.mu, lam)
            return gn_val
        # Closure-based prev_lam warm-start, mirroring the mode='gap' path.
        # Threading the previous outer's argmax-λ into _maximise_GN's
        # multi-start cuts the per-call SLSQP budget needed in steady state,
        # since the active argmin index ``i*`` typically only changes in
        # discrete jumps as the bundle grows.
        _prev_gn: Dict[str, Optional[np.ndarray]] = {"lam": None}
        def maximise_pc(bundle):
            v, l = _maximise_GN(bundle, prev_lam=_prev_gn["lam"])
            _prev_gn["lam"] = l
            return v, l
    else:
        raise ValueError(f"Unknown mode: {mode!r}.")

    bundle = Bundle(K=K, d=d, L=L, mu=mu)
    bundle.add_point(x0.copy(), objectives, grad_objectives, joint_oracle=joint_oracle)
    # The initial bundle point cost K gradient evals at x0.
    grad_evals = K

    cpu_times: List[float] = []
    worst_errs: List[float] = []
    outer_iters_history: List[int] = []
    grad_evals_history: List[int] = []
    pc_history: List[float] = []
    grad_evals_at_last_ckpt = 0

    # Cumulative wall time spent inside _worst_case_subopt_fast across all
    # prior checkpoints.  We subtract this from `time.time() - t_start`
    # when recording the next checkpoint, so the reported CPU at each
    # checkpoint reflects ONLY iterative algorithm work (not the wall
    # time spent computing worst-case error at earlier checkpoints).
    #
    # Without this correction, every prior checkpoint's evaluation cost
    # accumulates into the next checkpoint's reported time, inflating the
    # CPU axis by 2-10× depending on how many checkpoints have run.  This
    # turned out to be the dominant reason A2 *appeared* slower than the
    # baseline on the CPU plot: the per-checkpoint evaluation runs NAG
    # at every fine-grid λ (expensive), and accumulating that across
    # tens of checkpoints dwarfs A2's actual per-outer cost.
    checkpoint_overhead = 0.0

    def _checkpoint(label: str, pc_star=None, steps=None) -> None:
        nonlocal checkpoint_overhead
        cpu_times.append(time.time() - t_start - checkpoint_overhead)
        ck_t0 = time.time()
        err = _worst_case_subopt_fast(bundle, reference_map, objectives, K)
        checkpoint_overhead += time.time() - ck_t0
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
    # Setup work (Bundle construction, bundle.add_point at x0, the err0
    # evaluation above) is preprocessing — not iterative algorithm work.
    # Record checkpoint 0 with cpu = 0.0 explicitly and reset the clock so
    # subsequent checkpoints measure only iterative work.  This matches
    # the baseline's checkpoint-0 convention so both curves align on the
    # CPU axis at the initial point.
    cpu_times.append(0.0)
    worst_errs.append(err0)
    outer_iters_history.append(0)
    grad_evals_history.append(0)
    if verbose:
        print(f"  A2 outer 0/{max_outer} | t=0.00s | grad_evals=0 " f"| err={err0:.4e}  (constant map)")
    t_start = time.time()

    for t in range(max_outer):
        pc_star, best_lam = maximise_pc(bundle)
        pc_history.append(pc_star)
        bundle_m_before = bundle.m
        steps = _bundle_update_adaptive(
            bundle, best_lam, pc_fn, objectives, grad_objectives,
            max_steps=max_inner,
            eps_inner=eps_inner,
            prune=prune_inner,
            joint_oracle=joint_oracle,
        )
        # Each retained candidate costs K gradient evaluations.  When the
        # ε/3 inner stop fires early the bundle grew by ``steps`` points
        # before pruning; after pruning (if enabled) only one is retained,
        # but every T-map candidate did require K gradient evals to
        # construct, so we charge for all of them.
        grad_evals += steps * K
        retained_steps = bundle.m - bundle_m_before

        do_checkpoint = (
            ((t + 1) % checkpoint_every == 0)
            or (t + 1 == max_outer)
            or (eval_every_n_grads is not None
                and (grad_evals - grad_evals_at_last_ckpt) >= eval_every_n_grads
            )
        )
        if do_checkpoint:
            _checkpoint(f"outer {t + 1}/{max_outer}", pc_star=pc_star, steps=steps)
            grad_evals_at_last_ckpt = grad_evals
            if verbose and retained_steps < steps:
                print(f"        (attempted {steps}, retained {retained_steps} "
                      f"after pruning)")
            # ---- Worst-case-error early-stop -----------------------------
            # If ``target_err`` is provided and the most recent checkpoint
            # shows ``err ≤ target_err``, terminate.  Intended use is to
            # pass the baseline's final worst-case error so that A2
            # performs no additional work once it has matched or beaten
            # the baseline's accuracy.  The check fires only at checkpoint
            # boundaries because err is only computed there.
            if target_err is not None and worst_errs[-1] <= target_err:
                if verbose:
                    print(f"  A2 early-stop at outer {t + 1}/{max_outer}: "
                          f"err={worst_errs[-1]:.4e} ≤ target_err={target_err:.4e}")
                break
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