"""
algorithm.py  –  Adaptive Algorithm (Algorithm 6)
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from scipy.optimize import minimize as sp_minimize

# Optional IPOPT backend for the GN lambda-maximisation (used as K grows).
# cyipopt needs an IPOPT install; the import is guarded so the rest of the
# module still loads if it is absent, in which case `_maximise_GN` falls
# back to SLSQP with a one-time warning.

from cyipopt import minimize_ipopt as _ipopt_minimize
_HAS_IPOPT = True
import warnings

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


def _gn_value_and_jac_batched(Fmat: np.ndarray, Jmat: np.ndarray,
                              L: np.ndarray, mu: Optional[np.ndarray],
                              lam: np.ndarray
                              ) -> Tuple[float, np.ndarray, int]:
    """Batched evaluation and analytical λ-gradient of  GN(λ; B)  (Eq. 17).

    GN(λ) = min_i ‖J_i^T λ‖²,    where

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



# =====================================================================
#  PC-specific λ maximisation
# =====================================================================
# GN:  non-concave  →  multi-start local search
# ---------------------------------------------------------------------------
def _gn_multistart_set(K: int, prev_lam, max_starts: int,
                       seed: int = 0):
    """Deterministic multi-start set for the non-concave GN maximisation.

    Priority order, truncated to ``max_starts``:
        centroid > vertices > near-corner points > prev_lam > edge midpoints

    The first blocks are O(K); the edge-midpoint block is O(K^2) and is the one
    that must be bounded as K grows.  When the full structured set fits under
    ``max_starts`` (small / moderate K) every start is emitted and the set is
    identical to the original SLSQP implementation's.  When K is large the edge
    block is *lazily* sampled (never materialised in full), so the routine stays
    both time- and memory-bounded -- this is what keeps the per-checkpoint
    maximiser tractable as K is pushed higher.

    The near-corner points (lambda_k = 0.8, remainder spread uniformly) and the
    edge midpoints recover GN basins on the non-convex problem that the vertices
    and centroid alone miss; see the original implementation note.
    """
    rng = np.random.RandomState(seed)
    EPS, NEAR = 1e-8, 0.8
    starts = []

    def room():
        return max_starts - len(starts)

    # centroid
    if room() > 0:
        starts.append(np.full(K, 1.0 / K))

    # vertices (O(K)); subsample only if they would overflow the budget
    if room() > 0:
        idx = np.arange(K)
        if K > room():
            idx = rng.choice(K, size=room(), replace=False)
        for k in idx:
            e = np.full(K, EPS)
            e[k] = 1.0 - (K - 1) * EPS
            starts.append(e)

    # near-corner starts (O(K)); subsample only if needed
    if room() > 0:
        idx = np.arange(K)
        if K > room():
            idx = rng.choice(K, size=room(), replace=False)
        for k in idx:
            e = np.full(K, (1.0 - NEAR) / (K - 1))
            e[k] = NEAR
            starts.append(e)

    # warm start from the previous outer iteration's optimum (additional only)
    if room() > 0 and prev_lam is not None:
        starts.append(np.clip(prev_lam, EPS, 1.0))

    # edge midpoints (O(K^2)): enumerate when small, else lazily sample
    total_edges = K * (K - 1) // 2
    if room() > 0 and total_edges > 0:
        def edge_mid(a, b):
            e = np.full(K, EPS)
            e[a] = 0.5 - (K - 2) * 0.5 * EPS
            e[b] = 0.5 - (K - 2) * 0.5 * EPS
            return e
        if total_edges <= room():
            for a in range(K):
                for b in range(a + 1, K):
                    starts.append(edge_mid(a, b))
        else:
            seen = set()
            while room() > 0 and len(seen) < total_edges:
                a = int(rng.randint(0, K - 1))
                b = int(rng.randint(a + 1, K))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                starts.append(edge_mid(a, b))
    return starts


def _maximise_GN(bundle: Bundle, prev_lam: Optional[np.ndarray] = None,
                 solver: str = "ipopt", max_starts: int = 256
                 ) -> Tuple[float, np.ndarray]:
    """Find  lambda* = argmax_{lambda in Delta_K}  GN(lambda; B_m).

        GN(lambda; Bm) = min_i ||J_F(x_i)^T lambda||^2   (or its s.c. scaling)

    Solver backends
    ---------------
    The decision variable is lambda in Delta_K, so the dimensionality of *this*
    problem is K (the number of objectives), independent of d / m / n.

    * ``solver="ipopt"`` (default): each local solve is handed to IPOPT via
      cyipopt's ``minimize_ipopt``.  IPOPT is an interior-point NLP solver whose
      advantage grows with K (the regime this code is scaling toward).  GN is
      only piecewise-smooth (a pointwise min of quadratics, non-differentiable
      on the index-switching manifold), so we use IPOPT's limited-memory
      (L-BFGS) Hessian approximation rather than an exact Hessian, which would
      be discontinuous across the min's pieces.  The analytical Danskin gradient
      (``_gn_value_and_jac_batched``) is still supplied, giving IPOPT exact
      first-order information.
    * ``solver="slsqp"``: the original scipy multi-start SLSQP path, kept as a
      fallback and for benchmarking.  It is also used automatically (with a
      one-time warning) when cyipopt is not importable.

    Multi-start / scaling in K
    --------------------------
    GN is neither convex nor concave in lambda, so a single local solve only
    finds a local max; ``_gn_multistart_set`` supplies the global coverage.
    ``max_starts`` bounds the number of local solves so the O(K^2) edge-midpoint
    block does not blow up the per-checkpoint maximisation at large K; for small
    K the full structured set fits and behaviour matches the original.

    Each local solve is wrapped in try/except and the start point itself is
    scored first, so a failed or early-terminated solve never loses ground and
    the returned value is monotone in the starts actually evaluated.

    This routine is invoked by ``pc_star`` for metric evaluation (excluded from
    the plotted cost axes) and as the outer-loop lambda-selector.
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

    con_eq = {"type": "eq",
              "fun": lambda l: float(np.sum(l) - 1.0),
              "jac": lambda l: np.ones(K)}
    constraints = [con_eq]
    bounds = [(1e-8, 1.0)] * K

    starts = _gn_multistart_set(K, prev_lam, max_starts)

    use_ipopt = (solver == "ipopt") and _HAS_IPOPT
    if solver == "ipopt" and not _HAS_IPOPT:
        warnings.warn(
            "cyipopt is not installed; _maximise_GN is falling back to SLSQP. "
            "Install IPOPT + cyipopt (e.g. `conda install -c conda-forge "
            "cyipopt`) to enable the IPOPT backend.",
            RuntimeWarning, stacklevel=2,
        )

    best_val = np.inf            # minimum of neg_gn == -(max GN)
    best_lam = starts[0]
    for lam0 in starts:
        # Score the start point itself first so a failed / early-terminated
        # solve never loses ground (keeps the result monotone in the starts).
        v0 = neg_gn(lam0)
        if v0 < best_val:
            best_val, best_lam = float(v0), np.asarray(lam0, dtype=float)

        try:
            if use_ipopt:
                res = _ipopt_minimize(
                    neg_gn, lam0, jac=neg_gn_jac,
                    bounds=bounds, constraints=constraints,
                    options={
                        "print_level": 0,
                        "sb": "yes",                    # suppress IPOPT banner
                        "tol": 1e-8,
                        "max_iter": 100,
                        "hessian_approximation": "limited-memory",
                    },
                )
            else:
                res = sp_minimize(
                    neg_gn, lam0, jac=neg_gn_jac, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"ftol": 1e-6, "maxiter": 60},
                )
        except Exception:
            # A single failed local solve must not abort the maximisation.
            continue

        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = float(res.fun)
            best_lam = np.asarray(res.x, dtype=float).copy()

    best_lam = np.maximum(best_lam, 0.0)
    s = best_lam.sum()
    best_lam = best_lam / s if s > 0 else np.full(K, 1.0 / K)
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
      and recompute the active-λ PC after each addition; the inner loop
      terminates as soon as PC drops below ``eps_inner`` or after
      ``max_steps`` candidates (safety cap).

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
    when no stopping rule fires within the safety cap; smaller otherwise.
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
    # a second time.  The closures in ``algorithm_adaptive`` for
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
            # mode set up by ``algorithm_adaptive``:
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
#  Instrumented Adaptive Algorithm:  checkpoint after each outer iteration
# =====================================================================








# =====================================================================
#  Bundle-coverage stationarity metric (reference-map-free)
#
#  Used for the comparison described in the research note: instead of a
#  precomputed fine-resolution reference map, each method is scored by the
#  worst-case (over the simplex) progress criterion of its own bundle:
#
#    GN*(B)  = max_{lambda in Delta_K}  min_i ||grad F_lambda(x_i)||^2   (non-convex)
#
#  Both are computable directly from the bundle (function values +
#  gradients at the bundle points) with no reference map.  GAP* reuses the
#  GAP outer maximiser (variant 'lb2'); GN* reuses the GN outer maximiser
#  (which, when bundle.mu is None, maximises exactly min_i ||J^T lambda||^2).
# =====================================================================
def _gram_stack(Jmat: np.ndarray) -> np.ndarray:
    """Per-point Gram matrices G_i = J_i J_i^T,  shape (m, K, K)."""
    return np.einsum("mkd,mld->mkl", Jmat, Jmat)


def _gn_over_samples(Jmat: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Vectorised  min_i ||J_i^T lam||^2 = min_i lam^T G_i lam  for many lam.

    Jmat : (m, K, d) ; lam : (S, K).  Returns (S,) — the GN value at each lam.
    """
    G = _gram_stack(Jmat)                                   # (m, K, K)
    quad = np.einsum("sk,mkl,sl->sm", lam, G, lam)          # (S, m) = lam^T G_i lam
    return quad.min(axis=1)                                 # (S,)


def pc_star(bundle: Bundle, mode: str,
            prev_lam: Optional[np.ndarray] = None,
            n_random: int = 6000, seed: int = 0) -> Tuple[float, np.ndarray]:
    """Worst-case-over-the-simplex progress criterion of a bundle (robust).

    GN*(B)  = max_{lambda} min_i ||grad F_lambda(x_i)||^2   (non-convex, 'gn')

    The max over the simplex is **non-concave**, so a single multi-start SLSQP
    run is only a (noisy) lower bound on the true worst case.  Because this is
    *metric evaluation* (excluded from the plotted cost), we can afford to be
    thorough: we combine the gradient-based maximiser (``_maximise_GAP`` /
    ``_maximise_GN``) with a dense Dirichlet random-sample sweep of the simplex
    (cheap, fully vectorised over samples) and return the larger value.  This
    makes GN*/GAP* an accurate, near-monotone worst-case measure, so the
    head-to-head curves are trustworthy.
    """
    K = bundle.K
    Fmat, Jmat = _bundle_arrays(bundle)

    # (1) gradient-based multi-start maximiser (sharp local refinement)
    v_opt, lam_opt = _maximise_GN(bundle, prev_lam=prev_lam)

    # (2) dense vectorised random-sample sweep of the simplex (global coverage)
    if n_random and n_random > 0:
        rng = np.random.RandomState(seed)
        lam_s = rng.dirichlet(np.ones(K), size=n_random)         # (S, K)
        vals = _gn_over_samples(Jmat, lam_s)
        s_best = int(np.argmax(vals))
        if vals[s_best] > v_opt:
            v_opt, lam_opt = float(vals[s_best]), lam_s[s_best]

    return v_opt, lam_opt


# Alias used inside algorithm_adaptive, whose local variable ``pc_star``
# (the outer-loop PC* value) would otherwise shadow the function above.
_pc_star_metric = pc_star
def bundle_from_points(points: np.ndarray, K: int, d: int,
                       L: np.ndarray, mu: Optional[np.ndarray],
                       objectives: List[Callable],
                       grad_objectives: List[Callable],
                       joint_oracle: Optional[Callable] = None) -> Bundle:
    """Construct a Bundle from an array of points (one row per point).

    Evaluates all K objectives and gradients at each point (via the fused
    ``joint_oracle`` when provided).  This is how the uniform-discretisation
    method's "bundle of last-iterate points" is assembled at a checkpoint so
    that ``pc_star`` can score it.  The oracle calls here are *metric
    evaluation* (measurement), not algorithm work.
    """
    B = Bundle(K=K, d=d, L=np.asarray(L, dtype=float), mu=(None if mu is None else np.asarray(mu, dtype=float)))
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    for row in pts:
        B.add_point(row, objectives, grad_objectives, joint_oracle=joint_oracle)
    return B