"""
bundle.py  –  Core bundle data structure
==========================================================================

This module implements the *bundle* B_m from Section 3 of the paper:

    B_m = { (x_i, F_1(x_i), ..., F_K(x_i), ∇F_1(x_i), ..., ∇F_K(x_i)) }_{i=1}^m

and the three progress criteria from Example 2 (Section 5.2):

    1. UB  – upper bound (Eq. 12)
    2. GAP – gap = UB − LB  (Eq. 15, strongly convex case)
    3. GN  – scaled gradient norm (Eq. 17)

All quantities use the *λ-dependent* smoothness constants  Lλ = Σ_k λ_k L_k
and strong-convexity constants  µλ = Σ_k λ_k µ_k  (when applicable).

Illustrative example
--------------------
Suppose we have K = 2 objectives on R^2:

    F_1(x) = 0.5 * x^T A_1 x,   F_2(x) = 0.5 * x^T A_2 x

with A_1 = diag(2, 10) (so L_1=10, µ_1=2) and A_2 = diag(4, 6) (L_2=6, µ_2=4).

For λ = (0.5, 0.5) we get  Lλ = 8,  µλ = 3.
Starting from x_1 = (1, 1), the bundle stores F_k(x_1) and ∇F_k(x_1),
and we can immediately evaluate  UB(λ; B_1), GAP(λ; B_1), GN(λ; B_1).

After adding x_2 = T(λ; B_1)  (one gradient descent step picking the best
bundle point), each progress criterion is guaranteed to decrease or stay the
same (Assumption 3.1, global monotonicity).
"""

from __future__ import annotations
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass, field
from typing import List, Optional, Callable


# ---------------------------------------------------------------------------
# Shared Gurobi environment (created once, reused across all LB solves)
# ---------------------------------------------------------------------------
_GRB_ENV: Optional[gp.Env] = None


def _get_gurobi_env() -> gp.Env:
    """Return a shared Gurobi environment with logging suppressed.

    Creating a Gurobi Env is expensive (license check, thread pool init).
    We do it once and cache it in a module-level variable so that every
    subsequent LB call reuses the same environment.
    """
    global _GRB_ENV
    if _GRB_ENV is None:
        _GRB_ENV = gp.Env(empty=True)
        _GRB_ENV.setParam("OutputFlag", 0)     # suppress all console output
        _GRB_ENV.setParam("LogToConsole", 0)
        _GRB_ENV.start()
    return _GRB_ENV


# ---------------------------------------------------------------------------
# Bundle data structure
# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    """Stores zeroth- and first-order oracle information at visited points.

    Attributes
    ----------
    K : int
        Number of objective functions.
    d : int
        Dimension of the decision variable x ∈ R^d.
    points : list[np.ndarray]
        List of iterates  x_1, …, x_m  (each shape (d,)).
    fvals : list[np.ndarray]
        fvals[i] = (F_1(x_i), …, F_K(x_i))  shape (K,).
    grads : list[np.ndarray]
        grads[i] = J_F(x_i)  the K×d Jacobian at x_i.
    L : np.ndarray
        Smoothness constants (L_1, …, L_K), shape (K,).
    mu : np.ndarray | None
        Strong-convexity / PL constants (µ_1, …, µ_K).  None when unavailable.
    """

    K: int
    d: int
    L: np.ndarray                       # shape (K,)
    mu: Optional[np.ndarray] = None     # shape (K,) or None
    points: List[np.ndarray] = field(default_factory=list)
    fvals: List[np.ndarray] = field(default_factory=list)
    grads: List[np.ndarray] = field(default_factory=list)

    # ---- helpers ----
    @property
    def m(self) -> int:
        """Current bundle size."""
        return len(self.points)

    def L_lam(self, lam: np.ndarray) -> float:
        """Lλ = Σ_k λ_k L_k."""
        return float(lam @ self.L)

    def mu_lam(self, lam: np.ndarray) -> float:
        """µλ = Σ_k λ_k µ_k  (requires self.mu is not None)."""
        assert self.mu is not None, "mu not set (needed for strong convexity / PL)"
        return float(lam @ self.mu)

    def F_lam(self, idx: int, lam: np.ndarray) -> float:
        """Fλ(x_i) = λ^T F(x_i)."""
        return float(lam @ self.fvals[idx])

    def grad_F_lam(self, idx: int, lam: np.ndarray) -> np.ndarray:
        """∇Fλ(x_i) = J_F(x_i)^T λ,  shape (d,)."""
        return self.grads[idx].T @ lam   # (d, K) @ (K,) = (d,)

    def add_point(self, x: np.ndarray, objectives: List[Callable], grad_objectives: List[Callable],
                  joint_oracle: Optional[Callable] = None):
        """Evaluate all objectives and gradients at x and append to bundle.

        Parameters
        ----------
        x                 : iterate at which to evaluate.
        objectives        : list of K F_i closures (used when ``joint_oracle`` is None).
        grad_objectives   : list of K ∇F_i closures (used when ``joint_oracle`` is None).
        joint_oracle      : optional fused oracle ``θ → (fv, gv)`` returning
                            ``(K,)`` and ``(K, d)`` arrays in a single pass.  When
                            provided, eliminates the redundant forward-pass work that
                            otherwise occurs when ``F_i`` and ``∇F_i`` are called
                            sequentially.  See ``make_mlp_nonconvex`` /
                            ``make_logreg_strongly_convex`` for fused oracles.
        """
        if joint_oracle is not None:
            fv, gv = joint_oracle(x)
        else:
            fv = np.array([f(x) for f in objectives])
            gv = np.vstack([g(x) for g in grad_objectives])   # (K, d)
        self.points.append(x.copy())
        self.fvals.append(fv)
        self.grads.append(gv)

    def pop_point(self):
        """Pop the last element out of the oracle."""
        self.points.pop()
        self.fvals.pop()
        self.grads.pop()


# ---------------------------------------------------------------------------
# Progress criteria  (Section 5.2)
# ---------------------------------------------------------------------------
def UB(bundle: Bundle, lam: np.ndarray) -> float:
    """Upper bound progress criterion  (Eq. 12).

        UB(λ; B_m) = min_{i ∈ [m]}  { Fλ(x_i) − 1/(2Lλ) ‖∇Fλ(x_i)‖² }

    This is valid under any smoothness assumption (no convexity needed).
    """
    Ll = bundle.L_lam(lam)
    best = np.inf
    for i in range(bundle.m):
        fi = bundle.F_lam(i, lam)
        gi = bundle.grad_F_lam(i, lam)
        val = fi - 0.5 / Ll * np.dot(gi, gi)
        if val < best:
            best = val
    return best


# ---------------------------------------------------------------------------
# LB  –  lower bound (Eq. 14), solved via Gurobi
# ---------------------------------------------------------------------------
def _build_lb_data(bundle: Bundle, lam: np.ndarray):
    """Precompute the Gram matrix G and linear vector c for the LB QP.

    Returns (G, c, mul, m) where the LB problem is:

        max_{β ∈ Δ_m}  { −µλ/2  β^T G β  +  c^T β }

    Derivation (from Eq. 14):
        X̌_i  = x_i − (1/µλ) ∇Fλ(x_i)
        c_i   = Fλ(x_i) − ⟨∇Fλ(x_i), x_i⟩ + (µλ/2) ‖x_i‖²
        G     = X̌ X̌^T   (m × m, PSD)
    """
    m = bundle.m
    mul = bundle.mu_lam(lam)

    X = np.array([p for p in bundle.points])                   # (m, d)
    Fvec = np.array([fi @ lam for fi in bundle.fvals])         # (m,)
    Gmat_arr = np.array([J.T @ lam for J in bundle.grads])     # (m, d)

    Xcheck = X - Gmat_arr / mul                                # (m, d)
    diagHlam = np.sum(Gmat_arr * X, axis=1)
    dvec = np.sum(X * X, axis=1)
    c = Fvec - diagHlam + 0.5 * mul * dvec

    G = Xcheck @ Xcheck.T
    G = 0.5 * (G + G.T) + 1e-12 * np.eye(m)

    return G, c, mul, m


def _lb_gurobi(G: np.ndarray, c: np.ndarray, mul: float, m: int) -> float:
    """Solve the LB QP via Gurobi.

    Formulation (equivalent convex minimisation):

        min_β  { µλ/2  β^T G β  −  c^T β }
        s.t.   β_i ≥ 0  for all i,
               Σ_i β_i = 1

    Gurobi solves this as a convex QP to global optimality.  The
    original concave-maximisation value is recovered by negating
    the optimal objective.

    Implementation notes
    --------------------
    - Uses the matrix-oriented MVar API (addMVar) for efficient
      bulk variable creation and quadratic objective specification.
    - Reuses a shared Gurobi environment (_get_gurobi_env) so the
      license check and thread-pool initialisation happen only once.
    - The quadratic objective matrix is  H = µλ · G  which is PSD
      since G = X̌ X̌^T is a Gram matrix.  Gurobi verifies PSD-ness
      internally and solves via its barrier or dual-simplex QP solver.

    Illustrative example
    --------------------
    For m = 3 (three bundle points), β ∈ Δ_3 is a triangle.
    Gurobi finds the β* on that triangle minimising the convex
    quadratic  0.5 µλ β^T G β − c^T β,  typically in a fraction
    of a millisecond.  Negating the objective gives the tightest
    lower bound from the three supporting hyperplanes:

        LB = −(0.5 µλ β*^T G β* − c^T β*) = −0.5 µλ ‖X̌^T β*‖² + c^T β*
    """
    env = _get_gurobi_env()
    model = gp.Model(env=env)

    # Decision variable:  β ∈ R^m  with  0 ≤ β_i ≤ 1
    beta = model.addMVar(m, lb=0.0, ub=1.0, name="beta")

    # Constraint:  Σ β_i = 1  (simplex)
    model.addConstr(beta.sum() == 1.0, name="simplex")

    # Objective:  min  0.5 µλ β^T G β  −  c^T β
    # Gurobi's setObjective with a QuadExpr built from MVar:
    #   beta @ H @ beta  expands to  Σ_{i,j} H_{ij} β_i β_j
    #   which Gurobi internally reads as  β^T H β  (includes the
    #   factor of 2 in off-diagonals), so we pass 0.5 * H.
    H = mul * G
    model.setObjective(0.5 * (beta @ H @ beta) - c @ beta, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return float(-model.objVal)
    else:
        # Fallback: return best vertex value
        return float(np.max(c - 0.5 * mul * np.diag(G)))


def _LB_1(bundle: Bundle, lam: np.ndarray) -> float:
    """LB_1: aggregated strongly convex minorants  (Eq. 14, Section 5.2.1).

        LB_1(λ; B_m) = max_{β ∈ Δ_m}  { −µλ/2  β^T G β  +  c^T β }

    This is the tightest single-step aggregation of the m individual
    strongly convex minorants, solved as a concave QP over β ∈ Δ_m.

    Expensive (O(m²) Gurobi QP solve per call) but gives the best
    lower bound and therefore the smallest GAP at each iteration.
    """
    G, c, mul, m = _build_lb_data(bundle, lam)
    return _lb_gurobi(G, c, mul, m)


def _LB_2(bundle: Bundle, lam: np.ndarray) -> float:
    """LB_2: best single-index minorant  (derived from GAP_2 in Eq. 15).

        LB_2(λ; B_m) = max_{i ∈ [m]}  { Fλ(x_i) − (1/(2µλ)) ‖∇Fλ(x_i)‖² }

    Each bundle point x_i gives an individual strongly convex minorant:

        F_λ^* ≥ F_λ(x_i) − (1/(2µλ)) ‖∇F_λ(x_i)‖²    (for every i)

    LB_2 picks the single tightest one — i.e., it restricts β in LB_1
    to the vertices of Δ_m (β = e_i for some i).  Equivalently, it
    "biases all weights towards the best index."

    This corresponds to GAP_2 in the paper:

        GAP_2(λ; B_m) = UB(λ; B_m) − LB_2(λ; B_m)
                     = min_i {Fλ(x_i) − 1/(2Lλ) ‖∇Fλ(x_i)‖²}
                       − max_i {Fλ(x_i) − 1/(2µλ) ‖∇Fλ(x_i)‖²}

    Cost: O(m·K·d) — just a scan over bundle points.  Much faster
    than LB_1's QP solve, at the price of a slightly looser bound
    (since LB_2 ≤ LB_1 always).

    Illustrative example
    --------------------
    For a 3-point bundle, LB_1 finds the optimal β ∈ Δ_3 (a 2D
    triangle search), while LB_2 just picks the best single vertex:
    β ∈ {e_1, e_2, e_3}.  LB_2 ≤ LB_1, hence GAP_2 ≥ GAP_1.
    """
    mul = bundle.mu_lam(lam)
    best = -np.inf
    for i in range(bundle.m):
        fi = bundle.F_lam(i, lam)
        gi = bundle.grad_F_lam(i, lam)
        val = fi - 0.5 / mul * np.dot(gi, gi)
        if val > best:
            best = val
    return best


def LB(bundle: Bundle, lam: np.ndarray, variant: str = "lb1") -> float:
    """Lower bound for F*_λ from bundle information.

    Two variants are available:

    - "lb1"  (default):  aggregated minorants  (Eq. 14).
         LB_1(λ; B_m) = max_{β ∈ Δ_m} { −µλ/2 β^T G β + c^T β }
         Tightest bound.  Requires a QP solve per call.

    - "lb2":  best single-index minorant  (derived from GAP_2).
         LB_2(λ; B_m) = max_i { Fλ(x_i) − 1/(2µλ) ‖∇Fλ(x_i)‖² }
         Cheaper.  Corresponds to restricting β to vertices of Δ_m.

    Relationship:  LB_2(λ; B_m) ≤ LB_1(λ; B_m)  always.
    Hence  GAP_2 ≥ GAP_1,  so LB_2 gives a more conservative progress
    criterion.  Use LB_2 for speed; use LB_1 for the tightest bound.

    Parameters
    ----------
    bundle  : current Bundle object.
    lam     : weight vector λ ∈ Δ_K.
    variant : "lb1" (QP-based, tight) or "lb2" (scan-based, fast).
    """
    if bundle.m == 0:
        return -np.inf
    if variant == "lb1":
        return _LB_1(bundle, lam)
    elif variant == "lb2":
        return _LB_2(bundle, lam)
    else:
        raise ValueError(f"Unknown LB variant: {variant!r}. Use 'lb1' or 'lb2'.")


def GAP(bundle: Bundle, lam: np.ndarray, variant: str = "lb1") -> float:
    """Gap progress criterion  GAP = UB − LB  (Eq. 15).

    Only meaningful under strong convexity.

    Parameters
    ----------
    bundle  : current Bundle object.
    lam     : weight vector λ ∈ Δ_K.
    variant : which LB to use — "lb1" (tight, default) or "lb2" (fast).
              "lb1" gives GAP_1; "lb2" gives GAP_2 (Eq. 15 in the paper).
    """
    return UB(bundle, lam) - LB(bundle, lam, variant=variant)


def GN(bundle: Bundle, lam: np.ndarray) -> float:
    """Scaled gradient-norm progress criterion  (Eq. 17).

        GN(λ; B_m) = 1/2 (1/µλ − 1/Lλ) min_{i} ‖∇Fλ(x_i)‖²

    Uses both µλ and Lλ as in Example 2.
    In the generic non-convex case (no µ), we fall back to
        GN(λ; B_m) = min_i  ‖∇Fλ(x_i)‖²   (un-scaled).
    """
    min_gnorm_sq = np.inf
    for i in range(bundle.m):
        gi = bundle.grad_F_lam(i, lam)
        gnorm_sq = float(np.dot(gi, gi))
        if gnorm_sq < min_gnorm_sq:
            min_gnorm_sq = gnorm_sq

    if bundle.mu is not None:
        mul = bundle.mu_lam(lam)
        Ll = bundle.L_lam(lam)
        scale = 0.5 * (1.0 / mul - 1.0 / Ll)
        return scale * min_gnorm_sq
    else:
        return min_gnorm_sq


# ---------------------------------------------------------------------------
# Mapping  T(λ; B_m)  –  the new point to add  (Eq. 13)
# ---------------------------------------------------------------------------
def T_map(bundle: Bundle, lam: np.ndarray) -> np.ndarray:
    """Compute T(λ; B_m) = x_{i*} − (1/Lλ) ∇Fλ(x_{i*})

    where  i* = argmin_{i∈[m]}{ Fλ(x_i) − 1/(2Lλ) ‖∇Fλ(x_i)‖² }.
    (Eq. 13 – one step of gradient descent from the best bundle point.)
    """
    Ll = bundle.L_lam(lam)
    best_val = np.inf
    best_i = 0
    for i in range(bundle.m):
        fi = bundle.F_lam(i, lam)
        gi = bundle.grad_F_lam(i, lam)
        val = fi - 0.5 / Ll * np.dot(gi, gi)
        if val < best_val:
            best_val = val
            best_i = i
    xi = bundle.points[best_i]
    gi = bundle.grad_F_lam(best_i, lam)
    return xi - (1.0 / Ll) * gi