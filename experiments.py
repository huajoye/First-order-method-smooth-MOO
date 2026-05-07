"""
experiments.py  –  Numerical experiments for the bundle MOO algorithm
=====================================================================

All experiments use multi-class classification objectives following
the notation from the paper:

    K classes, weight vectors  w^1, …, w^K ∈ R^p,
    labelled data  {(y_j, x_j)}_{j=1}^n  with  y_j ∈ [K],
    per-class loss  F_i(W) = (1/n_i) Σ_{j: y_j=i} {−log P(Y=i|x_j; W)}.

Five experiments:

  Exp 1 – Regularised multi-class logreg                                     (PC = GAP₁, strongly convex)
  Exp 2 – Regularised multi-class logreg with interpolation                  (PC = UB,   interpolation + PL)
  Exp 3 – Single-hidden-layer MLP                                            (PC = GN,   generic non-convex)
  Exp 4 – Pareto front tracing (2-class logreg)

The algorithm uses:
  - PC-specific λ maximisation (SLSQP / multi-start) instead of grid search
  - Adaptive inner-loop with max_steps

Output:  ``experiment_results.png``
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from algorithm import algorithm2_progressive
from objectives import (
    make_logreg_strongly_convex,
    make_mlp_nonconvex,
)
from baseline import *


# =====================================================================
#  Utility:  simplex grid  (for Pareto front)
# =====================================================================
def simplex_grid(K: int, resolution: int = 20) -> np.ndarray:
    """Tile the unit simplex Δ_K with a uniform grid."""
    if K == 1:
        return np.array([[1.0]])
    if K == 2:
        ts = np.linspace(0, 1, resolution + 1)
        return np.column_stack([ts, 1 - ts])
    points = []

    def _recurse(remaining, depth, current):
        if depth == K - 1:
            current.append(remaining)
            points.append(current[:])
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            _recurse(remaining - v, depth + 1, current)
            current.pop()

    _recurse(resolution, 0, [])
    return np.array(points, dtype=float) / resolution

############################
##plot functions
#############################
def _plot_cpu_vs_accuracy(
    bl: Dict, a2: Dict,
    plot_path: str,
    problem_params: Dict,
    coarse_resolution: int,
    fine_resolution: int,
    pc_label: str = "GAP",
) -> None:
    """Plot CPU time vs worst-case suboptimality for both methods.

    ``pc_label`` selects the progress-criterion shown in the Algorithm 2
    legend entry — e.g. "GAP", "UB", "GN".  Defaults to "GAP" for
    backward compatibility with ``experiment_logreg_gap``.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.semilogy(
        a2["cpu_times"], a2["worst_errs"],
        "o-", color="#2563eb", markersize=5, linewidth=1.8,
        label=f"Algorithm 2 ({pc_label})",
    )
    ax.semilogy(
        bl["cpu_times"], bl["worst_errs"],
        "s-", color="#dc2626", markersize=5, linewidth=1.8,
        label=f"Uniform discretisation (r = {coarse_resolution})",
    )

    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel(r"$\sup_{\lambda \in G_{\mathrm{fine}}}\,"
                  r"[F_\lambda(\hat x(\lambda)) - F_\lambda^*]$")
    params_str = _format_params(problem_params)
    ax.set_title(
        f"CPU time vs worst-case suboptimality\n"
        f"{params_str}  |  G_fine res = {fine_resolution}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved to {plot_path}")


def _plot_grads_vs_accuracy(
    bl: Dict, a2: Dict,
    plot_path: str,
    problem_params: Dict,
    coarse_resolution: int,
    fine_resolution: int,
    pc_label: str = "GAP",
) -> None:
    """Plot gradient evaluations vs worst-case suboptimality.

    For each method, the x-axis is the cumulative number of gradient-
    oracle evaluations  ∇F_k(x) used so far (one scalarised GD step
    costs K such evaluations, since it computes ∇F_k for all k ∈ [K]).
    The y-axis is the worst-case function-value suboptimality of the
    method's solution map at that point in its execution.

    This is the "oracle complexity" view: how much gradient information
    does each method need to achieve a given accuracy?  Unlike the CPU-
    time view, this strips away algorithmic overhead and reflects only
    the information-theoretic cost.

    ``pc_label`` selects the progress-criterion shown in the Algorithm 2
    legend entry.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        a2["grad_evals_history"], a2["worst_errs"],
        "o-", color="#2563eb", markersize=5, linewidth=1.8,
        label=f"Algorithm 2 ({pc_label})",
    )
    ax.plot(
        bl["grad_evals_history"], bl["worst_errs"],
        "s-", color="#dc2626", markersize=5, linewidth=1.8,
        label=f"Uniform discretisation (r = {coarse_resolution})",
    )

    # Use a symlog x-axis so the shared initial point at grad_evals = 0
    # is plotted (a true log axis cannot represent zero).  Linear region
    # extends out to linthresh = 1, logarithmic beyond.
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")

    ax.set_xlabel("Number of total gradient evaluations")
    ax.set_ylabel(r"$\sup_{\lambda \in G_{\mathrm{fine}}}\,"
                  r"[F_\lambda(\hat x(\lambda)) - F_\lambda^*]$")
    params_str = _format_params(problem_params)
    ax.set_title(
        f"Worst-case suboptimality vs Total gradient evaluations \n"
        f"{params_str}  |  G_fine res = {fine_resolution}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")


def _plot_pc_history(
    a2: Dict,
    plot_path: str,
    problem_params: Dict,
    pc_label: str = "GAP",
) -> None:
    """Plot the progress-criterion history  PC*_t  of Algorithm 2.

    For each outer iteration  t = 1, 2, ..., the value
        PC*_t := max_{λ ∈ Δ_K} PC(λ; B_t)
    is recorded (this is the value attained by the inner λ-maximisation).
    Theorem 2 bounds PC*_t ≤ ε after at most  (C · Lip_PC / ε)^{K−1}
    iterations, which translates to  PC*_t = O(t^{-1/(K-1)}).

    A log-y axis makes the (empirically much faster than worst-case)
    decay rate easy to read.  The baseline has no analogous quantity,
    so only Algorithm 2's curve is shown.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    pc_values = a2["pc_history"]
    if len(pc_values) == 0:
        print(f"  Skipping {plot_path}: empty pc_history.")
        plt.close()
        return
    iters = np.arange(1, len(pc_values) + 1)

    ax.semilogy(
        iters, pc_values,
        "o-", color="#2563eb", markersize=5, linewidth=1.8,
        label=f"Algorithm 2 ({pc_label})",
    )

    ax.set_xlabel("Outer iteration  $t$")
    ax.set_ylabel(rf"$\mathrm{{PC}}^*_t \;=\; \max_{{\lambda \in \Delta_K}} \,"
                  rf"\mathrm{{{pc_label}}}(\lambda;\,B_t)$")
    params_str = _format_params(problem_params)
    ax.set_title(
        f"Progress-criterion decay across outer iterations\n"
        f"{params_str}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")


# =====================================================================
#  Experiment 1:  Regularised multi-class logreg  (PC = GAP)
# =====================================================================
def experiment_logreg_gap(
    verbose: bool = True,
    coarse_resolution: int = 10,
    fine_resolution: int = 20,
    n_passes: int = 3,
    steps_per_point_per_pass: int = 10,
    max_outer: int = 300,
    max_inner: int = 5,
    eval_every_n_grads: int = 1,
    plot_path_cpu: str = "logreg_cpu_vs_accuracy.png",
    plot_path_grads: str = "logreg_grads_vs_accuracy.png",
    plot_path_pc: str = "logreg_pc_history.png",
) -> Dict:
    """Compare Algorithm 2 vs the uniform-discretisation baseline.

    Two complementary plots are produced:
      * CPU time   vs worst-case suboptimality   ("real-time" comparison)
      * Gradient evals vs worst-case suboptimality ("oracle-cost" comparison)

    The CPU plot reflects practical wall-time cost (including Algorithm
    2's overhead from maximising PC, T_map calls, and bundle bookkeeping).
    The gradient-eval plot reflects pure oracle complexity: how much
    gradient information each algorithm needs to achieve a given
    accuracy, regardless of overhead.  This is the metric that the
    paper's Theorems 2-3 bound asymptotically.

    Protocol
    --------
    1. Precompute reference  F*_λ  on a very fine grid G_fine of Δ_K.
    2. Run baseline progressively, checkpointing at every M gradient
       evaluations (and at every pass boundary).
    3. Run Algorithm 2 progressively, same checkpointing rule.
    4. Plot two figures.
    """
    print("=" * 65)
    print("Exp 1: Regularised multi-class logreg — CPU time and grad-evals "
          "vs worst-case err")
    print("=" * 65)

    K, p, n, reg = 3, 3, 30, 0.1
    d = K * p
    objs, grads, L, mu = make_logreg_strongly_convex(
        K=K, p=p, n=n, reg=reg, seed=43,
    )
    W0 = np.zeros(d)

    print(f"  K={K}, p={p}, n={n}, reg={reg}, d={d}")
    print(f"  L = {np.round(L, 4)}, µ = {mu}")
    print(f"  Checkpoint cadence: M = {eval_every_n_grads} gradient evals")

    # --- 1. Precompute reference map on the fine grid ---
    if verbose:
        print(f"\n  Precomputing reference map on fine grid "
              f"(resolution = {fine_resolution}) ...")
    ref_t0 = time.time()
    reference_map = compute_reference_map(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, fine_resolution=fine_resolution,
        n_iters=20_000, grad_tol=1e-12, verbose=False,
    )
    ref_time = time.time() - ref_t0
    print(f"  Reference map ready: {len(reference_map['fine_grid'])} points, "
          f"{ref_time:.1f}s")

    # --- 2. Run the progressive baseline ---
    if verbose:
        print(f"\n  Running baseline (coarse resolution = {coarse_resolution}, "
              f"{n_passes} passes, {steps_per_point_per_pass} GD steps/point/pass) ...")
    bl = uniform_discretisation_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, resolution=coarse_resolution,
        reference_map=reference_map,
        n_passes=n_passes,
        steps_per_point_per_pass=steps_per_point_per_pass,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose,
    )

    # --- 3. Run Algorithm 2 with per-outer checkpoints ---
    if verbose:
        print(f"\n  Running Algorithm 2 ({max_outer} outer iters, "
              f"up to {max_inner} inner steps each) ...")
    a2 = algorithm2_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, reference_map=reference_map,
        mu=mu, mode="gap",
        max_outer=max_outer, max_inner=max_inner,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose,
    )

    # --- 4. Plots ---
    _plot_cpu_vs_accuracy(
        bl=bl, a2=a2, plot_path=plot_path_cpu,
        problem_params={"K": K, "p": p, "n": n, "d": d, "reg": reg},
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )
    _plot_grads_vs_accuracy(
        bl=bl, a2=a2, plot_path=plot_path_grads,
        problem_params={"K": K, "p": p, "n": n, "d": d, "reg": reg},
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )

    _plot_pc_history(
        a2=a2, plot_path=plot_path_pc,
        problem_params={"K": K, "p": p, "n": n, "d": d, "reg": reg},
        pc_label="GAP",
    )

    return {
        "reference_map": reference_map,
        "baseline": bl,
        "algorithm2": a2,
        "problem_params": {"K": K, "p": p, "n": n, "d": d, "reg": reg},
    }



# =====================================================================
#  Experiment 2:  Separable Gaussian mixture  +  multi-class logreg
#                 ("inverse logistic regression" — interpolation +
#                 sublevel-set PL regime, PC = UB)
# =====================================================================
def experiment_logreg_separable_gaussian(
    verbose: bool = True,
    coarse_resolution: int = 6,
    fine_resolution: int = 8,
    n_passes: int = 20,
    steps_per_point_per_pass: int = 1,
    max_outer: int = 25,
    max_inner: int = 200,
    eval_every_n_grads: int = 100,
    plot_path_cpu: str = "exp2_cpu_vs_accuracy.png",
    plot_path_grads: str = "exp2_grads_vs_accuracy.png",
) -> Dict:
    r"""Separable Gaussian mixture fit with unregularised multi-class logreg.

    Setting (the "inverse logistic regression" construction)
    --------------------------------------------------------
    Data:    K isotropic Gaussian clusters with shared covariance σ²·I and
             centres ‖μ_k − μ_l‖ = sep·√2.  By Bayes' rule, the posterior
             P(Y=k | X=x) is exactly softmax-linear in x — so multinomial
             logistic regression is well-specified, with planted weights
             w_k* = μ_k/σ² and biases  b_k* = −‖μ_k‖²/(2σ²).
    Loss:    F_i(W) = (1/n_i) Σ_{j: y_j=i} {−⟨w^i, x_j⟩ + log Σ_l exp(⟨w^l,x_j⟩)}
             — no ℓ₂ regulariser.

    Why interpolation holds (in the inf sense, Asn 5.1)
    ---------------------------------------------------
    F_i ≥ 0 trivially.  For separable clusters  inf F_i = 0  (drive the
    weights along a separating direction; the limit is reached only as
    ‖W‖ → ∞, never at a finite W — Soudry et al. 2018).  Both per-class
    objectives and every  F_λ  share the same property, so F_λ* = 0.

    Why strict global PL (Asn 5.2) FAILS
    ------------------------------------
    For one sample, set p := P(correct | x; w).  Then  F = −log p ∼ (1−p)
    while  ‖∇F‖² ∼ (1−p)²;  hence  ‖∇F‖²/F ∼ (1−p) → 0.  No global
    constant µ > 0 satisfies the PL inequality.

    Sublevel-set PL — what the algorithm actually uses
    --------------------------------------------------
    On the sublevel set  S_α := {W : F_λ(W) ≤ α},  separability gives a
    constant µ_λ(α) > 0 (generalized self-concordance, Bach 2014).
    Algorithm 2 starts at W_0 = 0 with F_i(W_0) = log K, so its iterates
    stay inside  S_{log K},  where ``make_logreg_separable_gaussian``
    reports a numerical estimate µ_i.  We pass that to algorithm2 in
    mode="ub" and check empirically how the upper bound evolves.

    Practical caveat
    ----------------
    Because separable softmax CE has  µ/L → 0  along the algorithm's
    trajectory (the very phenomenon that causes the global-PL failure),
    the inner gradient-descent iterates in mode="ub" make sub-percent
    UB reductions per step.  The default pruning rule in
    ``_bundle_update_adaptive`` rejects such steps, so the algorithm
    can stall after a few outer iterations.  This is reported faithfully
    in the convergence plot — it is *not* a bug, but the practical
    cost of the global-PL failure flagged in the docstring of
    ``make_logreg_separable_gaussian``.  A follow-up experiment with
    a small ℓ₂ regulariser and PC = GAP recovers fast convergence at
    the price of strict interpolation.
    """
    print("=" * 65)
    print("Exp 2: Separable Gaussian-mixture + softmax CE  (PC = UB)")
    print("=" * 65)

    K, p, n_per_class, sep, sigma = 3, 20, 30, 6.0, 1.0
    n_total = K * n_per_class
    objs, grads, L, mu = make_logreg_separable_gaussian(
        K=K, p=p, n_per_class=n_per_class, sep=sep, sigma=sigma, seed=17,
    )
    d = K * p
    W0 = np.zeros(d)

    print(f"  K={K}, p={p}, n={n_total}, sep={sep}, σ={sigma}, d={d}")
    print(f"  L = {np.round(L, 3)}")
    print(f"  µ (sublevel-set PL on {{F_i ≤ log K}}) = {np.round(mu, 4)}")
    print(f"  µ_min / L_max ≈ {mu.min()/L.max():.4f}    "
          f"(small ⇒ slow inner convergence, by global-PL failure)")
    print(f"  Checkpoint cadence: M = {eval_every_n_grads} gradient evals")

    # --- 1. Precompute reference map.  Note: separable softmax CE has
    #        no finite minimiser, so GD on F_λ converges only at rate
    #        O(1/log t).  We use a modest budget — the F*_λ estimates
    #        are correct to roughly the budget's tolerance, which is
    #        all we need for worst-case-suboptimality comparison.
    if verbose:
        print(f"\n  Precomputing reference map "
              f"(fine grid resolution = {fine_resolution}) ...")
    ref_t0 = time.time()
    reference_map = compute_reference_map(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, fine_resolution=fine_resolution,
        n_iters=5_000, grad_tol=1e-6, verbose=False,
    )
    ref_time = time.time() - ref_t0
    print(f"  Reference map ready: {len(reference_map['fine_grid'])} points, "
          f"{ref_time:.1f}s  (F*_λ range: "
          f"[{reference_map['F_star'].min():.4f}, "
          f"{reference_map['F_star'].max():.4f}])")

    # --- 2. Run progressive baseline ---
    if verbose:
        print(f"\n  Running baseline (coarse resolution = {coarse_resolution}, "
              f"{n_passes} passes) ...")
    bl = uniform_discretisation_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, resolution=coarse_resolution,
        reference_map=reference_map,
        n_passes=n_passes,
        steps_per_point_per_pass=steps_per_point_per_pass,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose,
    )

    # --- 3. Run Algorithm 2 in mode="ub" (interpolation + PL) ---
    if verbose:
        print(f"\n  Running Algorithm 2 mode=\"ub\" "
              f"({max_outer} outer iters, up to {max_inner} inner steps) ...")
    a2 = algorithm2_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=W0, reference_map=reference_map,
        mu=mu, mode="ub",
        max_outer=max_outer, max_inner=max_inner,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose,
    )

    # --- 4. Plots ---
    problem_params = {"K": K, "p": p, "n": n_total, "d": d,
                      "sep": sep, "sigma": sigma}
    _plot_cpu_vs_accuracy(
        bl=bl, a2=a2, plot_path=plot_path_cpu,
        problem_params=problem_params,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )
    _plot_grads_vs_accuracy(
        bl=bl, a2=a2, plot_path=plot_path_grads,
        problem_params=problem_params,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
    )

    return {
        "reference_map": reference_map,
        "baseline": bl,
        "algorithm2": a2,
        "problem_params": problem_params,
    }


# =====================================================================
#  Experiment 3:  Single-hidden-layer MLP  (PC = GN, generic non-convex)
# =====================================================================
def experiment_mlp_gn(
    verbose: bool = True,
    coarse_resolution: int = 10,
    fine_resolution: int = 20,
    n_passes: int = 25,
    steps_per_point_per_pass: int = 3,
    max_outer: int = 100,
    max_inner: int = 20,
    eval_every_n_grads: int = 1,
    plot_path_cpu: str = "MLP_cpu_vs_accuracy.png",
    plot_path_grads: str = "MLP_grads_vs_accuracy.png",
    plot_path_pc: str = "MLP_pc_history.png",
) -> Dict:
    """Compare Algorithm 2 (mode="gn") vs the uniform-discretisation baseline
    on a single-hidden-layer MLP with softmax cross-entropy (Exp 3).

    Architecture
    ------------
    x_j ∈ R^p  →  ReLU(W_1 x_j + b_1) ∈ R^h  →  W_2 a + b_2 ∈ R^K
    Parameters  θ = (W_1, b_1, W_2, b_2),  d = h·p + h + K·h + K.
    Per-class loss:
        F_i(θ) = (1/n_i) Σ_{j: y_j=i} { −z_j^{(i)} + log Σ_l exp(z_j^{(l)}) }.
    Non-convex due to the bilinear product W_2 · ReLU(W_1 x + b_1).

    Two complementary plots
    -----------------------
      * CPU time vs worst-case suboptimality (real wall-time comparison).
      * Gradient evals vs worst-case suboptimality (oracle-cost comparison).

    The protocol mirrors ``experiment_logreg_gap`` exactly so the two
    experiments are visually comparable on the same axes.

    Important caveat: the "reference" used for suboptimality
    -------------------------------------------------------
    Because the MLP problem is non-convex, ``compute_reference_map``
    cannot recover the true global optima  F*_λ.  What it actually
    computes is the value of  F_λ  at the gradient-descent stationary
    point reached from x_0 = 0 — an upper bound on F*_λ that may be
    loose if GD lands on a sub-optimal saddle or local min.  The
    "worst-case suboptimality" curves should therefore be read as
    "excess loss over the GD-from-zero reference," not as a bound on
    excess over the global optimum.  This is a fundamental limitation
    of the experimental protocol on non-convex objectives, not a flaw
    of either method.

    Note on the GN progress criterion and µ
    ---------------------------------------
    The gradient-norm criterion does not require a strong-convexity
    constant.  We pass ``mu=None`` to ``algorithm2_progressive``;
    if your current ``baseline.algorithm2_progressive`` raises
    ``ValueError`` for ``mode='gn'`` with ``mu=None``, relax that
    pre-condition (mode "gn" doesn't actually use ``mu``).

    Parameters
    ----------
    verbose                  : print progress for both methods.
    coarse_resolution        : grid resolution for the baseline.
    fine_resolution          : grid resolution for the reference map.
    n_passes                 : baseline outer passes.
    steps_per_point_per_pass : baseline GD steps per grid point per pass.
    max_outer, max_inner     : Algorithm 2 budget.
    eval_every_n_grads       : checkpoint cadence (gradient-oracle calls).
    plot_path_cpu            : output path for CPU-vs-accuracy plot.
    plot_path_grads          : output path for grads-vs-accuracy plot.
    """
    print("=" * 65)
    print("Exp 3: Single-hidden-layer MLP  (PC = GN) — CPU time and "
          "grad-evals vs worst-case err")
    print("=" * 65)

    K, p, n, h = 3, 10, 30, 8
    d = h * p + h + K * h + K               # d = 67
    objs, grads, L = make_mlp_nonconvex(K=K, p=p, n=n, h=h, seed=7)
    theta0 = np.zeros(d)

    print(f"  K={K}, p={p}, n={n}, h={h}, d={d}")
    print(f"  Estimated L = {np.round(L, 4)}     (no µ — non-convex)")
    print(f"  Checkpoint cadence: M = {eval_every_n_grads} gradient evals")

    # --- 1. Precompute the (approximate) reference map on the fine grid ---
    #
    # For non-convex F_λ this is not the global minimum but the value at
    # the GD-from-zero stationary point.  See the function docstring for
    # what the resulting "worst-case suboptimality" actually quantifies.
    if verbose:
        print(f"\n  Precomputing reference map on fine grid "
              f"(resolution = {fine_resolution}) ...")
        print(f"  [non-convex: this is the GD stationary value, "
              f"not the global optimum]")
    ref_t0 = time.time()
    reference_map = compute_reference_map(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=theta0, fine_resolution=fine_resolution,
        n_iters=20_000, grad_tol=1e-5, verbose=False,
    )
    ref_time = time.time() - ref_t0
    print(f"  Reference map ready: {len(reference_map['fine_grid'])} points, "
          f"{ref_time:.1f}s  (F_λ stationary range: "
          f"[{reference_map['F_star'].min():.4f}, "
          f"{reference_map['F_star'].max():.4f}])")

    # --- 2. Progressive baseline ---
    if verbose:
        print(f"\n  Running baseline (coarse resolution = {coarse_resolution}, "
              f"{n_passes} passes, "
              f"{steps_per_point_per_pass} GD steps/point/pass) ...")
    bl = uniform_discretisation_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=theta0, resolution=coarse_resolution,
        reference_map=reference_map,
        n_passes=n_passes,
        steps_per_point_per_pass=steps_per_point_per_pass,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose,
    )

    # --- 3. Algorithm 2 with mode="gn" ---
    #
    # mu is not required for the gradient-norm criterion; if your
    # algorithm2_progressive insists on a non-None mu for mode="gn",
    # that check should be relaxed (GN doesn't use mu).
    if verbose:
        print(f"\n  Running Algorithm 2 mode=\"gn\" "
              f"({max_outer} outer iters, up to {max_inner} inner steps) ...")
    a2 = algorithm2_progressive(
        K=K, d=d, objectives=objs, grad_objectives=grads,
        L=L, x0=theta0, reference_map=reference_map,
        mu=None, mode="gn",
        max_outer=max_outer, max_inner=max_inner,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose,
    )

    # --- 4. Plots ---
    problem_params = {"K": K, "p": p, "n": n, "h": h, "d": d}
    _plot_cpu_vs_accuracy(
        bl=bl, a2=a2, plot_path=plot_path_cpu,
        problem_params=problem_params,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        pc_label = 'GN'
    )
    _plot_grads_vs_accuracy(
        bl=bl, a2=a2, plot_path=plot_path_grads,
        problem_params=problem_params,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        pc_label='GN'
    )

    return {
        "reference_map": reference_map,
        "baseline": bl,
        "algorithm2": a2,
        "problem_params": problem_params,
    }


# # =====================================================================
# #  Experiment 4:  Pareto front tracing  (2-class logreg)
# # =====================================================================
# def experiment_pareto_front():
#     """Trace the Pareto front for 2-class regularised logistic regression.
#
#     After Algorithm 2 converges, we evaluate the solution map
#     Ŵ(λ) = T(λ; B_final) for a fine grid of λ ∈ Δ_2 and plot
#     the corresponding (F_1(Ŵ), F_2(Ŵ)) pairs.
#
#     The Pareto front shows the trade-off between per-class losses:
#     improving class-1 accuracy comes at the cost of class-2 accuracy.
#     """
#     print("=" * 65)
#     print("Exp 4: Pareto front  (2-class regularised logreg)")
#     print("=" * 65)
#
#     K, p, n, reg = 2, 5, 40, 0.05
#     d = K * p                                # d = 10
#     objs, grads, L, mu = make_logreg_strongly_convex(
#         K=K, p=p, n=n, reg=reg, seed=99,
#     )
#     W0 = np.zeros(d)
#     eps = 5e-2
#
#     print(f"  K={K}, p={p}, n={n}, reg={reg}, d={d}, ε={eps}")
#
#     res = algorithm2_progressive(
#         K=K, d=d, objectives=objs, grad_objectives=grads,
#         L=L, x0=W0, eps=eps, mode="gap", mu=mu,
#         max_outer=80, max_inner=200, verbose=False,
#     )
#     bundle = res["bundle"]
#
#     # Evaluate the approximate solution map  Ŵ(λ) = T(λ; B_final)
#     fine_grid = simplex_grid(K, 100)
#     f1_vals, f2_vals = [], []
#     for lam in fine_grid:
#         W_hat = T_map(bundle, lam)
#         f1_vals.append(objs[0](W_hat))
#         f2_vals.append(objs[1](W_hat))
#
#     print(f"  Outer iterations : {res['outer_iters']}")
#     print(f"  Oracle calls     : {res['oracle_calls']}")
#     print(f"  Bundle size      : {bundle.m} points\n")
#
#     return f1_vals, f2_vals, res


# =====================================================================
#  Plotting
# =====================================================================
def _format_params(params):
    """Format a params dict as a plain-text string for plot titles.

    Examples:
        {'K': 3, 'p': 4, 'n': 60, 'd': 12, 'reg': 0.1}
          -> 'K=3, p=4, n=60, d=12, reg=0.1'
        {'K': 3, 'p': 5, 'n': 45, 'd': 15, 'separable': True}
          -> 'K=3, p=5, n=45, d=15, separable'
    """
    parts = []
    for k, v in params.items():
        if isinstance(v, bool):
            if v:
                parts.append(str(k))
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)
def make_plots(res1, pareto_data, res2=None, res3=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Plot 1: GAP convergence (regularised logreg) ----
    ax = axes[0, 0]
    ax.semilogy(res1["pc_history"], "o-", color="#2563eb", markersize=4, linewidth=1.5,
                label="Algorithm 2 (GAP)")
    eps1 = res1["config"]["eps"]
    ax.axhline(y=eps1, color="grey", ls="--", lw=1, label=f"ε = {eps1}")
    ax.set_xlabel("Outer iteration t")
    ax.set_ylabel("max_λ GAP(λ; B_t)")
    ax.set_title(
        f"Exp 1: Regularised Logreg (GAP)\n"
        f"{_format_params(res1['config']['params'])}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: UB convergence (standard logreg) ----
    # ax = axes[0, 1]
    # ax.semilogy(res2["pc_history"], "s-", color="#dc2626", markersize=4, linewidth=1.5,
    #             label="Algorithm 2 (UB)")
    # eps2 = res2["config"]["eps"]
    # ax.axhline(y=eps2, color="grey", ls="--", lw=1, label=f"ε = {eps2}")
    # ax.set_xlabel("Outer iteration t")
    # ax.set_ylabel("max_λ UB(λ; B_t)")
    # ax.set_title(
    #     f"Exp 2: Standard Logreg, Separable (UB)\n"
    #     f"{_format_params(res2['config']['params'])}"
    # )
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # ---- Plot 3: GN convergence (MLP) ----
    # ax = axes[1, 0]
    # ax.semilogy(res3["pc_history"], "^-", color="#16a34a", markersize=4, linewidth=1.5,
    #             label="Algorithm 2 (GN)")
    # eps3 = res3["config"]["eps"]
    # ax.axhline(y=eps3, color="grey", ls="--", lw=1, label=f"ε = {eps3}")
    # ax.set_xlabel("Outer iteration t")
    # ax.set_ylabel("max_λ GN(λ; B_t)")
    # ax.set_title(
    #     f"Exp 3: Single-Hidden-Layer MLP (GN)\n"
    #     f"{_format_params(res3['config']['params'])}"
    # )
    # ax.legend()
    # ax.grid(True, alpha=0.3)

    # ---- Plot 4: Pareto front (2-class logreg) ----
    # f1, f2, _ = pareto_data
    # ax = axes[1, 1]
    # ax.scatter(f1, f2, s=10, c="#7c3aed", alpha=0.7)
    # ax.set_xlabel("F₁(Ŵ(λ))  [class 1 loss]")
    # ax.set_ylabel("F₂(Ŵ(λ))  [class 2 loss]")
    # ax.set_title("Exp 4: Pareto Front (2-class Logreg)\n"
    #              "K=2, p=5, n=40, d=10, reg=0.05")
    # ax.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig("experiment_results.png", dpi=150)
    # plt.close()
    # print("Plots saved to experiment_results.png")








# =====================================================================
if __name__ == "__main__":
    res1 = experiment_logreg_gap()
    print("✓ Experiment 1 completed.")
    #res2 = experiment_logreg_separable_gaussian()
    #print("✓ Experiment 2 (separable Gaussian mixture, UB) completed.")
    #res3 = experiment_mlp_gn()
    #print("✓ Experiment 3 completed.")