"""
experiments.py  –  Numerical experiments for the bundle MOO algorithm
=====================================================================

All experiments use multi-class classification objectives following
the notation from the paper:

    K classes, weight vectors  w^1, …, w^K ∈ R^p,
    labelled data  {(y_j, x_j)}_{j=1}^n  with  y_j ∈ [K],
    per-class loss  F_i(W) = (1/n_i) Σ_{j: y_j=i} {−log P(Y=i|x_j; W)}.

Two experiments:

  Exp 1 – Regularised multi-class logreg                                     (PC = GAP₁, strongly convex)
  Exp 2 – Single-hidden-layer MLP                                            (PC = GN,   generic non-convex)

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

    # Horizontal reference line at the baseline's final worst-case error.
    # This visualises the accuracy floor the baseline achieves at its
    # full budget — a fixed target that both algorithms can be measured
    # against.  Drawn first so the algorithm curves render on top.
    err_tol = 3e-3
    ax.axhline(
        y=err_tol,
        color="#059669", linestyle="--", linewidth=1.5,
        label=f"Error Tolerance = {err_tol}",
    )

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

    # Horizontal reference line at the baseline's final worst-case error
    # — the accuracy floor the baseline achieves at its full budget.
    err_tol = 3e-3
    ax.axhline(
        y=err_tol,
        color="#059669", linestyle="--", linewidth=1.5,
        label=f"Error Tolerance = {err_tol}",
    )

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
    coarse_resolution: int = 18,
    fine_resolution: int = 20,
    n_passes: int = 10,
    steps_per_point_per_pass: int = 20,
    max_outer: int = 1200,
    max_inner: int = 100,
    eval_every_n_grads: int = 500,
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

    K, p, n, reg = 4, 10, 40, 4.1
    d = K * p
    objs, grads, L, mu, joint_oracle = make_logreg_strongly_convex(K=K, p=p, n=n, reg=reg, seed=42,)
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
        n_iters=20_000, grad_tol=1e-12, mu=mu, verbose=False,
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
        eval_every_n_grads=None,
        mu=mu,
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
        checkpoint_every=20,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose, epsilon=1e-5,
        # Early-stop A2 once it reaches the baseline's final worst-case
        # error — A2 should never do more work than the baseline to
        # achieve a comparable accuracy level.  See algorithm.py for
        # the semantics of ``target_err``.
        target_err=bl["worst_errs"][-1],
        # Fused F+grad oracle: one forward pass per class instead of two
        # (one for F_i, one for ∇F_i) when bundle.add_point evaluates a
        # new point.  Tier 1 CPU optimisation; numerically identical to
        # the per-class path.
        joint_oracle=joint_oracle,
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
#  Experiment 2:  Single-hidden-layer MLP  (PC = GN, generic non-convex)
# =====================================================================
def experiment_mlp_gn(
    verbose: bool = True,
    K: int = 6,
    p: int = 10,
    n: int = 50,
    h: int = 16,
    coarse_resolution: int = 6,
    fine_resolution: int = 7,
    n_passes: int = 20,
    steps_per_point_per_pass: int = 100,
    max_outer: int = 1200,
    max_inner: int = 400,
    eval_every_n_grads: int = 500,
    epsilon: float = 1e-5,
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
    print("Exp 2: Single-hidden-layer MLP  (PC = GN) — CPU time and "
          "grad-evals vs worst-case err")
    print("=" * 65)

    K, p, n, h = int(K), int(p), int(n), int(h)
    d = h * p + h + K * h + K
    objs, grads, L, joint_oracle = make_mlp_nonconvex(K=K, p=p, n=n, h=h, seed=10)
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
        n_iters=500, grad_tol=1e-5, verbose=False,
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
        eval_every_n_grads=500,
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
        checkpoint_every=20,
        eval_every_n_grads=eval_every_n_grads,
        verbose=verbose, epsilon=epsilon,
        # Early-stop A2 once it reaches the baseline's final worst-case
        # error — A2 should never do more work than the baseline to
        # achieve a comparable accuracy level.  See algorithm.py for
        # the semantics of ``target_err``.
        target_err=bl["worst_errs"][-1],
        # Fused F+grad oracle:  one forward pass per class instead of
        # two (F_i + ∇F_i) on every bundle.add_point.  Tier 1 CPU
        # optimisation; numerically identical to per-class path.
        joint_oracle=joint_oracle,
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
    _plot_pc_history(
        a2=a2, plot_path=plot_path_pc,
        problem_params=problem_params,
        pc_label="GN",
    )

    return {
        "reference_map": reference_map,
        "baseline": bl,
        "algorithm2": a2,
        "problem_params": problem_params,
    }



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


# =====================================================================
if __name__ == "__main__":
    res1 = experiment_logreg_gap()
    print("✓ Experiment 1 completed.")
    #res2 = experiment_mlp_gn()
    #print("✓ Experiment 2 completed.")