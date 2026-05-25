# A First-Order Bundle Method for Smooth Multi-Objective Optimization

A Python implementation and empirical study of the bundle method for smooth multi-objective optimization (MOO), reproducing **Algorithm 2** ("Simple Adaptive Algorithm v2") from Grigas & Cheng's paper *A First-Order Bundle Method for Smooth Multi-Objective Optimization* and comparing it against the natural uniform-discretisation baseline on two problem families: regularised multi-class logistic regression (strongly convex) and a single-hidden-layer MLP (non-convex).

The goal is the *smooth solution map* problem: given $K$ smooth objectives $F_1, \dots, F_K : \mathbb{R}^d \to \mathbb{R}$, produce a map $\hat{x}(\lambda)$ over the simplex $\Delta_K$ such that, uniformly in $\lambda$,

$$
F_\lambda(\hat{x}(\lambda)) - F^*_\lambda \;\le\; \varepsilon,
\qquad F_\lambda(x) := \sum_{k=1}^K \lambda_k F_k(x).
$$

The paper's contribution is an adaptive bundle method that builds one shared first-order model of $F_\lambda$ across the simplex, instead of independently optimising at many grid points.

---

## What's in this repository

| File | Contents |
|---|---|
| `bundle.py` | Bundle data structure $B_m$ and the three progress criteria (`UB`, `GAP`, `GN`) from §5.2 of the paper. Includes both LB variants (`LB_1` via Gurobi QP, `LB_2` closed-form) and the T-map. |
| `algorithm.py` | Algorithm 2 with progressive checkpointing (`algorithm2_progressive`). Includes vectorised bundle helpers, an analytical Jacobian for the λ-maximisation, and a tier-2 fused per-class oracle used by `bundle.add_point`. |
| `baseline.py` | Uniform-discretisation baseline (NAG inner loop for strongly convex, GD inner loop for the non-convex case) with the same checkpoint accounting as Algorithm 2. Also exposes `compute_reference_map`, which builds the ground-truth $F^*_\lambda$ over a fine simplex grid. |
| `objectives.py` | Two objective factories: `make_logreg_strongly_convex` (multi-class logistic regression with $\ell_2$ regularisation, $\mu = \text{reg}$) and `make_mlp_nonconvex` (1-hidden-layer MLP with softmax cross-entropy). Both expose per-class loss and gradient closures plus a fused joint oracle. |
| `experiments.py` | Two end-to-end experiments (`experiment_logreg_gap`, `experiment_mlp_gn`) that build the data, run both algorithms with matching checkpoint accounting, and produce CPU-time-vs-accuracy and grad-evals-vs-accuracy plots. |

The paper PDF that this code follows is included as `A First-Order Bundle Method for Smooth Multi-objective Optimization.pdf`.

---

## Algorithm overview

The bundle $B_m = \{(x_i, F_k(x_i), \nabla F_k(x_i))\}_{i=1}^m$ encodes first-order information collected over $m$ iterates. From it the algorithm computes three quantities the paper calls *progress criteria*:

- **UB** (Eq. 12): upper bound on $\min_x F_\lambda(x)$ from a smoothness inequality.
- **GAP** = UB − LB (Eq. 15): used in the strongly convex case. Two variants of the lower bound — `LB_1` (aggregated QP over all bundle points; needs Gurobi) and `LB_2` (closed-form single-index minorant). The code uses `LB_2` by default — it's about 100× faster than `LB_1` and the GAP it produces still upper-bounds the true suboptimality.
- **GN** (Eq. 17): a scaled minimum gradient norm over the bundle, used in the non-convex / PL setting.

The outer loop of Algorithm 2 is:

1. Find $\lambda_t \in \arg\max_{\lambda \in \Delta_K} \text{PC}(\lambda; B_t)$ via SLSQP with a multi-start strategy (warm start from the previous $\lambda_{t-1}$ plus simplex vertices and centroid for safety).
2. Run an inner loop (`_bundle_update_adaptive`) of T-map steps at $\lambda_t$, appending each iterate to the bundle, until either `PC(λ_t; B_cur) ≤ ε/3` or `max_inner` is reached.
3. If $\max_\lambda \text{PC}(\lambda; B) \le 2\varepsilon/3$, stop.

The progressive variant interleaves the outer loop with periodic checkpoints that compute the worst-case suboptimality $\sup_{\lambda \in G_{\text{fine}}} [F_\lambda(\hat{x}(\lambda)) - F^*_\lambda]$ over a fine reference grid.

---

## Quick start

```bash
# Optional: create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Required dependencies
pip install numpy scipy matplotlib

# Optional: Gurobi for the LB_1 variant (the LB_2 default does not need it)
pip install gurobipy

# Run the logreg experiment (strongly convex; PC = GAP)
python experiments.py
```

By default `experiments.py` runs `experiment_logreg_gap()`. To run the MLP experiment instead, edit the bottom of the file:

```python
if __name__ == "__main__":
    res = experiment_mlp_gn()
```

Each experiment writes three PNGs (CPU-vs-accuracy, grad-evals-vs-accuracy, PC history) and returns a dict of results.

### Calling the algorithms directly

```python
import numpy as np
from objectives import make_logreg_strongly_convex
from baseline import compute_reference_map, uniform_discretisation_progressive
from algorithm import algorithm2_progressive

K, p, n, reg = 5, 10, 20, 1
objs, grads, L, mu, joint = make_logreg_strongly_convex(K=K, p=p, n=n, reg=reg, seed=42)
d = K * p
W0 = np.zeros(d)

# Build the reference map (one-time cost)
ref = compute_reference_map(
    K=K, d=d, objectives=objs, grad_objectives=grads,
    L=L, x0=W0, fine_resolution=20,
    n_iters=20_000, grad_tol=1e-12, mu=mu, verbose=False,
)

# Run baseline
bl = uniform_discretisation_progressive(
    K=K, d=d, objectives=objs, grad_objectives=grads,
    L=L, x0=W0, resolution=17, reference_map=ref,
    n_passes=10, steps_per_point_per_pass=20,
    eval_every_n_grads=500, mu=mu, verbose=False,
)

# Run Algorithm 2 (early-stops once it reaches the baseline's final accuracy)
a2 = algorithm2_progressive(
    K=K, d=d, objectives=objs, grad_objectives=grads,
    L=L, x0=W0, reference_map=ref,
    mu=mu, mode="gap",
    max_outer=1200, max_inner=100,
    epsilon=1e-6,
    eval_every_n_grads=500,
    target_err=bl["worst_errs"][-1],
    joint_oracle=joint,
)

print(f"BL final: cpu={bl['cpu_times'][-1]:.2f}s err={bl['worst_errs'][-1]:.4e} grad_evals={bl['grad_evals_history'][-1]}")
print(f"A2 final: cpu={a2['cpu_times'][-1]:.2f}s err={a2['worst_errs'][-1]:.4e} grad_evals={a2['grad_evals_history'][-1]} outer={len(a2['pc_history'])}")
```

---

## Experiments

### Experiment 1 — multi-class logistic regression (strongly convex, PC = GAP)

Typical configuration:

```
K = 4 or 5, p = 10, n = 20 or 40, reg = 1
PC = GAP (uses LB_2)
fine grid: simplex resolution 20  (binom(K-1+20, K-1) points)
baseline: 10 passes of 20 NAG steps per coarse-grid point
A2: max_outer=1200, max_inner=100, epsilon=1e-6
```

The strongly convex setting is where Algorithm 2 most clearly wins on gradient-evaluation efficiency: A2 reaches the baseline's final accuracy with roughly an order of magnitude fewer gradient evaluations. On CPU time the picture is more mixed — at small problem sizes each gradient is so cheap that A2's per-outer SLSQP and bundle overhead can dominate.

### Experiment 2 — single-hidden-layer MLP (non-convex, PC = GN)

Typical configuration:

```
K = 4–6, p = 10, n = 50, h = 16 or 32
PC = GN (paper Eq. 17, with the no-µ fallback documented below)
baseline: GD (not NAG) per coarse-grid point
```

In the non-convex MLP setting the comparison is harder, and two phenomena are worth knowing about:

- A "rise-after-plateau" pattern can appear in the baseline trajectory: the worst-case error initially drops, then can creep back up across passes as coarse-grid solutions become more strongly committed to their local basins while the basin structure differs across nearby λ. This is the basin-mismatch effect on non-convex problems.
- A2 is gradient-efficient but each outer iteration carries SLSQP and bundle overhead, so it competes well on grad-eval count but loses on CPU at small `d` until the per-gradient cost is large enough to amortise the overhead.

Each experiment writes two main plots (CPU-vs-err and grads-vs-err) plus a PC-history plot. Example output filenames already in the repo follow the pattern `MLP_cpu_K5p10n50h32fi6co5err5e2.png` (K=5, p=10, n=50, h=32, fine_r=6, coarse_r=5, err target = 5e-2).

---

## Implementation notes (deliberate choices and deviations from the paper)

**`LB_2` is the default lower bound.** Paper §5.2.1 defines `LB_1` as the aggregated QP solution. The code provides both, but defaults to `LB_2` (single-index minorant) inside `_maximise_GAP` because `LB_2` is closed-form, ~100× faster, and avoids a Gurobi dependency. Mathematically `LB_2 ≤ LB_1`, so the resulting GAP is a slightly looser bound on suboptimality but still valid.

**Inner-loop pruning (`prune_inner=True`).** Appendix B.1 of the paper assumes the inner BundleUpdate appends every T-map iterate. The code's default keeps only the iterate with the smallest gradient norm at each inner round (paper §7 heuristic). To recover the proof's exact semantics use `prune_inner=False`.

**GN's no-µ fallback.** Paper Eq. 17 defines GN as $\tfrac{1}{2}(1/\mu_\lambda - 1/L_\lambda) \min_i \|\nabla F_\lambda(x_i)\|^2$ and requires the strongly convex / PL setting. In `mode="gn"` on the MLP, where no $\mu$ is available, the code returns the un-scaled $\min_i \|\nabla F_\lambda(x_i)\|^2$. This is a sensible heuristic for the experiment but **does not inherit the paper's convergence-rate guarantees** — the rate constants depend on $1/\mu_\lambda - 1/L_\lambda$ and the proof relies on PL.

**Multi-start in `_maximise_GAP` / `_maximise_GN`.** The paper specifies the abstract problem $\max_\lambda \text{PC}(\lambda; B)$ but not the solver. The implementation uses SLSQP with a warm start (previous outer's argmax) plus simplex vertices and centroid as safety probes. Empirically the warm start usually wins, but on non-concave PC (the MLP / GN setting) the safety probes occasionally find better basins and are not skippable.

**Fused per-class oracle.** `bundle.add_point` accepts an optional `joint_oracle` argument that computes all K losses and gradients in a single shared forward pass. The objective factories return this as `joint_oracle.fused` (fused across classes on the full training set) and the underlying per-class fused version via `joint_oracle(theta)`. The fused version is used by default via the `use_fused_oracle=True` parameter in `algorithm2_progressive`. Verified byte-identical trajectory against the per-class path on both MLP and logreg.

**Checkpoint accounting.** The reported `cpu_times` in both `bl` and `a2` exclude the cost of `_worst_case_subopt_fast` / `worst_case_suboptimality_baseline` — they accumulate this into a `checkpoint_overhead` accumulator and subtract it from `time.time() - t_start`. Verified empirically: reported CPU is invariant under 50× changes in checkpoint cadence.

**`target_err` early-stop.** `algorithm2_progressive` accepts a `target_err` argument; if provided, the outer loop terminates the first time the checkpointed worst-case error drops below this threshold. The experiments pass `target_err = bl["worst_errs"][-1]` so that A2 stops as soon as it matches the baseline's final accuracy — without this, A2 will continue tightening PC long after err has plateaued.

---

## Reading the plots

Each experiment produces two main plots:

- **CPU time vs worst-case suboptimality.** Lower curves at a given x-value mean "this method reaches that error faster in wall time". The error axis is log-scale; the dashed horizontal line is the configured error tolerance.
- **Total gradient evaluations vs worst-case suboptimality.** Same y-axis but with cumulative gradient-oracle calls on the x-axis. This corresponds to the oracle-complexity quantity that the paper's bounds are stated in.

A point at the bottom-left in either plot is a method that has reached good accuracy with little resource. The two axes can disagree because gradient evaluations have different per-call costs in the two algorithms (the baseline's GD/NAG steps are one matrix-vector multiply each, while A2's per-outer cost also includes the SLSQP multi-start over the simplex).

---

## Dependencies

| Package | Required? | Used for |
|---|---|---|
| `numpy` | yes | All numerical kernels |
| `scipy` | yes | SLSQP for $\max_\lambda \text{PC}$ |
| `matplotlib` | yes | Output plots |
| `gurobipy` | optional | The `LB_1` lower bound only; the default `LB_2` path doesn't use it |

Tested with Python 3.12. Should work on any reasonably recent Python 3.

---

## Honest caveats

- This is research / study code, not a polished library. Function signatures and defaults are tuned for the two specific experiments above and may need attention if you apply this to other problems.
- The non-convex MLP experiment is interesting but the algorithm's theoretical guarantees do not directly apply there (see "GN's no-µ fallback" above). The plots are useful as empirical observations about behaviour, not as a verified rate test.
- Performance characteristics depend on the relative cost of one gradient evaluation versus one SLSQP-over-simplex call. On small problems (small $d$, small $n$) the SLSQP cost can dominate and Algorithm 2 will lose to the baseline on CPU time even when it wins on gradient-eval count. The repo's example configurations include both regimes.

---

## Citation

If you use this code, please cite the original paper that the algorithm is from (Grigas & Cheng, *A First-Order Bundle Method for Smooth Multi-Objective Optimization*; the PDF is included in the repository). This codebase is an independent reproduction; please do not cite it as an original contribution.

---

## License

Add the license you intend to use (MIT, Apache-2.0, BSD-3, etc.). I left this section as a placeholder — please pick the one you want.
