"""
The uniform-discretisation baseline performance plots
    non-convex   (MLP)    :  GN*(B)  = sup_{lambda in Delta_K} min_{x_i\in B_m} ||grad F_lambda(x_i)||^2
"""

import time
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from objectives import make_mlp_nonconvex
from baseline import uniform_discretisation

# plot conventions (consistent with the rest of the project)
_BL_KW = dict(color="#d62728", marker="s", ms=5, lw=1.6, label="uniform discretisation")


def _plot_coverage(bl: Dict, mode: str, title: str, out_path: str) -> str:
    """Two-panel plot: coverage metric vs CPU time, and vs gradient evals."""
    ylabel = (r"$\sup_{\lambda\in\Delta_K} [\min_{x_i\in\mathcal{B}_m} \|\nabla F_\lambda(x_i)\|^2]$"
              if mode == "gn" else
              r"$\sup_{\lambda\in\Delta_K}$ GAP$(\lambda; \mathcal{B}_m)$")
    fig, (ax_t, ax_g) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Per-call baseline style: append the coarse-grid resolution r to the
    # legend label so the discretisation density is visible on the plot.
    bl_kw = {**_BL_KW, "label": f"uniform discretisation (r={bl['resolution']})"}

    # Final worst-case error reached by the uniform baseline, drawn as a
    # green horizontal reference line on both panels.
    final_err = bl["cov_history"][-1]
    _FINAL_KW = dict(color="#2ca02c", ls="--", lw=1.4,
                     label=f"baseline final error = {final_err:.3e}")

    ax_t.plot(bl["cpu_times"], bl["cov_history"], **bl_kw)
    ax_t.axhline(final_err, **_FINAL_KW)
    ax_t.set_xlabel("CPU time (s)")
    ax_t.set_ylabel(ylabel)
    ax_t.set_yscale("log")
    ax_t.set_title("worst-case squared gradient norm vs CPU time" if mode == "gn" else "worst-case function suboptimality vs CPU time")
    ax_t.grid(True, which="both", alpha=0.25)
    ax_t.legend(frameon=False, fontsize=9)

    ax_g.plot(bl["grad_evals_history"], bl["cov_history"], **bl_kw)
    ax_g.axhline(final_err, **_FINAL_KW)
    ax_g.set_xlabel("total gradient evaluations")
    ax_g.set_ylabel(ylabel)
    ax_g.set_yscale("log")
    ax_g.set_title("worst-case squared gradient norm vs gradient evals" if mode == "gn" else "worst-case function suboptimality vs gradient evals")
    ax_g.grid(True, which="both", alpha=0.25)
    ax_g.legend(frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def experiment_mlp_gn_coverage(
    verbose: bool = True,
    K: int = 3, p: int = 10, n: int = 20, h: int = 8, seed: int = 10,
    coarse_resolution: int = 26,
    n_passes: int = 15, steps_per_point_per_pass: int = 50,
    eval_every_n_grads: int = 600, checkpoint_every: int = 3,
    out_path: str = "mlp.png",
) -> Dict:
    """Non-convex MLP: GN* coverage, adaptive vs uniform."""
    print("=" * 68)
    print("Coverage experiment — MLP (non-convex), metric = GN*")
    print("=" * 68)
    d = h * p + h + K * h + K
    objs, grads, L, joint = make_mlp_nonconvex(K=K, p=p, n=n, h=h, seed=seed)
    x0 = np.zeros(d)
    print(f"  K={K}, p={p}, n={n}, h={h}, d={d}  |  L={np.round(L,3)} ")

    if verbose:
        print("\n  [uniform discretisation] ...")
    bl = uniform_discretisation(
        K=K, objectives=objs, grad_objectives=grads, L=L, x0=x0,
        resolution=coarse_resolution, n_passes=n_passes,
        steps_per_point_per_pass=steps_per_point_per_pass,
        eval_every_n_grads=eval_every_n_grads,
        coverage_mode="gn", joint_oracle=joint, verbose=verbose)

    path = _plot_coverage(
        bl, mode="gn",
        title="MLP with parameters: K={}, p={}, n={}, h={}, d={}".format(K,p,n,h,d),
        out_path=out_path)
    print(f"\n  BL  final GN* = {bl['cov_history'][-1]:.4e}  "
          f"(ge={bl['grad_evals_history'][-1]}, cpu={bl['cpu_times'][-1]:.2f}s)")
    return {"baseline": bl, "plot": path}


if __name__ == "__main__":
    experiment_mlp_gn_coverage()