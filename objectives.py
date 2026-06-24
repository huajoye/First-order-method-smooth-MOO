"""Non-convex MLP multi-objective testbed.

Interface (matches ``make_mlp_nonconvex``):
    objectives      : list of K callables  F_i(theta) -> float
    grad_objectives : list of K callables  grad F_i(theta) -> ndarray (d,)
    L               : estimated smoothness constants, shape (K,)
    joint_oracle    : theta -> (fv (K,), gv (K, d)) in a single pass;
                      joint_oracle.fused is the same (kept for API parity).

theta is a NumPy array throughout (the optimisation code is NumPy)
"""
from typing import Callable, List, Tuple

import numpy as np
import torch

torch.set_default_dtype(torch.float64)        # match NumPy float64 precision
torch.set_num_threads(max(1, torch.get_num_threads()))

# ====================================================================
# Softmax utilities
# ====================================================================
def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis.

    Parameters
    ----------
    logits : shape (n, K)  –  row j contains ⟨w^1, x_j⟩, …, ⟨w^K, x_j⟩.

    Returns
    -------
    probs  : shape (n, K),  probs[j, i] = P(Y=i | X=x_j; W).
    """
    shift = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shift)
    return e / e.sum(axis=-1, keepdims=True)


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    """Numerically stable  log Σ_l exp(⟨w^l, x_j⟩)  for each sample j.

    Parameters
    ----------
    logits : shape (n, K)

    Returns
    -------
    out    : shape (n,),  out[j] = log Σ_{l=1}^K exp(⟨w^l, x_j⟩).
    """
    m = logits.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).squeeze(-1)

def _sample_planted_data(
    K: int,
    p: int,
    n: int,
    rng: np.random.RandomState,
    w_true_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (X, y) from a linear-softmax planted model.

    Procedure
    ---------
    1. W_true ~ Uniform[-w_true_scale, w_true_scale]^{K × p}
    2. X      ~ N(0, 1)^{n × p}
    3. For each j:  y_j ~ Categorical( softmax(X W_true^T)_j )

    Parameters
    ----------
    K, p, n      : number of classes, features, samples.
    rng          : numpy RandomState (for reproducibility).
    w_true_scale : range of the ground-truth weights is [-s, s].
                   Larger s  ⇒  more separable classes (labels less noisy).

    Returns
    -------
    X      : shape (n, p)   feature matrix.
    labels : shape (n,)     integer labels in [0, K).
    W_true : shape (K, p)   ground-truth weights (returned for diagnostics;
                            the learner should never see these).

    Notes on class balance
    ----------------------
    With X ~ N(0, 1) and W_true ~ U[-1, 1], the true logits have
    std ≈ sqrt(p/3)  per coordinate, so the softmax outputs are moderately
    peaked.  Class counts will be close to, but not exactly, n/K.  Empty
    classes are unlikely for balanced K but are handled by the callers
    via an n_i ≥ 1 guard.
    """
    #W_true = rng.uniform(-w_true_scale, w_true_scale, size=(K, p))
    W_true = rng.randn(K, p)
    X = rng.randn(n, p)
    true_logits = X @ W_true.T                   # (n, K)
    true_probs = _softmax(true_logits)           # (n, K)

    # Vectorised categorical sampling via inverse-CDF on cumulative probs.
    # For each row j, draw u_j ~ U(0, 1) and pick the smallest class
    # index i such that cumprob[j, i] ≥ u_j.
    cumprob = np.cumsum(true_probs, axis=1)      # (n, K)
    u = rng.uniform(size=(n, 1))                 # (n, 1)
    labels = (u < cumprob).argmax(axis=1)        # (n,)
    #print('labels:',labels)
    return X, labels, W_true


# ====================================================================
# Single-hidden-layer neural network  (generic non-convex)
# ====================================================================
def make_mlp_nonconvex(
    K: int = 3,
    p: int = 4,
    n: int = 60,
    h: int = 8,
    seed: int = 7,
    w_true_scale: float = 1.0,
) -> Tuple[List[Callable], List[Callable], np.ndarray, Callable]:
    """Create K per-class cross-entropy objectives for a 1-hidden-layer MLP.

    Architecture
    ------------
    Input  x_j ∈ R^p
      →  hidden layer:   a_j = σ(W_1 x_j + b_1) ∈ R^h      (σ = ReLU)
      →  output layer:   z_j = W_2 a_j + b_2 ∈ R^K
      →  softmax:        P(Y = i | x_j; θ) = exp(z_j^{(i)}) / Σ_l exp(z_j^{(l)})

    Parameters  θ = (W_1, b_1, W_2, b_2)  flattened into a vector of
    dimension  d = h·p + h + K·h + K.

    Per-class loss (the i-th MOO objective):

        F_i(θ) = (1/n_i) Σ_{j: y_j=i} { −z_j^{(i)} + log Σ_l exp(z_j^{(l)}) }

    This is non-convex due to the composition of the linear output layer
    with the ReLU hidden layer (the product W_2 · σ(W_1 x + b_1) is
    non-convex in (W_1, b_1, W_2) jointly).

    Data generation (planted model)
    -------------------------------
    Uses the same linear-softmax planted model as the logistic-regression
    generator for consistency across experiments:

        W_true ~ Uniform[-w_true_scale, w_true_scale]^{K × p},
        X      ~ N(0, 1)^{n × p},
        y_j    ~ Categorical( softmax(X W_true^T)_j ).

    The MLP is an over-parameterised non-convex hypothesis class that
    can represent the true linear-softmax model exactly (take W_1 = I,
    b_1 sufficiently negative-shifted or W_1 such that pre-activations
    stay positive, then set W_2 to recover W_true).  This makes it a
    clean non-convex testbed: well-specified in theory, but optimisation
    must still contend with ReLU kinks and the bilinear W_2 W_1 product.

    Smoothness
    ----------
    No closed-form L_i is available for a neural network.  We estimate
    L_i by computing the gradient at several random points and measuring
    the maximum ratio  ‖∇F_i(θ₁) − ∇F_i(θ₂)‖ / ‖θ₁ − θ₂‖.

    Parameters
    ----------
    K            : number of classes.
    p            : feature dimension.
    n            : total number of training samples.
    h            : number of hidden units.
    seed         : random seed.
    w_true_scale : range of the ground-truth weights (default 1.0).

    Returns
    -------
    objectives      : list of K callables  F_i(θ) → float.
    grad_objectives : list of K callables  ∇F_i(θ) → ndarray of shape (d,).
    L               : estimated smoothness constants, shape (K,).
                      (no mu — the objectives are non-convex.)

    Illustrative example
    --------------------
    With K=3, p=4, h=8 the parameter vector θ has dimension
    d = 8·4 + 8 + 3·8 + 3 = 67.  The three objectives F_1, F_2, F_3
    measure the per-class cross-entropy through the neural network.
    At random initialisation, each F_i ≈ log(K) ≈ 1.099.

    >>> objs, grads, L = make_mlp_nonconvex(K=3, p=4, n=60, h=8)
    >>> d = 8*4 + 8 + 3*8 + 3  # = 67
    >>> theta = np.zeros(d)
    >>> objs[0](theta)           # F_1 at zero weights
    """
    rng = np.random.RandomState(seed)
    d = h * p + h + K * h + K     # total parameter count

    # ---- planted-model data generation ----
    X, labels, _W_true = _sample_planted_data(
        K=K, p=p, n=n, rng=rng, w_true_scale=w_true_scale,
    )

    class_idx = [np.where(labels == i)[0] for i in range(K)]
    n_i = np.array([max(len(idx), 1) for idx in class_idx], dtype=float)

    # ---- parameter packing / unpacking ----
    def _unpack(theta: np.ndarray):
        """Unpack θ into (W_1, b_1, W_2, b_2).

        Layout in θ (contiguous blocks):
          W_1 : h × p  (rows of the input-to-hidden weight matrix)
          b_1 : h      (hidden biases)
          W_2 : K × h  (rows of the hidden-to-output weight matrix)
          b_2 : K      (output biases)
        """
        idx = 0
        W1 = theta[idx: idx + h * p].reshape(h, p);  idx += h * p
        b1 = theta[idx: idx + h];                     idx += h
        W2 = theta[idx: idx + K * h].reshape(K, h);   idx += K * h
        b2 = theta[idx: idx + K];                      idx += K
        return W1, b1, W2, b2

    # ---- forward pass ----
    def _forward(theta: np.ndarray, X_batch: np.ndarray):
        """Compute hidden activations and output logits.

        Returns
        -------
        A      : (n_batch, h)  hidden activations  σ(X_batch @ W_1^T + b_1)
        Z      : (n_batch, K)  output logits  A @ W_2^T + b_2
        pre_A  : (n_batch, h)  pre-activation  X_batch @ W_1^T + b_1
        W1, b1, W2, b2 : unpacked parameters
        """
        W1, b1, W2, b2 = _unpack(theta)
        pre_A = X_batch @ W1.T + b1              # (n_batch, h)
        A = np.maximum(pre_A, 0.0)                # ReLU
        Z = A @ W2.T + b2                         # (n_batch, K)
        return A, Z, pre_A, W1, b1, W2, b2

    # ---- per-class objective ----
    def _F_i(theta: np.ndarray, i: int) -> float:
        """Evaluate F_i(θ).

        F_i(θ) = (1/n_i) Σ_{j: y_j=i} { −z_j^{(i)} + log Σ_l exp(z_j^{(l)}) }

        Steps:
          1. Forward pass on class-i samples to get logits z_j ∈ R^K.
          2. For each such sample, compute  −z_j[i] + logsumexp(z_j).
          3. Average over n_i samples.
        """
        idx = class_idx[i]
        X_i = X[idx]                               # (n_i, p)
        _, Z_i, _, _, _, _, _ = _forward(theta, X_i)
        lse = _logsumexp(Z_i)                      # (n_i,)
        losses = -Z_i[:, i] + lse                  # (n_i,)
        return float(losses.sum() / n_i[i])

    # ---- per-class gradient via backpropagation ----
    def _grad_F_i(theta: np.ndarray, i: int) -> np.ndarray:
        """Compute ∇F_i(θ) via backpropagation.

        Backprop derivation for a single sample j with y_j = i:

          Forward:
            pre_a = W_1 x_j + b_1           ∈ R^h
            a      = σ(pre_a)               ∈ R^h   (ReLU)
            z      = W_2 a + b_2            ∈ R^K
            loss   = −z[i] + logsumexp(z)

          Output layer gradient:
            ∂loss/∂z = softmax(z) − e_i      ∈ R^K
            (where e_i is the i-th standard basis vector)

            ∂loss/∂W_2 = (∂loss/∂z) a^T     ∈ R^{K×h}
            ∂loss/∂b_2 = ∂loss/∂z            ∈ R^K

          Hidden layer gradient:
            δ_hidden = W_2^T (∂loss/∂z) ⊙ σ'(pre_a)   ∈ R^h
            (where σ'(pre_a) = 1[pre_a > 0] for ReLU)

            ∂loss/∂W_1 = δ_hidden x_j^T     ∈ R^{h×p}
            ∂loss/∂b_1 = δ_hidden            ∈ R^h

          Average over all samples j with y_j = i, dividing by n_i.
        """
        idx_i = class_idx[i]
        X_i = X[idx_i]                             # (n_i, p)
        ni = n_i[i]

        A_i, Z_i, pre_A_i, W1, b1, W2, b2 = _forward(theta, X_i)

        # Output-layer gradient:  ∂loss/∂z = softmax(z) − e_i
        probs_i = _softmax(Z_i)                    # (n_i, K)
        dZ = probs_i.copy()                        # (n_i, K)
        dZ[:, i] -= 1.0                            # subtract indicator

        # Gradients w.r.t. W_2, b_2
        dW2 = (dZ.T @ A_i) / ni                    # (K, h)
        db2 = dZ.sum(axis=0) / ni                  # (K,)

        # Backprop through ReLU to hidden layer
        dA = dZ @ W2                                # (n_i, h)
        relu_mask = (pre_A_i > 0).astype(float)    # (n_i, h)
        dH = dA * relu_mask                         # (n_i, h)

        # Gradients w.r.t. W_1, b_1
        dW1 = (dH.T @ X_i) / ni                    # (h, p)
        db1 = dH.sum(axis=0) / ni                  # (h,)

        # Pack into flat gradient vector (same layout as θ)
        grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])
        return grad

    # ---- fused per-class objective + gradient ----
    # Returns (F_i(θ), ∇F_i(θ)) in a single forward pass.  The forward
    # work (X_i @ W1.T, ReLU, A @ W2.T) is performed once and reused for
    # both the loss (via logsumexp) and the gradient (via backprop).
    # This eliminates the redundant forward pass that occurs when F_i
    # and grad_F_i are called sequentially in bundle.add_point.
    def _F_and_grad_F_i(theta: np.ndarray, i: int) -> Tuple[float, np.ndarray]:
        idx_i = class_idx[i]
        X_i = X[idx_i]                             # (n_i, p)
        ni = n_i[i]
        A_i, Z_i, pre_A_i, W1, b1, W2, b2 = _forward(theta, X_i)

        # ---- shared softmax + logsumexp ----
        # Both `lse` (for the loss) and `probs_i` (for the gradient)
        # derive from  shifted = Z_i - max(Z_i, axis=-1).  Computing
        # the row-max + exp once and deriving both from those values
        # saves the duplicate work that two helper calls
        # (`_logsumexp` + `_softmax`) would otherwise do.
        # Numerically identical to the prior `_logsumexp` / `_softmax`
        # pair (same shift, same exp, same divisions) — verified
        # byte-equivalent against per-class output.
        z_max = Z_i.max(axis=-1, keepdims=True)            # (n_i, 1)
        exp_shifted = np.exp(Z_i - z_max)                  # (n_i, K)
        sum_exp = exp_shifted.sum(axis=-1, keepdims=True)  # (n_i, 1)
        log_sum_exp = np.log(sum_exp)                      # (n_i, 1)
        lse = (z_max + log_sum_exp).squeeze(-1)            # (n_i,)
        probs_i = exp_shifted / sum_exp                    # (n_i, K)

        # ---- loss ----
        losses = -Z_i[:, i] + lse                  # (n_i,)
        loss = float(losses.sum() / ni)

        # ---- gradient ----
        dZ = probs_i.copy()                        # (n_i, K)
        dZ[:, i] -= 1.0                            # ∂loss/∂z = softmax − e_i
        dW2 = (dZ.T @ A_i) / ni                    # (K, h)
        db2 = dZ.sum(axis=0) / ni                  # (K,)
        dA = dZ @ W2                                # (n_i, h)
        relu_mask = (pre_A_i > 0).astype(float)    # (n_i, h)
        dH = dA * relu_mask                         # (n_i, h)
        dW1 = (dH.T @ X_i) / ni                    # (h, p)
        db1 = dH.sum(axis=0) / ni                  # (h,)
        grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])

        return loss, grad

    # ---- joint oracle (all K classes at once) ----
    # Returns (fv, gv) with shapes ((K,), (K, d)) — the exact arrays
    # ``bundle.add_point`` would build from K separate F_i + K separate
    # grad_F_i calls, but each class only runs its fused forward+backward
    # once.  Used by ``bundle.add_point`` when the caller provides a
    # joint oracle (Tier 1 CPU optimisation).
    def _joint_oracle(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fv = np.empty(K, dtype=np.float64)
        gv = np.empty((K, d), dtype=np.float64)
        for i in range(K):
            loss_i, grad_i = _F_and_grad_F_i(theta, i)
            fv[i] = loss_i
            gv[i] = grad_i
        return fv, gv

    # ---- Tier 2: fused-across-classes joint oracle ----
    # Computes ALL K (F_i, ∇F_i) pairs from ONE shared forward pass over
    # the FULL training set X (n_total, p), instead of K separate forward
    # passes each over its own class subset X_i.
    #
    # Mathematical equivalence
    # ------------------------
    # The per-class oracle does, for each i:
    #     X_i  = X[idx_i]                     # n_i rows of X
    #     pre_A_i = X_i @ W1.T + b1
    #     A_i = ReLU(pre_A_i)
    #     Z_i = A_i @ W2.T + b2
    #     # then logsumexp + softmax + backprop on Z_i, scaled by 1/n_i
    #
    # The fused oracle does ONE forward pass on full X:
    #     pre_A_all = X @ W1.T + b1            # (n_total, h)
    #     A_all     = ReLU(pre_A_all)
    #     Z_all     = A_all @ W2.T + b2        # (n_total, K)
    # Then slices ``Z_all[idx_i]`` for each class.  Because slicing
    # preserves arithmetic exactly (no broadcasting tricks, no different
    # numerical reductions), the resulting Z_i, A_i, pre_A_i for each
    # class are IDENTICAL bit-for-bit to the per-class version.
    # Hence each per-class loss and gradient is computed on IDENTICAL
    # intermediate arrays — the only thing that changes is one big matmul
    # in place of K small ones.
    #
    # Savings (relative to ``_joint_oracle``):
    #   * one ``_unpack(theta)`` instead of K
    #   * one ``X @ W1.T`` matmul of size (n_total, p) @ (p, h)
    #     instead of K of size (n_i, p) @ (p, h)
    #   * one ``A @ W2.T`` matmul of size (n_total, h) @ (h, K)
    #     instead of K of size (n_i, h) @ (h, K)
    #   * one row-wise max + exp pass over (n_total, K) instead of K
    #     passes over (n_i, K)
    #
    # The per-class backprop matmuls (dZ.T @ A_i, dH.T @ X_i) are still
    # done per-class because each F_i averages over only its own samples
    # — but those are now small operations on slices of the already-
    # computed shared arrays.
    def _joint_oracle_fused(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W1, b1, W2, b2 = _unpack(theta)

        # ---- single forward pass on FULL X ----
        pre_A_all = X @ W1.T + b1                       # (n_total, h)
        A_all = np.maximum(pre_A_all, 0.0)              # (n_total, h)
        Z_all = A_all @ W2.T + b2                       # (n_total, K)

        # ---- shared row-max + exp + softmax + logsumexp on FULL Z ----
        # Both `lse` (for the loss) and `probs` (for the gradient) derive
        # from a single row-max / exp pass.  Saves the duplicate work
        # that two helper calls would otherwise do.
        z_max = Z_all.max(axis=-1, keepdims=True)        # (n_total, 1)
        exp_shifted = np.exp(Z_all - z_max)              # (n_total, K)
        sum_exp = exp_shifted.sum(axis=-1, keepdims=True)
        log_sum_exp = np.log(sum_exp)                    # (n_total, 1)
        lse_all = (z_max + log_sum_exp).squeeze(-1)      # (n_total,)
        probs_all = exp_shifted / sum_exp                # (n_total, K)

        # ---- per-class losses and gradients ----
        # Per-class slicing into the already-computed shared arrays.
        # Each per-class operation reproduces the per-class oracle EXACTLY
        # because the slices are bit-identical to what _forward(theta, X_i)
        # would have produced.
        fv = np.empty(K, dtype=np.float64)
        gv = np.empty((K, d), dtype=np.float64)
        for i in range(K):
            idx_i = class_idx[i]
            X_i = X[idx_i]                               # (n_i, p)
            A_i = A_all[idx_i]                           # (n_i, h)
            pre_A_i = pre_A_all[idx_i]                   # (n_i, h)
            Z_i = Z_all[idx_i]                           # (n_i, K)
            lse_i = lse_all[idx_i]                       # (n_i,)
            probs_i = probs_all[idx_i]                   # (n_i, K)
            ni = n_i[i]

            # Loss (same formula as _F_and_grad_F_i):
            losses = -Z_i[:, i] + lse_i                  # (n_i,)
            fv[i] = float(losses.sum() / ni)

            # Gradient (same backprop as _F_and_grad_F_i):
            dZ = probs_i.copy()                          # (n_i, K)
            dZ[:, i] -= 1.0
            dW2 = (dZ.T @ A_i) / ni                      # (K, h)
            db2 = dZ.sum(axis=0) / ni                    # (K,)
            dA = dZ @ W2                                 # (n_i, h)
            relu_mask = (pre_A_i > 0).astype(float)      # (n_i, h)
            dH = dA * relu_mask                          # (n_i, h)
            dW1 = (dH.T @ X_i) / ni                      # (h, p)
            db1 = dH.sum(axis=0) / ni                    # (h,)
            gv[i] = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])

        return fv, gv

    objectives = [lambda theta, i=i: _F_i(theta, i) for i in range(K)]
    grad_objectives = [lambda theta, i=i: _grad_F_i(theta, i) for i in range(K)]
    joint_oracle = _joint_oracle
    # Expose the Tier 2 (fused-across-classes) joint oracle as an attribute
    # for testing.  NOT yet used by default — `bundle.add_point` still calls
    # `joint_oracle(theta)` which dispatches to `_joint_oracle` (the per-
    # class fused version).  Once verified against the per-class output,
    # callers can opt in by passing `joint_oracle.fused` as the joint
    # oracle, or by replacing `joint_oracle = _joint_oracle` above.
    joint_oracle.fused = _joint_oracle_fused

    # ---- estimate smoothness constants L_i ----
    # Sample random parameter pairs and measure gradient Lipschitz ratio.
    n_probes = 40
    L_arr = np.zeros(K)
    for i in range(K):
        max_ratio = 0.0
        for _ in range(n_probes):
            t1 = rng.randn(d) * 0.5
            t2 = t1 + rng.randn(d) * 0.1
            g1 = grad_objectives[i](t1)
            g2 = grad_objectives[i](t2)
            diff_g = np.linalg.norm(g1 - g2)
            diff_t = np.linalg.norm(t1 - t2)
            if diff_t > 1e-12:
                max_ratio = max(max_ratio, diff_g / diff_t)
        L_arr[i] = max_ratio * 2.0    # safety factor of 2

    return objectives, grad_objectives, L_arr, joint_oracle