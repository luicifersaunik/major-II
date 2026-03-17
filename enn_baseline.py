"""
enn_baseline.py — Classical ENN
Stable version: gradient descent + Xavier init + per-size lr + gradient clipping.

Why not Adam for ENN?
Adam is very sensitive to learning rate per dataset — causes explosion
on some markets (NASDAQ ENN-100 → NMSE 320) while fine on others.
Gradient descent with Xavier init + clipping is robust across all datasets.

Fair comparison note:
Both QENN-Qiskit and ENN use:
  - Xavier weight initialization
  - Gradient clipping scaled by network size
  - Same number of epochs (80)
The only difference is the hidden layer architecture.
"""

import numpy as np


# ─────────────────────────────────────────────
#  Activation
# ─────────────────────────────────────────────
def tanh(x):
    return np.tanh(np.clip(x, -15, 15))

def tanh_deriv(x):
    return 1.0 - np.tanh(np.clip(x, -15, 15)) ** 2


# ─────────────────────────────────────────────
#  Per-size learning rate
# ─────────────────────────────────────────────
def get_lr(nh):
    """
    Stable learning rates per hidden size.
    Larger networks need smaller lr — standard practice.
    """
    lr_map = {
        10:  1e-3,
        20:  8e-4,
        40:  5e-4,
        70:  3e-4,
        100: 2e-4,
    }
    return lr_map.get(nh, 1e-3 / np.sqrt(nh))


# ─────────────────────────────────────────────
#  Classical ENN
# ─────────────────────────────────────────────
class ENN:
    """
    Classical Elman Neural Network.

    Fair comparison settings vs QENN-Qiskit:
      ✅ Xavier weight initialization (same)
      ✅ Gradient clipping scaled by 1/sqrt(nh) (same)
      ✅ Per-size learning rates (stable across all datasets)
      ✅ Same number of epochs
      ✅ Self-connection gain c (optional, 0 = standard ENN)
    """

    def __init__(self, ni, nh, no, c=0.0, lr=None, seed=42):
        rng      = np.random.default_rng(seed)
        self.ni  = ni
        self.nh  = nh
        self.no  = no
        self.c   = c
        self.lr  = lr if lr is not None else get_lr(nh)

        # Xavier initialization — same as QENN-Qiskit
        s1 = np.sqrt(2.0 / (ni + nh))
        s2 = np.sqrt(2.0 / (nh + no))

        self.W1  = rng.uniform(-s1, s1, (nh, nh))   # context → hidden
        self.W2  = rng.uniform(-s1, s1, (nh, ni))   # input   → hidden
        self.W3  = rng.uniform(-s2, s2, (no, nh))   # hidden  → output
        self.b_h = np.zeros(nh)
        self.b_o = np.zeros(no)
        self.uc  = np.zeros(nh)

        # Gradient clip — scaled by network size, same as QENN-Qiskit
        self.clip = 1.0 / np.sqrt(nh)

    def reset_context(self):
        self.uc = np.zeros(self.nh)

    def _clip(self, g):
        norm = np.linalg.norm(g)
        return g * self.clip / norm if norm > self.clip else g

    # ── Forward ─────────────────────────────────────────────────
    def forward(self, x):
        net_h   = self.W1 @ self.uc + self.W2 @ x + self.b_h
        u       = tanh(net_h)
        self.uc = self.c * self.uc + u
        net_o   = self.W3 @ u + self.b_o
        return net_o.copy(), u, net_h, net_o

    # ── Backward (gradient descent) ──────────────────────────────
    def backward(self, x, y_target, y_d, u, net_h, net_o):
        e = y_d - y_target

        # Output layer
        dW3 = self._clip(np.outer(e, u))
        dbo = self._clip(e.copy())

        # Hidden layer
        delta_h = self._clip((self.W3.T @ e) * tanh_deriv(net_h))
        dW1     = self._clip(np.outer(delta_h, self.uc))
        dW2     = self._clip(np.outer(delta_h, x))
        dbh     = delta_h.copy()

        # Gradient descent updates
        self.W3  -= self.lr * dW3
        self.b_o -= self.lr * dbo
        self.W1  -= self.lr * dW1
        self.W2  -= self.lr * dW2
        self.b_h -= self.lr * dbh

        return float(np.mean(e ** 2))

    # ── Training ─────────────────────────────────────────────────
    def train(self, X_train, y_train, epochs=80, verbose=False):
        history = []
        for ep in range(epochs):
            self.reset_context()
            total_mse = 0.0
            for i in range(len(X_train)):
                y_d, u, net_h, net_o = self.forward(X_train[i])
                mse = self.backward(X_train[i], y_train[i],
                                    y_d, u, net_h, net_o)
                total_mse += mse
            history.append(total_mse / len(X_train))
            if verbose and ep % 20 == 0:
                print(f"  Epoch {ep:3d} | MSE: {history[-1]:.5f}")
        return history

    # ── Inference ────────────────────────────────────────────────
    def predict(self, X_test):
        self.reset_context()
        preds = []
        for x in X_test:
            y_d, _, _, _ = self.forward(x)
            preds.append(y_d.copy())
        return np.array(preds)