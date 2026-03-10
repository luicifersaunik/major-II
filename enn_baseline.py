"""
Classical Elman Neural Network (ENN) — Baseline Model
Implements the standard ENN (Eq. 1) for direct comparison with QENN.
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_deriv(x):
    return 1.0 - np.tanh(np.clip(x, -15, 15)) ** 2


class ENN:
    """
    Classical Elman Neural Network with optional self-connection on context.
    Architecture: ni → nh (+ context nh) → no
    """

    def __init__(self, ni: int, nh: int, no: int,
                 c: float = 0.0, lr: float = 1e-3, seed: int = 42):
        """
        Parameters
        ----------
        ni  : input size
        nh  : hidden (and context) size
        no  : output size
        c   : self-connection gain (0 = standard ENN, >0 = modified)
        lr  : learning rate
        """
        rng = np.random.default_rng(seed)
        self.ni, self.nh, self.no = ni, nh, no
        self.c = c
        self.lr = lr

        s = 0.1
        # Eq. 1 weights
        self.W1 = rng.uniform(-s, s, (nh, nh))    # context → hidden
        self.W2 = rng.uniform(-s, s, (nh, ni))    # input   → hidden
        self.W3 = rng.uniform(-s, s, (no, nh))    # hidden  → output
        self.b_h = rng.uniform(-s, s, (nh,))
        self.b_o = rng.uniform(-s, s, (no,))

        self.uc = np.zeros(nh)  # context state

    def reset_context(self):
        self.uc = np.zeros(self.nh)

    def forward(self, x):
        net_h = self.W1 @ self.uc + self.W2 @ x + self.b_h
        u = np.tanh(net_h)

        # Update context: uc(k) = u(k-1) [+ self-connection if c>0]
        self.uc = self.c * self.uc + u   # standard ENN: c=0 → uc = u

        net_o = self.W3 @ u + self.b_o
        y = net_o   # linear output for regression
        return y, u, net_h, net_o

    def backward(self, x, y_target, y_d, u, net_h, net_o):
        e = y_d - y_target
        N = 1

        dE_dW3 = (2/N) * np.outer(e, u)
        dE_db_o = (2/N) * e

        delta_h = (self.W3.T @ e) * tanh_deriv(net_h) * (2/N)
        dE_dW1 = np.outer(delta_h, self.uc)
        dE_dW2 = np.outer(delta_h, x)
        dE_db_h = delta_h

        self.W3 -= self.lr * dE_dW3
        self.b_o -= self.lr * dE_db_o
        self.W1 -= self.lr * dE_dW1
        self.W2 -= self.lr * dE_dW2
        self.b_h -= self.lr * dE_db_h

        return float(np.mean(e ** 2))

    def train(self, X_train, y_train, epochs: int = 200, verbose: bool = False):
        history = []
        for ep in range(epochs):
            self.reset_context()
            total_mse = 0.0
            for i in range(len(X_train)):
                y_d, u, net_h, net_o = self.forward(X_train[i])
                mse = self.backward(X_train[i], y_train[i], y_d, u, net_h, net_o)
                total_mse += mse
            epoch_mse = total_mse / len(X_train)
            history.append(epoch_mse)
            if verbose and ep % 20 == 0:
                print(f"  Epoch {ep:3d} | NMSE: {epoch_mse:.5f}")
        return history

    def predict(self, X_test):
        self.reset_context()
        preds = []
        for x in X_test:
            y_d, _, _, _ = self.forward(x)
            preds.append(y_d.copy())
        return np.array(preds)
