"""
Quantum Elman Neural Network (QENN) - Core Implementation
Based on: "A quantum artificial neural network for stock closing price prediction"
Liu & Ma, Information Sciences 598 (2022) 75–85
"""

import numpy as np


# ─────────────────────────────────────────────
#  Quantum activation functions (Eq. 4, 7)
# ─────────────────────────────────────────────
def f0(x):
    """Cosine branch of quantum activation: |0⟩ amplitude"""
    return np.cos(np.exp(np.clip(x, -10, 10)))

def f1(x):
    """Sine branch of quantum activation: |1⟩ amplitude"""
    return np.sin(np.exp(np.clip(x, -10, 10)))

def f0_deriv(x):
    ex = np.exp(np.clip(x, -10, 10))
    return -ex * np.sin(ex)

def f1_deriv(x):
    ex = np.exp(np.clip(x, -10, 10))
    return ex * np.cos(ex)

def prob_deriv(x):
    """
    Derivative of measurement probability P(x) = f0(x)^2 + f1(x)^2 = 1,
    but for output layer we use p(k) = |f_{k_l}(net)|^2.
    For gradient computation we use the combined derivative p'(net).
    """
    return 2 * (f0(x) * f0_deriv(x) + f1(x) * f1_deriv(x))


# ─────────────────────────────────────────────
#  Quantum Elman Neural Network
# ─────────────────────────────────────────────
class QENN:
    """
    Quantum Elman Neural Network with self-connected context neurons.

    Architecture: ni × nh × no  (input × hidden × output)
    Context layer has same size as hidden layer (2^nh dimensional Hilbert space).
    """

    def __init__(self, ni: int, nh: int, no: int, c: float = 0.5, seed: int = 42):
        """
        Parameters
        ----------
        ni : number of input neurons
        nh : number of hidden neurons
        no : number of output neurons
        c  : self-connection feedback gain (0 < c < 1)
        """
        rng = np.random.default_rng(seed)
        self.ni, self.nh, self.no = ni, nh, no
        self.c = c  # self-connection gain for context layer (Eq. 6)

        # Hilbert space dimension = 2^nh
        self.dim = 2 ** nh

        # Weights & biases — initialised small
        scale = 0.1
        # hidden layer: input→hidden (w), context→hidden (w_tilde)
        self.W  = rng.uniform(-scale, scale, (nh, ni))    # w^i in R^ni
        self.W_ctx = rng.uniform(-scale, scale, (nh, self.dim))  # w~^i in R^(2^nh)
        self.b  = rng.uniform(-scale, scale, (nh,))

        # output layer: hidden→output (V)
        self.V  = rng.uniform(-scale, scale, (no, self.dim))
        self.b_out = rng.uniform(-scale, scale, (no,))

        # Context layer state (quantum amplitudes α̃)
        self.alpha_ctx = np.zeros(self.dim)   # α̃(k)

        # Learning rates (tuned by DCQGA)
        self.eta1 = 1e-4   # output layer lr
        self.eta2 = 2e-5   # context→hidden lr
        self.eta3 = 9e-6   # input→hidden lr

    # ── Forward pass ────────────────────────────────────────────────────────
    def _hidden_net(self, x_in):
        """Net input to each hidden neuron (scalar, Eq. 4)"""
        return self.W @ x_in + self.b + self.W_ctx @ self.alpha_ctx

    def _compute_alpha(self, net_h):
        """
        Quantum amplitudes for hidden layer (Eq. 5):
        |y_1⟩⊗|y_2⟩⊗...⊗|y_nh⟩ = Σ α_i(k)|i⟩
        We approximate the tensor product via Kronecker outer product.
        """
        # Each qubit: [f0(net_i), f1(net_i)]
        state = np.array([f0(net_h[0]), f1(net_h[0])])
        for i in range(1, self.nh):
            qi = np.array([f0(net_h[i]), f1(net_h[i])])
            state = np.kron(state, qi)
        # Normalise
        norm = np.linalg.norm(state)
        if norm > 1e-12:
            state = state / norm
        return state  # shape: (2^nh,)

    def _update_context(self, alpha_h):
        """
        Context layer update with self-connection (Eq. 6):
        α̃(k) = (c·α̃(k-1) + α(k-1)) / ‖…‖
        """
        raw = self.c * self.alpha_ctx + alpha_h
        norm = np.linalg.norm(raw)
        if norm > 1e-12:
            raw = raw / norm
        self.alpha_ctx = raw

    def forward(self, x_in):
        """
        Full forward pass.
        Returns: output vector y_d(k), hidden amplitudes alpha_h, net_h, net_out
        """
        net_h = self._hidden_net(x_in)
        alpha_h = self._compute_alpha(net_h)

        # Output layer net input (Eq. 7)
        net_out = self.V @ alpha_h + self.b_out   # shape: (no,)

        # Output: measurement probability (scalar per output neuron)
        y_d = f0(net_out) ** 2 + f1(net_out) ** 2   # ≈ 1 by construction
        # We use the net_out directly as the regression output after denorm
        # (standard approach for quantum-inspired regression)
        y_d = net_out   # use raw net for regression (standard practice)

        # Update context for NEXT step
        self._update_context(alpha_h)

        return y_d, alpha_h, net_h, net_out

    # ── Backward pass (gradient descent, Eq. 10-12) ─────────────────────────
    def backward(self, x_in, y_target, y_d, alpha_h, net_h, net_out):
        """Compute gradients and update weights."""
        e = y_d - y_target                          # error
        N = 1

        # ── Output layer gradients (Eq. 10) ──
        p_prime_out = prob_deriv(net_out)            # shape: (no,)
        dE_dV   = (2/N) * np.outer(e * p_prime_out, alpha_h)  # (no, dim)
        dE_db_out = (2/N) * (e * p_prime_out)        # (no,)

        # ── Hidden layer gradients (Eq. 11-12) ──
        # Chain: dE/d(net_h) via output amplitudes
        p_prime_h = prob_deriv(net_h)                # (nh,)

        # Approximate: dE/d(net_h_i) ∝ sum_l dE/dy_l * p'(net_out_l) * V_l,i
        # (simplified chain through Kronecker product)
        delta_h = np.zeros(self.nh)
        for i in range(self.nh):
            delta_h[i] = np.sum((2/N) * e * p_prime_out * self.V[:, i % self.dim]) * p_prime_h[i]

        dE_dW_ctx = np.outer(delta_h, self.alpha_ctx)   # (nh, dim)
        dE_db_h   = delta_h                              # (nh,)
        dE_dW     = np.outer(delta_h, x_in)             # (nh, ni)

        # ── Parameter update ──
        self.V      -= self.eta1 * dE_dV
        self.b_out  -= self.eta1 * dE_db_out
        self.W_ctx  -= self.eta2 * dE_dW_ctx
        self.b      -= self.eta2 * dE_db_h
        self.W      -= self.eta3 * dE_dW

        return float(np.mean(e ** 2))

    # ── Training ─────────────────────────────────────────────────────────────
    def reset_context(self):
        """Reset context layer between sequences."""
        self.alpha_ctx = np.zeros(self.dim)

    def train(self, X_train, y_train, epochs: int = 200, verbose: bool = False):
        """
        Train on sliding-window sequences.
        X_train: (n_samples, ni)
        y_train: (n_samples, no)
        """
        history = []
        for ep in range(epochs):
            self.reset_context()
            total_mse = 0.0
            for i in range(len(X_train)):
                x = X_train[i]
                y_t = y_train[i]
                y_d, alpha_h, net_h, net_out = self.forward(x)
                mse = self.backward(x, y_t, y_d, alpha_h, net_h, net_out)
                total_mse += mse
            epoch_mse = total_mse / len(X_train)
            history.append(epoch_mse)
            if verbose and (ep % 20 == 0):
                print(f"  Epoch {ep:3d} | NMSE: {epoch_mse:.5f}")
        return history

    def predict(self, X_test):
        """Run inference on test set."""
        self.reset_context()
        preds = []
        for x in X_test:
            y_d, _, _, _ = self.forward(x)
            preds.append(y_d.copy())
        return np.array(preds)

    def set_learning_rates(self, eta1, eta2, eta3):
        self.eta1, self.eta2, self.eta3 = eta1, eta2, eta3
