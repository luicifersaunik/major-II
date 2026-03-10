"""
Double Chains Quantum Genetic Algorithm (DCQGA)
for tuning QENN learning rates (η1, η2, η3).

Based on Section 3.3 of Liu & Ma (2022).
"""

import numpy as np


class DCQGA:
    """
    Double Chains Quantum Genetic Algorithm.

    Each chromosome has TWO gene chains (cos-chain and sin-chain),
    representing quantum bit amplitudes. The rotation gate updates
    qubits toward the best solution found.

    Usage
    -----
    dcqga = DCQGA(n_bits=3, bounds=[(1e-5,1e-4), (2e-5,1e-4), (9e-6,8e-5)])
    best_params, best_fit = dcqga.run(fitness_fn, population_size=50, epochs=100)
    """

    def __init__(self, n_bits: int, bounds: list, seed: int = 0):
        """
        Parameters
        ----------
        n_bits  : number of quantum bits per chromosome (= number of params)
        bounds  : list of (min, max) for each parameter
        """
        self.n = n_bits
        self.bounds = np.array(bounds)  # (n_bits, 2)
        self.rng = np.random.default_rng(seed)
        self.dh0 = 0.01 * np.pi         # initial rotation angle step

    # ── Initialise population ──────────────────────────────────────────────
    def _init_population(self, m: int):
        """
        Eq. 14: tij = 2π·r,  chromosome = [cos(t), sin(t)]
        Returns theta: (m, n) angles
        """
        return 2 * np.pi * self.rng.uniform(0, 1, (m, self.n))

    # ── Decode chromosomes to real-valued parameters ───────────────────────
    def _decode(self, theta: np.ndarray):
        """
        Eq. 15-16: split into cos-chain and sin-chain, map to [a_j, b_j].
        We average both chains for the final solution.
        """
        cos_chain = np.cos(theta)   # (m, n)
        sin_chain = np.sin(theta)

        a = self.bounds[:, 0]  # lower bounds
        b = self.bounds[:, 1]  # upper bounds

        # Eq. 16: X^c_j = 0.5*(b*(1+alpha) + a*(1-alpha))
        X_c = 0.5 * (b * (1 + cos_chain) + a * (1 - cos_chain))
        X_s = 0.5 * (b * (1 + sin_chain) + a * (1 - sin_chain))

        # Average both chains
        return 0.5 * (X_c + X_s)   # (m, n)

    # ── Rotation gate update ──────────────────────────────────────────────
    def _rotation_angle(self, theta_i, theta_best, fitness_i, fitness_best, grad_norm):
        """
        Eq. 20-21: adaptive rotation based on gradient of fitness.
        """
        # Direction: sign based on difference from best
        diff = np.cos(theta_best) * np.sin(theta_i) - np.sin(theta_best) * np.cos(theta_i)
        direction = np.sign(diff)
        direction[direction == 0] = 1.0

        # Size: Eq. 21 (simplified — use gradient proxy via fitness diff)
        size = self.dh0 * np.exp(-grad_norm)
        return direction * size

    # ── Mutation via quantum NOT gate ─────────────────────────────────────
    def _mutate(self, theta: np.ndarray, prob: float = 0.1):
        """Eq. 22-23: swap cos/sin by flipping angle by π/2."""
        mask = self.rng.uniform(0, 1, theta.shape) < prob
        theta[mask] += np.pi / 2
        return theta

    # ── Main optimisation loop ────────────────────────────────────────────
    def run(self, fitness_fn, population_size: int = 50,
            epochs: int = 100, mutation_prob: float = 0.1,
            verbose: bool = False):
        """
        Parameters
        ----------
        fitness_fn : callable(params_array) -> float (LOWER is better)
        Returns (best_params, best_fitness, history)
        """
        m = population_size
        theta = self._init_population(m)           # (m, n)
        params = self._decode(theta)                # (m, n)

        # Evaluate initial population
        fitness = np.array([fitness_fn(params[i]) for i in range(m)])
        best_idx = np.argmin(fitness)
        best_theta = theta[best_idx].copy()
        best_params = params[best_idx].copy()
        best_fit = fitness[best_idx]
        history = [best_fit]

        for ep in range(epochs):
            # Gradient proxy: normalise fitness variance
            f_range = fitness.max() - fitness.min() + 1e-12
            grad_norm = (fitness - fitness.min()) / f_range  # (m,)

            for i in range(m):
                # Rotation (Eq. 20-21)
                dh = self._rotation_angle(
                    theta[i], best_theta,
                    fitness[i], best_fit,
                    grad_norm[i]
                )
                theta[i] += dh

                # Mutation (Eq. 22-23)
                theta[i] = self._mutate(theta[i], mutation_prob)

            # Re-evaluate
            params = self._decode(theta)
            fitness = np.array([fitness_fn(params[i]) for i in range(m)])
            new_best = np.argmin(fitness)
            if fitness[new_best] < best_fit:
                best_fit = fitness[new_best]
                best_theta = theta[new_best].copy()
                best_params = params[new_best].copy()

            history.append(best_fit)
            if verbose and ep % 10 == 0:
                print(f"  DCQGA Epoch {ep:3d} | Best fitness: {best_fit:.6f} "
                      f"| η=({best_params[0]:.2e},{best_params[1]:.2e},{best_params[2]:.2e})")

        return best_params, best_fit, history
