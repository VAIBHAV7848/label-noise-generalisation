"""Unit tests for Proposition 2A matrix inverse perturbation and risk estimation bounds."""

import unittest
import numpy as np


class TestProposition2Bound(unittest.TestCase):
    def test_proposition2a_perturbation_and_bias_bounds(self):
        """Verify Proposition 2A risk bias bound across randomized matrices and perturbations."""
        K = 10
        M = 2.5
        num_trials = 1000
        rng = np.random.default_rng(42)

        violations = 0
        for _ in range(num_trials):
            # 1. Generate diagonally dominant invertible transition matrix T
            raw = rng.uniform(0.1, 1.0, size=(K, K))
            np.fill_diagonal(raw, raw.diagonal() + 4.0)
            T = raw / raw.sum(axis=1, keepdims=True)

            try:
                T_inv = np.linalg.inv(T)
            except np.linalg.LinAlgError:
                continue

            norm_T_inv_2 = np.linalg.norm(T_inv, ord=2)
            max_eps = 0.5 / norm_T_inv_2

            # 2. Generate random perturbation matrix E with ||E||_F <= eps
            eps = rng.uniform(1e-4, max_eps)
            E_raw = rng.normal(0, 1, size=(K, K))
            E = E_raw / np.linalg.norm(E_raw, ord="fro") * eps

            T_hat = T + E
            T_hat_inv = np.linalg.inv(T_hat)

            # 3. Check matrix inverse operator norm difference
            diff_norm = np.linalg.norm(T_hat_inv - T_inv, ord=2)
            theoretical_inv_bound = (norm_T_inv_2**2 * eps) / (1.0 - norm_T_inv_2 * eps)
            self.assertLessEqual(diff_norm, theoretical_inv_bound + 1e-10)

            # 4. Check pointwise risk bias on arbitrary probability vector p and loss vector ell
            p = rng.dirichlet(np.ones(K))
            ell = rng.uniform(0, M, size=K)

            actual_bias = np.abs(p @ T @ (T_hat_inv - T_inv) @ ell)
            theoretical_bias_bound = (np.sqrt(K) * M * norm_T_inv_2**2 * eps) / (1.0 - norm_T_inv_2 * eps)
            self.assertLessEqual(actual_bias, theoretical_bias_bound + 1e-10)


if __name__ == "__main__":
    unittest.main()
