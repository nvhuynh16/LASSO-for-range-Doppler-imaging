"""
Theoretical conditions for sparse recovery in radar applications.

This module implements checks for key theoretical conditions that guarantee
successful sparse recovery using LASSO and related methods.
"""

import numpy as np
from typing import List, Dict, Any
import warnings
from itertools import combinations


class TheoreticalConditions:
    """Theoretical conditions checker for compressed sensing."""

    @staticmethod
    def mutual_incoherence(A: np.ndarray) -> float:
        """
        Compute mutual incoherence of measurement matrix.

        Parameters
        ----------
        A : ndarray
            Measurement matrix.

        Returns
        -------
        mu : float
            Mutual incoherence value.
        """
        # Normalize columns
        A_normalized = A / (np.linalg.norm(A, axis=0) + 1e-12)

        # Compute Gram matrix
        G = A_normalized.T @ A_normalized

        # Set diagonal to zero and find maximum off-diagonal element
        np.fill_diagonal(G, 0)
        mu = np.max(np.abs(G))

        return float(mu)

    @staticmethod
    def restricted_isometry_constant(A: np.ndarray, s: int) -> float:
        """
        Estimate restricted isometry constant (computationally expensive).

        Parameters
        ----------
        A : ndarray
            Measurement matrix.
        s : int
            Sparsity level.

        Returns
        -------
        delta_s : float
            Estimated RIC.
        """
        if s > 20:
            warnings.warn("RIC computation expensive for large s")
            return TheoreticalConditions._approximate_ric(A, s)

        # Exact computation for small s
        n_measurements, n_features = A.shape

        if s >= n_features:
            return 1.0

        min_eigenvalue = np.inf
        max_eigenvalue = 0.0

        # Sample subset of all possible s-sparse supports
        max_combinations = 1000
        all_combinations = list(combinations(range(n_features), s))

        if len(all_combinations) > max_combinations:
            # Random sampling
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            sampled_combinations = [all_combinations[i] for i in indices]
        else:
            sampled_combinations = all_combinations

        for support in sampled_combinations:
            A_subset = A[:, list(support)]

            if A_subset.shape[1] > 0:
                # Compute eigenvalues of A_S^T A_S
                gram_matrix = A_subset.T @ A_subset
                eigenvals = np.linalg.eigvals(gram_matrix)

                min_eigenvalue = min(min_eigenvalue, np.min(eigenvals))
                max_eigenvalue = max(max_eigenvalue, np.max(eigenvals))

        # RIC is max deviation from 1
        delta_s = max(abs(1 - min_eigenvalue), abs(max_eigenvalue - 1))

        return float(delta_s)

    @staticmethod
    def _approximate_ric(A: np.ndarray, s: int) -> float:
        """Approximate RIC using random sampling."""
        n_measurements, n_features = A.shape
        n_trials = min(500, 2**min(s, 10))

        delta_values = []

        for _ in range(n_trials):
            # Random s-sparse support
            support = np.random.choice(n_features, s, replace=False)
            A_subset = A[:, support]

            gram_matrix = A_subset.T @ A_subset
            eigenvals = np.linalg.eigvals(gram_matrix)

            min_eig = np.min(eigenvals)
            max_eig = np.max(eigenvals)

            delta = max(abs(1 - min_eig), abs(max_eig - 1))
            delta_values.append(delta)

        return float(np.mean(delta_values))

    @staticmethod
    def restricted_isometry_property(A: np.ndarray, s: int) -> float:
        """
        Alias for restricted_isometry_constant for backward compatibility.

        Parameters
        ----------
        A : ndarray
            Measurement matrix.
        s : int
            Sparsity level.

        Returns
        -------
        delta_s : float
            Estimated RIC.
        """
        return TheoreticalConditions.restricted_isometry_constant(A, s)

    @staticmethod
    def restricted_eigenvalue(A: np.ndarray, s: int) -> float:
        """
        Compute restricted eigenvalue constant.

        Parameters
        ----------
        A : ndarray
            Measurement matrix.
        s : int
            Sparsity level.

        Returns
        -------
        re_constant : float
            Restricted eigenvalue constant.
        """
        if s > 15:
            return TheoreticalConditions._approximate_re(A, s)

        n_measurements, n_features = A.shape
        min_eigenvalue = np.inf

        # Sample combinations
        max_combinations = 500
        for _ in range(max_combinations):
            support = np.random.choice(n_features, min(s, n_features), replace=False)
            A_subset = A[:, support]

            if A_subset.shape[1] > 0:
                eigenvals = np.linalg.eigvals(A_subset.T @ A_subset)
                min_eigenvalue = min(min_eigenvalue, np.min(eigenvals))

        return float(max(min_eigenvalue, 0.0))

    @staticmethod
    def _approximate_re(A: np.ndarray, s: int) -> float:
        """Approximate restricted eigenvalue."""
        # Simple approximation based on smallest singular values
        try:
            _, singular_values, _ = np.linalg.svd(A, full_matrices=False)
            return float(singular_values[-1]**2)  # Smallest squared singular value
        except:
            return 0.0

    @staticmethod
    def beta_min_condition(signal: np.ndarray, threshold: float = 1e-6) -> float:
        """
        Compute beta-min condition (minimum non-zero coefficient magnitude).

        Parameters
        ----------
        signal : ndarray
            Signal vector.
        threshold : float
            Threshold for considering coefficients as non-zero.

        Returns
        -------
        beta_min : float
            Minimum non-zero coefficient magnitude.
        """
        nonzero_coeffs = signal[np.abs(signal) > threshold]
        return float(np.min(np.abs(nonzero_coeffs))) if len(nonzero_coeffs) > 0 else 0.0

    @staticmethod
    def compatibility_condition(A: np.ndarray, signal: np.ndarray,
                              measurements: np.ndarray, noise_level: float) -> float:
        """
        Check compatibility condition between signal and measurements.

        Parameters
        ----------
        A : ndarray
            Measurement matrix.
        signal : ndarray
            True signal.
        measurements : ndarray
            Noisy measurements.
        noise_level : float
            Noise standard deviation.

        Returns
        -------
        compatibility : float
            Compatibility measure.
        """
        predicted_measurements = A @ signal
        residual = measurements - predicted_measurements
        residual_norm = np.linalg.norm(residual)

        # Expected residual norm from noise
        expected_norm = noise_level * np.sqrt(len(measurements))

        # Compatibility is how close residual is to expected noise level
        compatibility = residual_norm / expected_norm if expected_norm > 0 else np.inf

        return float(compatibility)

    @staticmethod
    def irrepresentable_condition(A: np.ndarray, active_set: List[int]) -> float:
        """
        Check irrepresentable condition for LASSO variable selection.

        Parameters
        ----------
        A : ndarray
            Measurement matrix.
        active_set : list of int
            Indices of active variables.

        Returns
        -------
        irrepresentable : float
            Irrepresentable condition value (should be < 1).
        """
        n_features = A.shape[1]
        inactive_set = [i for i in range(n_features) if i not in active_set]

        if not active_set or not inactive_set:
            return 0.0

        # Normalize columns
        A_normalized = A / (np.linalg.norm(A, axis=0) + 1e-12)

        A_active = A_normalized[:, active_set]
        A_inactive = A_normalized[:, inactive_set]

        try:
            # Gram matrix for active set
            gram_active = A_active.T @ A_active

            # Cross-correlation between active and inactive
            cross_corr = A_inactive.T @ A_active

            # Irrepresentable condition
            inv_gram = np.linalg.pinv(gram_active)
            condition_matrix = cross_corr @ inv_gram

            # Maximum L1 norm of rows
            irrepresentable = np.max(np.sum(np.abs(condition_matrix), axis=1))

            return float(irrepresentable)

        except:
            return np.inf

    @staticmethod
    def coherence_recovery_bounds(A: np.ndarray, mu: float) -> Dict[str, int]:
        """
        Compute recovery bounds based on mutual incoherence.

        Parameters
        ----------
        A : ndarray
            Measurement matrix.
        mu : float
            Mutual incoherence value.

        Returns
        -------
        bounds : dict
            Recovery bounds.
        """
        # Exact recovery bound
        exact_bound = int(np.floor((1 + 1/mu) / 2)) if mu > 0 else 1

        # Stable recovery bound (typically larger)
        stable_bound = int(np.floor(0.7 * (1 + 1/mu))) if mu > 0 else 1

        return {
            'exact_recovery_bound': max(exact_bound, 1),
            'stable_recovery_bound': max(stable_bound, 1)
        }

    @staticmethod
    def rip_recovery_bounds(delta_2s: float, s: int) -> Dict[str, Any]:
        """
        Compute recovery bounds based on RIP.

        Parameters
        ----------
        delta_2s : float
            RIP constant for sparsity 2s.
        s : int
            Sparsity level.

        Returns
        -------
        bounds : dict
            Recovery guarantees based on RIP.
        """
        # Standard RIP condition for exact recovery
        rip_threshold = np.sqrt(2) - 1  # ≈ 0.414

        recovery_guarantee = delta_2s < rip_threshold

        return {
            'recovery_guarantee': bool(recovery_guarantee),
            'rip_threshold': float(rip_threshold),
            'actual_rip': float(delta_2s),
            'sparsity_level': int(s)
        }


theoretical_conditions = TheoreticalConditions()