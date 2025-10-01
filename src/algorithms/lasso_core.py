"""
Core LASSO (Least Absolute Shrinkage and Selection Operator) algorithm for radar applications.

This module implements the fundamental LASSO algorithm optimized for sparse radar
range-Doppler imaging. It includes coordinate descent optimization, theoretical
condition checking, and radar-specific enhancements.

References
----------
.. [1] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
       Journal of the Royal Statistical Society, 58(1), 267-288.
.. [2] Baraniuk, R. G. (2007). Compressive sensing. IEEE Signal Processing Magazine,
       24(4), 118-121.
.. [3] Herman, M. A., & Strohmer, T. (2009). High-resolution radar via compressed sensing.
       IEEE Transactions on Signal Processing, 57(6), 2275-2284.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm
from typing import Optional, Tuple, Dict, Any, Union
import warnings


class LassoRadar:
    """
    LASSO solver for sparse radar range-Doppler imaging.

    This class implements the LASSO (Least Absolute Shrinkage and Selection
    Operator) algorithm specifically designed for radar applications. It solves
    the optimization problem:

    min_{x} (1/2)||Ax - y||_2^2 + λ||x||_1

    where A is the measurement matrix, y is the observation vector, x is the
    sparse range-Doppler scene to recover, and λ is the regularization parameter.

    Parameters
    ----------
    lambda_reg : float, default=0.01
        Regularization parameter (λ) controlling sparsity. Higher values lead to
        sparser solutions.
    max_iterations : int, default=1000
        Maximum number of coordinate descent iterations.
    tolerance : float, default=1e-6
        Convergence tolerance for the optimization algorithm.
    verbose : bool, default=False
        If True, print convergence information during optimization.
    warm_start : bool, default=False
        If True, reuse the solution of the previous call to fit as initialization.

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
        Estimated sparse coefficients after fitting.
    n_iterations_ : int
        Number of iterations performed during optimization.
    converged_ : bool
        True if the algorithm converged, False otherwise.
    dual_gap_ : float
        Final duality gap, indicating optimization quality.

    Examples
    --------
    >>> import numpy as np
    >>> from lasso_radar.algorithms.lasso_core import LassoRadar
    >>>
    >>> # Generate synthetic radar data
    >>> n_range, n_doppler = 64, 32
    >>> n_measurements, n_features = 100, n_range * n_doppler
    >>> A = np.random.randn(n_measurements, n_features) / np.sqrt(n_measurements)
    >>> true_scene = np.zeros(n_features)
    >>> true_scene[50:55] = [1.0, 0.8, 0.6, 0.4, 0.2]  # Sparse targets
    >>> y = A @ true_scene + 0.01 * np.random.randn(n_measurements)
    >>>
    >>> # Fit LASSO model
    >>> lasso = LassoRadar(lambda_reg=0.01)
    >>> lasso.fit(A, y)
    >>>
    >>> # Get range-Doppler map
    >>> rd_map = lasso.get_range_doppler_map(n_range, n_doppler)
    """

    def __init__(
        self,
        lambda_reg: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False,
        warm_start: bool = False
    ):
        self.lambda_reg = lambda_reg
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.warm_start = warm_start

        # Validate parameters
        self._validate_parameters()

        # Initialize state variables
        self.coefficients_ = None
        self.n_iterations_ = 0
        self.converged_ = False
        self.dual_gap_ = np.inf

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.lambda_reg <= 0:
            raise ValueError("Lambda regularization must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        warm_start: Optional[bool] = None
    ) -> 'LassoRadar':
        """
        Fit the LASSO model to radar measurement data.

        Parameters
        ----------
        X : array-like of shape (n_measurements, n_features)
            Measurement matrix (sensing matrix) relating the sparse scene
            to the observations.
        y : array-like of shape (n_measurements,)
            Observation vector (radar measurements).
        warm_start : bool, optional
            Override the instance warm_start parameter for this fit.

        Returns
        -------
        self : object
            Returns the instance itself for method chaining.

        Raises
        ------
        ValueError
            If input dimensions are incompatible.
        """
        # Validate inputs
        X, y = self._validate_inputs(X, y)

        # Initialize coefficients
        if warm_start is None:
            warm_start = self.warm_start

        if not warm_start or self.coefficients_ is None:
            self.coefficients_ = np.zeros(X.shape[1])
        elif len(self.coefficients_) != X.shape[1]:
            warnings.warn("Warm start coefficients have wrong shape. Reinitializing.")
            self.coefficients_ = np.zeros(X.shape[1])

        # Precompute quantities for efficiency
        XTX = X.T @ X
        XTy = X.T @ y

        # Coordinate descent optimization
        self._coordinate_descent(X, y, XTX, XTy)

        return self

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and convert inputs to appropriate format."""
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != len(y):
            raise ValueError("Number of samples in X and y must match")
        if X.shape[0] == 0:
            raise ValueError("Empty input data")

        # Check for NaN/Inf values
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values")

        return X, y

    def _coordinate_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        XTX: np.ndarray,
        XTy: np.ndarray
    ) -> None:
        """
        Perform coordinate descent optimization.

        Parameters
        ----------
        X : ndarray
            Measurement matrix.
        y : ndarray
            Observation vector.
        XTX : ndarray
            Precomputed X^T @ X.
        XTy : ndarray
            Precomputed X^T @ y.
        """
        n_features = X.shape[1]
        coeffs = self.coefficients_.copy()

        # Track convergence
        prev_objective = self._compute_objective(X, y, coeffs)

        for iteration in range(self.max_iterations):
            coeffs_old = coeffs.copy()

            # Update each coordinate
            for j in range(n_features):
                # Compute residual excluding j-th feature
                residual = XTy[j] - XTX[j, :] @ coeffs + XTX[j, j] * coeffs[j]

                # Soft thresholding update
                if XTX[j, j] > 0:
                    coeffs[j] = self._soft_threshold(residual, self.lambda_reg) / XTX[j, j]

            # Check convergence
            current_objective = self._compute_objective(X, y, coeffs)
            relative_change = abs(current_objective - prev_objective) / (abs(prev_objective) + 1e-10)

            if relative_change < self.tolerance:
                self.converged_ = True
                break

            prev_objective = current_objective

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Objective = {current_objective:.6f}")

        self.coefficients_ = coeffs
        self.n_iterations_ = iteration + 1
        self.dual_gap_ = self._compute_dual_gap(X, y, coeffs)

        if not self.converged_ and self.verbose:
            warnings.warn(f"LASSO did not converge after {self.max_iterations} iterations")

    @staticmethod
    def _soft_threshold(x: float, threshold: float) -> float:
        """
        Apply soft thresholding operator.

        Parameters
        ----------
        x : float
            Input value.
        threshold : float
            Threshold parameter.

        Returns
        -------
        float
            Soft-thresholded value.
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _compute_objective(self, X: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Compute LASSO objective function value."""
        residual = X @ coeffs - y
        data_fit = 0.5 * np.sum(residual**2)
        regularization = self.lambda_reg * np.sum(np.abs(coeffs))
        return data_fit + regularization

    def _compute_dual_gap(self, X: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Compute duality gap for convergence assessment."""
        residual = X @ coeffs - y
        gradient = X.T @ residual

        # Dual variable
        max_grad = np.max(np.abs(gradient))
        if max_grad > self.lambda_reg:
            dual_var = (self.lambda_reg / max_grad) * residual
        else:
            dual_var = residual

        # Primal and dual objectives
        primal_obj = self._compute_objective(X, y, coeffs)
        dual_obj = -0.5 * np.sum(dual_var**2) + np.sum(y * dual_var)

        return primal_obj - dual_obj

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted LASSO model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        if self.coefficients_ is None:
            raise AttributeError("Model must be fitted before making predictions")

        X = np.asarray(X)
        return X @ self.coefficients_

    def get_range_doppler_map(self, n_range: int, n_doppler: int) -> np.ndarray:
        """
        Reshape coefficients into range-Doppler map format.

        Parameters
        ----------
        n_range : int
            Number of range bins.
        n_doppler : int
            Number of Doppler bins.

        Returns
        -------
        rd_map : ndarray of shape (n_doppler, n_range)
            Range-Doppler map with estimated target amplitudes.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        ValueError
            If dimensions don't match coefficient vector length.
        """
        if self.coefficients_ is None:
            raise AttributeError("Model must be fitted before extracting range-Doppler map")

        if n_range * n_doppler != len(self.coefficients_):
            raise ValueError(f"Dimensions {n_range}x{n_doppler} don't match coefficient length {len(self.coefficients_)}")

        return self.coefficients_.reshape(n_doppler, n_range)

    def compute_mutual_incoherence(self, A: np.ndarray) -> float:
        """
        Compute mutual incoherence of measurement matrix.

        The mutual incoherence μ(A) is the largest absolute correlation between
        any two distinct columns of the normalized measurement matrix:

        μ(A) = max_{i≠j} |⟨a_i, a_j⟩|

        where a_i are the normalized columns of A.

        Parameters
        ----------
        A : array-like of shape (n_measurements, n_features)
            Measurement matrix.

        Returns
        -------
        mu : float
            Mutual incoherence value between 0 and 1.

        Notes
        -----
        Lower mutual incoherence indicates better conditions for sparse recovery.
        For exact recovery of s-sparse signals, we typically need μ < 1/(2s-1).
        """
        A = np.asarray(A)

        # Normalize columns
        A_normalized = A / (norm(A, axis=0) + 1e-12)

        # Compute Gram matrix
        G = A_normalized.T @ A_normalized

        # Set diagonal to zero and find maximum off-diagonal element
        np.fill_diagonal(G, 0)
        mu = np.max(np.abs(G))

        return float(mu)

    def check_restricted_eigenvalue(self, A: np.ndarray, s: int) -> float:
        """
        Check restricted eigenvalue condition for sparse recovery guarantees.

        The restricted eigenvalue (RE) condition provides recovery guarantees for
        LASSO. For a sparsity level s, we compute:

        κ(s) = min_{||Δ||₀≤s, Δ≠0} ||AΔ||₂² / ||Δ||₂²

        Parameters
        ----------
        A : array-like of shape (n_measurements, n_features)
            Measurement matrix.
        s : int
            Sparsity level for the RE condition.

        Returns
        -------
        re_constant : float
            Restricted eigenvalue constant. Higher values indicate better conditions.

        Notes
        -----
        This is a computationally expensive operation for large matrices as it
        requires checking all possible s-sparse subsets. For practical applications,
        this is often approximated or computed only for small s values.
        """
        A = np.asarray(A)
        n_measurements, n_features = A.shape

        if s >= n_features:
            warnings.warn("Sparsity level s should be less than number of features")
            return 0.0

        # Check computational feasibility based on number of combinations
        from math import comb

        if s > 20:  # Computational limit for sparsity
            warnings.warn("RE computation expensive for large s. Using approximation.")
            return self._approximate_restricted_eigenvalue(A, s)

        # Check if number of combinations is too large
        max_combinations = 10000  # Reasonable limit for exact computation
        try:
            num_combinations = comb(n_features, s)
            if num_combinations > max_combinations:
                warnings.warn(f"RE computation expensive: {num_combinations:,} combinations. Using approximation.")
                return self._approximate_restricted_eigenvalue(A, s)
        except (OverflowError, ValueError):
            # If we can't even compute the number of combinations, use approximation
            warnings.warn("RE computation: too many combinations. Using approximation.")
            return self._approximate_restricted_eigenvalue(A, s)

        # Exact computation for feasible cases
        from itertools import combinations

        min_eigenvalue = np.inf
        count = 0  # Add counter for safety
        max_count = max_combinations  # Safety limit

        for indices in combinations(range(n_features), s):
            count += 1
            if count > max_count:  # Safety break
                warnings.warn("RE computation: hit safety limit. Using partial result.")
                break
            A_subset = A[:, list(indices)]

            # Compute smallest eigenvalue of A_subset^T @ A_subset
            if A_subset.shape[1] > 0:
                eigenvals = np.linalg.eigvals(A_subset.T @ A_subset)
                min_eigenvalue = min(min_eigenvalue, np.min(eigenvals))

        return float(max(min_eigenvalue, 0.0))

    def _approximate_restricted_eigenvalue(self, A: np.ndarray, s: int) -> float:
        """Approximate RE constant using random sampling."""
        n_measurements, n_features = A.shape
        n_trials = min(1000, 2**s)  # Limit number of trials

        min_eigenvalue = np.inf

        for _ in range(n_trials):
            # Random s-sparse vector
            indices = np.random.choice(n_features, s, replace=False)
            A_subset = A[:, indices]

            eigenvals = np.linalg.eigvals(A_subset.T @ A_subset)
            min_eigenvalue = min(min_eigenvalue, np.min(eigenvals))

        return float(max(min_eigenvalue, 0.0))

    def compute_recovery_bounds(self, A: np.ndarray) -> Dict[str, Any]:
        """
        Compute theoretical recovery bounds based on matrix properties.

        Parameters
        ----------
        A : array-like of shape (n_measurements, n_features)
            Measurement matrix.

        Returns
        -------
        bounds : dict
            Dictionary containing various recovery bounds and conditions:
            - 'mutual_incoherence': Mutual incoherence value
            - 'coherence_bound': Exact recovery bound based on mutual incoherence
            - 'stable_recovery_bound': Stable recovery bound
            - 'recommended_lambda': Recommended regularization parameter
        """
        mu = self.compute_mutual_incoherence(A)

        # Coherence-based exact recovery bound
        coherence_bound = int(np.floor((1 + 1/mu) / 2)) if mu > 0 else 1

        # Stable recovery bound (typically larger)
        stable_bound = int(np.floor(0.7 * (1 + 1/mu))) if mu > 0 else 1

        # Recommended lambda based on noise level estimation
        residual_std = 0.01  # Default estimate, should be provided by user
        recommended_lambda = 2 * residual_std * np.sqrt(2 * np.log(A.shape[1]))

        return {
            'mutual_incoherence': mu,
            'coherence_bound': coherence_bound,
            'stable_recovery_bound': stable_bound,
            'recommended_lambda': recommended_lambda
        }

    def set_params(self, **params) -> 'LassoRadar':
        """Set parameters for the LASSO solver."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        self._validate_parameters()
        return self

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for the LASSO solver."""
        return {
            'lambda_reg': self.lambda_reg,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'verbose': self.verbose,
            'warm_start': self.warm_start
        }

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² coefficient of determination.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R² coefficient of determination.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    def __repr__(self) -> str:
        """String representation of the LASSO solver."""
        params = self.get_params()
        param_str = ', '.join(f"{k}={v}" for k, v in params.items())
        return f"LassoRadar({param_str})"