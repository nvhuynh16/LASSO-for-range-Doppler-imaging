"""
Elastic Net algorithm for radar range-Doppler imaging.

The Elastic Net combines L1 (LASSO) and L2 (Ridge) regularization to handle
correlated features better than pure LASSO. This is particularly useful in
radar applications where adjacent range-Doppler cells may be correlated.

The optimization problem is:
min_{x} (1/2)||Ax - y||_2^2 + λ(α||x||_1 + (1-α)/2 ||x||_2^2)

where α controls the balance between L1 and L2 regularization.

References
----------
.. [1] Zou, H., & Hastie, T. (2005). Regularization and variable selection via the
       elastic net. Journal of the Royal Statistical Society, 67(2), 301-320.
.. [2] Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
       generalized linear models via coordinate descent. Journal of Statistical
       Software, 33(1), 1-22.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
import warnings

from .lasso_core import LassoRadar


class ElasticNetRadar(LassoRadar):
    """
    Elastic Net solver for sparse radar range-Doppler imaging.

    This class extends the basic LASSO solver to include L2 regularization,
    creating the Elastic Net penalty. This helps with grouped variable selection
    and handling correlated features common in radar applications.

    Parameters
    ----------
    lambda_reg : float, default=0.01
        Overall regularization strength (λ in the objective function).
    alpha : float, default=0.5
        Mixing parameter between L1 and L2 regularization:
        - alpha=1.0: Pure LASSO (L1 only)
        - alpha=0.0: Pure Ridge (L2 only)
        - 0 < alpha < 1: Elastic Net combination
    max_iterations : int, default=1000
        Maximum number of coordinate descent iterations.
    tolerance : float, default=1e-6
        Convergence tolerance for optimization.
    verbose : bool, default=False
        If True, print convergence information.
    warm_start : bool, default=False
        If True, reuse previous solution as initialization.

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
        Estimated sparse coefficients after fitting.
    n_iterations_ : int
        Number of iterations performed during optimization.
    converged_ : bool
        True if the algorithm converged.
    dual_gap_ : float
        Final duality gap (approximated for Elastic Net).

    Examples
    --------
    >>> import numpy as np
    >>> from lasso_radar.algorithms.elastic_net import ElasticNetRadar
    >>>
    >>> # Generate correlated radar data
    >>> n_measurements, n_features = 100, 200
    >>> A = np.random.randn(n_measurements, n_features)
    >>> # Add correlation between adjacent features
    >>> for i in range(n_features-1):
    ...     A[:, i+1] = 0.7 * A[:, i] + 0.3 * A[:, i+1]
    >>> A = A / np.linalg.norm(A, axis=0)
    >>>
    >>> # Sparse target with grouped structure
    >>> true_scene = np.zeros(n_features)
    >>> true_scene[50:55] = [1.0, 0.8, 0.9, 0.7, 0.6]  # Grouped targets
    >>> y = A @ true_scene + 0.01 * np.random.randn(n_measurements)
    >>>
    >>> # Fit Elastic Net (balances L1 and L2)
    >>> elastic_net = ElasticNetRadar(lambda_reg=0.01, alpha=0.5)
    >>> elastic_net.fit(A, y)
    >>>
    >>> # Compare with pure LASSO
    >>> lasso = ElasticNetRadar(lambda_reg=0.01, alpha=1.0)
    >>> lasso.fit(A, y)
    """

    def __init__(
        self,
        lambda_reg: float = 0.01,
        alpha: float = 0.5,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False,
        warm_start: bool = False
    ):
        # Set alpha before calling super().__init__() because _validate_parameters() needs it
        self.alpha = alpha
        super().__init__(
            lambda_reg=lambda_reg,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose,
            warm_start=warm_start
        )

    def _validate_alpha(self) -> None:
        """Validate the alpha mixing parameter."""
        if not 0 <= self.alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")

    def _validate_parameters(self) -> None:
        """Validate all parameters including alpha."""
        super()._validate_parameters()
        self._validate_alpha()

    def _coordinate_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        XTX: np.ndarray,
        XTy: np.ndarray
    ) -> None:
        """
        Perform coordinate descent optimization for Elastic Net.

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

        # Compute regularization parameters
        l1_reg = self.alpha * self.lambda_reg
        l2_reg = (1 - self.alpha) * self.lambda_reg

        # Track convergence
        prev_objective = self._compute_objective(X, y, coeffs)

        for iteration in range(self.max_iterations):
            coeffs_old = coeffs.copy()

            # Update each coordinate
            for j in range(n_features):
                # Compute residual excluding j-th feature
                residual = XTy[j] - XTX[j, :] @ coeffs + XTX[j, j] * coeffs[j]

                # Elastic Net coordinate update
                denominator = XTX[j, j] + l2_reg
                if denominator > 0:
                    coeffs[j] = self._soft_threshold(residual, l1_reg) / denominator

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
        self.dual_gap_ = self._compute_approximate_dual_gap(X, y, coeffs)

        if not self.converged_ and self.verbose:
            warnings.warn(f"Elastic Net did not converge after {self.max_iterations} iterations")

    def _compute_objective(self, X: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Compute Elastic Net objective function value."""
        residual = X @ coeffs - y
        data_fit = 0.5 * np.sum(residual**2)

        # Elastic Net penalty
        l1_penalty = self.alpha * self.lambda_reg * np.sum(np.abs(coeffs))
        l2_penalty = 0.5 * (1 - self.alpha) * self.lambda_reg * np.sum(coeffs**2)

        return data_fit + l1_penalty + l2_penalty

    def _compute_approximate_dual_gap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coeffs: np.ndarray
    ) -> float:
        """
        Compute approximate duality gap for Elastic Net.

        Note: The exact dual for Elastic Net is more complex than LASSO.
        This provides an approximation for convergence monitoring.
        """
        residual = X @ coeffs - y
        gradient = X.T @ residual + (1 - self.alpha) * self.lambda_reg * coeffs

        # Approximate dual variable (simplified)
        l1_reg = self.alpha * self.lambda_reg
        max_grad = np.max(np.abs(gradient))

        if max_grad > l1_reg:
            dual_var = (l1_reg / max_grad) * residual
        else:
            dual_var = residual

        # Approximate primal-dual gap
        primal_obj = self._compute_objective(X, y, coeffs)
        dual_obj = -0.5 * np.sum(dual_var**2) + np.sum(y * dual_var)

        return max(primal_obj - dual_obj, 0.0)

    def compute_solution_path(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha_values: Optional[np.ndarray] = None,
        n_alphas: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Elastic Net solution path across different alpha values.

        This method fits the model for different mixing parameters α, keeping
        the overall regularization strength λ fixed. Useful for understanding
        the transition from Ridge (α=0) to LASSO (α=1) behavior.

        Parameters
        ----------
        X : array-like of shape (n_measurements, n_features)
            Measurement matrix.
        y : array-like of shape (n_measurements,)
            Observation vector.
        alpha_values : array-like, optional
            Array of alpha values to use. If None, creates linearly spaced values.
        n_alphas : int, default=50
            Number of alpha values to use if alpha_values is None.

        Returns
        -------
        alpha_path : ndarray of shape (n_alphas,)
            Alpha values used in the path.
        coef_path : ndarray of shape (n_features, n_alphas)
            Coefficient values along the path.

        Examples
        --------
        >>> # Compute solution path
        >>> alpha_path, coef_path = elastic_net.compute_solution_path(X, y)
        >>>
        >>> # Plot coefficient evolution
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(10, 6))
        >>> plt.plot(alpha_path, coef_path.T)
        >>> plt.xlabel('Alpha (L1 ratio)')
        >>> plt.ylabel('Coefficient value')
        >>> plt.title('Elastic Net Solution Path')
        """
        X, y = self._validate_inputs(X, y)

        if alpha_values is None:
            alpha_values = np.linspace(0, 1, n_alphas)
        else:
            alpha_values = np.asarray(alpha_values)

        n_features = X.shape[1]
        coef_path = np.zeros((n_features, len(alpha_values)))

        # Store original alpha
        original_alpha = self.alpha

        for i, alpha in enumerate(alpha_values):
            self.alpha = alpha
            self.fit(X, y, warm_start=(i > 0))
            coef_path[:, i] = self.coefficients_.copy()

        # Restore original alpha
        self.alpha = original_alpha

        return alpha_values, coef_path

    def compute_regularization_path(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_values: Optional[np.ndarray] = None,
        n_lambdas: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute regularization path across different lambda values.

        Parameters
        ----------
        X : array-like of shape (n_measurements, n_features)
            Measurement matrix.
        y : array-like of shape (n_measurements,)
            Observation vector.
        lambda_values : array-like, optional
            Array of lambda values. If None, creates log-spaced values.
        n_lambdas : int, default=50
            Number of lambda values if lambda_values is None.

        Returns
        -------
        lambda_path : ndarray of shape (n_lambdas,)
            Lambda values used in the path.
        coef_path : ndarray of shape (n_features, n_lambdas)
            Coefficient values along the path.
        """
        X, y = self._validate_inputs(X, y)

        if lambda_values is None:
            # Compute lambda_max (smallest lambda that gives zero solution)
            XTy = X.T @ y
            lambda_max = np.max(np.abs(XTy)) / self.alpha if self.alpha > 0 else 1.0
            lambda_min = 0.01 * lambda_max
            lambda_values = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n_lambdas)
        else:
            lambda_values = np.asarray(lambda_values)

        # Sort in decreasing order for warm starts
        lambda_values = np.sort(lambda_values)[::-1]

        n_features = X.shape[1]
        coef_path = np.zeros((n_features, len(lambda_values)))

        # Store original lambda
        original_lambda = self.lambda_reg

        for i, lam in enumerate(lambda_values):
            self.lambda_reg = lam
            self.fit(X, y, warm_start=(i > 0))
            coef_path[:, i] = self.coefficients_.copy()

        # Restore original lambda
        self.lambda_reg = original_lambda

        return lambda_values, coef_path

    def compute_effective_sparsity(self, threshold: float = 1e-6) -> Dict[str, float]:
        """
        Compute effective sparsity metrics for the Elastic Net solution.

        Parameters
        ----------
        threshold : float, default=1e-6
            Threshold below which coefficients are considered zero.

        Returns
        -------
        sparsity_metrics : dict
            Dictionary containing sparsity measures:
            - 'l0_norm': Number of non-zero coefficients
            - 'sparsity_ratio': Fraction of zero coefficients
            - 'l1_norm': L1 norm of coefficients
            - 'l2_norm': L2 norm of coefficients
            - 'effective_df': Effective degrees of freedom
        """
        if self.coefficients_ is None:
            raise AttributeError("Model must be fitted before computing sparsity")

        coeffs = self.coefficients_
        non_zero_mask = np.abs(coeffs) > threshold

        l0_norm = np.sum(non_zero_mask)
        sparsity_ratio = 1 - (l0_norm / len(coeffs))
        l1_norm = np.sum(np.abs(coeffs))
        l2_norm = np.sqrt(np.sum(coeffs**2))

        # Effective degrees of freedom (approximation)
        effective_df = l0_norm + (1 - self.alpha) * l2_norm**2 / self.lambda_reg

        return {
            'l0_norm': int(l0_norm),
            'sparsity_ratio': float(sparsity_ratio),
            'l1_norm': float(l1_norm),
            'l2_norm': float(l2_norm),
            'effective_df': float(effective_df)
        }

    def compare_with_lasso(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare Elastic Net solution with pure LASSO.

        Parameters
        ----------
        X : array-like of shape (n_measurements, n_features)
            Measurement matrix.
        y : array-like of shape (n_measurements,)
            Observation vector.

        Returns
        -------
        comparison : dict
            Dictionary comparing Elastic Net and LASSO solutions:
            - 'elastic_net_coefs': Elastic Net coefficients
            - 'lasso_coefs': LASSO coefficients
            - 'elastic_net_sparsity': Elastic Net sparsity metrics
            - 'lasso_sparsity': LASSO sparsity metrics
            - 'correlation': Correlation between solutions
        """
        # Fit Elastic Net
        original_alpha = self.alpha
        self.fit(X, y)
        elastic_net_coefs = self.coefficients_.copy()
        elastic_net_sparsity = self.compute_effective_sparsity()

        # Fit pure LASSO (alpha=1)
        self.alpha = 1.0
        self.fit(X, y)
        lasso_coefs = self.coefficients_.copy()

        # Compute LASSO sparsity
        lasso_sparsity = self.compute_effective_sparsity()

        # Restore original alpha
        self.alpha = original_alpha

        # Compute correlation between solutions
        correlation = np.corrcoef(elastic_net_coefs, lasso_coefs)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        return {
            'elastic_net_coefs': elastic_net_coefs,
            'lasso_coefs': lasso_coefs,
            'elastic_net_sparsity': elastic_net_sparsity,
            'lasso_sparsity': lasso_sparsity,
            'correlation': float(correlation)
        }

    def get_params(self) -> Dict[str, Any]:
        """Get parameters including alpha."""
        params = super().get_params()
        params['alpha'] = self.alpha
        return params

    def set_params(self, **params) -> 'ElasticNetRadar':
        """Set parameters including alpha."""
        if 'alpha' in params:
            self.alpha = params.pop('alpha')
            self._validate_alpha()
        return super().set_params(**params)

    def __repr__(self) -> str:
        """String representation including alpha parameter."""
        params = self.get_params()
        param_str = ', '.join(f"{k}={v}" for k, v in params.items())
        return f"ElasticNetRadar({param_str})"