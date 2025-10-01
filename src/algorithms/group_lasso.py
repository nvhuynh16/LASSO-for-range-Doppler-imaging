"""
Group LASSO algorithm for radar range-Doppler imaging.

Group LASSO extends standard LASSO to enforce group sparsity, where variables
are selected or discarded together in predefined groups. This is particularly
useful for radar applications where targets may span multiple range or Doppler
bins, creating natural grouping structures.

The optimization problem is:
min_{x} (1/2)||Ax - y||_2^2 + λ Σ_g w_g ||x_g||_2

where x_g represents the subvector of x corresponding to group g, and w_g are
group-specific weights.

References
----------
.. [1] Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression
       with grouped variables. Journal of the Royal Statistical Society, 68(1), 49-67.
.. [2] Meier, L., Van De Geer, S., & Bühlmann, P. (2008). The group lasso for
       logistic regression. Journal of the Royal Statistical Society, 70(1), 53-71.
.. [3] Simon, N., Friedman, J., Hastie, T., & Tibshirani, R. (2013). A sparse-group
       lasso. Journal of Computational and Graphical Statistics, 22(2), 231-245.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings

from .lasso_core import LassoRadar


class GroupLassoRadar(LassoRadar):
    """
    Group LASSO solver for structured sparse radar range-Doppler imaging.

    This class implements Group LASSO, which encourages sparsity at the group level
    rather than individual variables. Groups can represent spatial neighborhoods,
    frequency bands, or other structured patterns in radar data.

    Parameters
    ----------
    lambda_reg : float, default=0.01
        Overall regularization strength.
    max_iterations : int, default=1000
        Maximum number of iterations for the optimization algorithm.
    tolerance : float, default=1e-6
        Convergence tolerance.
    verbose : bool, default=False
        If True, print convergence information.
    warm_start : bool, default=False
        If True, reuse previous solution as initialization.
    adaptive_weights : bool, default=False
        If True, use adaptive weights based on group sizes.

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
        Estimated coefficients after fitting.
    groups_ : list of tuples
        Group structure used in the last fit.
    group_weights_ : ndarray
        Weights applied to each group.
    n_iterations_ : int
        Number of iterations performed during optimization.
    converged_ : bool
        True if the algorithm converged.

    Examples
    --------
    >>> import numpy as np
    >>> from lasso_radar.algorithms.group_lasso import GroupLassoRadar
    >>>
    >>> # Generate radar data with natural grouping
    >>> n_measurements, n_features = 100, 200
    >>> A = np.random.randn(n_measurements, n_features) / np.sqrt(n_measurements)
    >>>
    >>> # Create target scene with extended targets (groups)
    >>> true_scene = np.zeros(n_features)
    >>> true_scene[20:25] = [1.0, 0.8, 0.9, 0.7, 0.6]  # Group 1: Extended target
    >>> true_scene[80:85] = [0.6, 0.7, 0.8, 0.5, 0.4]  # Group 2: Another target
    >>> y = A @ true_scene + 0.01 * np.random.randn(n_measurements)
    >>>
    >>> # Define groups (e.g., spatial neighborhoods)
    >>> groups = []
    >>> for i in range(0, n_features, 5):
    ...     group_indices = list(range(i, min(i+5, n_features)))
    ...     groups.append((i//5, group_indices))
    >>>
    >>> # Fit Group LASSO
    >>> group_lasso = GroupLassoRadar(lambda_reg=0.01)
    >>> group_lasso.fit(A, y, groups=groups)
    """

    def __init__(
        self,
        lambda_reg: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False,
        warm_start: bool = False,
        adaptive_weights: bool = False
    ):
        super().__init__(
            lambda_reg=lambda_reg,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose,
            warm_start=warm_start
        )
        self.adaptive_weights = adaptive_weights

        # Group-specific attributes
        self.groups_ = None
        self.group_weights_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: List[Tuple[int, List[int]]],
        warm_start: Optional[bool] = None
    ) -> 'GroupLassoRadar':
        """
        Fit the Group LASSO model to radar measurement data.

        Parameters
        ----------
        X : array-like of shape (n_measurements, n_features)
            Measurement matrix.
        y : array-like of shape (n_measurements,)
            Observation vector.
        groups : list of tuples
            List of (group_id, feature_indices) tuples defining the group structure.
            Each tuple contains:
            - group_id: Unique identifier for the group
            - feature_indices: List of feature indices belonging to this group
        warm_start : bool, optional
            Override instance warm_start parameter.

        Returns
        -------
        self : object
            Returns the instance itself.

        Examples
        --------
        >>> # Define spatial groups for range-Doppler processing
        >>> groups = [
        ...     (0, [0, 1, 2, 3]),      # Range bins 0-3
        ...     (1, [4, 5, 6, 7]),      # Range bins 4-7
        ...     (2, [8, 9, 10, 11]),    # Range bins 8-11
        ... ]
        >>> group_lasso.fit(X, y, groups=groups)
        """
        # Validate inputs
        X, y = self._validate_inputs(X, y)
        groups = self._validate_groups(groups, X.shape[1])

        self.groups_ = groups

        # Initialize group weights
        self._initialize_group_weights(groups)

        # Call parent fit method with group-specific coordinate descent
        return super().fit(X, y, warm_start)

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and condition inputs for numerical stability."""
        X, y = super()._validate_inputs(X, y)

        # Additional validation for Group LASSO
        if X.shape[1] > 1000:
            warnings.warn("Large feature count may cause numerical instability in Group LASSO")

        # Check for extreme values that could cause overflow
        if np.max(np.abs(X)) > 1e3:
            warnings.warn("Large input values detected. Consider scaling features.")

        return X, y

    def _validate_groups(
        self,
        groups: List[Tuple[int, List[int]]],
        n_features: int
    ) -> List[Tuple[int, List[int]]]:
        """Validate group structure."""
        if not groups:
            raise ValueError("Groups list cannot be empty")

        # Check group structure
        all_indices = set()
        for group_id, indices in groups:
            if not isinstance(indices, (list, tuple, np.ndarray)):
                raise ValueError(f"Group {group_id} indices must be list, tuple, or array")

            indices = list(indices)
            if not indices:
                warnings.warn(f"Group {group_id} is empty")
                continue

            # Check for valid indices
            if any(idx < 0 or idx >= n_features for idx in indices):
                raise ValueError(f"Group {group_id} contains invalid feature indices")

            # Check for overlapping groups (warn but allow)
            overlap = all_indices.intersection(set(indices))
            if overlap:
                warnings.warn(f"Group {group_id} overlaps with previous groups at indices {overlap}")

            all_indices.update(indices)

        # Check coverage
        missing_features = set(range(n_features)) - all_indices
        if missing_features:
            warnings.warn(f"Features {missing_features} not assigned to any group")

        return groups

    def _initialize_group_weights(self, groups: List[Tuple[int, List[int]]]) -> None:
        """Initialize group weights."""
        n_groups = len(groups)
        self.group_weights_ = np.ones(n_groups)

        if self.adaptive_weights:
            # Set weights proportional to sqrt(group_size)
            for i, (group_id, indices) in enumerate(groups):
                self.group_weights_[i] = np.sqrt(len(indices))

    def _coordinate_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        XTX: np.ndarray,
        XTy: np.ndarray
    ) -> None:
        """
        Perform block coordinate descent for Group LASSO.

        Instead of updating individual coordinates, this method updates
        entire groups using the group soft thresholding operator.
        """
        n_features = X.shape[1]
        coeffs = self.coefficients_.copy()

        # Track convergence
        prev_objective = self._compute_objective(X, y, coeffs)

        for iteration in range(self.max_iterations):
            coeffs_old = coeffs.copy()

            # Update each group
            for group_idx, (group_id, indices) in enumerate(self.groups_):
                if not indices:
                    continue

                # Extract group information
                group_indices = np.array(indices)
                group_size = len(group_indices)

                # Compute residual for this group
                other_contribution = XTX[np.ix_(group_indices, range(n_features))] @ coeffs
                group_contribution = XTX[np.ix_(group_indices, group_indices)] @ coeffs[group_indices]
                residual = XTy[group_indices] - other_contribution + group_contribution

                # Group soft thresholding
                group_weight = self.group_weights_[group_idx]
                threshold = self.lambda_reg * group_weight

                # Solve group subproblem: (X_g^T X_g + μI) β_g = residual - threshold * β_g/||β_g||
                XgTXg = XTX[np.ix_(group_indices, group_indices)]

                # Use iterative method for group update
                coeffs[group_indices] = self._group_soft_threshold(
                    XgTXg, residual, threshold
                )

            # Check convergence
            current_objective = self._compute_objective(X, y, coeffs)

            # Check for numerical issues
            if not np.isfinite(current_objective):
                warnings.warn("Numerical instability detected. Stopping optimization.")
                break

            relative_change = abs(current_objective - prev_objective) / (abs(prev_objective) + 1e-10)

            if relative_change < self.tolerance:
                self.converged_ = True
                break

            prev_objective = current_objective

            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Objective = {current_objective:.6f}")

        self.coefficients_ = coeffs
        self.n_iterations_ = iteration + 1

        if not self.converged_ and self.verbose:
            warnings.warn(f"Group LASSO did not converge after {self.max_iterations} iterations")

    def _group_soft_threshold(
        self,
        XgTXg: np.ndarray,
        residual: np.ndarray,
        threshold: float,
        group_iterations: int = 10
    ) -> np.ndarray:
        """
        Apply group soft thresholding operator.

        Solves: min_β (1/2) β^T XgTXg β - residual^T β + threshold ||β||_2

        Parameters
        ----------
        XgTXg : ndarray
            X_g^T @ X_g for group g.
        residual : ndarray
            Residual vector for the group.
        threshold : float
            Group threshold parameter.
        group_iterations : int
            Number of iterations for group subproblem.

        Returns
        -------
        beta_group : ndarray
            Updated group coefficients.
        """
        group_size = len(residual)

        if group_size == 0:
            return np.array([])

        # Initialize with least squares solution if threshold is small
        try:
            if threshold < 1e-10:
                return np.linalg.solve(XgTXg + 1e-10 * np.eye(group_size), residual)
        except np.linalg.LinAlgError:
            pass

        # Iterative soft thresholding for group
        beta = np.zeros(group_size)

        for _ in range(group_iterations):
            # Gradient step
            grad = XgTXg @ beta - residual

            # Check for numerical issues
            if not np.all(np.isfinite(grad)):
                warnings.warn("Numerical instability in group gradient computation")
                break

            # Step size (conservative)
            if group_size > 0:
                eigenvalue_est = np.trace(XgTXg) / group_size
                step_size = 1.0 / (eigenvalue_est + 1e-6)  # More conservative regularization
            else:
                step_size = 1.0

            # Update
            beta_new = beta - step_size * grad

            # Check for numerical issues
            if not np.all(np.isfinite(beta_new)):
                warnings.warn("Numerical instability in group update step")
                break

            # Group soft thresholding
            beta_norm = np.linalg.norm(beta_new)
            if beta_norm > threshold * step_size:
                beta = beta_new * (1 - threshold * step_size / beta_norm)
            else:
                beta = np.zeros(group_size)

        return beta

    def _compute_objective(self, X: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> float:
        """Compute Group LASSO objective function value."""
        residual = X @ coeffs - y
        data_fit = 0.5 * np.sum(residual**2)

        # Group penalty
        group_penalty = 0.0
        for group_idx, (group_id, indices) in enumerate(self.groups_):
            if indices:
                group_norm = np.linalg.norm(coeffs[indices])
                group_weight = self.group_weights_[group_idx]
                group_penalty += self.lambda_reg * group_weight * group_norm

        return data_fit + group_penalty

    def get_active_groups(self, threshold: float = 1e-6) -> List[Tuple[int, List[int]]]:
        """
        Get groups with non-zero coefficients.

        Parameters
        ----------
        threshold : float, default=1e-6
            Threshold below which groups are considered inactive.

        Returns
        -------
        active_groups : list of tuples
            List of (group_id, feature_indices) for active groups.
        """
        if self.coefficients_ is None:
            raise AttributeError("Model must be fitted before getting active groups")

        active_groups = []
        for group_id, indices in self.groups_:
            if indices:
                group_norm = np.linalg.norm(self.coefficients_[indices])
                if group_norm > threshold:
                    active_groups.append((group_id, indices))

        return active_groups

    def compute_group_sparsity_metrics(self, threshold: float = 1e-6) -> Dict[str, Any]:
        """
        Compute group-level sparsity metrics.

        Parameters
        ----------
        threshold : float, default=1e-6
            Threshold for determining active groups.

        Returns
        -------
        metrics : dict
            Dictionary containing group sparsity metrics:
            - 'n_active_groups': Number of active groups
            - 'group_sparsity_ratio': Fraction of inactive groups
            - 'avg_group_norm': Average norm of active groups
            - 'group_norms': Norms of all groups
        """
        if self.coefficients_ is None:
            raise AttributeError("Model must be fitted before computing metrics")

        group_norms = []
        active_count = 0

        for group_id, indices in self.groups_:
            if indices:
                group_norm = np.linalg.norm(self.coefficients_[indices])
                group_norms.append(group_norm)
                if group_norm > threshold:
                    active_count += 1
            else:
                group_norms.append(0.0)

        total_groups = len(self.groups_)
        group_sparsity_ratio = 1 - (active_count / total_groups) if total_groups > 0 else 0

        active_norms = [norm for norm in group_norms if norm > threshold]
        avg_group_norm = np.mean(active_norms) if active_norms else 0.0

        return {
            'n_active_groups': active_count,
            'group_sparsity_ratio': float(group_sparsity_ratio),
            'avg_group_norm': float(avg_group_norm),
            'group_norms': group_norms
        }

    def compute_within_group_sparsity(self, threshold: float = 1e-6) -> Dict[str, Any]:
        """
        Compute within-group sparsity statistics.

        This analyzes how sparse the solution is within active groups,
        which can indicate if the group structure is appropriate.

        Parameters
        ----------
        threshold : float, default=1e-6
            Threshold for determining non-zero coefficients.

        Returns
        -------
        sparsity_stats : dict
            Dictionary containing within-group sparsity statistics.
        """
        if self.coefficients_ is None:
            raise AttributeError("Model must be fitted before computing within-group sparsity")

        within_group_sparsity = []
        active_groups = self.get_active_groups(threshold)

        for group_id, indices in active_groups:
            group_coeffs = self.coefficients_[indices]
            non_zero_count = np.sum(np.abs(group_coeffs) > threshold)
            group_size = len(indices)
            sparsity = 1 - (non_zero_count / group_size) if group_size > 0 else 0
            within_group_sparsity.append(sparsity)

        if within_group_sparsity:
            avg_within_sparsity = np.mean(within_group_sparsity)
            std_within_sparsity = np.std(within_group_sparsity)
        else:
            avg_within_sparsity = 0.0
            std_within_sparsity = 0.0

        return {
            'avg_within_group_sparsity': float(avg_within_sparsity),
            'std_within_group_sparsity': float(std_within_sparsity),
            'within_group_sparsities': within_group_sparsity
        }

    def create_spatial_groups(
        self,
        n_range: int,
        n_doppler: int,
        group_size: Tuple[int, int] = (4, 4),
        overlap: Tuple[int, int] = (0, 0)
    ) -> List[Tuple[int, List[int]]]:
        """
        Create spatial groups for range-Doppler processing.

        This utility method creates rectangular groups in the range-Doppler space,
        which is common for radar applications where targets may span multiple
        adjacent bins.

        Parameters
        ----------
        n_range : int
            Number of range bins.
        n_doppler : int
            Number of Doppler bins.
        group_size : tuple of int, default=(4, 4)
            Size of each group as (range_size, doppler_size).
        overlap : tuple of int, default=(0, 0)
            Overlap between groups as (range_overlap, doppler_overlap).

        Returns
        -------
        groups : list of tuples
            List of (group_id, feature_indices) for the spatial groups.

        Examples
        --------
        >>> # Create 4x4 groups with 2-bin overlap
        >>> groups = group_lasso.create_spatial_groups(64, 32, (4, 4), (2, 2))
        >>> group_lasso.fit(X, y, groups=groups)
        """
        range_size, doppler_size = group_size
        range_overlap, doppler_overlap = overlap

        range_step = range_size - range_overlap
        doppler_step = doppler_size - doppler_overlap

        groups = []
        group_id = 0

        for r_start in range(0, n_range, range_step):
            for d_start in range(0, n_doppler, doppler_step):
                # Define group boundaries
                r_end = min(r_start + range_size, n_range)
                d_end = min(d_start + doppler_size, n_doppler)

                # Convert 2D indices to 1D feature indices
                indices = []
                for r in range(r_start, r_end):
                    for d in range(d_start, d_end):
                        feature_idx = r * n_doppler + d
                        indices.append(feature_idx)

                if indices:  # Only add non-empty groups
                    groups.append((group_id, indices))
                    group_id += 1

        return groups

    def get_params(self) -> Dict[str, Any]:
        """Get parameters including adaptive_weights."""
        params = super().get_params()
        params['adaptive_weights'] = self.adaptive_weights
        return params

    def set_params(self, **params) -> 'GroupLassoRadar':
        """Set parameters including adaptive_weights."""
        if 'adaptive_weights' in params:
            self.adaptive_weights = params.pop('adaptive_weights')
        return super().set_params(**params)

    def __repr__(self) -> str:
        """String representation including group-specific parameters."""
        params = self.get_params()
        param_str = ', '.join(f"{k}={v}" for k, v in params.items())
        return f"GroupLassoRadar({param_str})"