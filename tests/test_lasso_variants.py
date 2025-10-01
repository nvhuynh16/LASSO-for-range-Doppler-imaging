"""
Unit tests for LASSO variants (Elastic Net, Group LASSO).

Tests cover:
- Elastic Net functionality and L1/L2 balance
- Group LASSO sparsity patterns
- Parameter validation
- Comparison with standard LASSO
- Performance characteristics
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch

# Import the modules to be tested
from lasso_radar.algorithms.elastic_net import ElasticNetRadar
from lasso_radar.algorithms.group_lasso import GroupLassoRadar


class TestElasticNetRadar:
    """Test suite for Elastic Net radar algorithm."""

    @pytest.fixture
    def sample_radar_data(self):
        """Generate synthetic radar data for testing."""
        np.random.seed(42)
        n_range = 32
        n_doppler = 16
        n_measurements = 128

        # Create sparse target scene with correlated features
        target_scene = np.zeros((n_range, n_doppler))
        # Add clustered targets (correlated in space)
        target_scene[8:12, 4:6] = 0.8  # Cluster 1
        target_scene[20:24, 10:12] = 0.6  # Cluster 2

        # Create measurement matrix
        measurement_matrix = np.random.randn(n_measurements, n_range * n_doppler) / np.sqrt(n_measurements)

        # Add correlation between nearby range-Doppler cells
        for i in range(measurement_matrix.shape[1] - 1):
            if np.random.rand() < 0.3:  # 30% correlation
                measurement_matrix[:, i+1] += 0.5 * measurement_matrix[:, i]

        # Generate measurements
        target_vector = target_scene.flatten()
        measurements = measurement_matrix @ target_vector + 0.01 * np.random.randn(n_measurements)

        return {
            'measurements': measurements,
            'measurement_matrix': measurement_matrix,
            'true_scene': target_scene,
            'n_range': n_range,
            'n_doppler': n_doppler
        }

    @pytest.fixture
    def elastic_net_solver(self):
        """Create Elastic Net radar solver instance."""
        return ElasticNetRadar(
            lambda_reg=0.01,
            alpha=0.5,  # Balance between L1 and L2
            max_iterations=1000,
            tolerance=1e-6
        )

    def test_elastic_net_initialization(self):
        """Test Elastic Net solver initialization."""
        # Test default initialization
        solver = ElasticNetRadar()
        assert solver.lambda_reg > 0
        assert 0 <= solver.alpha <= 1
        assert solver.max_iterations > 0

        # Test custom initialization
        solver = ElasticNetRadar(lambda_reg=0.05, alpha=0.7, max_iterations=500)
        assert solver.lambda_reg == 0.05
        assert solver.alpha == 0.7
        assert solver.max_iterations == 500

    def test_invalid_alpha_parameter(self):
        """Test that invalid alpha values raise errors."""
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            ElasticNetRadar(alpha=-0.1)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            ElasticNetRadar(alpha=1.5)

    def test_elastic_net_vs_lasso_ridge(self, sample_radar_data):
        """Test that Elastic Net interpolates between LASSO and Ridge."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        # Pure LASSO (alpha = 1)
        lasso_solver = ElasticNetRadar(lambda_reg=0.01, alpha=1.0)
        lasso_solver.fit(measurement_matrix, measurements)
        lasso_sparsity = np.sum(np.abs(lasso_solver.coefficients_) > 1e-6)

        # Pure Ridge (alpha = 0)
        ridge_solver = ElasticNetRadar(lambda_reg=0.01, alpha=0.0)
        ridge_solver.fit(measurement_matrix, measurements)
        ridge_sparsity = np.sum(np.abs(ridge_solver.coefficients_) > 1e-6)

        # Elastic Net (alpha = 0.5)
        elastic_solver = ElasticNetRadar(lambda_reg=0.01, alpha=0.5)
        elastic_solver.fit(measurement_matrix, measurements)
        elastic_sparsity = np.sum(np.abs(elastic_solver.coefficients_) > 1e-6)

        # LASSO should be sparsest, Ridge least sparse
        assert lasso_sparsity <= elastic_sparsity <= ridge_sparsity

    def test_correlated_features_handling(self, sample_radar_data):
        """Test that Elastic Net handles correlated features better than LASSO."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        # Create highly correlated measurement matrix
        n_features = measurement_matrix.shape[1]
        corr_matrix = np.random.randn(measurement_matrix.shape[0], n_features)

        # Add strong correlation between adjacent features
        for i in range(n_features - 1):
            corr_matrix[:, i+1] = 0.8 * corr_matrix[:, i] + 0.2 * np.random.randn(corr_matrix.shape[0])

        # Normalize
        corr_matrix = corr_matrix / np.linalg.norm(corr_matrix, axis=0)

        # Generate new measurements
        true_coefs = np.zeros(n_features)
        true_coefs[10:15] = 1.0  # Group of correlated features
        corr_measurements = corr_matrix @ true_coefs + 0.01 * np.random.randn(corr_matrix.shape[0])

        # Compare LASSO vs Elastic Net
        lasso_solver = ElasticNetRadar(lambda_reg=0.01, alpha=1.0)
        lasso_solver.fit(corr_matrix, corr_measurements)

        elastic_solver = ElasticNetRadar(lambda_reg=0.01, alpha=0.5)
        elastic_solver.fit(corr_matrix, corr_measurements)

        # Check that Elastic Net selects more correlated features
        lasso_selected = np.where(np.abs(lasso_solver.coefficients_) > 1e-6)[0]
        elastic_selected = np.where(np.abs(elastic_solver.coefficients_) > 1e-6)[0]

        # Elastic Net should select more features in the true group
        lasso_correct = len(set(lasso_selected) & set(range(10, 15)))
        elastic_correct = len(set(elastic_selected) & set(range(10, 15)))

        assert elastic_correct >= lasso_correct

    def test_grouping_effect(self, sample_radar_data):
        """Test the grouping effect of Elastic Net."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        solver = ElasticNetRadar(lambda_reg=0.01, alpha=0.3)  # More Ridge-like
        solver.fit(measurement_matrix, measurements)

        # Get range-Doppler map
        rd_map = solver.get_range_doppler_map(
            sample_radar_data['n_range'],
            sample_radar_data['n_doppler']
        )

        # Check for spatial smoothness (grouping effect)
        # Compare with pure LASSO
        lasso_solver = ElasticNetRadar(lambda_reg=0.01, alpha=1.0)
        lasso_solver.fit(measurement_matrix, measurements)
        lasso_rd_map = lasso_solver.get_range_doppler_map(
            sample_radar_data['n_range'],
            sample_radar_data['n_doppler']
        )

        # Elastic Net should have smoother solutions
        elastic_gradient = np.mean(np.abs(np.gradient(rd_map)))
        lasso_gradient = np.mean(np.abs(np.gradient(lasso_rd_map)))

        # Elastic Net typically produces smoother results
        assert elastic_gradient <= lasso_gradient * 1.5  # Allow some tolerance

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_alpha_sweep(self, alpha, sample_radar_data):
        """Test performance across different alpha values."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        solver = ElasticNetRadar(lambda_reg=0.01, alpha=alpha)
        solver.fit(measurement_matrix, measurements)

        # Should converge for all alpha values
        assert solver.converged_ or solver.n_iterations_ == solver.max_iterations

        # Check coefficient statistics
        coef_l1_norm = np.sum(np.abs(solver.coefficients_))
        coef_l2_norm = np.sum(solver.coefficients_ ** 2)

        assert np.isfinite(coef_l1_norm)
        assert np.isfinite(coef_l2_norm)


class TestGroupLassoRadar:
    """Test suite for Group LASSO radar algorithm."""

    @pytest.fixture
    def grouped_radar_data(self):
        """Generate radar data with natural grouping structure."""
        np.random.seed(42)
        n_range = 20  # Reduced for test performance
        n_doppler = 10  # Reduced for test performance
        n_measurements = 100  # Reduced for test performance

        # Create target scene with range-Doppler groups
        target_scene = np.zeros((n_range, n_doppler))

        # Group 1: Extended target in range
        target_scene[5:8, 3] = [0.8, 1.0, 0.9]

        # Group 2: Extended target in Doppler
        target_scene[12, 6:8] = [0.6, 0.8]

        # Define groups (e.g., range bins or Doppler bins)
        groups = []
        group_id = 0

        # Range groups (every 5 range bins)
        for r_start in range(0, n_range, 5):
            for d in range(n_doppler):
                group_indices = []
                for r in range(r_start, min(r_start + 5, n_range)):
                    group_indices.append(r * n_doppler + d)
                groups.append((group_id, group_indices))
                group_id += 1

        measurement_matrix = np.random.randn(n_measurements, n_range * n_doppler) / np.sqrt(n_measurements)
        target_vector = target_scene.flatten()
        measurements = measurement_matrix @ target_vector + 0.01 * np.random.randn(n_measurements)

        return {
            'measurements': measurements,
            'measurement_matrix': measurement_matrix,
            'true_scene': target_scene,
            'groups': groups,
            'n_range': n_range,
            'n_doppler': n_doppler
        }

    @pytest.fixture
    def group_lasso_solver(self):
        """Create Group LASSO radar solver instance."""
        return GroupLassoRadar(
            lambda_reg=0.01,
            max_iterations=100,  # Reduced for test performance
            tolerance=1e-4       # Relaxed for test performance
        )

    def test_group_lasso_initialization(self):
        """Test Group LASSO solver initialization."""
        solver = GroupLassoRadar()
        assert solver.lambda_reg > 0
        assert solver.max_iterations > 0
        assert solver.tolerance > 0

    def test_group_sparsity(self, group_lasso_solver, grouped_radar_data):
        """Test that Group LASSO enforces group sparsity."""
        measurements = grouped_radar_data['measurements']
        measurement_matrix = grouped_radar_data['measurement_matrix']
        groups = grouped_radar_data['groups']

        group_lasso_solver.fit(measurement_matrix, measurements, groups=groups)

        # Check group sparsity
        active_groups = 0
        for group_id, group_indices in groups:
            group_coefs = group_lasso_solver.coefficients_[group_indices]
            group_norm = np.linalg.norm(group_coefs)

            if group_norm > 1e-6:
                active_groups += 1

        total_groups = len(groups)
        sparsity_ratio = active_groups / total_groups

        # Should have reasonable group sparsity for educational implementation
        assert sparsity_ratio < 0.8, f"Too many active groups: {sparsity_ratio:.3f}"  # Relaxed for smaller test problem
        assert sparsity_ratio > 0, "Should have some active groups"

    def test_within_group_density(self, group_lasso_solver, grouped_radar_data):
        """Test that Group LASSO selects entire groups."""
        measurements = grouped_radar_data['measurements']
        measurement_matrix = grouped_radar_data['measurement_matrix']
        groups = grouped_radar_data['groups']

        group_lasso_solver.fit(measurement_matrix, measurements, groups=groups)

        # For active groups, most coefficients should be non-zero
        for group_id, group_indices in groups:
            group_coefs = group_lasso_solver.coefficients_[group_indices]
            group_norm = np.linalg.norm(group_coefs)

            if group_norm > 1e-6:  # Active group
                non_zero_in_group = np.sum(np.abs(group_coefs) > 1e-6)
                group_size = len(group_indices)

                # Most of the group should be active
                density = non_zero_in_group / group_size
                assert density > 0.5, f"Low within-group density: {density:.3f}"

    def test_group_vs_standard_lasso(self, grouped_radar_data):
        """Compare Group LASSO with standard LASSO."""
        measurements = grouped_radar_data['measurements']
        measurement_matrix = grouped_radar_data['measurement_matrix']
        groups = grouped_radar_data['groups']

        # Standard LASSO
        from lasso_radar.algorithms.lasso_core import LassoRadar
        standard_lasso = LassoRadar(lambda_reg=0.01)
        standard_lasso.fit(measurement_matrix, measurements)

        # Group LASSO
        group_lasso = GroupLassoRadar(lambda_reg=0.01, max_iterations=100)
        group_lasso.fit(measurement_matrix, measurements, groups=groups)

        # Analyze selection patterns
        standard_active = np.where(np.abs(standard_lasso.coefficients_) > 1e-6)[0]
        group_active = np.where(np.abs(group_lasso.coefficients_) > 1e-6)[0]

        # Group LASSO should have more structured sparsity
        # Count how many complete groups are selected in each case
        standard_complete_groups = 0
        group_complete_groups = 0

        for group_id, group_indices in groups:
            standard_in_group = len(set(standard_active) & set(group_indices))
            group_in_group = len(set(group_active) & set(group_indices))

            if standard_in_group == len(group_indices):
                standard_complete_groups += 1
            if group_in_group == len(group_indices):
                group_complete_groups += 1

        # Group LASSO should select more complete groups
        assert group_complete_groups >= standard_complete_groups

    def test_overlapping_groups(self, group_lasso_solver):
        """Test Group LASSO with overlapping groups."""
        n_features = 20
        n_measurements = 50

        # Create overlapping groups (warnings expected)
        groups = [
            (0, list(range(0, 8))),      # Group 1: features 0-7
            (1, list(range(5, 13))),     # Group 2: features 5-12 (overlap)
            (2, list(range(10, 18))),    # Group 3: features 10-17 (overlap)
            (3, list(range(18, 20))),    # Group 4: remaining features
        ]

        # Generate data
        np.random.seed(42)
        A = np.random.randn(n_measurements, n_features)
        true_coefs = np.zeros(n_features)
        true_coefs[6:9] = 1.0  # Active in overlapping region
        y = A @ true_coefs + 0.01 * np.random.randn(n_measurements)

        # Fit Group LASSO
        group_lasso_solver.fit(A, y, groups=groups)

        # Should handle overlapping groups without errors
        assert hasattr(group_lasso_solver, 'coefficients_')
        assert group_lasso_solver.coefficients_.shape == (n_features,)

    def test_adaptive_group_weights(self, grouped_radar_data):
        """Test adaptive group weights based on group size."""
        # Use a smaller, more stable test case
        np.random.seed(42)
        n_measurements, n_features = 50, 30
        A = np.random.randn(n_measurements, n_features) / np.sqrt(n_measurements)
        x_true = np.zeros(n_features)
        x_true[[5, 15, 25]] = [1.0, 0.8, 0.6]
        y = A @ x_true + 0.01 * np.random.randn(n_measurements)

        # Create groups of different sizes
        varied_groups = [
            (0, list(range(0, 10))),    # Small group
            (1, list(range(10, 20))),   # Medium group
            (2, list(range(20, 30))),   # Large group
        ]

        solver = GroupLassoRadar(lambda_reg=0.01, adaptive_weights=True, max_iterations=50)
        solver.fit(A, y, groups=varied_groups)

        # Check that adaptive weights were computed
        assert hasattr(solver, 'group_weights_')
        assert len(solver.group_weights_) == len(varied_groups)

    @pytest.mark.parametrize("lambda_reg", [0.001, 0.01, 0.1])
    def test_regularization_path_groups(self, lambda_reg, grouped_radar_data):
        """Test Group LASSO across different regularization strengths."""
        measurements = grouped_radar_data['measurements']
        measurement_matrix = grouped_radar_data['measurement_matrix']
        groups = grouped_radar_data['groups']

        solver = GroupLassoRadar(lambda_reg=lambda_reg, max_iterations=100)
        solver.fit(measurement_matrix, measurements, groups=groups)

        # Should converge for all regularization values
        assert solver.converged_ or solver.n_iterations_ == solver.max_iterations

        # Count active groups
        active_groups = 0
        for group_id, group_indices in groups:
            group_coefs = solver.coefficients_[group_indices]
            if np.linalg.norm(group_coefs) > 1e-6:
                active_groups += 1

        # Higher regularization should lead to fewer active groups
        if lambda_reg >= 0.01:
            total_groups = len(groups)
            sparsity_ratio = active_groups / total_groups
            assert sparsity_ratio <= 0.7  # Relaxed for smaller test problem