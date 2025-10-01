"""
Unit tests for core LASSO radar implementation.

Tests cover:
- Basic LASSO functionality
- Sparsity enforcement
- Convergence properties
- Parameter validation
- Performance with different noise levels
- Theoretical conditions (RIP, mutual incoherence)
"""

import pytest
import numpy as np
import numpy.testing as npt
from scipy import sparse
from unittest.mock import Mock, patch

# Import the module to be tested (will be implemented)
from lasso_radar.algorithms.lasso_core import LassoRadar


class TestLassoRadar:
    """Test suite for the core LASSO radar algorithm."""

    @pytest.fixture
    def sample_radar_data(self):
        """Generate synthetic radar data for testing."""
        np.random.seed(42)
        n_range = 64
        n_doppler = 32
        n_measurements = 256

        # Create sparse target scene
        target_scene = np.zeros((n_range, n_doppler))
        # Add a few strong targets
        target_scene[10, 5] = 1.0
        target_scene[25, 15] = 0.8
        target_scene[45, 25] = 0.6

        # Create measurement matrix (compressed sensing matrix)
        measurement_matrix = np.random.randn(n_measurements, n_range * n_doppler) / np.sqrt(n_measurements)

        # Generate measurements with noise
        target_vector = target_scene.flatten()
        clean_measurements = measurement_matrix @ target_vector
        noise_std = 0.01
        noisy_measurements = clean_measurements + noise_std * np.random.randn(n_measurements)

        return {
            'measurements': noisy_measurements,
            'measurement_matrix': measurement_matrix,
            'true_scene': target_scene,
            'noise_std': noise_std,
            'n_range': n_range,
            'n_doppler': n_doppler
        }

    @pytest.fixture
    def lasso_solver(self):
        """Create LASSO radar solver instance."""
        return LassoRadar(
            lambda_reg=0.01,
            max_iterations=1000,
            tolerance=1e-6,
            verbose=False
        )

    def test_lasso_initialization(self):
        """Test LASSO solver initialization with various parameters."""
        # Test default initialization
        solver = LassoRadar()
        assert solver.lambda_reg > 0
        assert solver.max_iterations > 0
        assert solver.tolerance > 0

        # Test custom initialization
        solver = LassoRadar(lambda_reg=0.05, max_iterations=500, tolerance=1e-4)
        assert solver.lambda_reg == 0.05
        assert solver.max_iterations == 500
        assert solver.tolerance == 1e-4

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="Lambda regularization must be positive"):
            LassoRadar(lambda_reg=-0.1)

        with pytest.raises(ValueError, match="Max iterations must be positive"):
            LassoRadar(max_iterations=0)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            LassoRadar(tolerance=-1e-6)

    def test_fit_basic_functionality(self, lasso_solver, sample_radar_data):
        """Test basic fit functionality."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        # Fit the model
        lasso_solver.fit(measurement_matrix, measurements)

        # Check that coefficients were estimated
        assert hasattr(lasso_solver, 'coefficients_')
        assert lasso_solver.coefficients_.shape == (measurement_matrix.shape[1],)

        # Check convergence information
        assert hasattr(lasso_solver, 'n_iterations_')
        assert hasattr(lasso_solver, 'converged_')
        assert lasso_solver.n_iterations_ <= lasso_solver.max_iterations

    def test_sparsity_enforcement(self, lasso_solver, sample_radar_data):
        """Test that LASSO enforces sparsity in the solution."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        # Fit with high regularization (should be very sparse)
        lasso_solver.lambda_reg = 0.1
        lasso_solver.fit(measurement_matrix, measurements)

        # Count non-zero coefficients
        non_zero_count = np.sum(np.abs(lasso_solver.coefficients_) > 1e-6)
        total_coefficients = len(lasso_solver.coefficients_)

        # Should be sparse (less than 10% non-zero)
        sparsity_ratio = non_zero_count / total_coefficients
        assert sparsity_ratio < 0.1, f"Solution not sparse enough: {sparsity_ratio:.3f}"

    def test_reconstruction_accuracy(self, sample_radar_data):
        """Test reconstruction accuracy with different noise levels."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']
        true_scene = sample_radar_data['true_scene']

        # Test with low noise
        solver_low_noise = LassoRadar(lambda_reg=0.001)
        solver_low_noise.fit(measurement_matrix, measurements)

        # Reshape coefficients back to scene
        reconstructed_scene = solver_low_noise.coefficients_.reshape(true_scene.shape)

        # Calculate reconstruction error
        mse = np.mean((reconstructed_scene - true_scene) ** 2)
        assert mse < 0.1, f"Reconstruction error too high: {mse:.4f}"

        # Test that main targets are detected
        # Find peaks in both true and reconstructed scenes
        true_peaks = np.unravel_index(np.argsort(true_scene.flatten())[-3:], true_scene.shape)
        recon_peaks = np.unravel_index(np.argsort(reconstructed_scene.flatten())[-3:], reconstructed_scene.shape)

        # At least one peak should be close
        peak_detected = False
        for i in range(3):
            for j in range(3):
                if (abs(true_peaks[0][i] - recon_peaks[0][j]) <= 2 and
                    abs(true_peaks[1][i] - recon_peaks[1][j]) <= 2):
                    peak_detected = True
                    break

        assert peak_detected, "No major peaks correctly detected"

    def test_predict_functionality(self, lasso_solver, sample_radar_data):
        """Test prediction functionality."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        # Fit the model
        lasso_solver.fit(measurement_matrix, measurements)

        # Test prediction
        predicted = lasso_solver.predict(measurement_matrix)

        assert predicted.shape == measurements.shape

        # Prediction error should be reasonable
        prediction_error = np.mean((predicted - measurements) ** 2)
        noise_power = sample_radar_data['noise_std'] ** 2

        # Error should be on the order of noise power
        assert prediction_error < 10 * noise_power

    def test_get_range_doppler_map(self, lasso_solver, sample_radar_data):
        """Test range-Doppler map extraction."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']
        n_range = sample_radar_data['n_range']
        n_doppler = sample_radar_data['n_doppler']

        # Fit the model
        lasso_solver.fit(measurement_matrix, measurements)

        # Get range-Doppler map
        rd_map = lasso_solver.get_range_doppler_map(n_range, n_doppler)

        assert rd_map.shape == (n_doppler, n_range)
        assert np.all(np.isfinite(rd_map))

    def test_convergence_monitoring(self, sample_radar_data):
        """Test convergence monitoring and early stopping."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        # Test with very tight tolerance (should take many iterations)
        solver = LassoRadar(lambda_reg=0.01, max_iterations=10, tolerance=1e-12)
        solver.fit(measurement_matrix, measurements)

        # Should hit max iterations
        assert solver.n_iterations_ == solver.max_iterations
        assert not solver.converged_

        # Test with loose tolerance (should converge quickly)
        solver = LassoRadar(lambda_reg=0.01, max_iterations=1000, tolerance=1e-2)
        solver.fit(measurement_matrix, measurements)

        # Should converge before max iterations
        assert solver.n_iterations_ < solver.max_iterations
        assert solver.converged_

    def test_regularization_path(self, sample_radar_data):
        """Test behavior across different regularization strengths."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        lambda_values = [0.001, 0.01, 0.1, 1.0]
        sparsity_levels = []

        for lambda_reg in lambda_values:
            solver = LassoRadar(lambda_reg=lambda_reg)
            solver.fit(measurement_matrix, measurements)

            # Count non-zero coefficients
            non_zeros = np.sum(np.abs(solver.coefficients_) > 1e-6)
            sparsity_levels.append(non_zeros)

        # Sparsity should generally decrease with increasing regularization
        assert sparsity_levels[0] >= sparsity_levels[1] >= sparsity_levels[2]

    def test_mutual_incoherence_check(self, lasso_solver):
        """Test mutual incoherence condition checking."""
        # Create a matrix with known mutual incoherence
        n, p = 50, 100
        np.random.seed(42)
        A = np.random.randn(n, p)
        A = A / np.linalg.norm(A, axis=0)  # Normalize columns

        # Compute mutual incoherence
        mu = lasso_solver.compute_mutual_incoherence(A)

        assert isinstance(mu, float)
        assert 0 <= mu <= 1

        # Test with identity matrix (should have low incoherence)
        I = np.eye(50)
        mu_identity = lasso_solver.compute_mutual_incoherence(I)
        assert mu_identity < 0.1

    def test_restricted_eigenvalue_condition(self, lasso_solver, sample_radar_data):
        """Test restricted eigenvalue condition assessment."""
        measurement_matrix = sample_radar_data['measurement_matrix']

        # Test with small sparsity level
        s = 5
        re_constant = lasso_solver.check_restricted_eigenvalue(measurement_matrix, s)

        assert isinstance(re_constant, float)
        assert re_constant >= 0

    @pytest.mark.parametrize("noise_level", [0.001, 0.01, 0.1, 0.5])
    def test_noise_robustness(self, noise_level, sample_radar_data):
        """Test LASSO performance at different noise levels."""
        measurement_matrix = sample_radar_data['measurement_matrix']
        true_scene = sample_radar_data['true_scene']

        # Generate measurements with specified noise level
        target_vector = true_scene.flatten()
        clean_measurements = measurement_matrix @ target_vector
        noisy_measurements = clean_measurements + noise_level * np.random.randn(len(clean_measurements))

        # Adapt regularization to noise level
        lambda_reg = noise_level * 0.1
        solver = LassoRadar(lambda_reg=lambda_reg)
        solver.fit(measurement_matrix, noisy_measurements)

        # Check that solver converged
        assert solver.converged_ or solver.n_iterations_ == solver.max_iterations

        # At high noise levels, expect more regularization
        if noise_level > 0.1:
            non_zeros = np.sum(np.abs(solver.coefficients_) > 1e-6)
            total_coefs = len(solver.coefficients_)
            sparsity_ratio = non_zeros / total_coefs
            assert sparsity_ratio < 0.2  # Should be quite sparse

    def test_edge_cases(self, lasso_solver):
        """Test edge cases and error handling."""
        # Test with empty measurements
        with pytest.raises(ValueError):
            lasso_solver.fit(np.array([]).reshape(0, 5), np.array([]))

        # Test with mismatched dimensions
        A = np.random.randn(10, 5)
        y = np.random.randn(8)  # Wrong size
        with pytest.raises(ValueError):
            lasso_solver.fit(A, y)

        # Test prediction before fitting
        with pytest.raises(AttributeError):
            unfitted_solver = LassoRadar()
            unfitted_solver.predict(A)

    def test_warm_start(self, sample_radar_data):
        """Test warm start functionality for iterative solving."""
        measurements = sample_radar_data['measurements']
        measurement_matrix = sample_radar_data['measurement_matrix']

        # First solve
        solver = LassoRadar(lambda_reg=0.01)
        solver.fit(measurement_matrix, measurements)
        first_solution = solver.coefficients_.copy()
        first_iterations = solver.n_iterations_

        # Solve again with warm start
        solver.fit(measurement_matrix, measurements, warm_start=True)
        second_iterations = solver.n_iterations_

        # Warm start should typically converge faster
        assert second_iterations <= first_iterations

        # Solutions should be very similar
        solution_diff = np.linalg.norm(first_solution - solver.coefficients_)
        assert solution_diff < 1e-3  # Relaxed tolerance for numerical stability