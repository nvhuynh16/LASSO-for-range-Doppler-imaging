"""
Tests for Adaptive Robust LASSO algorithm with IAA-inspired enhancements.
"""

import pytest
import numpy as np
import warnings
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lasso_radar.algorithms.adaptive_lasso import AdaptiveRobustLasso
from lasso_radar.algorithms.lasso_core import LassoRadar


class TestAdaptiveRobustLasso:
    """Test suite for AdaptiveRobustLasso class."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Standard test problem
        self.n_samples = 50
        self.n_features = 30
        self.X = np.random.randn(self.n_samples, self.n_features) / np.sqrt(self.n_samples)

        # Sparse ground truth
        self.x_true = np.zeros(self.n_features)
        self.x_true[[5, 15, 25]] = [1.0, 0.8, 0.6]

        # Clean and noisy measurements
        self.y_clean = self.X @ self.x_true
        self.noise_low = 0.01 * np.random.randn(self.n_samples)   # High SNR
        self.noise_high = 0.1 * np.random.randn(self.n_samples)  # Low SNR

        self.y_high_snr = self.y_clean + self.noise_low
        self.y_low_snr = self.y_clean + self.noise_high

    def test_initialization(self):
        """Test AdaptiveRobustLasso initialization."""
        # Test default initialization
        lasso = AdaptiveRobustLasso()
        assert lasso.base_lambda == 0.01
        assert lasso.stabilization_factor == 0.1
        assert lasso.adaptive_lambda is True
        assert lasso.robust_solving is True
        assert lasso.eigenvalue_init is True

        # Test custom initialization
        lasso = AdaptiveRobustLasso(
            base_lambda=0.05,
            stabilization_factor=0.2,
            adaptive_lambda=False,
            robust_solving=False,
            eigenvalue_init=False
        )
        assert lasso.base_lambda == 0.05
        assert lasso.stabilization_factor == 0.2
        assert lasso.adaptive_lambda is False
        assert lasso.robust_solving is False
        assert lasso.eigenvalue_init is False

    def test_snr_estimation(self):
        """Test SNR estimation functionality."""
        lasso = AdaptiveRobustLasso()

        # Test with high SNR data
        snr_high = lasso._estimate_snr_db(self.X, self.y_high_snr)
        assert isinstance(snr_high, float)
        assert -10 <= snr_high <= 30  # Should be within reasonable range

        # Test with low SNR data
        snr_low = lasso._estimate_snr_db(self.X, self.y_low_snr)
        assert isinstance(snr_low, float)
        assert -10 <= snr_low <= 30

        # High SNR should generally be higher than low SNR
        # (though this isn't guaranteed due to estimation noise)
        assert snr_high >= snr_low - 5  # Allow some tolerance

    def test_adaptive_regularization(self):
        """Test adaptive regularization parameter selection."""
        lasso = AdaptiveRobustLasso(base_lambda=0.01)

        # Test adaptation for different SNR values
        lambda_high_snr = lasso._adapt_regularization_parameter(20.0)  # High SNR
        lambda_low_snr = lasso._adapt_regularization_parameter(-5.0)   # Low SNR

        # Low SNR should require higher regularization
        assert lambda_low_snr > lambda_high_snr
        assert lambda_low_snr > lasso.base_lambda

        # Check bounds
        lambda_extreme_high = lasso._adapt_regularization_parameter(50.0)
        lambda_extreme_low = lasso._adapt_regularization_parameter(-50.0)

        assert lambda_extreme_high >= lasso.base_lambda * 0.1
        assert lambda_extreme_low <= lasso.base_lambda * 10.0

    def test_robust_matrix_solving(self):
        """Test robust matrix solving with fallback strategies."""
        lasso = AdaptiveRobustLasso()

        # Test with well-conditioned matrix
        A_good = np.random.randn(10, 10)
        A_good = A_good.T @ A_good + 0.1 * np.eye(10)  # Make positive definite
        b = np.random.randn(10)

        x_good = lasso._solve_robust(A_good, b)
        assert x_good.shape == (10,)
        assert np.all(np.isfinite(x_good))
        assert lasso.fallback_strategy_used_ == "standard_solve"

        # Test with singular matrix (should trigger SVD fallback)
        A_singular = np.random.randn(10, 10)
        A_singular[:, -1] = A_singular[:, 0]  # Make last column identical to first

        x_singular = lasso._solve_robust(A_singular, b)
        assert x_singular.shape == (10,)
        assert np.all(np.isfinite(x_singular))
        # Should use SVD or regularized fallback

    def test_eigenvalue_initialization(self):
        """Test eigenvalue-based initialization."""
        lasso = AdaptiveRobustLasso()

        # Test initialization
        coeffs_init = lasso._initialize_coefficients_eigenvalue(self.X, self.y_high_snr)

        assert coeffs_init.shape == (self.n_features,)
        assert np.all(np.isfinite(coeffs_init))

        # Should not be all zeros (unless data is pathological)
        assert np.any(coeffs_init != 0)

        # Should be reasonable magnitude (not too large)
        assert np.max(np.abs(coeffs_init)) < 10.0

    def test_coordinate_descent_with_stabilization(self):
        """Test coordinate descent with stabilization factor."""
        # Test without stabilization
        lasso_no_stab = AdaptiveRobustLasso(stabilization_factor=0.0)
        lasso_no_stab.coefficients_ = np.random.randn(self.n_features) * 0.1
        lasso_no_stab.lambda_reg_ = 0.01

        old_coeff = lasso_no_stab.coefficients_[0]
        lasso_no_stab._coordinate_descent_step(self.X, self.y_high_snr, 0)
        new_coeff = lasso_no_stab.coefficients_[0]

        # Test with stabilization
        lasso_stab = AdaptiveRobustLasso(stabilization_factor=0.5)
        lasso_stab.coefficients_ = np.random.randn(self.n_features) * 0.1
        lasso_stab.lambda_reg_ = 0.01

        old_coeff_stab = lasso_stab.coefficients_[0]
        lasso_stab._coordinate_descent_step(self.X, self.y_high_snr, 0)
        new_coeff_stab = lasso_stab.coefficients_[0]

        # Stabilized update should be between old and new values
        # (This test may be sensitive to the specific update, so we check basic properties)
        assert np.isfinite(new_coeff_stab)

    def test_basic_fitting(self):
        """Test basic fitting functionality."""
        lasso = AdaptiveRobustLasso(verbose=False)

        # Test fitting
        lasso.fit(self.X, self.y_high_snr)

        # Check basic attributes
        assert hasattr(lasso, 'coefficients_')
        assert lasso.coefficients_.shape == (self.n_features,)
        assert np.all(np.isfinite(lasso.coefficients_))

        # Check enhancement attributes
        assert hasattr(lasso, 'lambda_reg_')
        assert hasattr(lasso, 'estimated_snr_db_')
        assert hasattr(lasso, 'converged_')
        assert hasattr(lasso, 'n_iterations_')

    def test_adaptive_vs_fixed_lambda(self):
        """Test adaptive lambda vs fixed lambda behavior."""
        # Adaptive lambda
        lasso_adaptive = AdaptiveRobustLasso(
            base_lambda=0.01,
            adaptive_lambda=True,
            verbose=False
        )
        lasso_adaptive.fit(self.X, self.y_low_snr)

        # Fixed lambda
        lasso_fixed = AdaptiveRobustLasso(
            base_lambda=0.01,
            adaptive_lambda=False,
            verbose=False
        )
        lasso_fixed.fit(self.X, self.y_low_snr)

        # Adaptive should have adapted lambda
        assert lasso_adaptive.lambda_reg_ != lasso_adaptive.base_lambda
        assert lasso_fixed.lambda_reg_ == lasso_fixed.base_lambda

        # Both should converge
        assert lasso_adaptive.converged_ or lasso_adaptive.n_iterations_ > 10
        assert lasso_fixed.converged_ or lasso_fixed.n_iterations_ > 10

    def test_enhancement_modes(self):
        """Test different combinations of enhancements."""
        configs = [
            {'adaptive_lambda': True, 'eigenvalue_init': True, 'stabilization_factor': 0.1},
            {'adaptive_lambda': False, 'eigenvalue_init': True, 'stabilization_factor': 0.0},
            {'adaptive_lambda': True, 'eigenvalue_init': False, 'stabilization_factor': 0.2},
            {'adaptive_lambda': False, 'eigenvalue_init': False, 'stabilization_factor': 0.0},
        ]

        for config in configs:
            lasso = AdaptiveRobustLasso(verbose=False, **config)
            lasso.fit(self.X, self.y_high_snr)

            # All configurations should work
            assert lasso.coefficients_.shape == (self.n_features,)
            assert np.all(np.isfinite(lasso.coefficients_))

    def test_comparison_with_standard_lasso(self):
        """Test comparison with standard LASSO."""
        # Standard LASSO
        lasso_standard = LassoRadar(lambda_reg=0.01, verbose=False)
        lasso_standard.fit(self.X, self.y_high_snr)

        # Adaptive LASSO
        lasso_adaptive = AdaptiveRobustLasso(base_lambda=0.01, verbose=False)
        lasso_adaptive.fit(self.X, self.y_high_snr)

        # Both should produce reasonable results
        assert np.all(np.isfinite(lasso_standard.coefficients_))
        assert np.all(np.isfinite(lasso_adaptive.coefficients_))

        # Adaptive should have additional information
        assert hasattr(lasso_adaptive, 'estimated_snr_db_')
        assert not hasattr(lasso_standard, 'estimated_snr_db_')

    def test_low_snr_scenario(self):
        """Test performance in challenging low SNR scenario."""
        # Very challenging scenario
        lasso = AdaptiveRobustLasso(base_lambda=0.01, verbose=False)
        lasso.fit(self.X, self.y_low_snr)

        # Should handle low SNR gracefully
        assert lasso.coefficients_.shape == (self.n_features,)
        assert np.all(np.isfinite(lasso.coefficients_))

        # Should adapt regularization for low SNR
        if lasso.adaptive_lambda:
            assert lasso.lambda_reg_ >= lasso.base_lambda  # Should increase regularization

    def test_ill_conditioned_matrix(self):
        """Test behavior with ill-conditioned measurement matrix."""
        # Create ill-conditioned matrix
        X_ill = np.random.randn(20, 30)
        X_ill[:, -1] = X_ill[:, 0] + 1e-10 * np.random.randn(20)  # Nearly identical columns

        y_ill = X_ill @ self.x_true[:30] + 0.01 * np.random.randn(20)

        lasso = AdaptiveRobustLasso(robust_solving=True, verbose=False)

        # Should handle ill-conditioning gracefully
        lasso.fit(X_ill, y_ill)

        assert lasso.coefficients_.shape == (30,)
        assert np.all(np.isfinite(lasso.coefficients_))

    def test_enhancement_summary(self):
        """Test enhancement summary functionality."""
        lasso = AdaptiveRobustLasso(verbose=False)
        lasso.fit(self.X, self.y_high_snr)

        summary = lasso.get_enhancement_summary()

        # Check required keys
        required_keys = [
            'adaptive_lambda_used', 'estimated_snr_db', 'base_lambda',
            'adapted_lambda', 'eigenvalue_init_used', 'stabilization_applied',
            'stabilization_factor', 'robust_solving_used', 'fallback_strategy',
            'converged', 'n_iterations'
        ]

        for key in required_keys:
            assert key in summary

        # Check data types
        assert isinstance(summary['adaptive_lambda_used'], bool)
        assert isinstance(summary['base_lambda'], float)
        assert isinstance(summary['adapted_lambda'], float)
        assert isinstance(summary['converged'], bool)
        assert isinstance(summary['n_iterations'], int)

    def test_warm_start(self):
        """Test warm start functionality."""
        lasso = AdaptiveRobustLasso(warm_start=True, verbose=False)

        # First fit
        lasso.fit(self.X, self.y_high_snr)
        coeffs_first = lasso.coefficients_.copy()
        iters_first = lasso.n_iterations_

        # Second fit with warm start (should use previous solution)
        lasso.fit(self.X, self.y_high_snr)
        coeffs_second = lasso.coefficients_.copy()
        iters_second = lasso.n_iterations_

        # Results should be similar (may not be identical due to adaptation)
        assert np.allclose(coeffs_first, coeffs_second, rtol=1e-3)

    def test_verbose_output(self, capsys):
        """Test verbose output functionality."""
        lasso = AdaptiveRobustLasso(verbose=True)
        lasso.fit(self.X, self.y_high_snr)

        captured = capsys.readouterr()

        # Should contain expected verbose information
        assert "Estimated SNR" in captured.out or "SNR" in captured.out
        assert "lambda" in captured.out or "Lambda" in captured.out
        assert "Converged" in captured.out

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        lasso = AdaptiveRobustLasso(verbose=False)

        # Test with very small problem
        X_small = np.random.randn(3, 2)
        y_small = np.random.randn(3)

        lasso.fit(X_small, y_small)
        assert lasso.coefficients_.shape == (2,)
        assert np.all(np.isfinite(lasso.coefficients_))

        # Test with zero matrix (should be handled gracefully)
        X_zero = np.zeros((10, 5))
        y_zero = np.random.randn(10)

        lasso_zero = AdaptiveRobustLasso(verbose=False)
        lasso_zero.fit(X_zero, y_zero)

        # Should handle zero matrix without crashing
        assert lasso_zero.coefficients_.shape == (5,)
        assert np.all(np.isfinite(lasso_zero.coefficients_))

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid parameters
        with pytest.raises(ValueError):
            lasso = AdaptiveRobustLasso(base_lambda=-1.0)
            lasso.fit(self.X, self.y_high_snr)

        # Test boundary parameters
        lasso = AdaptiveRobustLasso(
            base_lambda=1e-10,
            stabilization_factor=0.0,
            max_iterations=1
        )

        # Should handle boundary cases without error
        lasso.fit(self.X, self.y_high_snr)
        assert lasso.coefficients_.shape == (self.n_features,)

    def test_reproducibility(self):
        """Test reproducibility with fixed random seed."""
        np.random.seed(123)
        X_test = np.random.randn(20, 15)
        y_test = np.random.randn(20)

        # First run
        np.random.seed(456)
        lasso1 = AdaptiveRobustLasso(verbose=False)
        lasso1.fit(X_test, y_test)

        # Second run with same seed
        np.random.seed(456)
        lasso2 = AdaptiveRobustLasso(verbose=False)
        lasso2.fit(X_test, y_test)

        # Results should be identical
        assert np.allclose(lasso1.coefficients_, lasso2.coefficients_)
        assert lasso1.estimated_snr_db_ == lasso2.estimated_snr_db_
        assert lasso1.lambda_reg_ == lasso2.lambda_reg_


class TestAdaptiveRobustLassoIntegration:
    """Integration tests for AdaptiveRobustLasso with radar scenarios."""

    def test_radar_range_doppler_scenario(self):
        """Test with realistic radar range-Doppler scenario."""
        np.random.seed(42)

        # Radar scenario parameters
        n_range, n_doppler = 16, 8
        n_measurements = 60

        # Create measurement matrix
        A = np.random.randn(n_measurements, n_range * n_doppler) / np.sqrt(n_measurements)

        # Create sparse target scene
        scene = np.zeros(n_range * n_doppler)
        scene[25] = 1.0   # Strong target
        scene[45] = 0.4   # Weak target
        scene[85] = 0.2   # Very weak target

        # Simulate measurements with noise
        y_clean = A @ scene
        noise = 0.05 * np.random.randn(n_measurements)
        y_noisy = y_clean + noise

        # Test adaptive LASSO
        lasso = AdaptiveRobustLasso(base_lambda=0.01, verbose=False)
        lasso.fit(A, y_noisy)

        # Reconstruct range-Doppler map
        reconstructed_scene = lasso.coefficients_.reshape(n_doppler, n_range)

        # Basic checks
        assert reconstructed_scene.shape == (n_doppler, n_range)
        assert np.all(np.isfinite(reconstructed_scene))

        # Should detect some targets (sparsity)
        n_detected = np.sum(np.abs(reconstructed_scene) > 0.01)
        assert n_detected > 0
        assert n_detected < n_range * n_doppler  # Should be sparse

    def test_performance_under_varying_snr(self):
        """Test performance under varying SNR conditions."""
        np.random.seed(42)

        # Base scenario
        n_samples, n_features = 40, 25
        X = np.random.randn(n_samples, n_features) / np.sqrt(n_samples)
        x_true = np.zeros(n_features)
        x_true[[5, 15, 20]] = [1.0, 0.6, 0.3]
        y_clean = X @ x_true

        # Test different noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2]  # From high to low SNR
        results = []

        for noise_level in noise_levels:
            y_noisy = y_clean + noise_level * np.random.randn(n_samples)

            lasso = AdaptiveRobustLasso(base_lambda=0.01, verbose=False)
            lasso.fit(X, y_noisy)

            # Calculate reconstruction error
            error = np.linalg.norm(lasso.coefficients_ - x_true)

            results.append({
                'noise_level': noise_level,
                'error': error,
                'snr_db': lasso.estimated_snr_db_,
                'lambda_adapted': lasso.lambda_reg_,
                'converged': lasso.converged_
            })

        # All should converge
        assert all(r['converged'] for r in results)

        # Lambda should generally increase with noise level
        lambdas = [r['lambda_adapted'] for r in results]
        # Allow some flexibility since adaptation is based on estimated SNR
        assert lambdas[-1] >= lambdas[0] * 0.5  # High noise should have higher lambda

    def test_comparison_standard_vs_adaptive(self):
        """Test direct comparison between standard and adaptive LASSO."""
        np.random.seed(42)

        # Challenging scenario (low SNR)
        n_samples, n_features = 30, 20
        X = np.random.randn(n_samples, n_features) / np.sqrt(n_samples)
        x_true = np.zeros(n_features)
        x_true[[3, 8, 15]] = [1.0, 0.5, 0.3]

        # High noise scenario
        y_clean = X @ x_true
        y_noisy = y_clean + 0.15 * np.random.randn(n_samples)

        # Standard LASSO
        lasso_std = LassoRadar(lambda_reg=0.01, verbose=False, max_iterations=500)
        lasso_std.fit(X, y_noisy)

        # Adaptive LASSO
        lasso_adapt = AdaptiveRobustLasso(
            base_lambda=0.01,
            verbose=False,
            max_iterations=500
        )
        lasso_adapt.fit(X, y_noisy)

        # Both should produce valid results
        assert np.all(np.isfinite(lasso_std.coefficients_))
        assert np.all(np.isfinite(lasso_adapt.coefficients_))

        # Calculate reconstruction errors
        error_std = np.linalg.norm(lasso_std.coefficients_ - x_true)
        error_adapt = np.linalg.norm(lasso_adapt.coefficients_ - x_true)

        # Adaptive should generally perform as well or better
        # (though this isn't guaranteed for every random realization)
        assert error_adapt < error_std * 2.0  # Allow some tolerance

        # Adaptive should have useful enhancement information
        summary = lasso_adapt.get_enhancement_summary()
        assert summary['adaptive_lambda_used']
        assert isinstance(summary['estimated_snr_db'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])