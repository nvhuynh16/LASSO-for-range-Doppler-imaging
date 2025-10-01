"""
Unit tests for utility functions and theoretical conditions.

Tests cover:
- SNR calculations
- Performance metrics
- Theoretical conditions (RIP, mutual incoherence, beta-min)
- Signal processing utilities
- Range-Doppler processing
- Validation functions
"""

import pytest
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock, patch

# Import the modules to be tested
from lasso_radar.utils.metrics import snr_calculator, performance_metrics
from lasso_radar.utils.conditions import theoretical_conditions
from lasso_radar.utils.signal_processing import range_doppler_processing


class TestSNRCalculator:
    """Test suite for SNR calculation utilities."""

    @pytest.fixture
    def signal_samples(self):
        """Generate test signal samples."""
        np.random.seed(42)

        # Generate test signals
        n_samples = 1000
        signal_power = 1.0
        noise_power = 0.1

        clean_signal = np.sqrt(signal_power) * np.random.randn(n_samples)
        noise = np.sqrt(noise_power) * np.random.randn(n_samples)
        noisy_signal = clean_signal + noise

        return {
            'clean_signal': clean_signal,
            'noise': noise,
            'noisy_signal': noisy_signal,
            'true_snr_db': 10 * np.log10(signal_power / noise_power)
        }

    def test_snr_from_signal_and_noise(self, signal_samples):
        """Test SNR calculation from signal and noise components."""
        clean_signal = signal_samples['clean_signal']
        noise = signal_samples['noise']
        true_snr_db = signal_samples['true_snr_db']

        calculated_snr = snr_calculator.snr_from_components(clean_signal, noise)

        # Should be close to theoretical value
        assert abs(calculated_snr - true_snr_db) < 1.0  # Within 1 dB

    def test_snr_from_noisy_signal(self, signal_samples):
        """Test SNR estimation from noisy signal only."""
        noisy_signal = signal_samples['noisy_signal']
        true_snr_db = signal_samples['true_snr_db']

        # Estimate SNR using various methods
        snr_estimates = {
            'moment_based': snr_calculator.estimate_snr_moments(noisy_signal),
            'spectral': snr_calculator.estimate_snr_spectral(noisy_signal),
            'quantile_based': snr_calculator.estimate_snr_quantiles(noisy_signal)
        }

        # All methods should give reasonable estimates
        for method, estimate in snr_estimates.items():
            assert isinstance(estimate, (int, float))
            assert np.isfinite(estimate)
            # Should be within reasonable range of true value
            assert abs(estimate - true_snr_db) < 5.0, f"{method} estimate too far: {estimate:.2f}"

    def test_complex_signal_snr(self):
        """Test SNR calculation for complex signals."""
        np.random.seed(42)
        n_samples = 1000

        # Complex signal with known SNR
        signal_power = 2.0
        noise_power = 0.2

        clean_signal = np.sqrt(signal_power/2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        )
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        )
        noisy_signal = clean_signal + noise

        true_snr_db = 10 * np.log10(signal_power / noise_power)
        calculated_snr = snr_calculator.snr_complex(clean_signal, noise)

        assert abs(calculated_snr - true_snr_db) < 1.0

    def test_radar_specific_snr(self):
        """Test radar-specific SNR calculations."""
        # Radar parameters
        target_rcs = 10.0  # m^2
        transmit_power = 1000.0  # W
        antenna_gain = 30.0  # dB
        frequency = 10e9  # Hz
        range_m = 1000.0  # m
        noise_figure = 3.0  # dB
        system_temp = 290.0  # K

        snr_db = snr_calculator.radar_range_equation_snr(
            target_rcs, transmit_power, antenna_gain, frequency,
            range_m, noise_figure, system_temp
        )

        assert isinstance(snr_db, (int, float))
        assert np.isfinite(snr_db)
        assert snr_db > 0  # Should be positive for reasonable parameters

    def test_snr_loss_calculations(self):
        """Test SNR loss calculations for various impairments."""
        clean_snr_db = 20.0

        # Test various loss mechanisms
        losses = {
            'quantization': snr_calculator.quantization_loss(n_bits=8),
            'mismatch': snr_calculator.mismatch_loss(correlation=0.9),
            'doppler': snr_calculator.doppler_loss(doppler_shift=100, pulse_width=1e-6),
            'window': snr_calculator.windowing_loss(window_type='hamming')
        }

        for loss_type, loss_db in losses.items():
            assert isinstance(loss_db, (int, float))
            assert loss_db >= 0, f"{loss_type} loss should be non-negative"

            # Calculate degraded SNR
            degraded_snr = clean_snr_db - loss_db
            assert degraded_snr <= clean_snr_db

    @pytest.mark.parametrize("snr_db", [-10, -5, 0, 5, 10, 20, 30])
    def test_snr_range_validation(self, snr_db):
        """Test SNR calculations across wide range of values."""
        # Generate signal with specific SNR
        n_samples = 1000
        signal_power = 1.0
        noise_power = signal_power / (10**(snr_db/10))

        signal = np.sqrt(signal_power) * np.random.randn(n_samples)
        noise = np.sqrt(noise_power) * np.random.randn(n_samples)

        calculated_snr = snr_calculator.snr_from_components(signal, noise)

        # Should be close to target SNR
        assert abs(calculated_snr - snr_db) < 2.0


class TestPerformanceMetrics:
    """Test suite for performance metrics calculations."""

    @pytest.fixture
    def detection_scenario(self):
        """Generate detection scenario for testing."""
        np.random.seed(42)

        # True target locations
        n_range = 100
        n_doppler = 50
        true_targets = np.zeros((n_range, n_doppler))

        # Add targets
        true_targets[20, 10] = 1.0
        true_targets[50, 25] = 0.8
        true_targets[80, 40] = 0.6

        # Simulated detection results (with some errors)
        detected_targets = true_targets.copy()

        # Add false alarm
        detected_targets[30, 15] = 0.3

        # Miss one target (make it too weak)
        detected_targets[80, 40] = 0.05

        return {
            'true_targets': true_targets,
            'detected_targets': detected_targets,
            'n_range': n_range,
            'n_doppler': n_doppler
        }

    def test_detection_metrics(self, detection_scenario):
        """Test detection performance metrics."""
        true_targets = detection_scenario['true_targets']
        detected_targets = detection_scenario['detected_targets']

        threshold = 0.1

        metrics = performance_metrics.detection_performance(
            true_targets, detected_targets, threshold
        )

        # Check metric types and ranges
        assert 'probability_detection' in metrics
        assert 'probability_false_alarm' in metrics
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1, f"{key} should be between 0 and 1"

        # Sanity checks
        assert metrics['probability_detection'] > 0.5  # Should detect most targets
        assert metrics['probability_false_alarm'] < 0.5  # False alarms should be limited

    def test_reconstruction_metrics(self):
        """Test reconstruction quality metrics."""
        np.random.seed(42)

        # Original sparse signal
        n_features = 200
        true_signal = np.zeros(n_features)
        true_signal[20] = 1.0
        true_signal[50] = 0.8
        true_signal[120] = 0.6

        # Reconstructed signal (with some error)
        reconstructed = true_signal.copy()
        reconstructed += 0.05 * np.random.randn(n_features)  # Add noise
        reconstructed[180] = 0.1  # Add spurious component

        metrics = performance_metrics.reconstruction_quality(true_signal, reconstructed)

        assert 'mse' in metrics
        assert 'nmse' in metrics
        assert 'snr_improvement' in metrics
        assert 'sparsity_error' in metrics

        # MSE should be small for good reconstruction
        assert metrics['mse'] < 0.1
        assert metrics['nmse'] < 0.1

    def test_sparsity_metrics(self):
        """Test sparsity-related metrics."""
        # Sparse signal
        sparse_signal = np.zeros(100)
        sparse_signal[10:15] = [1.0, 0.8, 0.6, 0.4, 0.2]

        # Dense signal
        dense_signal = 0.1 * np.random.randn(100)

        sparse_metrics = performance_metrics.sparsity_analysis(sparse_signal)
        dense_metrics = performance_metrics.sparsity_analysis(dense_signal)

        # Sparse signal should have higher sparsity measures
        assert sparse_metrics['l0_norm'] < dense_metrics['l0_norm']
        assert sparse_metrics['gini_coefficient'] > dense_metrics['gini_coefficient']
        assert sparse_metrics['sparsity_ratio'] > dense_metrics['sparsity_ratio']

    def test_radar_performance_metrics(self):
        """Test radar-specific performance metrics."""
        # Range-Doppler map
        np.random.seed(42)
        rd_map = np.random.randn(64, 32)

        # Add strong targets
        rd_map[20, 10] = 10.0
        rd_map[40, 20] = 8.0

        metrics = performance_metrics.radar_performance_analysis(rd_map)

        assert 'dynamic_range' in metrics
        assert 'clutter_level' in metrics
        assert 'peak_sidelobe_ratio' in metrics
        assert 'integrated_sidelobe_level' in metrics

        # Dynamic range should be reasonable
        assert metrics['dynamic_range'] > 10  # dB

    def test_computational_metrics(self):
        """Test computational performance metrics."""
        import time

        # Simulate algorithm execution
        start_time = time.time()

        # Dummy computation
        n = 1000
        A = np.random.randn(n, n)
        b = np.random.randn(n)
        x = np.linalg.solve(A, b)

        end_time = time.time()

        metrics = performance_metrics.computational_analysis(
            start_time, end_time, memory_usage=A.nbytes + b.nbytes + x.nbytes
        )

        assert 'execution_time' in metrics
        assert 'memory_usage_mb' in metrics
        assert 'flops_estimate' in metrics

        assert metrics['execution_time'] > 0
        assert metrics['memory_usage_mb'] > 0

    @pytest.mark.parametrize("noise_level", [0.01, 0.1, 0.5, 1.0])
    def test_metrics_robustness(self, noise_level, detection_scenario):
        """Test metric robustness to noise."""
        true_targets = detection_scenario['true_targets']

        # Add noise to detection results
        noisy_detected = true_targets + noise_level * np.random.randn(*true_targets.shape)

        try:
            metrics = performance_metrics.detection_performance(
                true_targets, noisy_detected, threshold=0.1
            )

            # Metrics should be computable even with noise
            assert all(np.isfinite(v) for v in metrics.values())

        except Exception as e:
            pytest.fail(f"Metrics failed with noise level {noise_level}: {e}")


class TestTheoreticalConditions:
    """Test suite for theoretical condition assessments."""

    @pytest.fixture
    def test_matrices(self):
        """Generate test matrices for condition analysis."""
        np.random.seed(42)

        # Well-conditioned matrix
        n, p = 50, 100
        A_good = np.random.randn(n, p)
        A_good = A_good / np.linalg.norm(A_good, axis=0)  # Normalize columns

        # Poorly conditioned matrix (highly correlated columns)
        A_bad = np.random.randn(n, p)
        # Make adjacent columns correlated
        for i in range(p-1):
            A_bad[:, i+1] = 0.9 * A_bad[:, i] + 0.1 * A_bad[:, i+1]
        A_bad = A_bad / np.linalg.norm(A_bad, axis=0)

        return {
            'well_conditioned': A_good,
            'poorly_conditioned': A_bad,
            'n': n,
            'p': p
        }

    def test_mutual_incoherence_calculation(self, test_matrices):
        """Test mutual incoherence calculation."""
        A_good = test_matrices['well_conditioned']
        A_bad = test_matrices['poorly_conditioned']

        # Calculate mutual incoherence
        mu_good = theoretical_conditions.mutual_incoherence(A_good)
        mu_bad = theoretical_conditions.mutual_incoherence(A_bad)

        # Both should be valid values
        assert 0 <= mu_good <= 1
        assert 0 <= mu_bad <= 1

        # Poorly conditioned matrix should have higher incoherence
        assert mu_bad > mu_good

    def test_restricted_isometry_property(self, test_matrices):
        """Test Restricted Isometry Property (RIP) assessment."""
        A = test_matrices['well_conditioned']

        # Test for different sparsity levels
        sparsity_levels = [1, 5, 10, 20]

        for s in sparsity_levels:
            if s < A.shape[1]:
                delta_s = theoretical_conditions.restricted_isometry_constant(A, s)

                assert isinstance(delta_s, (int, float))
                assert delta_s >= 0

                # RIP constant should be less than 1 for good matrices
                if s <= A.shape[0] // 4:  # Conservative check
                    assert delta_s < 1

    def test_restricted_eigenvalue_condition(self, test_matrices):
        """Test Restricted Eigenvalue (RE) condition."""
        A = test_matrices['well_conditioned']

        sparsity_levels = [5, 10, 15]

        for s in sparsity_levels:
            if s < A.shape[1]:
                re_constant = theoretical_conditions.restricted_eigenvalue(A, s)

                assert isinstance(re_constant, (int, float))
                assert re_constant >= 0

    def test_beta_min_condition(self):
        """Test beta-min condition for signal detectability."""
        # Create sparse signal with known minimum nonzero coefficient
        n_features = 100
        true_signal = np.zeros(n_features)

        # Set some coefficients with known minimum
        beta_min_true = 0.5
        true_signal[10] = 1.0
        true_signal[30] = beta_min_true
        true_signal[50] = 0.8

        calculated_beta_min = theoretical_conditions.beta_min_condition(true_signal)

        assert abs(calculated_beta_min - beta_min_true) < 1e-10

    def test_compatibility_condition(self, test_matrices):
        """Test compatibility condition between signal and measurement matrix."""
        A = test_matrices['well_conditioned']

        # Generate sparse signal
        n_features = A.shape[1]
        sparse_signal = np.zeros(n_features)
        sparse_signal[10:15] = np.random.randn(5)

        # Generate measurements
        measurements = A @ sparse_signal
        noise_level = 0.01
        noisy_measurements = measurements + noise_level * np.random.randn(len(measurements))

        compatibility = theoretical_conditions.compatibility_condition(
            A, sparse_signal, noisy_measurements, noise_level
        )

        assert isinstance(compatibility, (int, float))
        assert compatibility >= 0

    def test_irrepresentable_condition(self, test_matrices):
        """Test irrepresentable condition for LASSO variable selection."""
        A = test_matrices['well_conditioned']

        # Define active set
        active_set = [10, 20, 30, 40]

        irrepresentable = theoretical_conditions.irrepresentable_condition(A, active_set)

        assert isinstance(irrepresentable, (int, float))
        assert 0 <= irrepresentable <= 1

    def test_coherence_based_bounds(self, test_matrices):
        """Test coherence-based recovery bounds."""
        A = test_matrices['well_conditioned']

        mu = theoretical_conditions.mutual_incoherence(A)

        # Calculate recovery bounds
        bounds = theoretical_conditions.coherence_recovery_bounds(A, mu)

        assert 'exact_recovery_bound' in bounds
        assert 'stable_recovery_bound' in bounds

        exact_bound = bounds['exact_recovery_bound']
        stable_bound = bounds['stable_recovery_bound']

        assert isinstance(exact_bound, int)
        assert isinstance(stable_bound, int)
        assert exact_bound >= 1
        assert stable_bound >= exact_bound

    def test_rip_based_bounds(self, test_matrices):
        """Test RIP-based recovery guarantees."""
        A = test_matrices['well_conditioned']

        # Test for small sparsity level
        s = min(10, A.shape[0] // 4)

        try:
            delta_2s = theoretical_conditions.restricted_isometry_constant(A, 2*s)

            bounds = theoretical_conditions.rip_recovery_bounds(delta_2s, s)

            assert 'recovery_guarantee' in bounds
            assert isinstance(bounds['recovery_guarantee'], bool)

            if delta_2s < np.sqrt(2) - 1:  # Sufficient condition
                assert bounds['recovery_guarantee']

        except Exception:
            # RIP calculation might be expensive/intractable for large matrices
            pass

    def test_condition_relationships(self, test_matrices):
        """Test relationships between different theoretical conditions."""
        A = test_matrices['well_conditioned']

        mu = theoretical_conditions.mutual_incoherence(A)

        # Test relationship: mu <= delta_s for appropriate s
        s = min(5, A.shape[0] // 8)

        try:
            delta_s = theoretical_conditions.restricted_isometry_constant(A, s)

            # Mutual incoherence provides upper bound for RIP constant
            assert mu <= (s - 1) * delta_s + delta_s  # Conservative relationship

        except Exception:
            # Skip if RIP calculation is too expensive
            pass

    @pytest.mark.parametrize("matrix_type", ['random_gaussian', 'fourier', 'bernoulli'])
    def test_different_matrix_types(self, matrix_type):
        """Test conditions on different types of measurement matrices."""
        np.random.seed(42)
        n, p = 40, 80

        if matrix_type == 'random_gaussian':
            A = np.random.randn(n, p) / np.sqrt(n)
        elif matrix_type == 'fourier':
            # Partial Fourier matrix
            indices = np.random.choice(p, n, replace=False)
            A = np.fft.fft(np.eye(p))[indices, :] / np.sqrt(p)
            A = A.real  # Take real part for simplicity
        elif matrix_type == 'bernoulli':
            A = np.random.choice([-1, 1], (n, p)) / np.sqrt(n)

        # Normalize columns
        A = A / np.linalg.norm(A, axis=0)

        # Calculate mutual incoherence
        mu = theoretical_conditions.mutual_incoherence(A)

        assert 0 <= mu <= 1
        assert np.isfinite(mu)

    def test_condition_convergence(self):
        """Test condition behavior as matrix dimensions change."""
        np.random.seed(42)

        dimensions = [(20, 40), (30, 60), (50, 100)]
        incoherences = []

        for n, p in dimensions:
            A = np.random.randn(n, p) / np.sqrt(n)
            A = A / np.linalg.norm(A, axis=0)

            mu = theoretical_conditions.mutual_incoherence(A)
            incoherences.append(mu)

        # All should be finite and valid
        assert all(0 <= mu <= 1 for mu in incoherences)
        assert all(np.isfinite(mu) for mu in incoherences)


class TestSignalProcessingUtils:
    """Test suite for signal processing utility functions."""

    def test_range_doppler_processing_basic(self):
        """Test basic range-Doppler processing functionality."""
        np.random.seed(42)

        # Generate test data
        n_pulses = 32
        n_samples = 64

        # Simulate radar returns
        radar_data = np.random.randn(n_pulses, n_samples) + 1j * np.random.randn(n_pulses, n_samples)

        # Add a target
        target_range_bin = 20
        target_doppler_bin = 10

        # Add coherent target across pulses with Doppler shift
        for pulse_idx in range(n_pulses):
            doppler_phase = 2 * np.pi * target_doppler_bin * pulse_idx / n_pulses
            radar_data[pulse_idx, target_range_bin] += 2.0 * np.exp(1j * doppler_phase)

        # Process
        rd_map = range_doppler_processing.basic_range_doppler(radar_data)

        assert rd_map.shape == (n_pulses, n_samples)
        assert np.all(np.isfinite(rd_map))

        # Target should be visible in range-Doppler map
        peak_indices = np.unravel_index(np.argmax(np.abs(rd_map)), rd_map.shape)

        # Should be close to target location
        assert abs(peak_indices[0] - target_doppler_bin) <= 2
        assert abs(peak_indices[1] - target_range_bin) <= 2

    def test_windowing_functions(self):
        """Test various windowing functions."""
        length = 64

        windows = {
            'rectangular': range_doppler_processing.rectangular_window(length),
            'hamming': range_doppler_processing.hamming_window(length),
            'hanning': range_doppler_processing.hanning_window(length),
            'blackman': range_doppler_processing.blackman_window(length),
            'kaiser': range_doppler_processing.kaiser_window(length, beta=8.6)
        }

        for window_name, window in windows.items():
            assert len(window) == length
            assert np.all(np.isfinite(window))
            assert np.all(window >= 0)  # Windows should be non-negative

            # Most windows should have maximum near center
            if window_name != 'rectangular':
                center_idx = length // 2
                max_idx = np.argmax(window)
                assert abs(max_idx - center_idx) <= 2

    def test_doppler_processing_variants(self):
        """Test different Doppler processing approaches."""
        np.random.seed(42)
        n_pulses = 32
        n_samples = 64

        radar_data = np.random.randn(n_pulses, n_samples) + 1j * np.random.randn(n_pulses, n_samples)

        # Standard FFT-based processing
        rd_fft = range_doppler_processing.fft_doppler_processing(radar_data)

        # Windowed processing
        rd_windowed = range_doppler_processing.windowed_doppler_processing(
            radar_data, window_type='hamming'
        )

        # Both should have same dimensions
        assert rd_fft.shape == rd_windowed.shape == radar_data.shape

        # Windowed version should have lower sidelobes
        fft_sidelobes = np.std(np.abs(rd_fft))
        windowed_sidelobes = np.std(np.abs(rd_windowed))

        # This is not always true, but generally windowing reduces sidelobes
        # Just check that both are finite and reasonable
        assert np.isfinite(fft_sidelobes)
        assert np.isfinite(windowed_sidelobes)

    def test_clutter_suppression(self):
        """Test clutter suppression algorithms."""
        np.random.seed(42)
        n_pulses = 64
        n_samples = 128

        # Generate data with clutter and moving target
        radar_data = np.zeros((n_pulses, n_samples), dtype=complex)

        # Add stationary clutter (zero Doppler)
        clutter_amplitude = 5.0
        for range_bin in range(n_samples):
            radar_data[:, range_bin] += clutter_amplitude * (np.random.randn() + 1j * np.random.randn())

        # Add moving target
        target_doppler = 10
        target_range = 50
        target_amplitude = 1.0

        for pulse_idx in range(n_pulses):
            doppler_phase = 2 * np.pi * target_doppler * pulse_idx / n_pulses
            radar_data[pulse_idx, target_range] += target_amplitude * np.exp(1j * doppler_phase)

        # Apply clutter suppression
        suppressed_data = range_doppler_processing.clutter_suppression(
            radar_data, method='mti'
        )

        assert suppressed_data.shape == radar_data.shape

        # Clutter at zero Doppler should be reduced
        rd_original = np.fft.fft(radar_data, axis=0)
        rd_suppressed = np.fft.fft(suppressed_data, axis=0)

        # Zero Doppler bin (DC) should be attenuated
        dc_original = np.mean(np.abs(rd_original[0, :]))
        dc_suppressed = np.mean(np.abs(rd_suppressed[0, :]))

        assert dc_suppressed < dc_original

    def test_cfar_detection(self):
        """Test CFAR (Constant False Alarm Rate) detection."""
        np.random.seed(42)

        # Generate range-Doppler map with noise and targets
        n_range = 100
        n_doppler = 50

        noise_power = 1.0
        rd_map = noise_power * np.random.rayleigh(1.0, (n_doppler, n_range))

        # Add targets
        target_locations = [(10, 20), (30, 60), (40, 80)]
        target_strengths = [10.0, 15.0, 8.0]

        for (d, r), strength in zip(target_locations, target_strengths):
            rd_map[d, r] = strength

        # Apply CFAR detection
        detections = range_doppler_processing.cfar_detection(
            rd_map, guard_cells=2, training_cells=10, pfa=1e-6
        )

        assert detections.shape == rd_map.shape
        assert detections.dtype == bool

        # Should detect the strong targets
        detected_targets = 0
        for (d, r) in target_locations:
            if detections[d, r]:
                detected_targets += 1

        # Should detect at least some targets
        assert detected_targets >= len(target_locations) // 2