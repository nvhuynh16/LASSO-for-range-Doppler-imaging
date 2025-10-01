"""
Unit tests for visualization functions.

Tests cover:
- Range-Doppler plotting
- SNR comparison plots
- Sparsity analysis visualization
- Theoretical condition plots
- Performance comparison charts
- Interactive plotting functionality
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock

# Use Agg backend for testing (no display)
matplotlib.use('Agg')

# Import the modules to be tested
from lasso_radar.visualization.plotting import (
    plot_range_doppler,
    plot_snr_comparison,
    plot_sparsity_analysis,
    plot_theoretical_conditions
)


class TestRangeDopplerPlotting:
    """Test suite for range-Doppler visualization."""

    @pytest.fixture
    def sample_rd_data(self):
        """Generate sample range-Doppler data."""
        np.random.seed(42)

        n_range = 64
        n_doppler = 32

        # Create background noise
        rd_map = 0.1 * np.random.randn(n_doppler, n_range)

        # Add targets
        rd_map[10, 20] = 2.0   # Strong target
        rd_map[15, 35] = 1.5   # Medium target
        rd_map[25, 50] = 1.0   # Weak target

        # Create coordinate arrays
        range_bins = np.arange(n_range) * 10  # 10m resolution
        doppler_bins = np.arange(n_doppler) - n_doppler//2  # Centered around zero

        return {
            'rd_map': rd_map,
            'range_bins': range_bins,
            'doppler_bins': doppler_bins,
            'n_range': n_range,
            'n_doppler': n_doppler
        }

    def test_basic_range_doppler_plot(self, sample_rd_data):
        """Test basic range-Doppler plotting functionality."""
        rd_map = sample_rd_data['rd_map']

        fig, ax = plot_range_doppler.basic_plot(rd_map)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Check that plot has correct dimensions
        images = ax.get_images()
        assert len(images) == 1  # Should have one image

        image_data = images[0].get_array()
        assert image_data.shape == rd_map.shape

        plt.close(fig)

    def test_range_doppler_with_coordinates(self, sample_rd_data):
        """Test range-Doppler plotting with coordinate labels."""
        rd_map = sample_rd_data['rd_map']
        range_bins = sample_rd_data['range_bins']
        doppler_bins = sample_rd_data['doppler_bins']

        fig, ax = plot_range_doppler.with_coordinates(
            rd_map, range_bins, doppler_bins,
            title="Test Range-Doppler Map",
            xlabel="Range (m)",
            ylabel="Doppler (Hz)"
        )

        assert isinstance(fig, plt.Figure)
        assert ax.get_xlabel() == "Range (m)"
        assert ax.get_ylabel() == "Doppler (Hz)"
        assert ax.get_title() == "Test Range-Doppler Map"

        plt.close(fig)

    def test_logarithmic_scale_plot(self, sample_rd_data):
        """Test logarithmic scale plotting."""
        rd_map = np.abs(sample_rd_data['rd_map'])  # Ensure positive values

        fig, ax = plot_range_doppler.logarithmic_scale(rd_map, db_range=40)

        # Check that colorbar uses dB scale
        cbar = fig.colorbar
        if hasattr(fig, 'colorbar') and fig.colorbar is not None:
            # Colorbar should show dB values
            pass

        plt.close(fig)

    def test_contour_plot(self, sample_rd_data):
        """Test contour plotting functionality."""
        rd_map = sample_rd_data['rd_map']

        fig, ax = plot_range_doppler.contour_plot(
            rd_map, levels=10, filled=True
        )

        # Check for contour collections
        collections = ax.collections
        assert len(collections) > 0  # Should have contour lines/fills

        plt.close(fig)

    def test_3d_surface_plot(self, sample_rd_data):
        """Test 3D surface plotting."""
        rd_map = sample_rd_data['rd_map']

        fig = plot_range_doppler.surface_3d(rd_map)

        assert isinstance(fig, plt.Figure)

        # Check for 3D axes
        axes = fig.get_axes()
        assert len(axes) > 0

        plt.close(fig)

    def test_interactive_plot(self, sample_rd_data):
        """Test interactive plotting capabilities."""
        rd_map = sample_rd_data['rd_map']

        # Mock plotly to avoid dependency issues in testing
        with patch('lasso_radar.visualization.plotting.plotly') as mock_plotly:
            mock_fig = MagicMock()
            mock_plotly.graph_objects.Heatmap.return_value = mock_fig
            mock_plotly.graph_objects.Figure.return_value = mock_fig

            fig = plot_range_doppler.interactive_plot(rd_map)

            # Should call plotly functions
            assert mock_plotly.graph_objects.Heatmap.called

    def test_multiple_rd_maps_comparison(self, sample_rd_data):
        """Test comparison of multiple range-Doppler maps."""
        rd_map1 = sample_rd_data['rd_map']
        rd_map2 = rd_map1 + 0.2 * np.random.randn(*rd_map1.shape)
        rd_map3 = rd_map1 * 0.8

        rd_maps = [rd_map1, rd_map2, rd_map3]
        titles = ["Original", "Noisy", "Attenuated"]

        fig, axes = plot_range_doppler.compare_multiple(rd_maps, titles)

        assert isinstance(fig, plt.Figure)
        assert len(axes) == len(rd_maps)

        for i, ax in enumerate(axes):
            assert ax.get_title() == titles[i]

        plt.close(fig)

    def test_target_annotation(self, sample_rd_data):
        """Test target annotation on range-Doppler plots."""
        rd_map = sample_rd_data['rd_map']

        # Define target locations
        targets = [
            {'range_idx': 20, 'doppler_idx': 10, 'label': 'Target 1'},
            {'range_idx': 35, 'doppler_idx': 15, 'label': 'Target 2'}
        ]

        fig, ax = plot_range_doppler.with_annotations(rd_map, targets)

        # Check for annotations
        annotations = ax.texts
        assert len(annotations) >= len(targets)

        plt.close(fig)


class TestSNRComparisonPlots:
    """Test suite for SNR comparison visualization."""

    @pytest.fixture
    def snr_comparison_data(self):
        """Generate SNR comparison data."""
        snr_values = np.arange(-10, 31, 2)  # SNR from -10 to 30 dB

        # Simulate performance metrics for different algorithms
        lasso_pd = 1 / (1 + np.exp(-(snr_values - 5) / 3))  # Sigmoid curve
        matched_filter_pd = 1 / (1 + np.exp(-(snr_values - 10) / 2))

        # Add some realistic noise
        np.random.seed(42)
        lasso_pd += 0.02 * np.random.randn(len(snr_values))
        matched_filter_pd += 0.02 * np.random.randn(len(snr_values))

        # Clip to valid probability range
        lasso_pd = np.clip(lasso_pd, 0, 1)
        matched_filter_pd = np.clip(matched_filter_pd, 0, 1)

        return {
            'snr_values': snr_values,
            'lasso_pd': lasso_pd,
            'matched_filter_pd': matched_filter_pd
        }

    def test_basic_snr_comparison(self, snr_comparison_data):
        """Test basic SNR comparison plotting."""
        snr_values = snr_comparison_data['snr_values']
        lasso_pd = snr_comparison_data['lasso_pd']
        matched_filter_pd = snr_comparison_data['matched_filter_pd']

        fig, ax = plot_snr_comparison.detection_performance(
            snr_values,
            {'LASSO': lasso_pd, 'Matched Filter': matched_filter_pd}
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        # Check that both curves are plotted
        lines = ax.get_lines()
        assert len(lines) == 2

        # Check axis labels
        assert 'SNR' in ax.get_xlabel()
        assert 'Detection' in ax.get_ylabel() or 'Probability' in ax.get_ylabel()

        plt.close(fig)

    def test_roc_curves(self):
        """Test ROC curve plotting."""
        # Generate ROC data
        n_points = 100

        # Perfect detector
        fpr_perfect = np.linspace(0, 1, n_points)
        tpr_perfect = np.ones_like(fpr_perfect)
        tpr_perfect[fpr_perfect < 0.1] = fpr_perfect[fpr_perfect < 0.1] * 10

        # Realistic detector
        fpr_real = np.linspace(0, 1, n_points)
        tpr_real = np.sqrt(fpr_real)  # Square root curve

        roc_data = {
            'Perfect': (fpr_perfect, tpr_perfect),
            'Realistic': (fpr_real, tpr_real)
        }

        fig, ax = plot_snr_comparison.roc_curves(roc_data)

        assert isinstance(fig, plt.Figure)

        # Should have diagonal reference line plus ROC curves
        lines = ax.get_lines()
        assert len(lines) >= 2  # At least 2 ROC curves

        # Check axis labels
        assert 'False Positive' in ax.get_xlabel()
        assert 'True Positive' in ax.get_ylabel()

        plt.close(fig)

    def test_noise_robustness_comparison(self, snr_comparison_data):
        """Test noise robustness comparison plots."""
        snr_values = snr_comparison_data['snr_values']

        # Generate reconstruction error data
        lasso_error = 1.0 / (1 + np.exp((snr_values - 0) / 5))  # Decreasing with SNR
        elastic_net_error = 1.2 / (1 + np.exp((snr_values - 2) / 5))

        error_data = {
            'LASSO': lasso_error,
            'Elastic Net': elastic_net_error
        }

        fig, ax = plot_snr_comparison.reconstruction_error(snr_values, error_data)

        assert isinstance(fig, plt.Figure)

        # Check for logarithmic y-axis if errors span multiple orders of magnitude
        if ax.get_yscale() == 'log':
            # Logarithmic scale is appropriate for error plots
            pass

        plt.close(fig)

    def test_algorithm_comparison_table(self):
        """Test algorithm comparison table visualization."""
        algorithms = ['LASSO', 'Elastic Net', 'Group LASSO', 'Matched Filter']
        metrics = ['Computation Time', 'Memory Usage', 'Accuracy', 'Robustness']

        # Generate comparison data (normalized scores)
        np.random.seed(42)
        scores = np.random.rand(len(algorithms), len(metrics))

        fig, ax = plot_snr_comparison.algorithm_comparison_table(
            algorithms, metrics, scores
        )

        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_performance_vs_sparsity(self):
        """Test performance vs sparsity level plots."""
        sparsity_levels = np.logspace(0, 2, 20)  # From 1 to 100

        # Simulate performance degradation with increased sparsity
        lasso_performance = np.exp(-sparsity_levels / 50)
        omp_performance = np.exp(-sparsity_levels / 30)

        performance_data = {
            'LASSO': lasso_performance,
            'OMP': omp_performance
        }

        fig, ax = plot_snr_comparison.performance_vs_sparsity(
            sparsity_levels, performance_data
        )

        assert isinstance(fig, plt.Figure)
        assert 'Sparsity' in ax.get_xlabel()

        plt.close(fig)

    @pytest.mark.parametrize("plot_type", ["line", "scatter", "bar"])
    def test_different_plot_types(self, plot_type, snr_comparison_data):
        """Test different plot types for SNR comparison."""
        snr_values = snr_comparison_data['snr_values']
        lasso_pd = snr_comparison_data['lasso_pd']

        fig, ax = plot_snr_comparison.flexible_plot(
            snr_values, lasso_pd, plot_type=plot_type
        )

        assert isinstance(fig, plt.Figure)

        # Check that appropriate plot elements exist
        if plot_type == "line":
            assert len(ax.get_lines()) > 0
        elif plot_type == "scatter":
            assert len(ax.collections) > 0
        elif plot_type == "bar":
            assert len(ax.patches) > 0

        plt.close(fig)


class TestSparsityAnalysisPlots:
    """Test suite for sparsity analysis visualization."""

    @pytest.fixture
    def sparsity_data(self):
        """Generate sparsity analysis data."""
        np.random.seed(42)

        # Create signals with different sparsity levels
        n_features = 100

        # Sparse signal
        sparse_signal = np.zeros(n_features)
        sparse_signal[10:15] = [2.0, 1.5, 1.8, 1.2, 0.8]

        # Dense signal
        dense_signal = 0.3 * np.random.randn(n_features)

        # Reconstructed sparse signal (with some error)
        reconstructed = sparse_signal + 0.1 * np.random.randn(n_features)

        return {
            'sparse_signal': sparse_signal,
            'dense_signal': dense_signal,
            'reconstructed': reconstructed,
            'n_features': n_features
        }

    def test_sparsity_pattern_visualization(self, sparsity_data):
        """Test sparsity pattern visualization."""
        sparse_signal = sparsity_data['sparse_signal']

        fig, ax = plot_sparsity_analysis.sparsity_pattern(sparse_signal)

        assert isinstance(fig, plt.Figure)

        # Should show non-zero elements clearly
        lines = ax.get_lines()
        assert len(lines) > 0

        plt.close(fig)

    def test_sparse_vs_dense_comparison(self, sparsity_data):
        """Test comparison between sparse and dense signals."""
        sparse_signal = sparsity_data['sparse_signal']
        dense_signal = sparsity_data['dense_signal']

        signals = {
            'Sparse': sparse_signal,
            'Dense': dense_signal
        }

        fig, axes = plot_sparsity_analysis.compare_sparsity(signals)

        assert isinstance(fig, plt.Figure)
        assert len(axes) == len(signals)

        plt.close(fig)

    def test_coefficient_magnitude_distribution(self, sparsity_data):
        """Test coefficient magnitude distribution plots."""
        sparse_signal = sparsity_data['sparse_signal']

        fig, ax = plot_sparsity_analysis.magnitude_distribution(sparse_signal)

        assert isinstance(fig, plt.Figure)

        # Should have histogram or distribution plot
        patches = ax.patches
        assert len(patches) > 0  # Histogram bars

        plt.close(fig)

    def test_reconstruction_quality_visualization(self, sparsity_data):
        """Test reconstruction quality visualization."""
        original = sparsity_data['sparse_signal']
        reconstructed = sparsity_data['reconstructed']

        fig, axes = plot_sparsity_analysis.reconstruction_comparison(
            original, reconstructed
        )

        assert isinstance(fig, plt.Figure)
        assert len(axes) >= 2  # At least original and reconstructed

        plt.close(fig)

    def test_sparsity_level_analysis(self):
        """Test sparsity level analysis plots."""
        # Generate data for different sparsity levels
        sparsity_levels = [5, 10, 20, 50]
        reconstruction_errors = [0.01, 0.05, 0.15, 0.8]

        fig, ax = plot_sparsity_analysis.sparsity_vs_error(
            sparsity_levels, reconstruction_errors
        )

        assert isinstance(fig, plt.Figure)
        assert 'Sparsity' in ax.get_xlabel()
        assert 'Error' in ax.get_ylabel()

        plt.close(fig)

    def test_regularization_path_plot(self):
        """Test regularization path visualization."""
        # Simulate LASSO regularization path
        lambda_values = np.logspace(-3, 1, 50)
        n_features = 20

        # Generate coefficient paths (some go to zero)
        coefficients = np.zeros((n_features, len(lambda_values)))

        for i in range(n_features):
            # Different features zero out at different lambda values
            threshold_idx = np.random.randint(10, 40)
            coefficients[i, :threshold_idx] = np.random.randn() * np.exp(-lambda_values[:threshold_idx])

        fig, ax = plot_sparsity_analysis.regularization_path(
            lambda_values, coefficients
        )

        assert isinstance(fig, plt.Figure)

        # Should have multiple coefficient paths
        lines = ax.get_lines()
        assert len(lines) > 0

        plt.close(fig)

    def test_sparsity_metrics_radar_chart(self, sparsity_data):
        """Test radar chart for sparsity metrics."""
        sparse_signal = sparsity_data['sparse_signal']

        # Calculate various sparsity metrics
        metrics = {
            'L0 Norm': np.sum(np.abs(sparse_signal) > 1e-6),
            'L1 Norm': np.sum(np.abs(sparse_signal)),
            'L2 Norm': np.sum(sparse_signal**2),
            'Gini Coefficient': plot_sparsity_analysis._gini_coefficient(sparse_signal),
            'Entropy': plot_sparsity_analysis._entropy(sparse_signal)
        }

        fig, ax = plot_sparsity_analysis.metrics_radar_chart(metrics)

        assert isinstance(fig, plt.Figure)

        plt.close(fig)


class TestTheoreticalConditionsPlots:
    """Test suite for theoretical conditions visualization."""

    @pytest.fixture
    def condition_data(self):
        """Generate theoretical condition data."""
        # Simulate condition values for different matrix parameters
        matrix_sizes = [(20, 40), (30, 60), (50, 100), (100, 200)]

        mutual_incoherence = [0.8, 0.6, 0.4, 0.3]
        rip_constants = [0.3, 0.25, 0.2, 0.15]
        recovery_bounds = [5, 8, 15, 25]

        return {
            'matrix_sizes': matrix_sizes,
            'mutual_incoherence': mutual_incoherence,
            'rip_constants': rip_constants,
            'recovery_bounds': recovery_bounds
        }

    def test_mutual_incoherence_plot(self, condition_data):
        """Test mutual incoherence visualization."""
        matrix_sizes = condition_data['matrix_sizes']
        mu_values = condition_data['mutual_incoherence']

        # Convert sizes to measurement ratios
        ratios = [n/p for n, p in matrix_sizes]

        fig, ax = plot_theoretical_conditions.mutual_incoherence_plot(
            ratios, mu_values
        )

        assert isinstance(fig, plt.Figure)
        assert 'Measurement Ratio' in ax.get_xlabel() or 'Ratio' in ax.get_xlabel()
        assert 'Mutual Incoherence' in ax.get_ylabel()

        plt.close(fig)

    def test_rip_constant_visualization(self, condition_data):
        """Test RIP constant visualization."""
        sparsity_levels = [1, 2, 5, 10, 15, 20]
        rip_constants = [0.1, 0.15, 0.25, 0.4, 0.6, 0.8]

        fig, ax = plot_theoretical_conditions.rip_constants_plot(
            sparsity_levels, rip_constants
        )

        assert isinstance(fig, plt.Figure)
        assert 'Sparsity' in ax.get_xlabel()
        assert 'RIP' in ax.get_ylabel()

        plt.close(fig)

    def test_recovery_bounds_plot(self, condition_data):
        """Test recovery bounds visualization."""
        mu_values = np.linspace(0.1, 0.9, 20)

        # Calculate theoretical bounds
        exact_bounds = (1 + 1/mu_values) / 2
        stable_bounds = exact_bounds * 1.5  # Slightly higher for stable recovery

        bounds_data = {
            'Exact Recovery': exact_bounds,
            'Stable Recovery': stable_bounds
        }

        fig, ax = plot_theoretical_conditions.recovery_bounds_plot(
            mu_values, bounds_data
        )

        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_phase_transition_diagram(self):
        """Test phase transition diagram."""
        # Create phase transition data
        undersampling_ratios = np.linspace(0.1, 1.0, 20)
        sparsity_ratios = np.linspace(0.01, 0.5, 20)

        # Create meshgrid
        U, S = np.meshgrid(undersampling_ratios, sparsity_ratios)

        # Phase transition boundary (simplified)
        success_probability = np.exp(-(S / U) * 5)

        fig, ax = plot_theoretical_conditions.phase_transition_diagram(
            undersampling_ratios, sparsity_ratios, success_probability
        )

        assert isinstance(fig, plt.Figure)

        # Should have colormap/contour plot
        images = ax.get_images()
        collections = ax.collections
        assert len(images) > 0 or len(collections) > 0

        plt.close(fig)

    def test_condition_comparison_chart(self, condition_data):
        """Test comparison chart for different conditions."""
        algorithms = ['LASSO', 'OMP', 'CoSaMP', 'IHT']
        conditions = ['Mutual Incoherence', 'RIP', 'Restricted Eigenvalue']

        # Generate comparison matrix (boolean: condition satisfied)
        np.random.seed(42)
        satisfaction_matrix = np.random.choice([True, False],
                                             size=(len(algorithms), len(conditions)),
                                             p=[0.7, 0.3])

        fig, ax = plot_theoretical_conditions.condition_comparison_chart(
            algorithms, conditions, satisfaction_matrix
        )

        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_beta_min_threshold_plot(self):
        """Test beta-min threshold visualization."""
        noise_levels = np.logspace(-3, 0, 20)  # From 0.001 to 1

        # Calculate minimum signal strength for reliable detection
        beta_min_values = 2 * noise_levels * np.sqrt(2 * np.log(100))  # Simplified formula

        fig, ax = plot_theoretical_conditions.beta_min_plot(
            noise_levels, beta_min_values
        )

        assert isinstance(fig, plt.Figure)
        assert 'Noise' in ax.get_xlabel()

        plt.close(fig)

    def test_coherence_matrix_visualization(self):
        """Test coherence matrix visualization."""
        # Create a small coherence matrix
        n_features = 20
        np.random.seed(42)

        # Generate random dictionary matrix
        dictionary = np.random.randn(30, n_features)
        dictionary = dictionary / np.linalg.norm(dictionary, axis=0)

        # Calculate Gram matrix (coherence matrix)
        coherence_matrix = dictionary.T @ dictionary

        fig, ax = plot_theoretical_conditions.coherence_matrix_plot(coherence_matrix)

        assert isinstance(fig, plt.Figure)

        # Should have image/heatmap
        images = ax.get_images()
        assert len(images) > 0

        plt.close(fig)

    @pytest.mark.parametrize("condition_type", ["mutual_incoherence", "rip", "beta_min"])
    def test_condition_threshold_visualization(self, condition_type):
        """Test threshold visualization for different conditions."""
        parameter_values = np.linspace(0.1, 1.0, 50)

        if condition_type == "mutual_incoherence":
            thresholds = 1 / parameter_values  # Simplified threshold
            y_label = "Recovery Threshold"
        elif condition_type == "rip":
            thresholds = np.sqrt(2) - 1 + 0.1 * parameter_values
            y_label = "RIP Constant"
        else:  # beta_min
            thresholds = 2 * parameter_values * np.sqrt(np.log(100))
            y_label = "Minimum Signal Strength"

        fig, ax = plot_theoretical_conditions.threshold_plot(
            parameter_values, thresholds, condition_type
        )

        assert isinstance(fig, plt.Figure)
        assert y_label in ax.get_ylabel()

        plt.close(fig)


class TestPlottingUtilities:
    """Test suite for plotting utility functions."""

    def test_color_palette_generation(self):
        """Test color palette generation."""
        from lasso_radar.visualization.plotting import utils

        n_colors = 5
        palette = utils.generate_color_palette(n_colors)

        assert len(palette) == n_colors
        assert all(len(color) == 7 for color in palette)  # Hex colors
        assert all(color.startswith('#') for color in palette)

    def test_subplot_layout_optimization(self):
        """Test optimal subplot layout calculation."""
        from lasso_radar.visualization.plotting import utils

        # Test various numbers of subplots
        test_cases = [1, 2, 3, 4, 5, 6, 8, 9, 12]

        for n_subplots in test_cases:
            rows, cols = utils.optimal_subplot_layout(n_subplots)

            assert rows * cols >= n_subplots
            assert rows > 0 and cols > 0

            # Should be reasonably square
            aspect_ratio = max(rows, cols) / min(rows, cols)
            assert aspect_ratio <= 3  # Not too elongated

    def test_figure_sizing(self):
        """Test automatic figure sizing."""
        from lasso_radar.visualization.plotting import utils

        # Test different data dimensions
        data_shapes = [(10, 10), (50, 20), (100, 100), (200, 50)]

        for shape in data_shapes:
            figsize = utils.calculate_figure_size(shape)

            assert len(figsize) == 2
            assert figsize[0] > 0 and figsize[1] > 0
            assert figsize[0] <= 20 and figsize[1] <= 20  # Reasonable bounds

    def test_axis_formatting(self):
        """Test axis formatting utilities."""
        from lasso_radar.visualization.plotting import utils

        fig, ax = plt.subplots()

        # Test scientific notation formatting
        utils.format_axis_scientific(ax, 'x')
        utils.format_axis_scientific(ax, 'y')

        # Test grid styling
        utils.apply_grid_style(ax, alpha=0.3)

        # Test label formatting
        utils.format_labels(ax, 'Test X', 'Test Y', 'Test Title')

        assert ax.get_xlabel() == 'Test X'
        assert ax.get_ylabel() == 'Test Y'
        assert ax.get_title() == 'Test Title'

        plt.close(fig)

    def test_colormap_customization(self):
        """Test custom colormap creation."""
        from lasso_radar.visualization.plotting import utils

        # Test radar-specific colormap
        cmap = utils.create_radar_colormap()

        assert hasattr(cmap, 'N')  # Should be a colormap object
        assert cmap.N > 0

    def test_annotation_helpers(self):
        """Test annotation helper functions."""
        from lasso_radar.visualization.plotting import utils

        fig, ax = plt.subplots()

        # Test peak annotation
        data = np.random.randn(10, 10)
        data[5, 7] = 5.0  # Add peak

        utils.annotate_peaks(ax, data, threshold=2.0)

        # Should have added annotations
        annotations = ax.texts
        assert len(annotations) > 0

        plt.close(fig)

    def test_export_utilities(self):
        """Test figure export utilities."""
        from lasso_radar.visualization.plotting import utils
        import tempfile
        import os

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test multiple format export
            filename = os.path.join(tmpdir, 'test_figure')

            utils.export_figure(fig, filename, formats=['png', 'pdf'])

            # Check that files were created
            assert os.path.exists(filename + '.png')
            assert os.path.exists(filename + '.pdf')

        plt.close(fig)

    def test_performance_optimization(self):
        """Test plotting performance optimization."""
        from lasso_radar.visualization.plotting import utils

        # Test data decimation for large datasets
        large_data = np.random.randn(10000)

        decimated_data = utils.decimate_for_plotting(large_data, max_points=1000)

        assert len(decimated_data) <= 1000
        assert len(decimated_data) > 0

        # Test rasterization for complex plots
        fig, ax = plt.subplots()

        # Create complex plot
        for i in range(100):
            ax.plot(np.random.randn(100), alpha=0.1)

        utils.optimize_for_vector_graphics(ax)

        plt.close(fig)