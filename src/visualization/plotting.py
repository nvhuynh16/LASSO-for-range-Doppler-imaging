"""
Visualization functions for LASSO radar analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple


def plot_range_doppler(rd_map: np.ndarray, title: str = "Range-Doppler Map",
                      range_bins: Optional[np.ndarray] = None,
                      doppler_bins: Optional[np.ndarray] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Plot range-Doppler map."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if range_bins is None:
        range_bins = np.arange(rd_map.shape[1])
    if doppler_bins is None:
        doppler_bins = np.arange(rd_map.shape[0])

    im = ax.imshow(np.abs(rd_map), aspect='auto', origin='lower',
                   extent=[range_bins[0], range_bins[-1], doppler_bins[0], doppler_bins[-1]])

    ax.set_xlabel('Range')
    ax.set_ylabel('Doppler')
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label='Magnitude')

    return fig, ax


def plot_snr_comparison(snr_values: np.ndarray, performance_data: Dict[str, np.ndarray],
                       title: str = "SNR Performance Comparison") -> Tuple[plt.Figure, plt.Axes]:
    """Plot SNR vs performance comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm, performance in performance_data.items():
        ax.plot(snr_values, performance, marker='o', label=algorithm)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Detection Probability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


def plot_sparsity_analysis(signal: np.ndarray, title: str = "Sparsity Analysis") -> Tuple[plt.Figure, plt.Axes]:
    """Plot sparsity pattern of signal."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Coefficient magnitude plot
    ax1.stem(np.arange(len(signal)), np.abs(signal), basefmt=" ")
    ax1.set_xlabel('Coefficient Index')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'{title} - Coefficient Magnitudes')
    ax1.grid(True, alpha=0.3)

    # Histogram of magnitudes
    nonzero_magnitudes = np.abs(signal)[np.abs(signal) > 1e-6]
    if len(nonzero_magnitudes) > 0:
        ax2.hist(nonzero_magnitudes, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Coefficient Magnitude')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{title} - Magnitude Distribution')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_theoretical_conditions(conditions_data: Dict[str, Any],
                               title: str = "Theoretical Conditions") -> Tuple[plt.Figure, plt.Axes]:
    """Plot theoretical conditions analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    condition_names = list(conditions_data.keys())
    condition_values = list(conditions_data.values())

    bars = ax.bar(condition_names, condition_values)
    ax.set_ylabel('Condition Value')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)

    # Color bars based on values (green for good, red for poor)
    for bar, value in zip(bars, condition_values):
        if isinstance(value, (int, float)):
            if value < 0.5:
                bar.set_color('green')
            elif value < 1.0:
                bar.set_color('orange')
            else:
                bar.set_color('red')

    plt.tight_layout()
    return fig, ax


# Add namespace-like access
class RangeDopplerPlots:
    """Range-Doppler plotting utilities."""

    @staticmethod
    def basic_plot(rd_map: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        return plot_range_doppler(rd_map)

    @staticmethod
    def with_coordinates(rd_map: np.ndarray, range_bins: np.ndarray,
                        doppler_bins: np.ndarray, title: str = "Range-Doppler Map",
                        xlabel: str = "Range", ylabel: str = "Doppler") -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plot_range_doppler(rd_map, title, range_bins, doppler_bins)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def logarithmic_scale(rd_map: np.ndarray, db_range: float = 40) -> Tuple[plt.Figure, plt.Axes]:
        """Plot in logarithmic scale (dB)."""
        rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-10)
        max_val = np.max(rd_map_db)
        rd_map_db = np.clip(rd_map_db, max_val - db_range, max_val)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(rd_map_db, aspect='auto', origin='lower')
        ax.set_title('Range-Doppler Map (dB)')
        plt.colorbar(im, ax=ax, label='Magnitude (dB)')

        return fig, ax


    @staticmethod
    def contour_plot(rd_map: np.ndarray, levels: int = 10, filled: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """Create contour plot of range-Doppler map."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if filled:
            cs = ax.contourf(np.abs(rd_map), levels=levels)
        else:
            cs = ax.contour(np.abs(rd_map), levels=levels)

        ax.set_title('Range-Doppler Contour Plot')
        plt.colorbar(cs, ax=ax)
        return fig, ax

    @staticmethod
    def surface_3d(rd_map: np.ndarray) -> plt.Figure:
        """Create 3D surface plot."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(range(rd_map.shape[1]), range(rd_map.shape[0]))
        ax.plot_surface(X, Y, np.abs(rd_map), cmap='viridis')

        ax.set_title('3D Range-Doppler Surface')
        return fig

    @staticmethod
    def interactive_plot(rd_map: np.ndarray):
        """Create interactive plot (mock implementation)."""
        # Mock implementation for testing
        return {'type': 'interactive_plot', 'data': rd_map.shape}

    @staticmethod
    def compare_multiple(rd_maps: List[np.ndarray], titles: List[str]) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Compare multiple range-Doppler maps."""
        n_maps = len(rd_maps)
        fig, axes = plt.subplots(1, n_maps, figsize=(5*n_maps, 5))

        if n_maps == 1:
            axes = [axes]

        for i, (rd_map, title) in enumerate(zip(rd_maps, titles)):
            im = axes[i].imshow(np.abs(rd_map), aspect='auto', origin='lower')
            axes[i].set_title(title)
            plt.colorbar(im, ax=axes[i])

        return fig, axes

    @staticmethod
    def with_annotations(rd_map: np.ndarray, targets: List[Dict]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot with target annotations."""
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(np.abs(rd_map), aspect='auto', origin='lower')

        for target in targets:
            ax.annotate(target['label'],
                       (target['range_idx'], target['doppler_idx']),
                       color='red', fontsize=10)

        plt.colorbar(im, ax=ax)
        return fig, ax


# Add all methods to the plot_range_doppler namespace
plot_range_doppler.basic_plot = RangeDopplerPlots.basic_plot
plot_range_doppler.with_coordinates = RangeDopplerPlots.with_coordinates
plot_range_doppler.logarithmic_scale = RangeDopplerPlots.logarithmic_scale
plot_range_doppler.contour_plot = RangeDopplerPlots.contour_plot
plot_range_doppler.surface_3d = RangeDopplerPlots.surface_3d
plot_range_doppler.interactive_plot = RangeDopplerPlots.interactive_plot
plot_range_doppler.compare_multiple = RangeDopplerPlots.compare_multiple
plot_range_doppler.with_annotations = RangeDopplerPlots.with_annotations


# Additional plotting functions referenced in tests
class SNRComparisonPlots:
    """SNR comparison plotting utilities."""

    @staticmethod
    def detection_performance(snr_values: np.ndarray, performance_data: Dict[str, np.ndarray]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot detection performance vs SNR."""
        return plot_snr_comparison(snr_values, performance_data)

    @staticmethod
    def roc_curves(roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot ROC curves."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)

        for name, (fpr, tpr) in roc_data.items():
            ax.plot(fpr, tpr, label=name, linewidth=2)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def reconstruction_error(snr_values: np.ndarray, error_data: Dict[str, np.ndarray]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot reconstruction error vs SNR."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for algorithm, errors in error_data.items():
            ax.semilogy(snr_values, errors, marker='o', label=algorithm)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Reconstruction Error vs SNR')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def algorithm_comparison_table(algorithms: List[str], metrics: List[str], scores: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Create comparison table visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(scores, cmap='RdYlGn', aspect='auto')

        ax.set_xticks(range(len(metrics)))
        ax.set_yticks(range(len(algorithms)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(algorithms)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_title('Algorithm Comparison')
        plt.colorbar(im, ax=ax)

        return fig, ax

    @staticmethod
    def performance_vs_sparsity(sparsity_levels: np.ndarray, performance_data: Dict[str, np.ndarray]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot performance vs sparsity level."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for algorithm, performance in performance_data.items():
            ax.semilogx(sparsity_levels, performance, marker='o', label=algorithm)

        ax.set_xlabel('Sparsity Level')
        ax.set_ylabel('Performance')
        ax.set_title('Performance vs Sparsity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def flexible_plot(x_data: np.ndarray, y_data: np.ndarray, plot_type: str = "line") -> Tuple[plt.Figure, plt.Axes]:
        """Flexible plotting with different types."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == "line":
            ax.plot(x_data, y_data, marker='o')
        elif plot_type == "scatter":
            ax.scatter(x_data, y_data)
        elif plot_type == "bar":
            ax.bar(x_data, y_data)

        return fig, ax


# Add SNR comparison functions
plot_snr_comparison.detection_performance = SNRComparisonPlots.detection_performance
plot_snr_comparison.roc_curves = SNRComparisonPlots.roc_curves
plot_snr_comparison.reconstruction_error = SNRComparisonPlots.reconstruction_error
plot_snr_comparison.algorithm_comparison_table = SNRComparisonPlots.algorithm_comparison_table
plot_snr_comparison.performance_vs_sparsity = SNRComparisonPlots.performance_vs_sparsity
plot_snr_comparison.flexible_plot = SNRComparisonPlots.flexible_plot


# Sparsity analysis plots
class SparsityAnalysisPlots:
    """Sparsity analysis plotting utilities."""

    @staticmethod
    def sparsity_pattern(signal: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot sparsity pattern."""
        return plot_sparsity_analysis(signal, "Sparsity Pattern")

    @staticmethod
    def compare_sparsity(signals: Dict[str, np.ndarray]) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Compare sparsity between signals."""
        n_signals = len(signals)
        fig, axes = plt.subplots(1, n_signals, figsize=(5*n_signals, 5))

        if n_signals == 1:
            axes = [axes]

        for i, (name, signal) in enumerate(signals.items()):
            axes[i].stem(range(len(signal)), np.abs(signal), basefmt=" ")
            axes[i].set_title(f"{name} - Sparsity Pattern")
            axes[i].set_xlabel('Index')
            axes[i].set_ylabel('Magnitude')

        return fig, axes

    @staticmethod
    def magnitude_distribution(signal: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot magnitude distribution."""
        fig, ax = plt.subplots(figsize=(8, 6))

        magnitudes = np.abs(signal)[np.abs(signal) > 1e-6]
        if len(magnitudes) > 0:
            ax.hist(magnitudes, bins=20, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Count')
        ax.set_title('Magnitude Distribution')

        return fig, ax

    @staticmethod
    def reconstruction_comparison(original: np.ndarray, reconstructed: np.ndarray) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Compare original and reconstructed signals."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.stem(range(len(original)), np.abs(original), basefmt=" ")
        ax1.set_title('Original Signal')
        ax1.set_ylabel('Magnitude')

        ax2.stem(range(len(reconstructed)), np.abs(reconstructed), basefmt=" ")
        ax2.set_title('Reconstructed Signal')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Magnitude')

        return fig, [ax1, ax2]

    @staticmethod
    def sparsity_vs_error(sparsity_levels: List[int], errors: List[float]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot sparsity vs reconstruction error."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(sparsity_levels, errors, marker='o')
        ax.set_xlabel('Sparsity Level')
        ax.set_ylabel('Reconstruction Error')
        ax.set_title('Sparsity vs Reconstruction Error')
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def regularization_path(lambda_values: np.ndarray, coefficients: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot regularization path."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(coefficients.shape[0]):
            ax.semilogx(lambda_values, coefficients[i, :], alpha=0.7)

        ax.set_xlabel('Regularization Parameter')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Regularization Path')
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def metrics_radar_chart(metrics: Dict[str, float]) -> Tuple[plt.Figure, plt.Axes]:
        """Create radar chart for sparsity metrics."""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = list(metrics.values())

        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles)
        ax.set_xticklabels(list(metrics.keys()))
        ax.set_title('Sparsity Metrics')

        return fig, ax

    @staticmethod
    def _gini_coefficient(signal: np.ndarray) -> float:
        """Calculate Gini coefficient."""
        magnitude = np.abs(signal)
        sorted_vals = np.sort(magnitude)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n

    @staticmethod
    def _entropy(signal: np.ndarray) -> float:
        """Calculate entropy of signal."""
        magnitude = np.abs(signal)
        magnitude = magnitude / np.sum(magnitude)
        magnitude = magnitude[magnitude > 0]
        return -np.sum(magnitude * np.log2(magnitude))


# Add sparsity analysis functions
plot_sparsity_analysis.sparsity_pattern = SparsityAnalysisPlots.sparsity_pattern
plot_sparsity_analysis.compare_sparsity = SparsityAnalysisPlots.compare_sparsity
plot_sparsity_analysis.magnitude_distribution = SparsityAnalysisPlots.magnitude_distribution
plot_sparsity_analysis.reconstruction_comparison = SparsityAnalysisPlots.reconstruction_comparison
plot_sparsity_analysis.sparsity_vs_error = SparsityAnalysisPlots.sparsity_vs_error
plot_sparsity_analysis.regularization_path = SparsityAnalysisPlots.regularization_path
plot_sparsity_analysis.metrics_radar_chart = SparsityAnalysisPlots.metrics_radar_chart
plot_sparsity_analysis._gini_coefficient = SparsityAnalysisPlots._gini_coefficient
plot_sparsity_analysis._entropy = SparsityAnalysisPlots._entropy


# Theoretical conditions plots
class TheoreticalConditionsPlots:
    """Theoretical conditions plotting utilities."""

    @staticmethod
    def mutual_incoherence_plot(ratios: List[float], mu_values: List[float]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot mutual incoherence vs measurement ratio."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(ratios, mu_values, marker='o', linewidth=2)
        ax.set_xlabel('Measurement Ratio (m/n)')
        ax.set_ylabel('Mutual Incoherence')
        ax.set_title('Mutual Incoherence vs Measurement Ratio')
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def rip_constants_plot(sparsity_levels: List[int], rip_constants: List[float]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot RIP constants vs sparsity."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(sparsity_levels, rip_constants, marker='s', linewidth=2)
        ax.axhline(y=np.sqrt(2) - 1, color='red', linestyle='--', label='RIP Threshold')
        ax.set_xlabel('Sparsity Level')
        ax.set_ylabel('RIP Constant (δₛ)')
        ax.set_title('Restricted Isometry Property Constants')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def recovery_bounds_plot(mu_values: np.ndarray, bounds_data: Dict[str, np.ndarray]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot recovery bounds vs mutual incoherence."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, bounds in bounds_data.items():
            ax.plot(mu_values, bounds, marker='o', label=name, linewidth=2)

        ax.set_xlabel('Mutual Incoherence')
        ax.set_ylabel('Recovery Bound')
        ax.set_title('Recovery Bounds vs Mutual Incoherence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def phase_transition_diagram(undersampling_ratios: np.ndarray, sparsity_ratios: np.ndarray,
                               success_probability: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot phase transition diagram."""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.contourf(undersampling_ratios, sparsity_ratios, success_probability,
                        levels=np.linspace(0, 1, 11), cmap='RdYlGn')

        ax.set_xlabel('Undersampling Ratio (m/n)')
        ax.set_ylabel('Sparsity Ratio (k/n)')
        ax.set_title('LASSO Phase Transition Diagram')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Success Probability')

        return fig, ax

    @staticmethod
    def condition_comparison_chart(algorithms: List[str], conditions: List[str],
                                 satisfaction_matrix: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot condition satisfaction comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(satisfaction_matrix.astype(float), cmap='RdYlGn', aspect='auto')

        ax.set_xticks(range(len(conditions)))
        ax.set_yticks(range(len(algorithms)))
        ax.set_xticklabels(conditions)
        ax.set_yticklabels(algorithms)

        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(conditions)):
                text = '✓' if satisfaction_matrix[i, j] else '✗'
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=12)

        ax.set_title('Theoretical Condition Satisfaction')
        plt.colorbar(im, ax=ax)

        return fig, ax

    @staticmethod
    def beta_min_plot(noise_levels: np.ndarray, beta_min_values: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot beta-min threshold vs noise level."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.loglog(noise_levels, beta_min_values, marker='o', linewidth=2)
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Minimum Signal Strength (β-min)')
        ax.set_title('Beta-Min Condition vs Noise Level')
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def coherence_matrix_plot(coherence_matrix: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
        """Plot coherence matrix heatmap."""
        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(np.abs(coherence_matrix), cmap='viridis')
        ax.set_title('Coherence Matrix (|G|)')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')

        plt.colorbar(im, ax=ax)

        return fig, ax

    @staticmethod
    def threshold_plot(parameter_values: np.ndarray, thresholds: np.ndarray,
                      condition_type: str) -> Tuple[plt.Figure, plt.Axes]:
        """Plot threshold vs parameter for different conditions."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(parameter_values, thresholds, marker='o', linewidth=2)
        ax.set_xlabel('Parameter Value')

        if condition_type == "mutual_incoherence":
            ax.set_ylabel('Recovery Threshold')
            ax.set_title('Mutual Incoherence Recovery Threshold')
        elif condition_type == "rip":
            ax.set_ylabel('RIP Constant')
            ax.set_title('RIP Constant Threshold')
        elif condition_type == "beta_min":
            ax.set_ylabel('Minimum Signal Strength')
            ax.set_title('Beta-Min Threshold')

        ax.grid(True, alpha=0.3)

        return fig, ax


# Add theoretical conditions functions
plot_theoretical_conditions.mutual_incoherence_plot = TheoreticalConditionsPlots.mutual_incoherence_plot
plot_theoretical_conditions.rip_constants_plot = TheoreticalConditionsPlots.rip_constants_plot
plot_theoretical_conditions.recovery_bounds_plot = TheoreticalConditionsPlots.recovery_bounds_plot
plot_theoretical_conditions.phase_transition_diagram = TheoreticalConditionsPlots.phase_transition_diagram
plot_theoretical_conditions.condition_comparison_chart = TheoreticalConditionsPlots.condition_comparison_chart
plot_theoretical_conditions.beta_min_plot = TheoreticalConditionsPlots.beta_min_plot
plot_theoretical_conditions.coherence_matrix_plot = TheoreticalConditionsPlots.coherence_matrix_plot
plot_theoretical_conditions.threshold_plot = TheoreticalConditionsPlots.threshold_plot


# Plotting utility functions
class PlottingUtils:
    """Utility functions for plotting and visualization."""

    @staticmethod
    def generate_color_palette(n_colors: int) -> List[str]:
        """Generate a color palette with specified number of colors."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            # Convert HSV to RGB to hex
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            colors.append(hex_color)
        return colors

    @staticmethod
    def optimal_subplot_layout(n_subplots: int) -> Tuple[int, int]:
        """Calculate optimal subplot layout (rows, cols) for given number of subplots."""
        if n_subplots == 1:
            return 1, 1

        # Find factors close to square root
        sqrt_n = int(np.sqrt(n_subplots))

        # Try to find a good rectangular layout
        for cols in range(sqrt_n, n_subplots + 1):
            if n_subplots % cols == 0:
                rows = n_subplots // cols
                return rows, cols

        # If no perfect factorization, use ceiling
        cols = sqrt_n + 1
        rows = int(np.ceil(n_subplots / cols))
        return rows, cols

    @staticmethod
    def calculate_figure_size(data_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate appropriate figure size based on data dimensions."""
        height, width = data_shape

        # Base size
        base_size = 6.0

        # Calculate aspect ratio
        aspect_ratio = width / height

        if aspect_ratio > 2:
            # Wide data
            figsize = (base_size * 1.5, base_size * 0.75)
        elif aspect_ratio < 0.5:
            # Tall data
            figsize = (base_size * 0.75, base_size * 1.5)
        else:
            # Roughly square
            figsize = (base_size, base_size)

        # Ensure reasonable bounds
        figsize = (
            max(4, min(20, figsize[0])),
            max(4, min(20, figsize[1]))
        )

        return figsize

    @staticmethod
    def format_axis_scientific(ax: plt.Axes, axis: str) -> None:
        """Format axis to use scientific notation."""
        from matplotlib.ticker import ScalarFormatter

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))

        if axis.lower() == 'x':
            ax.xaxis.set_major_formatter(formatter)
        elif axis.lower() == 'y':
            ax.yaxis.set_major_formatter(formatter)

    @staticmethod
    def apply_grid_style(ax: plt.Axes, alpha: float = 0.3) -> None:
        """Apply consistent grid styling to axes."""
        ax.grid(True, alpha=alpha, linewidth=0.5, linestyle='-')
        ax.set_axisbelow(True)

    @staticmethod
    def format_labels(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
        """Apply consistent label formatting."""
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

    @staticmethod
    def create_radar_colormap():
        """Create custom colormap for radar applications."""
        from matplotlib.colors import LinearSegmentedColormap

        # Define colors: dark blue -> blue -> green -> yellow -> red
        colors = ['#000033', '#000080', '#0080ff', '#00ff80', '#ffff00', '#ff8000', '#ff0000']
        n_bins = 256

        cmap = LinearSegmentedColormap.from_list('radar', colors, N=n_bins)
        return cmap

    @staticmethod
    def annotate_peaks(ax: plt.Axes, data: np.ndarray, threshold: float = 0.8) -> None:
        """Annotate peaks in 2D data above threshold."""
        # Find peaks above threshold
        normalized_data = data / np.max(np.abs(data))
        peak_indices = np.where(normalized_data > threshold)

        for i, j in zip(peak_indices[0], peak_indices[1]):
            ax.annotate(f'({i},{j})',
                       xy=(j, i),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    @staticmethod
    def export_figure(fig: plt.Figure, filename: str, formats: List[str] = None) -> None:
        """Export figure in multiple formats."""
        if formats is None:
            formats = ['png']

        for fmt in formats:
            full_filename = f"{filename}.{fmt}"
            fig.savefig(full_filename, format=fmt, dpi=300, bbox_inches='tight')

    @staticmethod
    def decimate_for_plotting(data: np.ndarray, max_points: int = 1000) -> np.ndarray:
        """Decimate data for efficient plotting of large datasets."""
        if len(data) <= max_points:
            return data

        # Simple decimation by taking every nth point
        step = len(data) // max_points
        return data[::step]

    @staticmethod
    def optimize_for_vector_graphics(ax: plt.Axes) -> None:
        """Optimize plot for vector graphics output."""
        # Rasterize complex elements to reduce file size
        for collection in ax.collections:
            if len(collection.get_offsets() if hasattr(collection, 'get_offsets') else []) > 1000:
                collection.set_rasterized(True)


# Create utils namespace for backward compatibility
class Utils:
    """Namespace for plotting utilities."""
    generate_color_palette = PlottingUtils.generate_color_palette
    optimal_subplot_layout = PlottingUtils.optimal_subplot_layout
    calculate_figure_size = PlottingUtils.calculate_figure_size
    format_axis_scientific = PlottingUtils.format_axis_scientific
    apply_grid_style = PlottingUtils.apply_grid_style
    format_labels = PlottingUtils.format_labels
    create_radar_colormap = PlottingUtils.create_radar_colormap
    annotate_peaks = PlottingUtils.annotate_peaks
    export_figure = PlottingUtils.export_figure
    decimate_for_plotting = PlottingUtils.decimate_for_plotting
    optimize_for_vector_graphics = PlottingUtils.optimize_for_vector_graphics


# Create utils instance for imports
utils = Utils()