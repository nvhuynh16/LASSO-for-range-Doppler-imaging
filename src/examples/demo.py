#!/usr/bin/env python3
"""
LASSO Radar Demo Script

This demo showcases the basic functionality of the LASSO radar package
with a simple range-Doppler imaging example.
"""

import numpy as np
import matplotlib.pyplot as plt
from lasso_radar.algorithms.lasso_core import LassoRadar
from lasso_radar.algorithms.elastic_net import ElasticNetRadar
from lasso_radar.utils.metrics import snr_calculator
from lasso_radar.visualization.plotting import RangeDopplerPlots


def create_demo_scenario():
    """Create a simple radar scenario for demonstration."""
    print("Creating demo radar scenario...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Scenario parameters
    n_range, n_doppler = 32, 16
    n_measurements = 128

    # Create sparse target scene
    scene = np.zeros((n_doppler, n_range))

    # Add targets: [Doppler_bin, Range_bin] = amplitude
    scene[5, 10] = 1.0    # Strong target
    scene[8, 20] = 0.6    # Medium target
    scene[12, 25] = 0.3   # Weak target

    print(f"  - Scene size: {n_range} range bins × {n_doppler} Doppler bins")
    print(f"  - Number of targets: {np.sum(scene > 0)}")
    print(f"  - Measurements: {n_measurements}")

    # Generate measurement matrix
    A = np.random.randn(n_measurements, n_range * n_doppler)
    A = A / np.linalg.norm(A, axis=0)  # Normalize columns

    # Simulate radar measurements
    scene_vector = scene.flatten()
    clean_measurements = A @ scene_vector

    # Add noise
    noise_level = 0.01
    noisy_measurements = clean_measurements + noise_level * np.random.randn(n_measurements)

    # Calculate SNR
    signal_power = np.var(clean_measurements)
    noise_power = noise_level**2
    snr_db = snr_calculator(signal_power, noise_power)
    print(f"  - SNR: {snr_db:.1f} dB")

    return {
        'scene': scene,
        'measurement_matrix': A,
        'measurements': noisy_measurements,
        'n_range': n_range,
        'n_doppler': n_doppler,
        'snr_db': snr_db
    }


def run_lasso_demo(scenario):
    """Run LASSO reconstruction demo."""
    print("\nRunning LASSO reconstruction...")

    # Standard LASSO
    lasso = LassoRadar(lambda_reg=0.005, verbose=True)
    lasso.fit(scenario['measurement_matrix'], scenario['measurements'])

    # Get range-Doppler map
    lasso_scene = lasso.get_range_doppler_map(
        scenario['n_range'],
        scenario['n_doppler']
    )

    # Calculate performance metrics
    reconstruction_error = np.linalg.norm(lasso_scene - scenario['scene'])
    sparsity_achieved = np.sum(np.abs(lasso_scene) > 0.01)

    print(f"  - Converged: {lasso.converged_}")
    print(f"  - Iterations: {lasso.n_iterations_}")
    print(f"  - Reconstruction error: {reconstruction_error:.4f}")
    print(f"  - Sparsity: {sparsity_achieved} non-zero elements")

    return lasso_scene


def run_elastic_net_demo(scenario):
    """Run Elastic Net reconstruction demo."""
    print("\nRunning Elastic Net reconstruction...")

    # Elastic Net (balance between L1 and L2)
    elastic_net = ElasticNetRadar(lambda_reg=0.005, alpha=0.5, verbose=True)
    elastic_net.fit(scenario['measurement_matrix'], scenario['measurements'])

    # Get range-Doppler map
    elastic_scene = elastic_net.get_range_doppler_map(
        scenario['n_range'],
        scenario['n_doppler']
    )

    # Calculate performance metrics
    reconstruction_error = np.linalg.norm(elastic_scene - scenario['scene'])
    sparsity_achieved = np.sum(np.abs(elastic_scene) > 0.01)

    print(f"  - Converged: {elastic_net.converged_}")
    print(f"  - Iterations: {elastic_net.n_iterations_}")
    print(f"  - Reconstruction error: {reconstruction_error:.4f}")
    print(f"  - Sparsity: {sparsity_achieved} non-zero elements")

    return elastic_scene


def visualize_results(scenario, lasso_scene, elastic_scene):
    """Create visualization of results."""
    print("\nGenerating visualization...")

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original scene
        im1 = axes[0].imshow(np.abs(scenario['scene']), aspect='auto', origin='lower')
        axes[0].set_title('Original Scene')
        axes[0].set_xlabel('Range Bin')
        axes[0].set_ylabel('Doppler Bin')
        plt.colorbar(im1, ax=axes[0])

        # LASSO reconstruction
        im2 = axes[1].imshow(np.abs(lasso_scene), aspect='auto', origin='lower')
        axes[1].set_title('LASSO Reconstruction')
        axes[1].set_xlabel('Range Bin')
        axes[1].set_ylabel('Doppler Bin')
        plt.colorbar(im2, ax=axes[1])

        # Elastic Net reconstruction
        im3 = axes[2].imshow(np.abs(elastic_scene), aspect='auto', origin='lower')
        axes[2].set_title('Elastic Net Reconstruction')
        axes[2].set_xlabel('Range Bin')
        axes[2].set_ylabel('Doppler Bin')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()

        print("  - Visualization displayed")

    except Exception as e:
        print(f"  - Visualization failed: {e}")
        print("  - (This is normal in headless environments)")


def print_educational_summary():
    """Print educational summary about LASSO in radar."""
    print("\n" + "="*60)
    print("EDUCATIONAL SUMMARY")
    print("="*60)
    print("""
LASSO (Least Absolute Shrinkage and Selection Operator) for radar applications:

KEY STRENGTHS:
• Sparse recovery: Exploits the fact that radar scenes typically contain few targets
• Super-resolution: Can resolve targets beyond traditional Rayleigh limit
• Compressed sensing: Works with fewer measurements than conventional methods
• Sidelobe suppression: L1 penalty naturally reduces spurious peaks

KEY LIMITATIONS:
• Low SNR performance: Degrades significantly below ~5 dB SNR
• Computational cost: O(N²) vs O(N log N) for matched filtering
• Parameter sensitivity: Requires careful regularization tuning
• Dense scene failure: Breaks down when sparsity assumption is violated

WHEN TO USE LASSO:
✓ High SNR scenarios (>10 dB)
✓ Sparse target environments
✓ Super-resolution requirements
✓ Hardware constraints (sub-Nyquist sampling)

WHEN TO AVOID LASSO:
✗ Low SNR conditions (<5 dB)
✗ Dense clutter environments
✗ Real-time processing constraints
✗ When traditional methods meet requirements

ELASTIC NET ADVANTAGES:
• Handles correlated features better than pure LASSO
• Provides grouping effect for clustered targets
• More stable selection in high-dimensional problems
• Balances sparsity (L1) with smoothness (L2)
""")
    print("="*60)


def main():
    """Main demo function."""
    print("LASSO Radar Package Demo")
    print("=" * 40)
    print("This demo showcases sparse reconstruction for radar range-Doppler imaging.")
    print()

    try:
        # Create scenario
        scenario = create_demo_scenario()

        # Run algorithms
        lasso_scene = run_lasso_demo(scenario)
        elastic_scene = run_elastic_net_demo(scenario)

        # Visualize results
        visualize_results(scenario, lasso_scene, elastic_scene)

        # Educational summary
        print_educational_summary()

        print("\nDemo completed successfully!")
        print("For more advanced examples, see the documentation at:")
        print("https://github.com/your-username/lasso-radar")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Please check your installation and try again.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())