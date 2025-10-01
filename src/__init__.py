"""LASSO and Variants for Radar Range-Doppler Imaging.

This package implements LASSO (Least Absolute Shrinkage and Selection Operator)
and its variants for sparse radar range-Doppler imaging. It provides tools for
compressed sensing in radar applications, including theoretical analysis of
conditions like Restricted Eigenvalue, Mutual Incoherence, and Beta-min.

Modules:
    algorithms: Core LASSO algorithms and variants
    signal_design: Coprime signal design and waveform generation
    utils: Utility functions for radar signal processing
    visualization: Plotting and analysis tools
"""

__version__ = "1.0.0"
__author__ = "Radar Signal Processing Team"
__email__ = "contact@example.com"

# Import key classes and functions for easy access
from .algorithms import (
    LassoRadar,
    ElasticNetRadar,
    GroupLassoRadar,
    MatchedFilter,
)
from .signal_design import (
    CoprimeSignalDesigner,
    WaveformGenerator,
)
from .utils import (
    snr_calculator,
    performance_metrics,
    theoretical_conditions,
)
from .visualization import (
    plot_range_doppler,
    plot_snr_comparison,
    plot_sparsity_analysis,
)

__all__ = [
    "LassoRadar",
    "ElasticNetRadar", 
    "GroupLassoRadar",
    "MatchedFilter",
    "CoprimeSignalDesigner",
    "WaveformGenerator",
    "snr_calculator",
    "performance_metrics",
    "theoretical_conditions",
    "plot_range_doppler",
    "plot_snr_comparison",
    "plot_sparsity_analysis",
]