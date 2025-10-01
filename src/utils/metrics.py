"""
Performance metrics and evaluation utilities for radar LASSO algorithms.
"""

import numpy as np
from typing import Dict, Any, Tuple
import warnings


def snr_calculator():
    """SNR calculation utilities namespace."""
    pass


def snr_from_components(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate SNR from signal and noise components."""
    signal_power = np.var(signal)
    noise_power = np.var(noise)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf


def estimate_snr_moments(noisy_signal: np.ndarray) -> float:
    """Estimate SNR using moment-based method."""
    # Simple moment-based estimator
    signal_power = np.var(noisy_signal)
    noise_power = np.median(np.abs(noisy_signal - np.mean(noisy_signal))**2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0


def estimate_snr_spectral(noisy_signal: np.ndarray) -> float:
    """Estimate SNR using spectral method."""
    spectrum = np.abs(np.fft.fft(noisy_signal))**2
    signal_power = np.max(spectrum)
    noise_power = np.median(spectrum)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0


def estimate_snr_quantiles(noisy_signal: np.ndarray) -> float:
    """Estimate SNR using quantile-based method."""
    magnitude = np.abs(noisy_signal)
    signal_level = np.percentile(magnitude, 95)
    noise_level = np.percentile(magnitude, 50)
    return 20 * np.log10(signal_level / noise_level) if noise_level > 0 else 0.0


def snr_complex(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate SNR for complex signals."""
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = np.mean(np.abs(noise)**2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf


# Attach functions to namespace
snr_calculator.snr_from_components = snr_from_components
snr_calculator.estimate_snr_moments = estimate_snr_moments
snr_calculator.estimate_snr_spectral = estimate_snr_spectral
snr_calculator.estimate_snr_quantiles = estimate_snr_quantiles
snr_calculator.snr_complex = snr_complex


def radar_range_equation_snr(target_rcs: float, transmit_power: float,
                           antenna_gain: float, frequency: float,
                           range_m: float, noise_figure: float,
                           system_temp: float) -> float:
    """Calculate SNR using radar range equation."""
    wavelength = 3e8 / frequency
    gain_linear = 10**(antenna_gain / 10)

    # Simplified radar equation
    received_power = (transmit_power * gain_linear**2 * wavelength**2 * target_rcs) / \
                    ((4 * np.pi)**3 * range_m**4)

    # Thermal noise
    k_boltzmann = 1.38e-23
    noise_power = k_boltzmann * system_temp * 1e6  # 1 MHz bandwidth assumption
    noise_power *= 10**(noise_figure / 10)

    return 10 * np.log10(received_power / noise_power)


# Additional utility functions for snr_calculator
snr_calculator.radar_range_equation_snr = radar_range_equation_snr


def quantization_loss(n_bits: int) -> float:
    """Calculate quantization loss in dB."""
    return 6.02 * n_bits - 1.76  # Simplified formula


def mismatch_loss(correlation: float) -> float:
    """Calculate mismatch loss due to imperfect correlation."""
    return -10 * np.log10(correlation**2) if correlation > 0 else np.inf


def doppler_loss(doppler_shift: float, pulse_width: float) -> float:
    """Calculate Doppler loss."""
    return -10 * np.log10(np.sinc(doppler_shift * pulse_width)**2)


def windowing_loss(window_type: str) -> float:
    """Calculate windowing loss for different window types."""
    losses = {
        'rectangular': 0.0,
        'hamming': 1.34,
        'hanning': 1.42,
        'blackman': 2.38
    }
    return losses.get(window_type.lower(), 0.0)


# Add loss functions to namespace
snr_calculator.quantization_loss = quantization_loss
snr_calculator.mismatch_loss = mismatch_loss
snr_calculator.doppler_loss = doppler_loss
snr_calculator.windowing_loss = windowing_loss


class PerformanceMetrics:
    """Performance metrics for radar algorithms."""

    @staticmethod
    def detection_performance(true_targets: np.ndarray, detected_targets: np.ndarray,
                            threshold: float) -> Dict[str, float]:
        """Calculate detection performance metrics."""
        true_binary = (true_targets > threshold).astype(int)
        detected_binary = (detected_targets > threshold).astype(int)

        tp = np.sum(true_binary * detected_binary)
        fp = np.sum((1 - true_binary) * detected_binary)
        fn = np.sum(true_binary * (1 - detected_binary))
        tn = np.sum((1 - true_binary) * (1 - detected_binary))

        pd = tp / (tp + fn) if (tp + fn) > 0 else 0
        pfa = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = pd
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'probability_detection': pd,
            'probability_false_alarm': pfa,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    @staticmethod
    def reconstruction_quality(true_signal: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """Calculate reconstruction quality metrics."""
        mse = np.mean((true_signal - reconstructed)**2)
        signal_power = np.var(true_signal)
        nmse = mse / signal_power if signal_power > 0 else np.inf

        snr_improvement = 10 * np.log10(signal_power / mse) if mse > 0 else np.inf

        true_nonzeros = np.sum(np.abs(true_signal) > 1e-6)
        recon_nonzeros = np.sum(np.abs(reconstructed) > 1e-6)
        sparsity_error = abs(true_nonzeros - recon_nonzeros)

        return {
            'mse': mse,
            'nmse': nmse,
            'snr_improvement': snr_improvement,
            'sparsity_error': sparsity_error
        }

    @staticmethod
    def sparsity_analysis(signal: np.ndarray, threshold: float = 1e-6) -> Dict[str, float]:
        """Analyze sparsity properties of signal."""
        magnitude = np.abs(signal)
        nonzero_count = np.sum(magnitude > threshold)

        # L0 "norm" (number of non-zeros)
        l0_norm = nonzero_count

        # Sparsity ratio
        sparsity_ratio = 1 - (nonzero_count / len(signal))

        # Gini coefficient (inequality measure)
        sorted_vals = np.sort(magnitude)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n

        return {
            'l0_norm': l0_norm,
            'sparsity_ratio': sparsity_ratio,
            'gini_coefficient': gini
        }

    @staticmethod
    def radar_performance_analysis(rd_map: np.ndarray) -> Dict[str, float]:
        """Analyze radar-specific performance metrics."""
        magnitude = np.abs(rd_map)

        # Dynamic range
        max_val = np.max(magnitude)
        min_val = np.min(magnitude[magnitude > 0]) if np.any(magnitude > 0) else 1e-10
        dynamic_range = 20 * np.log10(max_val / min_val)

        # Clutter level (median background level)
        clutter_level = np.median(magnitude)

        # Peak sidelobe ratio (simplified)
        peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        # Create mask excluding main peak region
        mask = np.ones_like(magnitude, dtype=bool)
        r_start, r_end = max(0, peak_idx[0]-2), min(magnitude.shape[0], peak_idx[0]+3)
        c_start, c_end = max(0, peak_idx[1]-2), min(magnitude.shape[1], peak_idx[1]+3)
        mask[r_start:r_end, c_start:c_end] = False

        sidelobes = magnitude[mask]
        peak_sidelobe = np.max(sidelobes) if len(sidelobes) > 0 else 0
        peak_sidelobe_ratio = 20 * np.log10(peak_sidelobe / max_val) if peak_sidelobe > 0 else -np.inf

        # Integrated sidelobe level
        main_peak_energy = np.sum(magnitude[~mask]**2)
        sidelobe_energy = np.sum(sidelobes**2)
        total_energy = main_peak_energy + sidelobe_energy
        integrated_sidelobe_level = 10 * np.log10(sidelobe_energy / total_energy) if total_energy > 0 else -np.inf

        return {
            'dynamic_range': dynamic_range,
            'clutter_level': float(clutter_level),
            'peak_sidelobe_ratio': peak_sidelobe_ratio,
            'integrated_sidelobe_level': integrated_sidelobe_level
        }

    @staticmethod
    def computational_analysis(start_time: float, end_time: float, memory_usage: int) -> Dict[str, float]:
        """Analyze computational performance metrics."""
        execution_time = end_time - start_time
        memory_usage_mb = memory_usage / (1024 * 1024)

        # Rough FLOPS estimate (very approximate)
        flops_estimate = memory_usage * 2  # Assume 2 FLOPS per byte processed

        return {
            'execution_time': execution_time,
            'memory_usage_mb': memory_usage_mb,
            'flops_estimate': float(flops_estimate)
        }


performance_metrics = PerformanceMetrics()