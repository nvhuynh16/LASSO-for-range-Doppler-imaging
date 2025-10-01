"""
Signal processing utilities for radar range-Doppler processing.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple


def range_doppler_processing():
    """Range-Doppler processing utilities namespace."""
    pass


def basic_range_doppler(radar_data: np.ndarray) -> np.ndarray:
    """
    Basic range-Doppler processing using FFT.

    Parameters
    ----------
    radar_data : ndarray of shape (n_pulses, n_samples)
        Radar data matrix.

    Returns
    -------
    rd_map : ndarray
        Range-Doppler map.
    """
    # Range compression already assumed done
    # Apply Doppler FFT
    rd_map = np.fft.fft(radar_data, axis=0)
    return rd_map


def fft_doppler_processing(radar_data: np.ndarray) -> np.ndarray:
    """FFT-based Doppler processing."""
    return np.fft.fft(radar_data, axis=0)


def windowed_doppler_processing(radar_data: np.ndarray,
                               window_type: str = 'hamming') -> np.ndarray:
    """Windowed Doppler processing."""
    n_pulses = radar_data.shape[0]

    if window_type == 'hamming':
        window = signal.windows.hamming(n_pulses)
    elif window_type == 'hanning':
        window = signal.windows.hann(n_pulses)
    else:
        window = np.ones(n_pulses)

    windowed_data = radar_data * window[:, np.newaxis]
    return np.fft.fft(windowed_data, axis=0)


def rectangular_window(length: int) -> np.ndarray:
    """Generate rectangular window."""
    return np.ones(length)


def hamming_window(length: int) -> np.ndarray:
    """Generate Hamming window."""
    return signal.windows.hamming(length)


def hanning_window(length: int) -> np.ndarray:
    """Generate Hann window."""
    return signal.windows.hann(length)


def blackman_window(length: int) -> np.ndarray:
    """Generate Blackman window."""
    return signal.windows.blackman(length)


def kaiser_window(length: int, beta: float = 8.6) -> np.ndarray:
    """Generate Kaiser window."""
    return signal.windows.kaiser(length, beta)


def clutter_suppression(radar_data: np.ndarray, method: str = 'mti') -> np.ndarray:
    """Apply clutter suppression."""
    if method == 'mti':
        # Simple MTI filter (first difference)
        suppressed = np.zeros_like(radar_data)
        suppressed[1:] = radar_data[1:] - radar_data[:-1]
        return suppressed
    else:
        return radar_data


def cfar_detection(rd_map: np.ndarray, guard_cells: int = 2,
                  training_cells: int = 10, pfa: float = 1e-6) -> np.ndarray:
    """CFAR detection on range-Doppler map."""
    # Simplified CFAR implementation
    threshold_multiplier = -np.log(pfa)  # Exponential approximation

    detections = np.zeros_like(rd_map, dtype=bool)

    for r in range(guard_cells + training_cells, rd_map.shape[0] - guard_cells - training_cells):
        for d in range(guard_cells + training_cells, rd_map.shape[1] - guard_cells - training_cells):
            # Get training cells (excluding guard cells)
            training_samples = []

            # Collect training samples around the cell under test
            for dr in range(-guard_cells-training_cells, guard_cells+training_cells+1):
                for dd in range(-guard_cells-training_cells, guard_cells+training_cells+1):
                    if abs(dr) > guard_cells or abs(dd) > guard_cells:  # Outside guard region
                        if 0 <= r+dr < rd_map.shape[0] and 0 <= d+dd < rd_map.shape[1]:
                            training_samples.append(rd_map[r+dr, d+dd])

            if len(training_samples) > 0:
                noise_level = np.mean(training_samples)
                threshold = threshold_multiplier * noise_level

                if rd_map[r, d] > threshold:
                    detections[r, d] = True

    return detections


# Attach functions to namespace
range_doppler_processing.basic_range_doppler = basic_range_doppler
range_doppler_processing.fft_doppler_processing = fft_doppler_processing
range_doppler_processing.windowed_doppler_processing = windowed_doppler_processing
range_doppler_processing.rectangular_window = rectangular_window
range_doppler_processing.hamming_window = hamming_window
range_doppler_processing.hanning_window = hanning_window
range_doppler_processing.blackman_window = blackman_window
range_doppler_processing.kaiser_window = kaiser_window
range_doppler_processing.clutter_suppression = clutter_suppression
range_doppler_processing.cfar_detection = cfar_detection