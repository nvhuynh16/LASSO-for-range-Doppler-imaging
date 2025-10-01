"""
Matched Filter implementation for radar range-Doppler processing.

The matched filter is the optimal detector for known signals in white Gaussian noise.
It serves as a baseline comparison for sparse reconstruction methods like LASSO.
This implementation includes both coherent and non-coherent processing variants.

The matched filter output for a signal s(t) in noise n(t) is:
y(τ) = ∫ [s(t) + n(t)] s*(t-τ) dt

For radar applications, this extends to range-Doppler processing where the filter
is matched to the transmitted waveform and accounts for Doppler shifts.

References
----------
.. [1] Richards, M. A. (2005). Fundamentals of Radar Signal Processing.
       McGraw-Hill Professional.
.. [2] Skolnik, M. I. (2008). Radar Handbook. McGraw-Hill Professional.
.. [3] Mahafza, B. R. (2000). Radar Systems Analysis and Design Using MATLAB.
       Chapman & Hall/CRC.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Optional, Tuple, Dict, Any, Union
import warnings


class MatchedFilter:
    """
    Matched filter implementation for radar signal processing.

    This class implements coherent and non-coherent matched filtering for
    radar range and range-Doppler processing. It serves as a performance
    baseline for comparison with sparse reconstruction methods.

    Parameters
    ----------
    reference_waveform : array-like
        Reference waveform to match against (transmitted signal).
    fs : float
        Sampling frequency in Hz.
    normalize : bool, default=True
        If True, normalize the matched filter for unit gain.

    Attributes
    ----------
    reference_waveform : ndarray
        Time-reversed and conjugated reference waveform.
    fs : float
        Sampling frequency.
    waveform_length : int
        Length of the reference waveform.

    Examples
    --------
    >>> import numpy as np
    >>> from lasso_radar.algorithms.matched_filter import MatchedFilter
    >>>
    >>> # Generate linear chirp waveform
    >>> duration = 1e-6  # 1 microsecond
    >>> bandwidth = 10e6  # 10 MHz
    >>> fs = 50e6  # 50 MHz
    >>> t = np.arange(0, duration, 1/fs)
    >>> chirp = np.exp(1j * np.pi * (bandwidth/duration) * t**2)
    >>>
    >>> # Create matched filter
    >>> mf = MatchedFilter(chirp, fs)
    >>>
    >>> # Process received signal
    >>> received_signal = chirp + 0.1 * np.random.randn(len(chirp))
    >>> compressed_pulse = mf.process_pulse(received_signal)
    """

    def __init__(
        self,
        reference_waveform: np.ndarray,
        fs: float,
        normalize: bool = True
    ):
        self.fs = fs
        self.waveform_length = len(reference_waveform)

        # Store time-reversed and conjugated reference (matched filter impulse response)
        self.reference_waveform = np.conj(reference_waveform[::-1])

        if normalize:
            # Normalize for unit gain
            norm_factor = np.sqrt(np.sum(np.abs(reference_waveform)**2))
            if norm_factor > 0:
                self.reference_waveform /= norm_factor

    def process_pulse(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Process a single pulse through the matched filter.

        Parameters
        ----------
        received_signal : array-like
            Received radar signal to be processed.

        Returns
        -------
        compressed_pulse : ndarray
            Matched filter output (compressed pulse).

        Notes
        -----
        The output length will be len(received_signal) + len(reference_waveform) - 1.
        The peak location corresponds to the delay of the target return.
        """
        received_signal = np.asarray(received_signal)

        # Convolve with matched filter
        compressed_pulse = signal.convolve(received_signal, self.reference_waveform, mode='full')

        return compressed_pulse

    def range_doppler_processing(
        self,
        radar_data: np.ndarray,
        window_type: str = 'hamming',
        zero_padding: Optional[int] = None
    ) -> np.ndarray:
        """
        Perform range-Doppler processing using matched filtering.

        This method applies matched filtering across range (fast-time) and
        FFT processing across Doppler (slow-time) to create a range-Doppler map.

        Parameters
        ----------
        radar_data : array-like of shape (n_pulses, n_samples)
            Radar data matrix where each row is a received pulse.
        window_type : str, default='hamming'
            Window function to apply before Doppler FFT.
            Options: 'hamming', 'hanning', 'blackman', 'rectangular'
        zero_padding : int, optional
            Zero padding factor for Doppler FFT. If None, uses n_pulses.

        Returns
        -------
        rd_map : ndarray of shape (n_doppler_bins, n_range_bins)
            Range-Doppler map with complex amplitudes.

        Examples
        --------
        >>> # Multi-pulse processing
        >>> n_pulses, n_samples = 64, 200
        >>> radar_data = np.random.randn(n_pulses, n_samples) + 1j * np.random.randn(n_pulses, n_samples)
        >>> rd_map = mf.range_doppler_processing(radar_data)
        """
        radar_data = np.asarray(radar_data)

        if radar_data.ndim != 2:
            raise ValueError("Radar data must be 2D array (n_pulses, n_samples)")

        n_pulses, n_samples = radar_data.shape

        # Step 1: Range compression (matched filtering)
        range_compressed = np.zeros((n_pulses, n_samples + self.waveform_length - 1), dtype=complex)

        for pulse_idx in range(n_pulses):
            range_compressed[pulse_idx] = self.process_pulse(radar_data[pulse_idx])

        # Truncate to original sample length (or adjust as needed)
        # Take the central portion to maintain timing
        start_idx = self.waveform_length // 2
        end_idx = start_idx + n_samples
        range_compressed = range_compressed[:, start_idx:end_idx]

        # Step 2: Doppler processing (coherent integration)
        doppler_fft_size = zero_padding if zero_padding is not None else n_pulses

        # Apply window function
        if window_type.lower() != 'rectangular':
            window = self._get_window(window_type, n_pulses)
            windowed_data = range_compressed * window[:, np.newaxis]
        else:
            windowed_data = range_compressed

        # Zero pad if requested
        if doppler_fft_size > n_pulses:
            padding = ((0, doppler_fft_size - n_pulses), (0, 0))
            windowed_data = np.pad(windowed_data, padding, mode='constant')

        # Doppler FFT
        rd_map = fft(windowed_data, axis=0)

        return rd_map

    def _get_window(self, window_type: str, length: int) -> np.ndarray:
        """Get window function for Doppler processing."""
        window_type = window_type.lower()

        if window_type == 'hamming':
            return signal.windows.hamming(length)
        elif window_type == 'hanning':
            return signal.windows.hann(length)
        elif window_type == 'blackman':
            return signal.windows.blackman(length)
        elif window_type == 'rectangular':
            return np.ones(length)
        else:
            warnings.warn(f"Unknown window type '{window_type}', using rectangular")
            return np.ones(length)

    def coherent_integration(
        self,
        radar_data: np.ndarray,
        n_coherent: int
    ) -> np.ndarray:
        """
        Perform coherent integration over multiple pulses.

        Parameters
        ----------
        radar_data : array-like of shape (n_pulses, n_samples)
            Radar data matrix.
        n_coherent : int
            Number of pulses to coherently integrate.

        Returns
        -------
        integrated_data : ndarray
            Coherently integrated radar data.
        """
        radar_data = np.asarray(radar_data)
        n_pulses, n_samples = radar_data.shape

        if n_coherent > n_pulses:
            raise ValueError("Cannot integrate more pulses than available")

        # Reshape and integrate
        n_integrations = n_pulses // n_coherent
        reshaped_data = radar_data[:n_integrations * n_coherent].reshape(
            n_integrations, n_coherent, n_samples
        )

        # Coherent sum
        integrated_data = np.sum(reshaped_data, axis=1)

        return integrated_data

    def non_coherent_integration(
        self,
        radar_data: np.ndarray,
        n_non_coherent: int,
        detection_type: str = 'square_law'
    ) -> np.ndarray:
        """
        Perform non-coherent integration (energy detection).

        Parameters
        ----------
        radar_data : array-like of shape (n_pulses, n_samples)
            Radar data matrix.
        n_non_coherent : int
            Number of pulses to non-coherently integrate.
        detection_type : str, default='square_law'
            Type of detection: 'square_law' or 'envelope'.

        Returns
        -------
        integrated_data : ndarray
            Non-coherently integrated radar data.
        """
        radar_data = np.asarray(radar_data)
        n_pulses, n_samples = radar_data.shape

        if n_non_coherent > n_pulses:
            raise ValueError("Cannot integrate more pulses than available")

        # Apply detection
        if detection_type == 'square_law':
            detected_data = np.abs(radar_data)**2
        elif detection_type == 'envelope':
            detected_data = np.abs(radar_data)
        else:
            raise ValueError("detection_type must be 'square_law' or 'envelope'")

        # Reshape and integrate
        n_integrations = n_pulses // n_non_coherent
        reshaped_data = detected_data[:n_integrations * n_non_coherent].reshape(
            n_integrations, n_non_coherent, n_samples
        )

        # Non-coherent sum
        integrated_data = np.sum(reshaped_data, axis=1)

        return integrated_data

    def estimate_snr_improvement(
        self,
        input_signal: np.ndarray,
        output_signal: np.ndarray
    ) -> float:
        """
        Estimate SNR improvement provided by matched filtering.

        Parameters
        ----------
        input_signal : array-like
            Input signal before matched filtering.
        output_signal : array-like
            Output signal after matched filtering.

        Returns
        -------
        snr_improvement_db : float
            SNR improvement in dB.
        """
        # Estimate signal and noise powers
        input_power = np.var(input_signal)
        output_signal_power = np.max(np.abs(output_signal)**2)
        output_noise_power = np.median(np.abs(output_signal)**2)

        # Calculate SNRs
        input_snr = 1.0  # Assume normalized input
        output_snr = output_signal_power / output_noise_power if output_noise_power > 0 else 1.0

        # SNR improvement in dB
        snr_improvement_db = 10 * np.log10(output_snr / input_snr)

        return float(snr_improvement_db)

    def analyze_sidelobe_performance(
        self,
        compressed_pulse: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze sidelobe performance of the matched filter output.

        Parameters
        ----------
        compressed_pulse : array-like
            Matched filter output (compressed pulse).

        Returns
        -------
        sidelobe_metrics : dict
            Dictionary containing sidelobe analysis:
            - 'peak_sidelobe_ratio_db': Peak sidelobe level relative to main peak
            - 'integrated_sidelobe_level_db': Integrated sidelobe level
            - 'mainlobe_width_samples': Width of the main lobe
        """
        compressed_pulse = np.asarray(compressed_pulse)
        magnitude = np.abs(compressed_pulse)

        # Find main peak
        peak_idx = np.argmax(magnitude)
        peak_value = magnitude[peak_idx]

        # Estimate mainlobe width (first nulls on either side)
        mainlobe_width = self._estimate_mainlobe_width(magnitude, peak_idx)

        # Define sidelobe region (exclude mainlobe)
        mainlobe_half_width = mainlobe_width // 2
        sidelobe_mask = np.ones(len(magnitude), dtype=bool)
        start_idx = max(0, peak_idx - mainlobe_half_width)
        end_idx = min(len(magnitude), peak_idx + mainlobe_half_width + 1)
        sidelobe_mask[start_idx:end_idx] = False

        sidelobe_region = magnitude[sidelobe_mask]

        # Peak sidelobe ratio
        if len(sidelobe_region) > 0:
            peak_sidelobe = np.max(sidelobe_region)
            peak_sidelobe_ratio_db = 20 * np.log10(peak_sidelobe / peak_value)
        else:
            peak_sidelobe_ratio_db = -np.inf

        # Integrated sidelobe level
        mainlobe_energy = np.sum(magnitude[start_idx:end_idx]**2)
        sidelobe_energy = np.sum(sidelobe_region**2)
        total_energy = mainlobe_energy + sidelobe_energy

        if total_energy > 0:
            integrated_sidelobe_level_db = 10 * np.log10(sidelobe_energy / total_energy)
        else:
            integrated_sidelobe_level_db = -np.inf

        return {
            'peak_sidelobe_ratio_db': float(peak_sidelobe_ratio_db),
            'integrated_sidelobe_level_db': float(integrated_sidelobe_level_db),
            'mainlobe_width_samples': int(mainlobe_width)
        }

    def _estimate_mainlobe_width(
        self,
        magnitude: np.ndarray,
        peak_idx: int,
        null_threshold: float = 0.1
    ) -> int:
        """Estimate mainlobe width by finding first nulls."""
        peak_value = magnitude[peak_idx]
        threshold = null_threshold * peak_value

        # Find first null to the left
        left_null = 0
        for i in range(peak_idx - 1, -1, -1):
            if magnitude[i] < threshold:
                left_null = i
                break

        # Find first null to the right
        right_null = len(magnitude) - 1
        for i in range(peak_idx + 1, len(magnitude)):
            if magnitude[i] < threshold:
                right_null = i
                break

        return right_null - left_null

    def estimate_complexity(self, data_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Estimate computational complexity of matched filter processing.

        Parameters
        ----------
        data_shape : tuple
            Shape of radar data as (n_pulses, n_samples).

        Returns
        -------
        complexity : dict
            Dictionary containing complexity estimates:
            - 'range_compression_ops': Operations for range compression
            - 'doppler_processing_ops': Operations for Doppler processing
            - 'total_ops': Total operations
        """
        n_pulses, n_samples = data_shape

        # Range compression (convolution)
        range_ops = n_pulses * n_samples * self.waveform_length

        # Doppler processing (FFT)
        doppler_ops = n_samples * n_pulses * np.log2(n_pulses)

        total_ops = range_ops + doppler_ops

        return {
            'range_compression_ops': float(range_ops),
            'doppler_processing_ops': float(doppler_ops),
            'total_ops': float(total_ops)
        }

    def compute_ambiguity_function(
        self,
        max_delay_samples: int = 50,
        max_doppler_bins: int = 50
    ) -> np.ndarray:
        """
        Compute ambiguity function of the reference waveform.

        The ambiguity function characterizes the response of the matched filter
        to targets at different delays and Doppler shifts.

        Parameters
        ----------
        max_delay_samples : int, default=50
            Maximum delay in samples for ambiguity function.
        max_doppler_bins : int, default=50
            Maximum Doppler bins for ambiguity function.

        Returns
        -------
        ambiguity : ndarray of shape (2*max_doppler_bins+1, 2*max_delay_samples+1)
            Ambiguity function magnitude.
        """
        # Get original reference waveform (undo time reversal and conjugation)
        original_waveform = np.conj(self.reference_waveform[::-1])

        # Delay axis
        delays = np.arange(-max_delay_samples, max_delay_samples + 1)

        # Doppler axis (normalized frequency)
        doppler_freqs = np.linspace(-0.5, 0.5, 2 * max_doppler_bins + 1)

        # Initialize ambiguity function
        ambiguity = np.zeros((len(doppler_freqs), len(delays)), dtype=complex)

        for i, doppler in enumerate(doppler_freqs):
            # Apply Doppler shift
            doppler_shifted = original_waveform * np.exp(
                1j * 2 * np.pi * doppler * np.arange(len(original_waveform))
            )

            for j, delay in enumerate(delays):
                # Apply delay and compute correlation
                if delay >= 0:
                    delayed_waveform = np.concatenate([
                        np.zeros(delay),
                        doppler_shifted
                    ])
                else:
                    delayed_waveform = doppler_shifted[-delay:]

                # Ensure same length for correlation
                min_length = min(len(original_waveform), len(delayed_waveform))
                correlation = np.sum(
                    original_waveform[:min_length] * np.conj(delayed_waveform[:min_length])
                )

                ambiguity[i, j] = correlation

        return np.abs(ambiguity)

    def compare_with_theoretical(
        self,
        snr_db: float,
        theoretical_pd: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compare matched filter performance with theoretical predictions.

        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in dB.
        theoretical_pd : float, optional
            Theoretical detection probability for comparison.

        Returns
        -------
        comparison : dict
            Dictionary containing performance comparison metrics.
        """
        # Theoretical matched filter SNR improvement
        theoretical_improvement = 10 * np.log10(self.waveform_length)

        # Theoretical processing gain
        processing_gain = theoretical_improvement

        return {
            'theoretical_snr_improvement_db': float(theoretical_improvement),
            'processing_gain_db': float(processing_gain),
            'input_snr_db': float(snr_db),
            'output_snr_db': float(snr_db + theoretical_improvement)
        }

    def __repr__(self) -> str:
        """String representation of the matched filter."""
        return (f"MatchedFilter(waveform_length={self.waveform_length}, "
                f"fs={self.fs:.0f})")