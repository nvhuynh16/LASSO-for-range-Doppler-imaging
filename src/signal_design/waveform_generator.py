"""
Waveform generation utilities for radar applications.

This module provides comprehensive waveform generation capabilities including
linear frequency modulated (chirp) signals, phase-coded waveforms, and
composite signals optimized for compressed sensing applications.

The module supports various waveform types commonly used in radar:
- Linear and nonlinear frequency modulated signals
- Phase-coded sequences (Barker, Zadoff-Chu, polyphase)
- Coprime-based waveforms
- Composite waveforms combining multiple modulation types

References
----------
.. [1] Levanon, N., & Mozeson, E. (2004). Radar Signals. John Wiley & Sons.
.. [2] Cook, C. E., & Bernfeld, M. (1967). Radar Signals: An Introduction to
       Theory and Application. Academic Press.
.. [3] Nathanson, F. E. (1999). Radar Design Principles. SciTech Publishing.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings


class WaveformGenerator:
    """
    Comprehensive waveform generator for radar applications.

    This class provides methods to generate various types of radar waveforms
    optimized for different applications, including compressed sensing and
    sparse reconstruction scenarios.

    Examples
    --------
    >>> from lasso_radar.signal_design.waveform_generator import WaveformGenerator
    >>>
    >>> # Create waveform generator
    >>> wg = WaveformGenerator()
    >>>
    >>> # Generate linear chirp
    >>> chirp = wg.generate_chirp(duration=1e-6, bandwidth=10e6, fs=50e6)
    >>>
    >>> # Generate Barker code
    >>> barker = wg.generate_barker_code(length=13)
    >>>
    >>> # Generate composite waveform
    >>> composite = wg.generate_composite_waveform(1e-6, 10e6, 50e6, 16)
    """

    def __init__(self):
        # Pre-computed Barker sequences
        self._barker_sequences = {
            2: [1, -1],
            3: [1, 1, -1],
            4: [1, 1, -1, 1],
            5: [1, 1, 1, -1, 1],
            7: [1, 1, 1, -1, -1, 1, -1],
            11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
            13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
        }

    def generate_chirp(
        self,
        duration: float,
        bandwidth: float,
        fs: float,
        f0: float = 0.0,
        chirp_type: str = 'linear'
    ) -> np.ndarray:
        """
        Generate frequency modulated (chirp) waveform.

        Parameters
        ----------
        duration : float
            Pulse duration in seconds.
        bandwidth : float
            Chirp bandwidth in Hz.
        fs : float
            Sampling frequency in Hz.
        f0 : float, default=0.0
            Starting frequency in Hz.
        chirp_type : str, default='linear'
            Type of chirp: 'linear', 'quadratic', or 'logarithmic'.

        Returns
        -------
        waveform : ndarray
            Complex chirp waveform.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> chirp = wg.generate_chirp(1e-6, 10e6, 50e6)
        >>> # Linear chirp with 1 microsecond duration and 10 MHz bandwidth
        """
        n_samples = int(duration * fs)
        t = np.arange(n_samples) / fs

        if chirp_type == 'linear':
            # Linear frequency modulation
            chirp_rate = bandwidth / duration
            phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t**2)

        elif chirp_type == 'quadratic':
            # Quadratic frequency modulation
            chirp_rate = bandwidth / (duration**2)
            phase = 2 * np.pi * (f0 * t + (chirp_rate / 3) * t**3)

        elif chirp_type == 'logarithmic':
            # Logarithmic frequency modulation
            k = bandwidth / np.log(1 + duration)
            phase = 2 * np.pi * k * (t + np.log(1 + t))

        else:
            raise ValueError(f"Unknown chirp type: {chirp_type}")

        # Generate complex waveform
        waveform = np.exp(1j * phase)

        return waveform

    def generate_barker_code(self, length: int) -> np.ndarray:
        """
        Generate Barker code sequence.

        Barker codes are binary sequences with optimal autocorrelation properties.
        The sidelobes of the autocorrelation function are ±1 for all known
        Barker sequences.

        Parameters
        ----------
        length : int
            Length of Barker code. Valid lengths: 2, 3, 4, 5, 7, 11, 13.

        Returns
        -------
        barker_code : ndarray
            Complex Barker code sequence (±1 phase modulation).

        Raises
        ------
        ValueError
            If requested length is not a known Barker sequence length.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> barker_13 = wg.generate_barker_code(13)
        >>> # 13-element Barker sequence with optimal autocorrelation
        """
        if length not in self._barker_sequences:
            raise ValueError(f"Barker sequence of length {length} does not exist. "
                           f"Available lengths: {list(self._barker_sequences.keys())}")

        sequence = np.array(self._barker_sequences[length])
        # Convert to complex phase modulation
        barker_code = sequence.astype(complex)

        return barker_code

    def generate_zadoff_chu(self, length: int, root: int) -> np.ndarray:
        """
        Generate Zadoff-Chu sequence.

        Zadoff-Chu sequences have constant magnitude and excellent autocorrelation
        properties. They are particularly useful for radar and communication
        applications.

        Parameters
        ----------
        length : int
            Sequence length (preferably prime for optimal properties).
        root : int
            Root parameter that determines the sequence. Must be coprime to length.

        Returns
        -------
        zc_sequence : ndarray
            Complex Zadoff-Chu sequence with unit magnitude.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> zc = wg.generate_zadoff_chu(31, 3)
        >>> # 31-element Zadoff-Chu sequence with root 3
        """
        if np.gcd(root, length) != 1:
            warnings.warn(f"Root {root} and length {length} are not coprime. "
                         "Autocorrelation properties may be suboptimal.")

        n = np.arange(length)

        if length % 2 == 1:  # Odd length
            phase = np.pi * root * n * (n + 1) / length
        else:  # Even length
            phase = np.pi * root * n**2 / length

        zc_sequence = np.exp(1j * phase)

        return zc_sequence

    def generate_polyphase_code(self, length: int, n_phases: int) -> np.ndarray:
        """
        Generate polyphase code sequence.

        Polyphase codes use multiple phase levels (not just binary) to achieve
        better autocorrelation properties and larger sequence sets.

        Parameters
        ----------
        length : int
            Sequence length.
        n_phases : int
            Number of phase levels (e.g., 4 for QPSK-like modulation).

        Returns
        -------
        poly_code : ndarray
            Complex polyphase code sequence.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> poly = wg.generate_polyphase_code(16, 4)
        >>> # 16-element sequence with 4 phase levels (0, π/2, π, 3π/2)
        """
        # Generate pseudo-random phase indices
        np.random.seed(42)  # For reproducibility
        phase_indices = np.random.randint(0, n_phases, length)

        # Convert to phases
        phases = 2 * np.pi * phase_indices / n_phases

        # Generate complex sequence
        poly_code = np.exp(1j * phases)

        return poly_code

    def generate_coprime_phase_code(
        self,
        moduli: List[int],
        code_length: int
    ) -> np.ndarray:
        """
        Generate phase code using coprime moduli.

        This method creates phase codes based on the Chinese Remainder Theorem
        using coprime moduli, providing excellent cross-correlation properties.

        Parameters
        ----------
        moduli : list of int
            Coprime moduli for code generation.
        code_length : int
            Length of the generated code.

        Returns
        -------
        phase_code : ndarray
            Phase code sequence in radians.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> phase_code = wg.generate_coprime_phase_code([31, 37], 100)
        """
        from .coprime_designer import CoprimeSignalDesigner

        designer = CoprimeSignalDesigner(moduli=moduli)
        phase_code = designer.generate_phase_code(code_length)

        return phase_code

    def quantize_phases(self, phases: np.ndarray, n_levels: int) -> np.ndarray:
        """
        Quantize continuous phases to discrete levels.

        Parameters
        ----------
        phases : array-like
            Continuous phase values in radians.
        n_levels : int
            Number of quantization levels.

        Returns
        -------
        quantized_phases : ndarray
            Quantized phase values.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> continuous_phases = np.random.uniform(0, 2*np.pi, 50)
        >>> quantized = wg.quantize_phases(continuous_phases, 8)
        """
        phases = np.asarray(phases)

        # Normalize phases to [0, 2π)
        normalized_phases = phases % (2 * np.pi)

        # Quantize
        phase_step = 2 * np.pi / n_levels
        quantized_indices = np.round(normalized_phases / phase_step).astype(int)
        quantized_indices = quantized_indices % n_levels

        quantized_phases = quantized_indices * phase_step

        return quantized_phases

    def generate_composite_waveform(
        self,
        duration: float,
        bandwidth: float,
        fs: float,
        phase_code_length: int,
        modulation_types: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate composite waveform combining multiple modulation types.

        This creates waveforms that combine frequency modulation (chirp) with
        phase coding, providing both range resolution and coding gain.

        Parameters
        ----------
        duration : float
            Pulse duration in seconds.
        bandwidth : float
            Chirp bandwidth in Hz.
        fs : float
            Sampling frequency in Hz.
        phase_code_length : int
            Length of phase code to apply.
        modulation_types : list of str, optional
            Types of modulation to combine. Default: ['chirp', 'phase_code'].

        Returns
        -------
        composite_waveform : ndarray
            Complex composite waveform.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> composite = wg.generate_composite_waveform(1e-6, 10e6, 50e6, 16)
        """
        if modulation_types is None:
            modulation_types = ['chirp', 'phase_code']

        n_samples = int(duration * fs)

        # Initialize with ones
        waveform = np.ones(n_samples, dtype=complex)

        if 'chirp' in modulation_types:
            # Apply chirp modulation
            chirp = self.generate_chirp(duration, bandwidth, fs)
            # Resample if necessary
            if len(chirp) != n_samples:
                # Simple resampling
                indices = np.linspace(0, len(chirp) - 1, n_samples)
                chirp_resampled = np.interp(indices, np.arange(len(chirp)), chirp.real) + \
                                 1j * np.interp(indices, np.arange(len(chirp)), chirp.imag)
                waveform *= chirp_resampled
            else:
                waveform *= chirp

        if 'phase_code' in modulation_types:
            # Apply phase coding
            if phase_code_length <= 13 and phase_code_length in self._barker_sequences:
                # Use Barker code if available
                phase_code = self.generate_barker_code(phase_code_length)
            else:
                # Use polyphase code
                phase_code = self.generate_polyphase_code(phase_code_length, 4)

            # Expand phase code to match waveform length
            samples_per_chip = n_samples // phase_code_length
            expanded_code = np.repeat(phase_code, samples_per_chip)

            # Handle length mismatch
            if len(expanded_code) < n_samples:
                # Pad with last value
                padding = n_samples - len(expanded_code)
                expanded_code = np.concatenate([expanded_code, np.repeat(expanded_code[-1], padding)])
            elif len(expanded_code) > n_samples:
                # Truncate
                expanded_code = expanded_code[:n_samples]

            waveform *= expanded_code

        return waveform

    def optimize_phase_code(
        self,
        length: int,
        n_phases: int,
        criterion: str = 'autocorr',
        n_iterations: int = 1000
    ) -> np.ndarray:
        """
        Optimize phase code for specific performance criteria.

        Parameters
        ----------
        length : int
            Code length.
        n_phases : int
            Number of phase levels.
        criterion : str, default='autocorr'
            Optimization criterion: 'autocorr', 'cross_corr', or 'ambiguity'.
        n_iterations : int, default=1000
            Number of optimization iterations.

        Returns
        -------
        optimized_code : ndarray
            Optimized phase code sequence.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> optimized = wg.optimize_phase_code(32, 8, 'autocorr')
        """
        # Initialize with random code
        best_code = self.generate_polyphase_code(length, n_phases)
        best_metric = self._evaluate_code_performance(best_code, criterion)

        for _ in range(n_iterations):
            # Generate candidate code
            candidate_code = self.generate_polyphase_code(length, n_phases)
            metric = self._evaluate_code_performance(candidate_code, criterion)

            # Check if better (lower is better for most criteria)
            if metric < best_metric:
                best_code = candidate_code.copy()
                best_metric = metric

        return best_code

    def _evaluate_code_performance(self, code: np.ndarray, criterion: str) -> float:
        """Evaluate performance of a phase code."""
        if criterion == 'autocorr':
            # Minimize autocorrelation sidelobes
            autocorr = np.correlate(code, code, mode='full')
            # Remove main peak and compute sidelobe energy
            center = len(autocorr) // 2
            sidelobes = np.concatenate([autocorr[:center], autocorr[center+1:]])
            return np.sum(np.abs(sidelobes)**2)

        elif criterion == 'cross_corr':
            # Minimize cross-correlation with shifted versions
            total_cross_corr = 0.0
            for shift in range(1, min(len(code), 10)):
                shifted_code = np.roll(code, shift)
                cross_corr = np.abs(np.sum(code * np.conj(shifted_code)))
                total_cross_corr += cross_corr
            return total_cross_corr

        elif criterion == 'ambiguity':
            # Minimize ambiguity function sidelobes (simplified)
            # This is computationally expensive, so we use a simplified version
            ambiguity_metric = 0.0
            for delay in range(1, min(len(code), 5)):
                for doppler in range(1, 5):
                    doppler_shift = np.exp(1j * 2 * np.pi * doppler * np.arange(len(code)) / len(code))
                    shifted_code = np.roll(code * doppler_shift, delay)
                    correlation = np.abs(np.sum(code * np.conj(shifted_code)))
                    ambiguity_metric += correlation
            return ambiguity_metric

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def analyze_waveform_properties(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Analyze properties of a generated waveform.

        Parameters
        ----------
        waveform : array-like
            Complex waveform to analyze.

        Returns
        -------
        properties : dict
            Dictionary containing waveform properties:
            - 'peak_to_average_power': Peak-to-average power ratio
            - 'bandwidth_efficiency': Spectral efficiency measure
            - 'autocorr_properties': Autocorrelation analysis
            - 'ambiguity_volume': Ambiguity function volume
        """
        waveform = np.asarray(waveform)

        # Peak-to-average power ratio
        instantaneous_power = np.abs(waveform)**2
        peak_power = np.max(instantaneous_power)
        avg_power = np.mean(instantaneous_power)
        papr = peak_power / avg_power if avg_power > 0 else np.inf

        # Bandwidth efficiency (spectral compactness)
        spectrum = np.abs(fft(waveform))**2
        spectrum_centered = np.fft.fftshift(spectrum)

        # Find -3dB bandwidth
        max_spectrum = np.max(spectrum_centered)
        half_power_indices = np.where(spectrum_centered >= max_spectrum / 2)[0]
        if len(half_power_indices) > 0:
            bandwidth_efficiency = (half_power_indices[-1] - half_power_indices[0]) / len(spectrum_centered)
        else:
            bandwidth_efficiency = 1.0

        # Autocorrelation properties
        autocorr = np.correlate(waveform, waveform, mode='full')
        autocorr_normalized = autocorr / np.max(np.abs(autocorr))

        center = len(autocorr) // 2
        main_peak = np.abs(autocorr_normalized[center])
        sidelobes = np.concatenate([
            autocorr_normalized[:center],
            autocorr_normalized[center+1:]
        ])

        if len(sidelobes) > 0:
            peak_sidelobe = np.max(np.abs(sidelobes))
            integrated_sidelobes = np.sum(np.abs(sidelobes)**2)
        else:
            peak_sidelobe = 0.0
            integrated_sidelobes = 0.0

        # Simplified ambiguity volume
        ambiguity_volume = self._compute_ambiguity_volume(waveform)

        return {
            'peak_to_average_power': float(papr),
            'bandwidth_efficiency': float(bandwidth_efficiency),
            'autocorr_properties': {
                'peak_sidelobe_ratio': float(peak_sidelobe / main_peak) if main_peak > 0 else 0.0,
                'integrated_sidelobe_level': float(integrated_sidelobes)
            },
            'ambiguity_volume': float(ambiguity_volume)
        }

    def _compute_ambiguity_volume(self, waveform: np.ndarray) -> float:
        """Compute simplified ambiguity function volume."""
        # This is a simplified computation for efficiency
        max_delay = min(10, len(waveform) // 4)
        max_doppler = min(10, len(waveform) // 4)

        total_volume = 0.0

        for delay in range(-max_delay, max_delay + 1):
            for doppler in range(-max_doppler, max_doppler + 1):
                if delay == 0 and doppler == 0:
                    continue  # Skip main peak

                # Apply delay
                if delay >= 0:
                    delayed_waveform = np.concatenate([np.zeros(delay), waveform])
                else:
                    delayed_waveform = waveform[-delay:]

                # Apply Doppler shift
                doppler_shift = np.exp(1j * 2 * np.pi * doppler * np.arange(len(delayed_waveform)) / len(waveform))
                doppler_waveform = delayed_waveform * doppler_shift

                # Compute correlation
                min_length = min(len(waveform), len(doppler_waveform))
                correlation = np.abs(np.sum(waveform[:min_length] * np.conj(doppler_waveform[:min_length])))

                total_volume += correlation**2

        # Normalize by main peak energy
        main_peak_energy = np.sum(np.abs(waveform)**2)**2
        return total_volume / main_peak_energy if main_peak_energy > 0 else 0.0

    def generate_waveform_family(
        self,
        base_params: Dict[str, Any],
        family_size: int,
        diversity_type: str = 'frequency'
    ) -> List[np.ndarray]:
        """
        Generate family of related waveforms for diversity applications.

        Parameters
        ----------
        base_params : dict
            Base parameters for waveform generation.
        family_size : int
            Number of waveforms in the family.
        diversity_type : str, default='frequency'
            Type of diversity: 'frequency', 'phase', or 'hybrid'.

        Returns
        -------
        waveform_family : list of ndarray
            List of diverse waveforms.

        Examples
        --------
        >>> wg = WaveformGenerator()
        >>> params = {'duration': 1e-6, 'bandwidth': 10e6, 'fs': 50e6}
        >>> family = wg.generate_waveform_family(params, 4, 'frequency')
        """
        waveform_family = []

        for i in range(family_size):
            if diversity_type == 'frequency':
                # Frequency diversity
                modified_params = base_params.copy()
                f_offset = i * base_params.get('bandwidth', 10e6) * 0.1
                modified_params['f0'] = f_offset
                waveform = self.generate_chirp(**modified_params)

            elif diversity_type == 'phase':
                # Phase code diversity
                length = base_params.get('phase_code_length', 16)
                if length <= 13 and length in self._barker_sequences:
                    # Use different Barker codes
                    available_lengths = [l for l in self._barker_sequences.keys() if l <= length + 5]
                    selected_length = available_lengths[i % len(available_lengths)]
                    waveform = self.generate_barker_code(selected_length)
                else:
                    # Use different polyphase codes
                    np.random.seed(i)  # Different seed for each waveform
                    waveform = self.generate_polyphase_code(length, 4)

            elif diversity_type == 'hybrid':
                # Hybrid diversity combining frequency and phase
                modified_params = base_params.copy()
                f_offset = i * base_params.get('bandwidth', 10e6) * 0.05
                modified_params['f0'] = f_offset

                phase_length = base_params.get('phase_code_length', 16)
                composite = self.generate_composite_waveform(
                    modified_params.get('duration', 1e-6),
                    modified_params.get('bandwidth', 10e6),
                    modified_params.get('fs', 50e6),
                    phase_length + i
                )
                waveform = composite

            else:
                raise ValueError(f"Unknown diversity type: {diversity_type}")

            waveform_family.append(waveform)

        return waveform_family

    def __repr__(self) -> str:
        """String representation of the waveform generator."""
        return "WaveformGenerator()"