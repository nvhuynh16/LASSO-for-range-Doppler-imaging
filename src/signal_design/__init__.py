"""Coprime signal design and waveform generation for radar."""

from .coprime_designer import CoprimeSignalDesigner
from .waveform_generator import WaveformGenerator

__all__ = [
    "CoprimeSignalDesigner",
    "WaveformGenerator",
]