"""Neuron recordings related functions."""
from __future__ import annotations
from bluecellulab.type_aliases import NeuronSection


def section_to_voltage_recording_str(section: NeuronSection, segment=0.5) -> str:
    """Converts a section and segment to voltage recording string."""
    return f"neuron.h.{section.name()}({segment})._ref_v"
