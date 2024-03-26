"""Tests for the recording module."""
from neuron import h

def section_to_voltage_recording_str(section: h.Section, segment=0.5) -> str:
    """Converts a section and segment to voltage recording string."""
    return f"neuron.h.{section.name()}({segment})._ref_v"


def test_section_to_voltage_recording_str():
    section = h.Section(name='test_section')
    recording_str = section_to_voltage_recording_str(section, 0.5)
    expected_str = "neuron.h.test_section(0.5)._ref_v"
    assert recording_str == expected_str
