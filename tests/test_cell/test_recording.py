"""Tests for the recording module."""
from neuron import h
from bluecellulab.cell.recording import section_to_voltage_recording_str


def test_section_to_voltage_recording_str():
    section = h.Section(name='test_section')
    recording_str = section_to_voltage_recording_str(section, 0.5)
    expected_str = "neuron.h.test_section(0.5)._ref_v"
    assert recording_str == expected_str
