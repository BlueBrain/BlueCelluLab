# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the recording module."""
from neuron import h
from bluecellulab.cell.recording import section_to_voltage_recording_str


def test_section_to_voltage_recording_str():
    section = h.Section(name='test_section')
    recording_str = section_to_voltage_recording_str(section, 0.5)
    expected_str = "neuron.h.test_section(0.5)._ref_v"
    assert recording_str == expected_str
