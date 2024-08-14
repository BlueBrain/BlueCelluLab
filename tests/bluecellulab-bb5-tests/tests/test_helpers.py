# Copyright 2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests of the helper functions used in tests."""

import os
from helpers.circuit import blueconfig_append_path

script_dir = os.path.dirname(__file__)


def test_blueconfig_append_path():
    """Tools: Test blueconfig_append_path."""
    conf_pre_path = os.path.join(script_dir, "examples", "sim_twocell_empty")
    blueconfig_path = os.path.join(conf_pre_path, "BlueConfig")

    fields = [
        "MorphologyPath",
        "METypePath",
        "CircuitPath",
        "nrnPath",
        "CurrentDir",
        "OutputRoot",
        "TargetFile",
    ]

    modified_config = blueconfig_append_path(
        blueconfig_path, conf_pre_path, fields=fields
    )

    for field in fields:
        field_val = modified_config.Run.__getattr__(field)
        assert os.path.isabs(field_val)
