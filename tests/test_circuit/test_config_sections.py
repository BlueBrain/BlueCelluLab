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
"""Unit tests for circuit/config/sections.py."""

import pytest
from bluecellulab.circuit.config.sections import string_to_bool


def test_string_to_bool():
    """Unit test for the string_to_bool function."""
    assert string_to_bool("True")
    assert string_to_bool("true")
    assert string_to_bool("TRUE")
    assert string_to_bool("1")
    assert string_to_bool("False") is False
    assert string_to_bool("false") is False
    assert string_to_bool("FALSE") is False
    assert string_to_bool("0") is False
    with pytest.raises(ValueError):
        string_to_bool("invalid")
