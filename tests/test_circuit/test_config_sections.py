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
