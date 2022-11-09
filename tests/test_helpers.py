"""Tests of the helper functions used in tests."""

import os
from tests.helpers.circuit import blueconfig_append_path

script_dir = os.path.dirname(__file__)


def test_blueconfig_append_path():
    """Tools: Test blueconfig_append_path."""
    conf_pre_path = os.path.join(
        script_dir, "examples", "sim_twocell_empty"
    )
    blueconfig_path = os.path.join(conf_pre_path, "BlueConfig")

    fields = ["MorphologyPath", "METypePath", "CircuitPath",
              "nrnPath", "CurrentDir", "OutputRoot", "TargetFile"]

    modified_config = blueconfig_append_path(
        blueconfig_path, conf_pre_path, fields=fields
    )

    for field in fields:
        field_val = modified_config.Run.__getattr__(field)
        assert os.path.isabs(field_val)
