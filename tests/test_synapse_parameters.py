"""Tests the synapse parameters."""

from pathlib import Path

import numpy as np

import bglibpy
from tests.helpers.circuit import blueconfig_append_path

tests_dir = Path(__file__).resolve().parent


def test_syn_dict_example_sims():
    """Test synapses dict produced in example simulations."""
    gid = 1
    conf_pre_path = tests_dir / "examples" / "sim_twocell_all"

    modified_conf = blueconfig_append_path(
        conf_pre_path / "BlueConfig", conf_pre_path
    )

    ssim = bglibpy.SSim(modified_conf)
    syn_descriptions = ssim.get_syn_descriptions(gid)
    assert set(syn_descriptions.index) == {
        ("", 0),
        ("", 1),
        ("", 2),
        ("", 3),
        ("", 4),
    }

    first_syn_description = np.array(
        [2.00000000e+00, 4.37500000e+00, 1.98000000e+02, 0.00000000e+00,
         2.99810982e+00, 3.17367077e-01, 5.01736701e-01, 6.72000000e+02,
         1.70000000e+01, 1.75563037e+00, 1.13000000e+02, 0, 0])

    assert np.allclose(syn_descriptions.loc[("", 0)], first_syn_description)
