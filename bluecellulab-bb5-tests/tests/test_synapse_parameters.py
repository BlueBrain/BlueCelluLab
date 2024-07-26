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
"""Tests the synapse parameters."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bluecellulab import CircuitSimulation
from bluecellulab.circuit import SynapseProperty

from helpers.circuit import blueconfig_append_path

tests_dir = Path(__file__).resolve().parent

proj1_path = "/gpfs/bbp.cscs.ch/project/proj1/"
proj55_path = "/gpfs/bbp.cscs.ch/project/proj55/"
proj64_path = "/gpfs/bbp.cscs.ch/project/proj64/"
proj83_path = "/gpfs/bbp.cscs.ch/project/proj83/"


def assert_series_almost_equal(series1, series2, rtol=1e-05, atol=1e-08):
    # Check if both series have the same length
    assert len(series1) == len(series2), "The series have different lengths."

    # Iterate over each pair of elements
    for i in range(len(series1)):
        # Check for string first
        if isinstance(series1[i], str) or isinstance(series2[i], str):
            assert series1[i] == series2[i], f"Elements at index {i} are not equal."
        else:
            # For non-string values, use np.isclose for comparison
            assert np.isclose(series1[i], series2[i], rtol=rtol, atol=atol)


@pytest.mark.v5
def test_syn_dict_proj1_sim1():
    """Test the synapse dict produced in renccv2 simulation."""
    gid = 75936
    blueconfig = os.path.join(
        proj1_path,
        "simulations/ReNCCv2/k_ca_scan_dense/K5p0/Ca1p25_synreport/",
        "BlueConfig",
    )

    circuit_sim = CircuitSimulation(blueconfig)
    syn_descriptions = circuit_sim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 4304

    a_syn_idx = ("", 4157)
    syn_params_gt = pd.Series(
        [
            1.84112000e05,
            2.77500000e00,
            2.85000000e02,
            4.00000000e01,
            6.98333502e-01,
            5.14560044e-01,
            4.89490896e-01,
            6.62000000e02,
            1.40000000e01,
            1.64541817e00,
            1.16000000e02,
            "",
            0,
            0,
        ]
    )
    res_series = syn_descriptions.loc[a_syn_idx]
    assert_series_almost_equal(res_series, syn_params_gt)


@pytest.mark.v6
def test_syn_dict_proj64_sim1():
    """Tests the synapse dict produced in proj64 sim1."""
    gid = 8709
    blueconfig = os.path.join(
        proj64_path, "circuits/S1HL-200um/20171002/simulations/003", "BlueConfig"
    )
    circuit_sim = CircuitSimulation(blueconfig)
    syn_descriptions = circuit_sim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 153

    a_syn_idx = ("", 50)
    syn_params_gt = pd.Series(
        [
            6.27400000e03,
            1.05000000e00,
            7.12000000e02,
            1.10000000e01,
            1.20243251e00,
            1.32327390e00,
            2.79871672e-01,
            5.87988770e02,
            1.79656506e01,
            9.63626862e00,
            1.00000000e00,
            "",
            0,
            0,
        ]
    )

    # comparing first n elements of pd.series with np.array
    res_series = syn_descriptions.loc[a_syn_idx]
    assert_series_almost_equal(res_series, syn_params_gt)


@pytest.mark.v6
def test_syn_dict_proj64_sim2():
    """Tests the synapse dict produced in proj64 sim3."""
    gid = 1326
    blueconfig = os.path.join(
        proj64_path,
        "home/vangeit/simulations/",
        "random123_tests/",
        "random123_tests_newneurod_rnd123",
        "BlueConfig",
    )
    circuit_sim = CircuitSimulation(blueconfig)
    syn_descriptions = circuit_sim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 250
    a_syn_idx = ("", 28)

    syn_params_gt = pd.Series(
        [
            625.0,
            2.275,
            368.0,
            10.0,
            3.7314558029174805,
            1.5232694149017334,
            0.3078206479549408,
            332.729736328125,
            17.701997756958008,
            9.191937446594238,
            1.0,
            1.0,
            "",
            0,
            0,
        ],
    )
    # comparing first n elements of pd.series with np.array
    res_series = syn_descriptions.loc[a_syn_idx]

    assert_series_almost_equal(res_series, syn_params_gt)


@pytest.mark.v6
def test_syn_dict_proj83_sim1():
    """Test the synapse dict produced in proj83's single vesicle simulation."""
    gid = 4138379
    blueconfig = os.path.join(
        proj83_path, "home/tuncel/bglibpy-tests/single-vesicle-AIS", "BlueConfig"
    )

    circuit_sim = CircuitSimulation(blueconfig)
    syn_descriptions = circuit_sim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 592
    a_syn_idx = ("", 313)

    syn_params_gt = pd.Series(
        [
            2.52158100e06,
            7.25000000e-01,
            1.29000000e02,
            9.00000000e00,
            7.61569560e-01,
            4.97424483e-01,
            6.66551819e02,
            1.65781898e01,
            1.79162169e00,
            1.14000000e02,
            1.00000000e00,
            2.78999996e00,
            6.99999988e-01,
            0.469802,
            "",
            0,
            0,
        ]
    )
    res_series = syn_descriptions.loc[a_syn_idx]
    assert_series_almost_equal(res_series, syn_params_gt)


@pytest.mark.thal
def test_syn_dict_proj55_sim1():
    """Test the synapse dict produced in a proj55 thalamus simulation."""
    gid = 35089
    blueconfig = os.path.join(
        proj55_path,
        "tuncel/simulations/release",
        "2020-08-06-v2",
        "bglibpy-thal-test-with-projections",
        "BlueConfig",
    )

    circuit_sim = CircuitSimulation(blueconfig)
    syn_descriptions = circuit_sim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 355
    a_syn_idx = ("", 234)
    syn_params_gt = pd.Series(
        [
            32088.0,
            4.075,
            626.0,
            79.0,
            0.5477614402770996,
            0.39246755838394165,
            408.77203369140625,
            0.0,
            6.245694637298584,
            4.0,
            1.0,
            0.504438,
            "",
            0,
            0,
        ],
    )
    res_series = syn_descriptions.loc[a_syn_idx]

    assert_series_almost_equal(res_series, syn_params_gt)


@pytest.mark.thal
def test_syn_dict_proj55_sim1_with_projections():
    """Test the synapse dict produced in a proj55 thalamus simulation."""
    gid = 35089
    blueconfig = os.path.join(
        proj55_path,
        "tuncel/simulations/release",
        "2020-08-06-v2",
        "bglibpy-thal-test-with-projections",
        "BlueConfig",
    )

    circuit_sim = CircuitSimulation(blueconfig)
    syn_descriptions = circuit_sim.get_syn_descriptions(
        gid, projections=["ML_afferents", "CT_afferents"]
    )

    synapse_index = [
        SynapseProperty.PRE_GID,
        SynapseProperty.AXONAL_DELAY,
        SynapseProperty.POST_SECTION_ID,
        SynapseProperty.POST_SEGMENT_ID,
        SynapseProperty.POST_SEGMENT_OFFSET,
        SynapseProperty.G_SYNX,
        SynapseProperty.U_SYN,
        SynapseProperty.D_SYN,
        SynapseProperty.F_SYN,
        SynapseProperty.DTC,
        SynapseProperty.TYPE,
        "source_population_name",
        "source_popid",
        "target_popid",
    ]

    assert len(syn_descriptions) == 843
    ml_afferent_syn_idx = ("ML_afferents", 5)
    ml_afferent_res = syn_descriptions.loc[
        ml_afferent_syn_idx
    ]
    ml_afferent_res = ml_afferent_res[synapse_index]  # reorder based on synapse index

    ml_afferent_syn_params_gt = pd.Series(
        [
            1.00018400e06,
            1.07500000e00,
            5.60000000e02,
            3.60000000e01,
            1.94341523e-01,
            4.27851178e00,
            5.03883432e-01,
            6.66070118e02,
            1.18374201e01,
            1.72209917e00,
            1.20000000e02,
            "",
            1,
            0,
        ]
    )
    assert_series_almost_equal(ml_afferent_res, ml_afferent_syn_params_gt)

    ct_afferent_syn_idx = ("CT_afferents", 49)
    ct_afferent_res = syn_descriptions.loc[
        ct_afferent_syn_idx
    ]
    ct_afferent_res = ct_afferent_res[synapse_index]  # reorder based on synapse index

    ct_afferent_syn_params_gt = pd.Series(
        [
            2.00921500e06,
            1.40000000e00,
            5.53000000e02,
            9.00000000e01,
            7.34749257e-01,
            1.79559005e-01,
            1.64240128e-01,
            3.69520016e02,
            1.80942025e02,
            2.89750268e00,
            1.20000000e02,
            "",
            2,
            0,
        ]
    )
    assert_series_almost_equal(ct_afferent_res, ct_afferent_syn_params_gt)


def test_syn_dict_example_sims():
    """Test synapses dict produced in example simulations."""
    gid = 1
    conf_pre_path = tests_dir / "examples" / "sim_twocell_all"

    modified_conf = blueconfig_append_path(conf_pre_path / "BlueConfig", conf_pre_path)

    circuit_sim = CircuitSimulation(modified_conf)
    syn_descriptions = circuit_sim.get_syn_descriptions(gid)
    assert set(syn_descriptions.index) == {
        ("", 0),
        ("", 1),
        ("", 2),
        ("", 3),
        ("", 4),
    }

    first_syn_description = pd.Series(
        [
            2.00000000e00,
            4.37500000e00,
            1.98000000e02,
            0.00000000e00,
            2.99810982e00,
            3.17367077e-01,
            5.01736701e-01,
            6.72000000e02,
            1.70000000e01,
            1.75563037e00,
            1.13000000e02,
            "",
            0,
            0,
        ]
    )
    res_series = syn_descriptions.loc[("", 0)]

    assert_series_almost_equal(res_series, first_syn_description)
