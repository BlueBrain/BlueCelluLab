"""Tests the synapse parameters."""

import os
from pathlib import Path

import bluepy
import numpy as np
import pandas as pd
import pytest
from bluepy.enums import Synapse as BLPSynapse

import bglibpy

tests_dir = Path(__file__).resolve().parent

proj1_path = "/gpfs/bbp.cscs.ch/project/proj1/"
proj55_path = "/gpfs/bbp.cscs.ch/project/proj55/"
proj64_path = "/gpfs/bbp.cscs.ch/project/proj64/"
proj83_path = "/gpfs/bbp.cscs.ch/project/proj83/"


def test_check_nrrp_value():
    """Unit test for check nrrp value."""
    synapses = pd.DataFrame(data={BLPSynapse.NRRP: [15.0, 16.0]})

    bglibpy.synapse.check_nrrp_value(synapses)

    synapses[BLPSynapse.NRRP].loc[0] = 15.1
    with pytest.raises(ValueError):
        bglibpy.synapse.check_nrrp_value(synapses)

    synapses[BLPSynapse.NRRP].loc[0] = -1

    with pytest.raises(ValueError):
        bglibpy.synapse.check_nrrp_value(synapses)


def test_get_connectomes_dict():
    """Test creation of connectome dict."""
    conf_pre_path = tests_dir / "examples" / "sim_twocell_all"
    modified_conf = bglibpy.tools.blueconfig_append_path(
        conf_pre_path / "BlueConfig", conf_pre_path
    )
    bc_simulation = bluepy.Simulation(modified_conf)
    bc_circuit = bc_simulation.circuit

    connectomes_dict = bglibpy.synapse.get_connectomes_dict(bc_circuit, None, None)
    assert connectomes_dict.keys() == {""}

    with pytest.raises(ValueError):
        bglibpy.synapse.get_connectomes_dict(bc_circuit, "projection1", ["projection2"])


@pytest.mark.thal
def test_get_connectomes_dict_with_projections():
    """Test the retrieval of projection and the local connectomes."""
    test_thalamus_path = os.path.join(
        proj55_path,
        "tuncel/simulations/release",
        "2020-08-06-v2",
        "bglibpy-thal-test-with-projections",
        "BlueConfig",
    )
    bc_simulation = bluepy.Simulation(test_thalamus_path)
    bc_circuit = bc_simulation.circuit

    # empty
    assert bglibpy.synapse.get_connectomes_dict(bc_circuit, None, None).keys() == {""}

    connectomes_dict = bglibpy.synapse.get_connectomes_dict(
        bc_circuit, "ML_afferents", None)

    # single projection
    assert connectomes_dict.keys() == {"", "ML_afferents"}

    # multiple projections
    all_connectomes = bglibpy.synapse.get_connectomes_dict(
        bc_circuit, None, ["ML_afferents", "CT_afferents"])
    assert all_connectomes.keys() == {"", "ML_afferents", "CT_afferents"}


def test_get_synapses_by_connectomes():
    """Test get_synapses_by_connectomes function."""
    conf_pre_path = tests_dir / "examples" / "sim_twocell_all"
    modified_conf = bglibpy.tools.blueconfig_append_path(
        conf_pre_path / "BlueConfig", conf_pre_path
    )
    bc_simulation = bluepy.Simulation(modified_conf)
    bc_circuit = bc_simulation.circuit

    gid = 1
    connectomes_dict = bglibpy.synapse.get_connectomes_dict(bc_circuit, None, None)
    all_properties = [
        BLPSynapse.PRE_GID,
        BLPSynapse.AXONAL_DELAY,
        BLPSynapse.POST_SECTION_ID,
        BLPSynapse.POST_SEGMENT_ID,
        BLPSynapse.POST_SEGMENT_OFFSET,
        BLPSynapse.G_SYNX,
        BLPSynapse.U_SYN,
        BLPSynapse.D_SYN,
        BLPSynapse.F_SYN,
        BLPSynapse.DTC,
        BLPSynapse.TYPE,
        BLPSynapse.NRRP,
        BLPSynapse.U_HILL_COEFFICIENT,
        BLPSynapse.CONDUCTANCE_RATIO]

    synapses = bglibpy.synapse.get_synapses_by_connectomes(
        connectomes_dict, all_properties, gid)

    proj_id, syn_idx = '', 0
    assert synapses.index[0] == (proj_id, syn_idx)
    assert synapses.shape == (5, 11)


def test_syn_dict_example_sims():
    """Test synapses dict produced in example simulations."""
    gid = 1
    conf_pre_path = tests_dir / "examples" / "sim_twocell_all"

    modified_conf = bglibpy.tools.blueconfig_append_path(
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


@pytest.mark.v5
def test_syn_dict_proj1_sim1():
    """Test the synapse dict produced in renccv2 simulation."""
    gid = 75936
    blueconfig = os.path.join(
        proj1_path,
        "simulations/ReNCCv2/k_ca_scan_dense/K5p0/Ca1p25_synreport/",
        "BlueConfig")

    ssim = bglibpy.SSim(blueconfig)
    syn_descriptions = ssim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 4304

    a_syn_idx = ('', 4157)
    syn_params_gt = np.array(
        [1.84112000e+05, 2.77500000e+00, 2.85000000e+02, 4.00000000e+01,
         6.98333502e-01, 5.14560044e-01, 4.89490896e-01, 6.62000000e+02,
         1.40000000e+01, 1.64541817e+00, 1.16000000e+02, 0, 0])

    assert np.allclose(syn_descriptions.loc[a_syn_idx], syn_params_gt)


@pytest.mark.v6
def test_syn_dict_proj64_sim1():
    """Tests the synapse dict produced in proj64 sim1."""
    gid = 8709
    blueconfig = os.path.join(
        proj64_path,
        "circuits/S1HL-200um/20171002/simulations/003",
        "BlueConfig")
    ssim = bglibpy.SSim(blueconfig)
    syn_descriptions = ssim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 153

    a_syn_idx = ('', 50)
    syn_params_gt = np.array(
        [6.27400000e+03, 1.05000000e+00, 7.12000000e+02, 1.10000000e+01,
         1.20243251e+00, 1.32327390e+00, 2.79871672e-01, 5.87988770e+02,
         1.79656506e+01, 9.63626862e+00, 1.00000000e+00, 0, 0])

    assert np.allclose(syn_descriptions.loc[a_syn_idx], syn_params_gt)


@pytest.mark.v6
def test_syn_dict_proj64_sim2():
    """Tests the synapse dict produced in proj64 sim3."""
    gid = 1326
    blueconfig = os.path.join(proj64_path,
                              "home/vangeit/simulations/",
                              "random123_tests/",
                              "random123_tests_newneurod_rnd123",
                              "BlueConfig")
    ssim = bglibpy.SSim(blueconfig)
    syn_descriptions = ssim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 250
    a_syn_idx = ('', 28)

    syn_params_gt = np.array(
        [625.0, 2.275, 368.0, 10.0, 3.7314558029174805, 1.5232694149017334,
         0.3078206479549408, 332.729736328125,
         17.701997756958008, 9.191937446594238, 1.0, 1.0, 0, 0], dtype=np.float)

    assert np.array_equal(
        syn_descriptions.loc[a_syn_idx], syn_params_gt, equal_nan=True)


@pytest.mark.v6
def test_syn_dict_proj83_sim1():
    """Test the synapse dict produced in proj83's single vesicle simulation."""
    gid = 4138379
    blueconfig = os.path.join(
        proj83_path, "home/tuncel/bglibpy-tests/single-vesicle-AIS",
        "BlueConfig"
    )

    ssim = bglibpy.SSim(blueconfig)
    syn_descriptions = ssim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 592
    a_syn_idx = ('', 313)

    syn_params_gt = np.array(
        [2.52158100e+06, 7.25000000e-01, 1.29000000e+02, -1.00000000e+00,
         4.69802350e-01, 7.61569560e-01, 4.97424483e-01, 6.66551819e+02,
         1.65781898e+01, 1.79162169e+00, 1.14000000e+02, 1.00000000e+00,
         2.78999996e+00, 6.99999988e-01, 0, 0])

    assert np.allclose(syn_descriptions.loc[a_syn_idx], syn_params_gt)


@pytest.mark.thal
def test_syn_dict_proj55_sim1():
    """Test the synapse dict produced in a proj55 thalamus simulation."""
    gid = 35089
    blueconfig = os.path.join(proj55_path,
                              "tuncel/simulations/release",
                              "2020-08-06-v2",
                              "bglibpy-thal-test-with-projections",
                              "BlueConfig")

    ssim = bglibpy.SSim(blueconfig)
    syn_descriptions = ssim.get_syn_descriptions(gid)

    assert len(syn_descriptions) == 355
    a_syn_idx = ('', 234)

    syn_params_gt = np.array(
        [32088.0, 4.075, 626.0, -1.0, 0.5044383406639099,
         0.5477614402770996, 0.39246755838394165, 408.77203369140625, 0.0,
         6.245694637298584, 4.0, 1.0, 0, 0], dtype=np.float)

    assert np.array_equal(
        syn_descriptions.loc[a_syn_idx], syn_params_gt, equal_nan=True)


@pytest.mark.thal
def test_syn_dict_proj55_sim1_with_projections():
    """Test the synapse dict produced in a proj55 thalamus simulation."""
    gid = 35089
    blueconfig = os.path.join(proj55_path,
                              "tuncel/simulations/release",
                              "2020-08-06-v2",
                              "bglibpy-thal-test-with-projections",
                              "BlueConfig")

    ssim = bglibpy.SSim(blueconfig)
    syn_descriptions = ssim.get_syn_descriptions(
        gid, projections=["ML_afferents", "CT_afferents"]
    )

    synapse_index = ([
        BLPSynapse.PRE_GID, BLPSynapse.AXONAL_DELAY,
        BLPSynapse.POST_SECTION_ID, BLPSynapse.POST_SEGMENT_ID,
        BLPSynapse.POST_SEGMENT_OFFSET, BLPSynapse.G_SYNX, BLPSynapse.U_SYN,
        BLPSynapse.D_SYN, BLPSynapse.F_SYN, BLPSynapse.DTC, BLPSynapse.TYPE,
        'source_popid', 'target_popid'])

    assert len(syn_descriptions) == 843
    ml_afferent_syn_idx = ("ML_afferents", 5)
    ml_afferent_res = syn_descriptions.loc[ml_afferent_syn_idx].dropna()  # drop nan values
    ml_afferent_res = ml_afferent_res[synapse_index]  # reorder based on synapse index

    ml_afferent_syn_params_gt = np.array(
        [1.00018400e+06, 1.07500000e+00, 5.60000000e+02, 3.60000000e+01,
         1.94341523e-01, 4.27851178e+00, 5.03883432e-01, 6.66070118e+02,
         1.18374201e+01, 1.72209917e+00, 1.20000000e+02, 1, 0])
    ml_afferent_gt = pd.Series(data=ml_afferent_syn_params_gt, index=synapse_index)

    assert np.allclose(ml_afferent_res, ml_afferent_gt)

    ct_afferent_syn_idx = ("CT_afferents", 49)
    ct_afferent_res = syn_descriptions.loc[ct_afferent_syn_idx].dropna()  # drop nan values
    ct_afferent_res = ct_afferent_res[synapse_index]  # reorder based on synapse index

    ct_afferent_syn_params_gt = np.array(
        [2.00921500e+06, 1.40000000e+00, 5.53000000e+02, 9.00000000e+01,
         7.34749257e-01, 1.79559005e-01, 1.64240128e-01, 3.69520016e+02,
         1.80942025e+02, 2.89750268e+00, 1.20000000e+02, 2, 0])

    ct_afferent_gt = pd.Series(data=ct_afferent_syn_params_gt, index=synapse_index)

    assert np.allclose(ct_afferent_res, ct_afferent_gt)
