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
"""Unit tests for SSim"""

# pylint: disable=E1101,W0201,F0401,E0611,W0212

import itertools
import os

import numpy as np
import pytest

import bluecellulab
from bluecellulab import CircuitSimulation
from bluecellulab.circuit import SynapseProperty
from bluecellulab.circuit.node_id import create_cell_id
from bluecellulab.circuit.circuit_access import get_synapse_connection_parameters
from bluecellulab.circuit.config.sections import Conditions
from .test_ssim import rms

script_dir = os.path.dirname(__file__)

# BB5 compiled NEURON is using the legacy units!
bluecellulab.neuron.h.nrnunit_use_legacy(True)


def proj_path(n):
    return f"/gpfs/bbp.cscs.ch/project/proj{n}/"


# Example ReNCCv2 sim used in BluePy use cases
renccv2_bc_1_path = os.path.join(
    proj_path(1),
    "simulations/ReNCCv2/k_ca_scan_dense/K5p0/Ca1p25_synreport/",
    "BlueConfig",
)

test_single_vesicle_path = os.path.join(
    proj_path(83), "home/tuncel/bglibpy-tests/single-vesicle-AIS", "BlueConfig"
)

plasticity_sim_path = os.path.join(
    proj_path(96), "home/tuncel/simulations/glusynapse-pairsim/p_Ca_2_0/", "BlueConfig"
)

risetime_sim_path = os.path.join(
    proj_path(96),
    "home/tuncel/simulations/glusynapse-pairsim/p_Ca_2_0_randomize_risetime/",
    "BlueConfig",
)

no_rand_risetime_sim_path = os.path.join(
    proj_path(96),
    "home/tuncel/simulations/glusynapse-pairsim/p_Ca_2_0_randomize_risetime/",
    "BlueConfigNoRandRise",
)


test_shotnoise_path = os.path.join(
    proj_path(83), "home/tuncel/bglibpy-tests/shotnoise-sim", "BlueConfig"
)

test_relative_shotnoise_path = os.path.join(
    proj_path(83), "home/bolanos/bglibpy-tests/relative_shotnoise", "BlueConfig"
)

test_relative_shotnoise_conductance_path = os.path.join(
    proj_path(83),
    "home/bolanos/bglibpy-tests/relative_shotnoise_conductance",
    "BlueConfig",
)

test_ornstein_path = os.path.join(
    proj_path(83), "home/tuncel/bglibpy-tests/ornstein_uhlenbeck", "BlueConfig"
)

test_relative_ornstein_path = os.path.join(
    proj_path(83),
    "home/tuncel/bglibpy-tests/relative_ornstein_uhlenbeck/",
    "BlueConfig",
)


@pytest.mark.v5
class TestSSimBaseClass_full_run:
    """Class to test SSim with full circuit"""

    def setup_method(self):
        """Setup"""
        self.t_start = 0
        self.t_stop = 500
        self.record_dt = 0.2
        self.len_voltage = self.t_stop / self.record_dt
        self.ssim = CircuitSimulation(renccv2_bc_1_path, record_dt=self.record_dt)
        assert isinstance(self.ssim, CircuitSimulation)

    def test_run(self):
        """SSim: Check if a full replay of a simulation run with forwardskip
        gives the same output trace as on BGQ"""
        gid = 75936
        self.ssim.instantiate_gids(
            [gid], add_synapses=True, add_minis=True, add_replay=True, add_stimuli=True
        )

        self.ssim.run(self.t_stop)

        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
        assert len(time_bglibpy) == self.len_voltage
        assert len(voltage_bglibpy) == self.len_voltage

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid, self.t_start, self.t_stop, self.record_dt
        )

        rms_error = rms(voltage_bglibpy, voltage_bglib)

        assert rms_error < 2.0


@pytest.mark.v5
class TestSSimBaseClass_full_realconn:

    """Class to test SSim with full circuit and multiple cells
    instantiate with real connections"""

    def setup_method(self):
        """Setup"""
        self.t_start = 0
        self.t_stop = 200
        self.record_dt = 0.2
        self.len_voltage = self.t_stop / self.record_dt
        self.ssim = CircuitSimulation(renccv2_bc_1_path, record_dt=self.record_dt)
        assert isinstance(self.ssim, CircuitSimulation)

    def test_run(self):
        """SSim: Check if a multi - cell full replay of a simulation
        gives the same output trace as on BGQ"""
        gid = 75936
        gids = range(gid, gid + 5)
        self.ssim.instantiate_gids(
            gids,
            add_synapses=True,
            add_minis=True,
            add_replay=True,
            add_stimuli=True,
            interconnect_cells=True,
        )
        self.ssim.run(self.t_stop)
        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gids[0])
        assert len(time_bglibpy) == self.len_voltage
        assert len(voltage_bglibpy) == self.len_voltage

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gids[0], self.t_start, self.t_stop, self.record_dt
        )

        rms_error = rms(voltage_bglibpy, voltage_bglib)

        assert rms_error < 2.0


@pytest.mark.v6
class TestSSimBaseClassSingleVesicleMinis:

    """Test SSim with MinisSingleVesicle, SpikeThreshold, V_Init, Celsius"""

    def setup_method(self):
        """Setup"""
        self.t_start = 0
        self.t_stop = 500
        self.record_dt = 0.1
        self.len_voltage = self.t_stop / self.record_dt

        self.ssim = CircuitSimulation(
            test_single_vesicle_path, record_dt=self.record_dt, print_cellstate=True
        )

        self.gid = 4138379
        self.ssim.instantiate_gids(
            [self.gid], add_synapses=True, add_minis=True, add_stimuli=True
        )

        self.cell = self.ssim.cells[self.gid]
        self.cell.add_ais_recording(dt=self.cell.record_dt)

    def test_fetch_emodel_name(self):
        """Test to check if the emodel name is correct."""
        cell_id = create_cell_id(self.gid)
        emodel_name = self.ssim.circuit_access._fetch_emodel_name(cell_id)
        assert emodel_name == "cADpyr_L5TPC"

    def test_fetch_cell_kwargs_emodel_properties(self):
        """Test to verify the kwargs threshold and holding currents."""
        cell_id = create_cell_id(self.gid)
        kwargs = self.ssim.fetch_cell_kwargs(cell_id)
        assert kwargs["emodel_properties"].threshold_current == 0.181875
        assert kwargs["emodel_properties"].holding_current == pytest.approx(
            -0.06993715097820541
        )

    def test_create_cell_from_circuit(self):
        """Test to verify the kwargs threshold and holding currents."""
        cell_id = create_cell_id(self.gid)
        cell = self.ssim.create_cell_from_circuit(cell_id)
        assert cell.cell_id.id == self.gid
        assert cell.record_dt == self.record_dt
        assert cell.threshold == 0.181875
        assert cell.hypamp == pytest.approx(-0.06993715097820541)

    def test_run(self):
        """SSim: Check if a full replay with MinisSingleVesicle
        SpikeThreshold, V_Init, Celcius produce the same voltage"""
        self.ssim.run(self.t_stop)

        voltage_bglibpy = self.ssim.get_voltage_trace(self.gid)
        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            self.gid, self.t_start, self.t_stop, self.record_dt
        )

        assert len(voltage_bglibpy) == len(voltage_bglib)
        rms_error = rms(voltage_bglibpy, voltage_bglib)

        assert rms_error < 4.38

        self.check_ais_voltages()

    def check_ais_voltages(self):
        """Makes sure recording at AIS from bglibpy and ND produce the same."""
        ais_voltage_bglibpy = self.cell.get_ais_voltage()

        ais_report = self.ssim.circuit_access._bluepy_sim.report("axon_SONATA")
        ais_voltage_mainsim = ais_report.get_gid(self.gid).to_numpy()

        assert len(ais_voltage_bglibpy) == len(ais_voltage_mainsim)

        rms_error = rms(ais_voltage_bglibpy, ais_voltage_mainsim)

        assert rms_error < 14.91


@pytest.mark.v6
def test_ssim_glusynapse():
    """Test the glusynapse mod and helper on a plasticity simulation."""
    ssim = CircuitSimulation(plasticity_sim_path, record_dt=0.1)
    gids = [3424064, 3424037]
    ssim.instantiate_gids(
        gids,
        add_synapses=True,
        add_stimuli=True,
        add_replay=False,
        intersect_pre_gids=[3424064],
    )
    tstop = 750
    ssim.run(tstop)
    cell = gids[1]  # postcell
    voltage_bglibpy = ssim.get_voltage_trace(cell)
    voltage_bglib = ssim.get_mainsim_voltage_trace(cell)[: len(voltage_bglibpy)]
    rms_error = rms(voltage_bglibpy, voltage_bglib)
    assert rms_error < 1e-3


@pytest.mark.v6
@pytest.mark.parametrize(
    "sim_path,expected_val",
    [
        (plasticity_sim_path, (0.12349485646988291, 0.04423470638285594)),
        (risetime_sim_path, (0.12349485646988291, 0.04423470638285594)),
        (no_rand_risetime_sim_path, (0.2, 0.2)),
    ],
)
def test_ssim_rand_gabaab_risetime(sim_path, expected_val):
    """Test for randomize_Gaba_risetime in BlueConfig Conditions block."""
    ssim = CircuitSimulation(sim_path, record_dt=0.1)
    gid = 3424037
    ssim.instantiate_gids([gid], intersect_pre_gids=[481868], add_synapses=True)

    cell = ssim.cells[gid]
    assert cell.synapses[("", 388)].hsynapse.tau_r_GABAA == expected_val[0]
    assert cell.synapses[("", 389)].hsynapse.tau_r_GABAA == expected_val[1]


@pytest.mark.v5
class TestSSimBaseClass_full:

    """Class to test SSim with full circuit"""

    def setup_method(self):
        """Setup"""
        self.ssim = CircuitSimulation(renccv2_bc_1_path)
        assert isinstance(self.ssim, CircuitSimulation)

    def test_generate_mtype_list(self):
        """SSim: Test generate_mtype_list"""
        mtypes_list = [["L23_BTC"], ["L23_BTC", "L23_LBC"], ["L5_TTPC1", "L6_TPC_L1"]]

        for mtypes in mtypes_list:
            mtypes_gids = self.ssim.circuit_access.get_gids_of_mtypes(mtypes=mtypes)
            mtypes_gids = {x.id for x in mtypes_gids}

            mtypes_filename = os.path.join(
                script_dir, "examples/mtype_lists", "%s.%s" % ("_".join(mtypes), "txt")
            )

            expected_gids = set(np.loadtxt(mtypes_filename))
            assert expected_gids == mtypes_gids

    def test_evaluate_connection_parameters(self):
        """SSim: Check if Connection block parsers yield expected output"""

        target_params = [
            (
                "L5_TTPC1",
                "L5_TTPC1",
                {
                    "SpontMinis": 0.067000000000000004,
                    "add_synapse": True,
                    "SynapseConfigure": [
                        "%s.mg = 1.0",
                        "%s.Use *= 0.185409696687",
                        "%s.NMDA_ratio = 0.4",
                        "%s.NMDA_ratio = 0.71",
                    ],
                    "Weight": 1.0,
                },
            ),
            (
                "L5_MC",
                "L5_MC",
                {
                    "SpontMinis": 0.012,
                    "add_synapse": True,
                    "SynapseConfigure": [
                        "%s.e_GABAA = -80.0 " "%s.e_GABAB = -75.8354310081",
                        "%s.Use *= 0.437475790642",
                    ],
                    "Weight": 1.0,
                },
            ),
            (
                "L5_LBC",
                "L5_LBC",
                {
                    "SpontMinis": 0.012,
                    "add_synapse": True,
                    "SynapseConfigure": [
                        "%s.e_GABAA = -80.0 " "%s.e_GABAB = -75.8354310081",
                        "%s.Use *= 0.437475790642",
                    ],
                    "Weight": 1.0,
                },
            ),
            (
                "L1_HAC",
                "L23_PC",
                {
                    "SpontMinis": 0.012,
                    "add_synapse": True,
                    "SynapseConfigure": [
                        "%s.e_GABAA = -80.0 %s.e_GABAB = -75.8354310081",
                        "%s.Use *= 0.437475790642",
                        "%s.GABAB_ratio = 0.75",
                    ],
                    "Weight": 1.0,
                },
            ),
        ]

        for pre_target, post_target, params in target_params:
            pre_gid, post_gid, syn_ids = list(
                itertools.islice(
                    self.ssim.circuit_access._bluepy_circuit.connectome.iter_connections(
                        pre_target, post_target, return_synapse_ids=True
                    ),
                    1,
                )
            )[0]
            syn_id = syn_ids[0][1]

            syn_desc = self.ssim.get_syn_descriptions(post_gid).loc[("", syn_id)]

            assert pre_gid == syn_desc[SynapseProperty.PRE_GID]

            pre_gid = create_cell_id(pre_gid)
            post_gid = create_cell_id(post_gid)
            evaluated_params = get_synapse_connection_parameters(
                self.ssim.circuit_access, pre_gid, post_gid
            )
            assert params == evaluated_params

    def test_add_replay_synapse_SynapseConfigure(self):
        """SSim: Check if SynapseConfigure works correctly"""
        gid = self.ssim.circuit_access.get_cell_ids_of_targets(["L5_MC"]).pop()
        self.ssim.instantiate_gids([gid], add_synapses=False, add_minis=False)
        pre_datas = self.ssim.get_syn_descriptions(gid)
        first_inh_syn = pre_datas[pre_datas[SynapseProperty.TYPE] < 100].iloc[0]
        sid = int(first_inh_syn.name[1])
        connection_parameters = {
            "SynapseConfigure": [
                "%s.e_GABAA = -80.5 %s.e_GABAB = -101.0",
                "%s.tau_d_GABAA = 10.0 %s.tau_r_GABAA = 1.0",
                "%s.e_GABAA = -80.6",
            ],
            "Weight": 2.0,
        }
        condition_entries = Conditions.init_empty()
        popids = (0, 0)
        extracellular_calcium = None
        self.ssim.cells[gid].add_replay_synapse(
            ("", sid),
            first_inh_syn,
            connection_parameters,
            condition_entries,
            popids,
            extracellular_calcium
        )

        assert self.ssim.cells[gid].synapses[("", sid)].hsynapse.e_GABAA == -80.6
        assert self.ssim.cells[gid].synapses[("", sid)].hsynapse.e_GABAB == -101.0
        assert self.ssim.cells[gid].synapses[("", sid)].hsynapse.tau_d_GABAA == 10.0
        assert self.ssim.cells[gid].synapses[("", sid)].hsynapse.tau_r_GABAA == 1.0


@pytest.mark.v6
def test_shotnoise():
    """Test injection of relative shot noise."""
    ssim = CircuitSimulation(test_shotnoise_path, record_dt=0.1)
    gids = [2886525, 3099746, 3425774, 3868780]
    ssim.instantiate_gids(gids, add_synapses=False, add_stimuli=True, add_replay=False)
    tstop = 250
    ssim.run(tstop)

    rms_errors = []
    for cell in gids:
        voltage_bglibpy = ssim.get_voltage_trace(cell)
        voltage_bglib = ssim.get_mainsim_voltage_trace(cell)[: len(voltage_bglibpy)]
        rms_error = rms(voltage_bglibpy, voltage_bglib)
        rms_errors.append(rms_error < 0.025)

    assert all(rms_errors)


@pytest.mark.v6
def test_relative_shotnoise():
    """Test injection of relative shot noise."""
    ssim = CircuitSimulation(test_relative_shotnoise_path, record_dt=0.1)
    gids = [2886525, 3099746, 3425774, 3868780]
    ssim.instantiate_gids(gids, add_synapses=False, add_stimuli=True, add_replay=False)
    tstop = 420
    ssim.run(tstop)

    rms_errors = []
    for cell in gids:
        voltage_bglibpy = ssim.get_voltage_trace(cell)
        voltage_bglib = ssim.get_mainsim_voltage_trace(cell)[: len(voltage_bglibpy)]
        voltage_diff = voltage_bglibpy - voltage_bglib
        rms_error = np.sqrt(np.mean(voltage_diff**2))
        rms_errors.append(rms_error < 0.025)

    assert all(rms_errors)


@pytest.mark.v6
def test_relative_shotnoise_conductance():
    """Test injection of relative shot noise in conductance mode."""
    ssim = CircuitSimulation(test_relative_shotnoise_conductance_path, record_dt=0.1)
    gids = [2886525, 3099746, 3425774, 3868780]
    ssim.instantiate_gids(gids, add_synapses=False, add_stimuli=True, add_replay=False)
    tstop = 500
    ssim.run(tstop)

    rms_errors = []
    for cell in gids:
        voltage_bglibpy = ssim.get_voltage_trace(cell)
        voltage_bglib = ssim.get_mainsim_voltage_trace(cell)[: len(voltage_bglibpy)]
        rms_error = rms(voltage_bglibpy, voltage_bglib)
        rms_errors.append(rms_error < 0.025)

    assert all(rms_errors)


@pytest.mark.v6
def test_ornstein_uhlenbeck():
    """Test injection of Ornstein-Uhlenbeck noise."""
    ssim = CircuitSimulation(test_ornstein_path, record_dt=0.025)
    gid = 2886525  # from the 'small' target
    ssim.instantiate_gids([gid], add_synapses=False, add_stimuli=True, add_replay=False)
    tstop = 500
    ssim.run(tstop)

    voltage_bglibpy = ssim.get_voltage_trace(gid)
    voltage_bglib = ssim.get_mainsim_voltage_trace(gid)[: len(voltage_bglibpy)]
    rms_error = rms(voltage_bglibpy, voltage_bglib)
    assert rms_error < 0.025


@pytest.mark.v6
def test_relative_ornstein_uhlenbeck():
    """Test injection of relative Ornstein-Uhlenbeck noise."""
    ssim = CircuitSimulation(test_relative_ornstein_path, record_dt=0.025)
    gids = [3425774, 3868780, 2886525, 3099746]
    ssim.instantiate_gids(gids, add_synapses=False, add_stimuli=True, add_replay=False)
    tstop = 300
    ssim.run(tstop)

    rms_errors = []
    for cell in gids:
        voltage_bglibpy = ssim.get_voltage_trace(cell)
        voltage_bglib = ssim.get_mainsim_voltage_trace(cell)[: len(voltage_bglibpy)]
        rms_error = rms(voltage_bglibpy, voltage_bglib)
        rms_errors.append(rms_error < 0.025)

    assert all(rms_errors)
