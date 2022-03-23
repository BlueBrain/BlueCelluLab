"""Unit tests for SSim"""

# pylint: disable=E1101,W0201,F0401,E0611,W0212

import os
import itertools

import pytest
import numpy as np
from bluepy.enums import Synapse as BLPSynapse

import bglibpy

script_dir = os.path.dirname(__file__)

def proj_path(n):
    return f"/gpfs/bbp.cscs.ch/project/proj{n}/"

# Example ReNCCv2 sim used in BluePy use cases
renccv2_bc_1_path = os.path.join(
    proj_path(1),
    "simulations/ReNCCv2/k_ca_scan_dense/K5p0/Ca1p25_synreport/",
    "BlueConfig")

v6_test_bc_1_path = os.path.join(
    proj_path(64),
    "circuits/S1HL-200um/20171002/simulations/003",
    "BlueConfig")

# v6_test_bc_3_path = os.path.join(proj64_path,
#                                 "home/king/sim/Basic",
#                                 "BlueConfig")
v6_test_bc_4_path = os.path.join(proj_path(64),
                                 "home/king/sim/Basic",
                                 "BlueConfig")


v6_test_bc_rnd123_1_path = os.path.join(proj_path(64),
                                        "home/vangeit/simulations/",
                                        "random123_tests/",
                                        "random123_tests_newneurod_rnd123",
                                        "BlueConfig")

test_thalamus_path = os.path.join(proj_path(55),
                                  "tuncel/simulations/release",
                                  "2020-08-06-v2",
                                  "bglibpy-thal-test-with-projections",
                                  "BlueConfig")

test_thalamus_no_population_id_path = os.path.join(proj_path(55),
                                                   "tuncel/simulations/release",
                                                   "2020-08-06-v2",
                                                   "bglibpy-thal-test-with-projections",
                                                   "BlueConfigNoPopulationID")

test_single_vesicle_path = os.path.join(
    proj_path(83), "home/tuncel/bglibpy-tests/single-vesicle-AIS",
    "BlueConfig"
)

test_sscx2020_packages_path = os.path.join(
    proj_path(83), "home/tuncel/sscx2020-sim/single-vesicle-minis-sim/",
    "BlueConfig"
)

hip20180219_1_path = os.path.join(
    proj_path(42),
    "circuits/O1/20180219",
    "CircuitConfig")

plasticity_sim_path = os.path.join(
    proj_path(96),
    "home/tuncel/simulations/glusynapse-pairsim/p_Ca_2_0/", "BlueConfig"
)

risetime_sim_path = os.path.join(
    proj_path(96),
    "home/tuncel/simulations/glusynapse-pairsim/p_Ca_2_0_randomize_risetime/",
    "BlueConfig"
)

no_rand_risetime_sim_path = os.path.join(
    proj_path(96),
    "home/tuncel/simulations/glusynapse-pairsim/p_Ca_2_0_randomize_risetime/",
    "BlueConfigNoRandRise"
)

test_relative_shotnoise_path = os.path.join(
    proj_path(83), "home/bolanos/bglibpy-tests/relative_shotnoise",
    "BlueConfig"
)

@pytest.mark.v5
class TestSSimBaseClass_full_run:

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.t_start = 0
        self.t_stop = 500
        self.record_dt = 0.2
        self.len_voltage = self.t_stop / self.record_dt
        self.ssim = bglibpy.ssim.SSim(
            renccv2_bc_1_path, record_dt=self.record_dt
        )
        assert (isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        self.ssim.delete()
        assert (bglibpy.tools.check_empty_topology())

    def test_run(self):
        """SSim: Check if a full replay of a simulation run """ \
            """with forwardskip """ \
            """gives the same output trace as on BGQ"""
        gid = 75936
        self.ssim.instantiate_gids(
            [gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True)

        self.ssim.run(self.t_stop)

        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
        assert len(time_bglibpy) == self.len_voltage
        assert len(voltage_bglibpy) == self.len_voltage

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid, self.t_start, self.t_stop, self.record_dt)

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        assert (rms_error < 2.0)


@pytest.mark.v5
class TestSSimBaseClass_full_realconn:

    """Class to test SSim with full circuit and multiple cells """ \
        """instantiate with real connections"""

    def setup(self):
        """Setup"""
        self.t_start = 0
        self.t_stop = 200
        self.record_dt = 0.2
        self.len_voltage = self.t_stop / self.record_dt
        self.ssim = bglibpy.ssim.SSim(
            renccv2_bc_1_path, record_dt=self.record_dt
        )
        assert(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        self.ssim.delete()
        assert(bglibpy.tools.check_empty_topology())

    def test_run(self):
        """SSim: Check if a multi - cell full replay of a simulation """ \
            """gives the same output trace as on BGQ"""
        gid = 75936
        # gids = 116390, 116392
        gids = range(gid, gid + 5)
        self.ssim.instantiate_gids(
            gids,
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True,
            interconnect_cells=True)
        self.ssim.run(self.t_stop)
        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gids[0])
        assert len(time_bglibpy) == self.len_voltage
        assert len(voltage_bglibpy) == self.len_voltage

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gids[0], self.t_start, self.t_stop, self.record_dt)

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        assert(rms_error < 2.0)


'''
Reenable this once the MOD files are separated out of neurodamus
@attr('gpfs', 'hip', 'debugtest')
class TestSSimBaseClass_hip_20180219:

    """Class to test SSim with full circuit and multiple cells """ \
        """instantiate with real connections"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(hip20180219_1_path)
        assert(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_run(self):
        """SSim: Check if a hippocampal cell can be instantiated"""
        gid = 1111
        self.ssim.instantiate_gids(
            [gid],
            add_synapses=True)
        print(self.ssim.cells[gid])
'''


@pytest.mark.v6
class TestSSimBaseClass_v6_full_run:

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.t_start = 0
        self.t_stop = 500
        self.record_dt = 0.1
        self.len_voltage = self.t_stop / self.record_dt
        self.ssim = bglibpy.ssim.SSim(
            v6_test_bc_1_path, record_dt=self.record_dt
        )
        assert(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        self.ssim.delete()

        assert(bglibpy.tools.check_empty_topology())

    def test_run(self):
        """SSim: Check if a full replay of a simulation run """ \
            """with forwardskip """ \
            """gives the same output trace as on BGQ for v6"""
        gid = 8709
        self.ssim.instantiate_gids(
            [gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True)

        self.ssim.run(self.t_stop)

        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
        assert len(time_bglibpy) == self.len_voltage
        assert len(voltage_bglibpy) == self.len_voltage

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid, self.t_start, self.t_stop, self.record_dt)

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        assert(rms_error < 0.5)


'''
@attr('gpfs', 'v6', 'debugtest')
class TestSSimBaseClass_v6_mvr_run:

    """Class to test SSim with full mvr circuit"""

    def setup(self):
        """Setup"""
        self.ssim = None

    def teardown(self):
        """Teardown"""
        pass

    def test_run(self):
        """SSim: Check if a full replay of a simulation """ \
            """gives the same output trace for O1v6a"""
        gids = [29561, 127275, 105081, 41625]
        for i, gid in enumerate(gids):
            self.ssim = bglibpy.ssim.SSim(v6_test_bc_4_path, record_dt=0.1)

            self.ssim.instantiate_gids(
                [gid],
                synapse_detail=2,
                add_replay=True,
                add_stimuli=True)

            # Point check of one synapse
            # Manual examination of nrn.h5 showed it has to have Nrrp == 3
            if gid == 29561:
                one_synapse = self.ssim.cells[gid].synapses[150]
                assert(hasattr(one_synapse, 'Nrrp'))
                assert one_synapse.Nrrp == 3

            self.ssim.run(500)

            time_bglibpy = self.ssim.get_time_trace()
            voltage_bglibpy = self.ssim.get_voltage_trace(gid)
            assert len(time_bglibpy) == 5000
            assert len(voltage_bglibpy) == 5000

            voltage_bglib = self.ssim.get_mainsim_voltage_trace(
                gid)[:len(voltage_bglibpy)]

            """
            time_bglib = self.ssim.get_mainsim_time_trace()[
                :len(voltage_bglibpy)]

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.plot(time_bglibpy, voltage_bglibpy, 'o', label='bglibpy')
            plt.plot(time_bglib, voltage_bglib, label='neurodamus')
            plt.legend()
            plt.savefig('O1v6a_%d.png' % i)
            """
            rms_error = np.sqrt(
                np.mean(
                    (voltage_bglibpy - voltage_bglib) ** 2))

            assert rms_error < 10.0

            del self.ssim
'''


@pytest.mark.thal
class TestSSimBaseClass_thalamus:
    """Class to test SSim for thalamus with 5 cells of interest"""

    def setup(self):
        """Setup"""
        self.ssim = None
        self.t_start = 0
        self.t_stop = 300
        self.record_dt = 0.1
        self.len_voltage = self.t_stop / self.record_dt

    def teardown(self):
        """Teardown"""
        pass

    def test_run(self):
        """SSim: Check if replay of thalamus simulation for the cells of
        interest gives similar output to the main simulation
        """

        gids = [35089, 37922, 38466, 40190, 42227]

        for gid in gids:

            ssim = bglibpy.ssim.SSim(
                test_thalamus_path,
                record_dt=self.record_dt)

            ssim.instantiate_gids(
                [gid],
                add_synapses=True,
                add_minis=True,
                add_stimuli=True,
                add_noise_stimuli=True,
                add_hyperpolarizing_stimuli=True,
                add_replay=True,
                add_projections=True,
            )
            ssim.run(self.t_stop)

            voltage_bglibpy = ssim.get_voltage_trace(gid)[1:]

            voltage_bglib = ssim.get_mainsim_voltage_trace(
                gid, self.t_start, self.t_stop, self.record_dt)[1:]

            assert len(voltage_bglibpy) == self.len_voltage - 1
            assert len(voltage_bglib) == self.len_voltage - 1

            rms_error = np.sqrt(
                np.mean(
                    (voltage_bglibpy - voltage_bglib) ** 2))

            assert rms_error < 0.055

    def test_population_id(self):
        """Tests the behaviour when the population id is missing."""

        gid = 35089
        ssim = bglibpy.ssim.SSim(
            test_thalamus_no_population_id_path,
            record_dt=0.1)

        with pytest.raises(bglibpy.PopulationIDMissingError):
            ssim.instantiate_gids(
                [gid], add_synapses=True, add_projections=True)

        ssim2 = bglibpy.ssim.SSim(
            test_thalamus_no_population_id_path,
            record_dt=0.1, ignore_populationid_error=True)
        # no exception
        ssim2.instantiate_gids(
            [gid],
            add_synapses=True,
            add_projections=True
        )


@pytest.mark.v6
class TestSSimBaseClass_v6_rnd123_1:

    """Class to test SSim with 1000 cell random123 circuit"""

    def setup(self):
        """Setup"""
        self.ssim = None
        self.t_start = 0
        self.t_stop = 200
        self.record_dt = 0.1
        self.len_voltage = self.t_stop / self.record_dt

    def teardown(self):
        """Teardown"""
        pass

    def test_run(self):
        """SSim: Check if a full replay with random 123 of a simulation """ \
            """gives the same output trace for O1v6a"""
        # gids = [1326, 67175, 160384]
        gids = [1326, 67175, 160384]
        # gids = [160384]
        # bglibpy.set_verbose(100)
        for gid in gids:
            self.ssim = bglibpy.ssim.SSim(
                v6_test_bc_rnd123_1_path,
                record_dt=self.record_dt)

            self.ssim.instantiate_gids(
                [gid],
                add_synapses=True,
                add_replay=True,
                add_minis=True,
                add_stimuli=True)

            self.ssim.run(self.t_stop)

            time_bglibpy = self.ssim.get_time_trace()
            voltage_bglibpy = self.ssim.get_voltage_trace(gid)
            assert len(time_bglibpy) == self.len_voltage
            assert len(voltage_bglibpy) == self.len_voltage

            voltage_bglib = self.ssim.get_mainsim_voltage_trace(
                gid, self.t_start, self.t_stop, self.record_dt)

            '''

            time_bglib = self.ssim.get_mainsim_time_trace()[
                :len(voltage_bglibpy)]

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.plot(time_bglibpy, voltage_bglibpy, label='bglibpy %d' % gid)
            plt.plot(time_bglib, voltage_bglib, label='neurodamus %d' % gid)
            plt.legend()
            plt.savefig('O1v6a_rng_%d.png' % gid)
            # import pickle
            # pickle.dump(plt.gcf(), open('O1v6a_rng_%d.pickle' % gid, 'wb'))
            # plt.clear()
            '''

            rms_error = np.sqrt(
                np.mean(
                    (voltage_bglibpy - voltage_bglib) ** 2))

            assert rms_error < 10.0

            self.ssim.delete()
            assert(bglibpy.tools.check_empty_topology())


@pytest.mark.v6
class TestSSimBaseClassSingleVesicleMinis:

    """Test SSim with MinisSingleVesicle, SpikeThreshold, V_Init, Celsius"""

    def setup(self):
        """Setup"""
        self.t_start = 0
        self.t_stop = 500
        self.record_dt = 0.1
        self.len_voltage = self.t_stop / self.record_dt

        self.ssim = bglibpy.ssim.SSim(
            test_single_vesicle_path,
            record_dt=self.record_dt)

        self.gid = 4138379
        self.ssim.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True)

        self.cell = self.ssim.cells[self.gid]
        self.cell.add_ais_recording(dt=self.cell.record_dt)

    def teardown(self):
        """Teardown"""
        del self.cell
        self.ssim.delete()
        assert(bglibpy.tools.check_empty_topology())

    def test_fetch_emodel_name(self):
        """Test to check if the emodel name is correct."""
        emodel_name = self.ssim.fetch_emodel_name(self.gid)
        assert emodel_name == "cADpyr_L5TPC"

    def test_fetch_cell_kwargs_extra_values(self):
        """Test to verify the kwargs threshold and holding currents."""
        kwargs = self.ssim.fetch_cell_kwargs(self.gid)
        assert kwargs["extra_values"]["threshold_current"] == 0.181875
        assert kwargs["extra_values"]["holding_current"] == pytest.approx(
            -0.06993715097820541)

    def test_run(self):
        """SSim: Check if a full replay with MinisSingleVesicle """ \
            """SpikeThreshold, V_Init, Celcius produce the same voltage"""
        self.ssim.run(self.t_stop)

        voltage_bglibpy = self.ssim.get_voltage_trace(self.gid)
        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            self.gid, self.t_start, self.t_stop, self.record_dt)

        assert len(voltage_bglibpy) == len(voltage_bglib)
        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        assert rms_error < 4.38

        self.check_ais_voltages()

    def check_ais_voltages(self):
        """Makes sure recording at AIS from bglibpy and ND produce the same."""
        ais_voltage_bglibpy = self.cell.get_ais_voltage()

        ais_report = self.ssim.bc_simulation.report('axon_SONATA')
        ais_voltage_mainsim = ais_report.get_gid(self.gid).to_numpy()

        assert len(ais_voltage_bglibpy) == len(ais_voltage_mainsim)
        voltage_diff = ais_voltage_bglibpy - ais_voltage_mainsim
        rms_error = np.sqrt(np.mean(voltage_diff ** 2))

        assert rms_error < 14.91


@pytest.mark.v6
def test_sscx_merge_pre_spike_trains():
    """Test for reproducing empty cell_info_dict['connections'] bug."""
    ssim = bglibpy.ssim.SSim(test_sscx2020_packages_path, record_dt=0.1)
    gid = 4138379
    ssim.instantiate_gids([gid], synapse_detail=2, add_replay=True,
                          add_stimuli=False, add_synapses=None,
                          intersect_pre_gids=None)
    cell_info_dict = ssim.cells[gid].info_dict
    assert cell_info_dict["connections"] != {}


@pytest.mark.v6
def test_ssim_glusynapse():
    """Test the glusynapse mod and helper on a plasticity simulation."""
    ssim = bglibpy.SSim(plasticity_sim_path, record_dt=0.1)
    gids = [3424064, 3424037]
    ssim.instantiate_gids(gids, add_synapses=True, add_stimuli=True,
        add_replay=False, intersect_pre_gids=[3424064])
    tstop = 750
    ssim.run(tstop)
    cell = gids[1]  # postcell
    voltage_bglibpy = ssim.get_voltage_trace(cell)
    voltage_bglib = ssim.get_mainsim_voltage_trace(cell)[:len(voltage_bglibpy)]
    voltage_diff = voltage_bglibpy - voltage_bglib
    rms_error = np.sqrt(np.mean(voltage_diff ** 2))
    assert rms_error < 0.025


@pytest.mark.v6
@pytest.mark.parametrize("sim_path,expected_val",[
    (plasticity_sim_path, (0.12349485646988291, 0.04423470638285594)),
    (risetime_sim_path, (0.12349485646988291, 0.04423470638285594)),
    (no_rand_risetime_sim_path, (0.2, 0.2))
    ])
def test_ssim_rand_gabaab_risetime(sim_path, expected_val):
    """Test for randomize_Gaba_risetime in BlueConfig Conditions block."""
    ssim = bglibpy.SSim(sim_path, record_dt=0.1)
    gid = 3424037
    ssim.instantiate_gids([gid], intersect_pre_gids=[481868], add_synapses=True)

    cell = ssim.cells[gid]
    assert cell.synapses[('', 388)].hsynapse.tau_r_GABAA == expected_val[0]
    assert cell.synapses[('', 389)].hsynapse.tau_r_GABAA == expected_val[1]


@pytest.mark.v5
class TestSSimBaseClass_full:

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(renccv2_bc_1_path)
        assert(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        self.ssim.delete()
        assert(bglibpy.tools.check_empty_topology())

    def test_generate_mtype_list(self):
        """SSim: Test generate_mtype_list"""

        mtypes_list = [
            ['L23_BTC'], ['L23_BTC', 'L23_LBC'], ['L5_TTPC1', 'L6_TPC_L1']]

        for mtypes in mtypes_list:
            mtypes_gids = self.ssim.get_gids_of_mtypes(mtypes=mtypes)

            mtypes_filename = os.path.join(
                script_dir, 'examples/mtype_lists', '%s.%s' %
                ('_'.join(mtypes), 'txt'))
            # np.savetxt(mtypes_filename, mtypes_gids)
            expected_gids = np.loadtxt(mtypes_filename)

            np.testing.assert_array_equal(
                expected_gids, mtypes_gids)

    def test_evaluate_connection_parameters(self):
        """SSim: Check if Connection block parsers yield expected output"""

        target_params = [
            ('L5_TTPC1',
             'L5_TTPC1',
             {'SpontMinis': 0.067000000000000004,
              'add_synapse': True,
              'SynapseConfigure': ['%s.mg = 1.0',
                                   '%s.Use *= 0.185409696687',
                                   '%s.NMDA_ratio = 0.4',
                                   '%s.NMDA_ratio = 0.71'],
              'Weight': 1.0}),
            ('L5_MC',
             'L5_MC',
             {'SpontMinis': 0.012,
              'add_synapse': True,
              'SynapseConfigure': ['%s.e_GABAA = -80.0 '
                                   '%s.e_GABAB = -75.8354310081',
                                   '%s.Use *= 0.437475790642'],
              'Weight': 1.0}),
            ('L5_LBC',
             'L5_LBC',
             {'SpontMinis': 0.012,
              'add_synapse': True,
              'SynapseConfigure': ['%s.e_GABAA = -80.0 '
                                   '%s.e_GABAB = -75.8354310081',
                                   '%s.Use *= 0.437475790642'],
              'Weight': 1.0}),
            ('L1_HAC',
             'L23_PC',
             {'SpontMinis': 0.012,
              'add_synapse': True,
              'SynapseConfigure': [
                  '%s.e_GABAA = -80.0 %s.e_GABAB = -75.8354310081',
                  '%s.Use *= 0.437475790642',
                  '%s.GABAB_ratio = 0.75'],
              'Weight': 1.0})
        ]

        for pre_target, post_target, params in target_params:
            pre_gid, post_gid, syn_ids = list(itertools.islice(
                self.ssim.bc_circuit.connectome.iter_connections(
                    pre_target, post_target, return_synapse_ids=True), 1))[0]
            syn_id = syn_ids[0][1]

            syn_desc = self.ssim.get_syn_descriptions(post_gid)[syn_id]

            assert pre_gid == syn_desc[0][BLPSynapse.PRE_GID]
            syn_type = syn_desc[0][BLPSynapse.TYPE]

            evaluated_params = self.ssim._evaluate_connection_parameters(
                pre_gid,
                post_gid,
                syn_type)
            assert params == evaluated_params

    def test_add_replay_synapse_SynapseConfigure(self):
        """SSim: Check if SynapseConfigure works correctly"""
        gid = int(self.ssim.get_gids_of_targets(['L5_MC'])[0])
        self.ssim.instantiate_gids([gid], synapse_detail=0)
        pre_datas = [x[0] for x in self.ssim.get_syn_descriptions(gid)]
        first_inh_syn = next(x for x in pre_datas if x[BLPSynapse.TYPE] < 100)
        sid = int(first_inh_syn.name[1])
        syn_params = pre_datas[sid]
        connection_parameters = {
            'SynapseConfigure': [
                '%s.e_GABAA = -80.5 %s.e_GABAB = -101.0',
                '%s.tau_d_GABAA = 10.0 %s.tau_r_GABAA = 1.0',
                '%s.e_GABAA = -80.6'],
            'Weight': 2.0}
        self.ssim.cells[gid].add_replay_synapse(
            ('', sid),
            syn_params,
            connection_parameters)

        assert(
            self.ssim.cells[gid].synapses[('', sid)].hsynapse.e_GABAA == -80.6
        )
        assert(
            self.ssim.cells[gid].synapses[('', sid)].hsynapse.e_GABAB == -101.0
        )
        assert(
            self.ssim.cells[gid].synapses[('', sid)].hsynapse.tau_d_GABAA == 10.0
        )
        assert(
            self.ssim.cells[gid].synapses[('', sid)].hsynapse.tau_r_GABAA == 1.0
        )


@pytest.mark.v6
def test_get_voltage_trace():
    """Test the filtering behaviour of get_voltage_trace with forwardskip."""
    blueconfig = f"{script_dir}/examples/gpfs_minimal_forwardskip/BlueConfig"
    ssim = bglibpy.ssim.SSim(blueconfig)
    gid = 693207
    ssim.instantiate_gids([693207])
    ssim.run()

    real_voltage = ssim.cells[gid].get_soma_voltage()[
        np.where(ssim.get_time() >= 0.0)
    ]
    voltage_trace = ssim.get_voltage_trace(gid)

    assert np.array_equal(real_voltage, voltage_trace)
    assert sum(ssim.get_time() < 0) == 10
    assert sum(ssim.get_time_trace() < 0) == 0
    assert len(voltage_trace) == len(ssim.get_time_trace())


@pytest.mark.v6
def test_relative_shotnoise():
    """Test injection of relative shot noise."""
    ssim = bglibpy.SSim(test_relative_shotnoise_path, record_dt=0.1)
    gids = [2886525, 3099746, 3425774, 3868780]
    ssim.instantiate_gids(gids, add_synapses=False, add_stimuli=True, add_replay=False)
    tstop = 500
    ssim.run(tstop)

    rms_tests = []
    for cell in gids:
        voltage_bglibpy = ssim.get_voltage_trace(cell)
        voltage_bglib = ssim.get_mainsim_voltage_trace(cell)[:len(voltage_bglibpy)]
        voltage_diff = voltage_bglibpy - voltage_bglib
        rms_error = np.sqrt(np.mean(voltage_diff ** 2))
        rms_tests.append(rms_error < 0.025)

    assert all(rms_tests)
