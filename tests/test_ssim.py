"""Unit tests for SSim"""

# pylint: disable=E1101,W0201,F0401,E0611,W0212

import os

import numpy as np
from bluepy_configfile.configfile import BlueConfig

import bglibpy
from tests.helpers.circuit import blueconfig_append_path

script_dir = os.path.dirname(__file__)


def test_merge_pre_spike_trains():
    """SSim: Testing merge_pre_spike_trains"""

    train1 = {1: [5, 100], 2: [5, 8, 120]}
    train2 = {2: [7], 3: [8]}
    train3 = {1: [5, 100]}

    trains_merged = {1: [5, 5, 100, 100], 2: [5, 7, 8, 120], 3: [8]}

    np.testing.assert_equal(
        {},
        bglibpy.ssim.SSim.merge_pre_spike_trains(None))
    np.testing.assert_equal(
        train1,
        bglibpy.ssim.SSim.merge_pre_spike_trains(train1))
    np.testing.assert_equal(
        train1,
        bglibpy.ssim.SSim.merge_pre_spike_trains(
            None,
            train1))
    np.testing.assert_equal(
        trains_merged,
        bglibpy.ssim.SSim.merge_pre_spike_trains(
            train1,
            None,
            train2,
            train3))


class TestSonataNodeInput:
    """Tests the Sonata nodes.h5 input specified as CellLibraryFile."""

    def setup(self):
        """Modify the BlueConfig to have absolute path to sonata.

        Bluepy requires absolute path for the CellLibraryFile.
        """

        prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_sonata_node" % script_dir)

        with open("BlueConfigTemplate") as f_template_config:
            config = BlueConfig(f_template_config)

        nodes_h5_abs_path = os.path.abspath("./nodes.h5")
        circuit_path_abs_path = os.path.abspath("../circuit_twocell_example1")

        config.Run.CellLibraryFile = nodes_h5_abs_path
        config.Run.CircuitPath = circuit_path_abs_path
        config.Run.MorphologyPath = os.path.join(
            circuit_path_abs_path, "morphologies"
        )
        config.Run.METypePath = os.path.join(circuit_path_abs_path, "ccells")
        config.Run.nrnPath = os.path.join(
            circuit_path_abs_path, "ncsFunctionalAllRecipePathways"
        )

        with open("./BlueConfig", "w") as f:
            f.writelines(str(config))

        os.chdir(prev_cwd)

    def test_sim_with_sonata_node(self):
        """Test instantiation of SSim using sonata nodes file."""
        blueconfig = "%s/examples/sim_sonata_node/BlueConfig" % script_dir

        ssim = bglibpy.SSim(blueconfig)

        assert ssim.circuit_access.node_properties_available


class TestSSimBaseClass_twocell_forwardskip:
    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_empty")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )
        self.ssim_bglibpy = bglibpy.SSim(modified_conf)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run(forward_skip=True, forward_skip_value=5000)

    def test_compare_traces(self):
        """SSim: Test get_time_trace and get_voltage_trace return pos values"""

        time = self.ssim_bglibpy.get_time()
        time_trace = self.ssim_bglibpy.get_time_trace()
        voltage_trace = self.ssim_bglibpy.get_voltage_trace(self.gid)

        time_len = len(time)
        time_trace_len = len(time_trace)
        assert len(time[np.where(time >= 0.0)]) == time_trace_len
        assert len(time[np.where(time < 0.0)]) == 10

        assert time_len == time_trace_len + 10

        assert len(voltage_trace) == time_trace_len

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_empty:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_empty")

        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )

        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """\
            """BGLibPy for two cell circuit"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)
        time_bglib = self.ssim_bglibpy.get_mainsim_time_trace()
        assert len(voltage_bglib) == len(time_bglib) == 1000

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)[
            0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        assert rms_error < 0.2

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_replay:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_replay")

        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )

        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutreplay = bglibpy.SSim(
            modified_conf,
            record_dt=0.1)
        self.ssim_bglibpy_withoutreplay.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=False)
        self.ssim_bglibpy_withoutreplay.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        assert len(voltage_bglib) == 1000

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)[
            0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        assert rms_error < .2

    def test_disable_replay(self):
        """SSim: Check if disabling the stimuli creates a different result"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        voltage_bglibpy_withoutreplay = \
            self.ssim_bglibpy_withoutreplay.get_voltage_trace(self.gid)[
                0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy_withoutreplay - voltage_bglib) ** 2))
        assert rms_error > .2

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy_withoutreplay.delete()
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_all_realconn:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_all")
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )
        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True,
            interconnect_cells=False)
        self.ssim_bglibpy.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit, spike replay, minis, """ \
            """noisy stimulus and real connections between cells"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        assert len(voltage_bglib) == 1000

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)[
            0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        assert rms_error < 1.0

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_all:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_all")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )

        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1, print_cellstate=True)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay, """ \
            """minis and noisy stimulus"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        assert len(voltage_bglib) == 1000

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)[
            0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        assert rms_error < 1.0

    def test_pre_gids(self):
        """SSim: Test pre_gids() of the cells for a two cell circuit"""

        pre_gids = self.ssim_bglibpy.cells[self.gid].pre_gids()

        assert len(pre_gids) == 1
        assert pre_gids[0] == 2

    def test_pre_gid_synapse_ids(self):
        """SSim: Test pre_gid_synapse_ids() of the cells for a two """ \
            """cell circuit"""

        assert self.ssim_bglibpy.cells[self.gid].pre_gids() == [2]

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


def rms(trace1, trace2):
    """Calculate rms error"""

    rms = np.sqrt(np.mean((trace1 - trace2) ** 2))
    return rms


class TestSSimBaseClass_twocell_all_intersect:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_all")

        # make the paths absolute
        self.modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )

    def test_compare_traces(self):
        """SSim: Check trace generated using intersect_pre_gid"""

        traces = {}

        for option, intersect in [('intersect', [2]),
                                  ('no_intersect', None),
                                  ('wrong_intersect', [3])]:
            ssim_bglibpy = bglibpy.SSim(self.modified_conf, record_dt=0.1)
            ssim_bglibpy.instantiate_gids(
                [self.gid],
                add_synapses=True,
                add_minis=False,
                add_replay=True,
                intersect_pre_gids=intersect)
            ssim_bglibpy.run()

            traces[option] = ssim_bglibpy.get_voltage_trace(self.gid)
            ssim_bglibpy.delete()

        assert (rms(traces['intersect'], traces['no_intersect']) == 0.0)
        assert (rms(traces['intersect'], traces['wrong_intersect']) > 0.0)

    def teardown(self):
        """Teardown"""
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_all_presynspiketrains:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_all")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )
        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_stimuli=False,
            add_synapses=True,
            add_replay=False,
            add_minis=False,
            pre_spike_trains={2: [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]})
        self.ssim_bglibpy.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and and pre_spike_train"""

        time_bglibpy = self.ssim_bglibpy.get_time_trace()
        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)

        epsp_level = np.max(
            voltage_bglibpy[
                (time_bglibpy < 20) & (
                    time_bglibpy > 10)])

        assert (epsp_level > -66)

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_all_mvr:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_all")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )
        self.ssim_bglib_all = bglibpy.SSim(modified_conf)
        self.voltage_bglib_all = self.ssim_bglib_all.get_mainsim_voltage_trace(
            gid=self.gid)

        conf_pre_path_mvr = os.path.join(
            script_dir, "examples", "sim_twocell_all_mvr")
        modified_conf_mvr = blueconfig_append_path(
            os.path.join(conf_pre_path_mvr, "BlueConfig"), conf_pre_path_mvr
        )
        self.ssim_bglibpy_mvr = bglibpy.SSim(modified_conf_mvr, record_dt=0.1)
        self.ssim_bglibpy_mvr.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy_mvr.run()
        self.voltage_bglib_mvr = \
            self.ssim_bglibpy_mvr.get_mainsim_voltage_trace(gid=self.gid)

    def test_mvr_trace_diff(self):
        """SSim: make sure MVR generates diff in neurodamus"""

        rms_error = np.sqrt(
            np.mean(
                (self.voltage_bglib_all - self.voltage_bglib_mvr) ** 2))

        assert rms_error > 10

        '''
        import matplotlib
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt
        plt.plot(self.voltage_bglib_all, label='No MVR')
        plt.plot(self.voltage_bglib_mvr, label='MVR')

        plt.legend()
        plt.savefig('mvr.png')
        '''

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay, """ \
            """minis and noisy stimulus and mvr"""

        assert len(self.voltage_bglib_mvr) == 1000

        voltage_bglibpy_mvr = self.ssim_bglibpy_mvr.get_voltage_trace(
            self.gid
        )[0:len(self.voltage_bglib_mvr)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy_mvr - self.voltage_bglib_mvr) ** 2))

        assert rms_error < 1.0

    def test_synapseconfigure(self):
        """SSim: Test if synapseconfigure with mvr works correctly"""

        first_synapse = self.ssim_bglibpy_mvr.cells[self.gid].synapses[('', 0)]
        assert (
            '%s.Nrrp = 3.0' == first_synapse.synapseconfigure_cmds[-1])

    def teardown(self):
        """Teardown"""
        self.ssim_bglib_all.delete()
        self.ssim_bglibpy_mvr.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_minis_replay:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_minis_replay")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )

        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutminis = bglibpy.SSim(
            modified_conf,
            record_dt=0.1)
        self.ssim_bglibpy_withoutminis.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=False,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy_withoutminis.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay and minis"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        assert len(voltage_bglib) == 1000

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)[
            0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        assert rms_error < 1.0

    def test_disable_minis(self):
        """SSim: Check if disabling the minis creates a different result"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        voltage_bglibpy_withoutminis = \
            self.ssim_bglibpy_withoutminis.get_voltage_trace(self.gid)[
                0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy_withoutminis - voltage_bglib) ** 2))
        assert rms_error > 1.0

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy_withoutminis.delete()
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_noisestim:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_noisestim")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )

        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutstim = bglibpy.SSim(
            modified_conf,
            record_dt=0.1)
        self.ssim_bglibpy_withoutstim.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=False,
            add_replay=True)
        self.ssim_bglibpy_withoutstim.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and noise stimulus"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        assert len(voltage_bglib) == 1000

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)[
            0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        assert rms_error < 1.0

    def test_disable_stimuli(self):
        """SSim: Check if disabling the stimuli creates a different result"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        voltage_bglibpy_withoutstim = \
            self.ssim_bglibpy_withoutstim.get_voltage_trace(self.gid)[
                0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy_withoutstim - voltage_bglib) ** 2))
        assert rms_error > 1.0

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy_withoutstim.delete()
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_twocell_pulsestim:

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_pulsestim")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )
        self.ssim_bglibpy = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutstim = bglibpy.SSim(
            modified_conf,
            record_dt=0.1)
        self.ssim_bglibpy_withoutstim.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=False,
            add_replay=True)
        self.ssim_bglibpy_withoutstim.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and pulse stimulus"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        assert len(voltage_bglib) == 1000

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_trace(self.gid)[
            0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        assert rms_error < 2.5

    def test_disable_stimuli(self):
        """SSim: Check if disabling the pulse stimuli creates a different """ \
            """result"""

        voltage_bglib = self.ssim_bglibpy.get_mainsim_voltage_trace(self.gid)

        voltage_bglibpy_withoutstim = \
            self.ssim_bglibpy_withoutstim.get_voltage_trace(self.gid)[
                0:len(voltage_bglib)]

        rms_error = np.sqrt(
            np.mean(
                (voltage_bglibpy_withoutstim - voltage_bglib) ** 2))
        assert rms_error > 20.0

    def teardown(self):
        """Teardown"""
        self.ssim_bglibpy_withoutstim.delete()
        self.ssim_bglibpy.delete()
        assert bglibpy.tools.check_empty_topology()


class TestSSimBaseClass_syns:

    """Class to test the syns / hsynapses property of Cell"""

    def setup(self):
        """Setup"""
        self.gid = 1
        conf_pre_path = os.path.join(
            script_dir, "examples", "sim_twocell_all")

        # make the paths absolute
        modified_conf = blueconfig_append_path(
            os.path.join(conf_pre_path, "BlueConfig"), conf_pre_path
        )
        self.ssim = bglibpy.SSim(modified_conf, record_dt=0.1)
        self.ssim.instantiate_gids(
            [self.gid],
            add_synapses=True,
            add_minis=True,
            add_stimuli=True,
            add_replay=True,
            interconnect_cells=False)

    def teardown(self):
        """Teardown"""
        self.ssim.delete()
        assert bglibpy.tools.check_empty_topology()

    def test_run(self):
        """SSim: Check if Cells.hsynapses return the right dictionary."""
        assert (
            isinstance(
                self.ssim.cells[self.gid].hsynapses[('', 3)].Use,
                float))
