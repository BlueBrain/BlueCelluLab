"""Unit tests for SSim"""

# pylint: disable=E1101,W0201,F0401,E0611,W0212

import os
import nose.tools as nt
# from nose.plugins.attrib import attr
import numpy
import bglibpy

script_dir = os.path.dirname(__file__)


def test_parse_outdat():
    """SSim: Testing parsing of out.dat"""
    try:
        outdat = bglibpy.ssim._parse_outdat(
            "%s/examples/sim_twocell_empty/output_doesntexist" % script_dir)
    except IOError:
        nt.assert_true(True)
    else:
        nt.assert_true(False)

    outdat = bglibpy.ssim._parse_outdat(
        "%s/examples/sim_twocell_minis_replay/output" % script_dir)
    nt.assert_true(45 in outdat[2])


class TestSSimBaseClass_twocell_empty(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_empty" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglib = bglibpy.SSim("BlueConfig")
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """\
            """BGLibPy for two cell circuit"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)
        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 0.2)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_replay(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_replay" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutreplay = bglibpy.SSim(
            "BlueConfig",
            record_dt=0.1)
        self.ssim_bglibpy_withoutreplay.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=False)
        self.ssim_bglibpy_withoutreplay.run()

        self.ssim_bglib = bglibpy.SSim("BlueConfig")

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < .2)

    def test_disable_replay(self):
        """SSim: Check if disabling the stimuli creates a different result"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        voltage_bglibpy_withoutreplay = \
            self.ssim_bglibpy_withoutreplay.get_voltage_traces()[1][
                0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy_withoutreplay - voltage_bglib) ** 2))
        nt.assert_true(rms_error > .2)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy_withoutreplay
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_all_realconn(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_all" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True,
            interconnect_cells=False)
        self.ssim_bglibpy.run()

        self.ssim_bglib = bglibpy.SSim("BlueConfig")

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit, spike replay, minis, """ \
            """noisy stimulus and real connections between cells"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_all(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_all" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglib = bglibpy.SSim("BlueConfig")

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay, """ \
            """minis and noisy stimulus"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(gid=1)

        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        nt.assert_true(rms_error < 1.0)

    def test_pre_gids(self):
        """SSim: Test pre_gids() of the cells for a two cell circuit"""

        pre_gids = self.ssim_bglibpy.cells[1].pre_gids()

        nt.assert_true(len(pre_gids) == 1)
        nt.assert_true(pre_gids[0] == 2)

    def test_pre_gid_synapse_ids(self):
        """SSim: Test pre_gid_synapse_ids() of the cells for a two """ \
            """cell circuit"""

        nt.assert_equal(self.ssim_bglibpy.cells[1].pre_gids(), [2])

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_all_mvr(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_all" % script_dir)
        self.ssim_bglib = bglibpy.SSim("BlueConfig")
        self.voltage_bglib_all = self.ssim_bglib.get_mainsim_voltage_trace(
            gid=1)
        os.chdir(self.prev_cwd)

        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_all_mvr" % script_dir)
        self.ssim_bglibpy_mvr = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy_mvr.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy_mvr.run()
        self.voltage_bglib_mvr = \
            self.ssim_bglibpy_mvr.get_mainsim_voltage_trace(gid=1)

        self.ssim_bglib_mvr = bglibpy.SSim("BlueConfig")

    def test_mvr_trace_diff(self):
        """SSim: make sure MVR generates diff in neurodamus"""

        # voltage_bglibpy_mvr = self.ssim_bglibpy_mvr.get_voltage_traces()[1][
        #    0:len(voltage_bglib_mvr)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (self.voltage_bglib_all - self.voltage_bglib_mvr) ** 2))

        nt.assert_true(rms_error > 10)

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

        nt.assert_equal(len(self.voltage_bglib_mvr), 1000)

        voltage_bglibpy_mvr = self.ssim_bglibpy_mvr.get_voltage_traces()[1][
            0:len(self.voltage_bglib_mvr)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy_mvr - self.voltage_bglib_mvr) ** 2))

        nt.assert_true(rms_error < 1.0)

    def test_synapseconfigure(self):
        """SSim: Test if synapseconfigure with mvr works correctly"""

        nt.assert_equal(
            '%s.Nrrp = 3.0', self.ssim_bglibpy_mvr.cells[1].synapses[0].
            synapseconfigure_cmds[-1])

    def teardown(self):
        """Teardown"""
        del self.ssim_bglib
        del self.ssim_bglibpy_mvr
        del self.ssim_bglib_mvr
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_synapseid(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_synapseid" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglib = bglibpy.SSim("BlueConfig")

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay, minis, """ \
            """noisy stimulus and SynapseID"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(gid=1)

        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        os.chdir("../sim_twocell_all")
        ssim_bglib_all = bglibpy.SSim("BlueConfig")

        voltage_bglib_all = ssim_bglib_all.get_mainsim_voltage_trace(gid=1)

        nt.assert_equal(len(voltage_bglib_all), 1000)

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < .5)

        rms_error_all = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib_all) ** 2))

        nt.assert_true(rms_error_all > 10.0)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_minis_replay(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_minis_replay" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutminis = bglibpy.SSim(
            "BlueConfig",
            record_dt=0.1)
        self.ssim_bglibpy_withoutminis.instantiate_gids(
            [1],
            synapse_detail=1,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy_withoutminis.run()

        self.ssim_bglib = bglibpy.SSim("BlueConfig")

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and spike replay and minis"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)

    def test_disable_minis(self):
        """SSim: Check if disabling the minis creates a different result"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        voltage_bglibpy_withoutminis = \
            self.ssim_bglibpy_withoutminis.get_voltage_traces()[1][
                0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy_withoutminis - voltage_bglib) ** 2))
        nt.assert_true(rms_error > 1.0)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy_withoutminis
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_noisestim(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_noisestim" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutstim = bglibpy.SSim(
            "BlueConfig",
            record_dt=0.1)
        self.ssim_bglibpy_withoutstim.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=False,
            add_replay=True)
        self.ssim_bglibpy_withoutstim.run()

        self.ssim_bglib = bglibpy.SSim("BlueConfig")

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and noise stimulus"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)

    def test_disable_stimuli(self):
        """SSim: Check if disabling the stimuli creates a different result"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        voltage_bglibpy_withoutstim = \
            self.ssim_bglibpy_withoutstim.get_voltage_traces()[1][
                0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy_withoutstim - voltage_bglib) ** 2))
        nt.assert_true(rms_error > 1.0)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy_withoutstim
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_pulsestim(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_pulsestim" % script_dir)
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglibpy.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True)
        self.ssim_bglibpy.run()

        self.ssim_bglibpy_withoutstim = bglibpy.SSim(
            "BlueConfig",
            record_dt=0.1)
        self.ssim_bglibpy_withoutstim.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=False,
            add_replay=True)
        self.ssim_bglibpy_withoutstim.run()

        self.ssim_bglib = bglibpy.SSim("BlueConfig")

    def test_compare_traces(self):
        """SSim: Compare the output traces of BGLib against those of """ \
            """BGLibPy for two cell circuit and pulse stimulus"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        nt.assert_true(rms_error < 2.5)

    def test_disable_stimuli(self):
        """SSim: Check if disabling the pulse stimuli creates a different """ \
            """result"""

        voltage_bglib = self.ssim_bglib.get_mainsim_voltage_trace(1)

        voltage_bglibpy_withoutstim = \
            self.ssim_bglibpy_withoutstim.get_voltage_traces()[1][
                0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy_withoutstim - voltage_bglib) ** 2))
        nt.assert_true(rms_error > 20.0)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy_withoutstim
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_syns(object):

    """Class to test the syns / hsynapses property of Cell"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("%s/examples/sim_twocell_all" % script_dir)
        self.ssim = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim.instantiate_gids(
            [1],
            synapse_detail=2,
            add_stimuli=True,
            add_replay=True,
            interconnect_cells=False)

    def teardown(self):
        """Teardown"""
        del self.ssim
        os.chdir(self.prev_cwd)

    def test_run(self):
        """SSim: Check if Cells.hsynapses and Cells.syns return """ \
            """the right dictionary"""
        gid = 1
        nt.assert_true(
            isinstance(
                self.ssim.cells[gid].hsynapses[3].Use,
                float))
        import warnings
        with warnings.catch_warnings(True) as w:
            nt.assert_true(isinstance(self.ssim.cells[gid].syns[4].Use, float))
            nt.assert_true(len(w) == 1)
            nt.assert_true(issubclass(w[-1].category, DeprecationWarning))
            nt.assert_true("deprecated" in str(w[-1].message))
