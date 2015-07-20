"""Unit tests for SSim"""

# pylint: disable=E1101,W0201,F0401,E0611,W0212

import nose.tools as nt
import numpy
import bglibpy
import os
from nose.plugins.attrib import attr

# todo make two cell circuit independent of bgscratch


def test_parse_outdat():
    """SSim: Testing parsing of out.dat"""
    try:
        outdat = bglibpy.ssim._parse_outdat(
            "examples/sim_twocell_empty/output_doesntexist")
    except IOError:
        nt.assert_true(True)
    else:
        nt.assert_true(False)

    outdat = bglibpy.ssim._parse_outdat(
        "examples/sim_twocell_minis_replay/output")
    nt.assert_true(45 in outdat[2])


class TestSSimBaseClass_twocell_empty(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("examples/sim_twocell_empty")
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

        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
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
        os.chdir("examples/sim_twocell_replay")
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
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < .2)

    def test_disable_replay(self):
        """SSim: Check if disabling the stimuli creates a different result"""
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
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
        os.chdir("examples/sim_twocell_all")
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
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
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
        os.chdir("examples/sim_twocell_all")
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
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
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

        print self.ssim_bglibpy.cells[1].pre_gids()

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)


class TestSSimBaseClass_twocell_synapseid(object):

    """Class to test SSim with two cell circuit"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("examples/sim_twocell_synapseid")
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
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        os.chdir("../sim_twocell_all")
        ssim_bglib_all = bglibpy.SSim("BlueConfig")
        voltage_bglib_all = \
            ssim_bglib_all.bc_simulation.reports.soma.time_series(1)
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
        os.chdir("examples/sim_twocell_minis_replay")
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
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)

    def test_disable_minis(self):
        """SSim: Check if disabling the minis creates a different result"""
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
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
        os.chdir("examples/sim_twocell_noisestim")
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
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
        nt.assert_equal(len(voltage_bglib), 1000)

        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][
            0:len(voltage_bglib)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)

    def test_disable_stimuli(self):
        """SSim: Check if disabling the stimuli creates a different result"""
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(
            1)
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


@attr('bgscratch')
class TestSSimBaseClass_full(object):

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(
            "/bgscratch/bbp/l5/projects/proj1/2013.01.14/simulations/"
            "SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/"
            "Control_Mg0p5/BlueConfig")
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_generate_mtype_list(self):
        """SSim: Test generate_mtype_list"""
        L23_BTC_gids = self.ssim.get_gids_of_mtypes(mtypes=['L23_BTC'])
        L2_several_gids = self.ssim.get_gids_of_mtypes(
            mtypes=[
                'L23_BTC',
                'L23_LBC'])
        L56_gids = self.ssim.get_gids_of_mtypes(
            mtypes=[
                'L5_TTPC1',
                'L6_TPC_L1'])

        import pickle
        l23_btc = pickle.load(open('examples/mtype_lists/l23_btc_gids.pkl'))
        l23_several = pickle.load(
            open('examples/mtype_lists/l23_several_gids.pkl'))
        l56 = pickle.load(open('examples/mtype_lists/l56_gids.pkl'))

        # print 'len(L23): ', len(L23_BTC_gids)
        # print 'len(l23): ', len(l23_btc)

        nt.eq_(
            len(L23_BTC_gids),
            len(l23_btc),
            "len of the list should be the same")
        nt.eq_(
            len(L2_several_gids),
            len(l23_several),
            "len of the list should be the same")
        nt.eq_(len(L56_gids), len(l56), "len of the list should be the same")

    def test_evaluate_connection_parameters(self):
        """SSim: Check if Connection block parsers yield expected output"""

        # check a TTPC1 pair
        pre_gid, post_gid = list(
            self.ssim.bc_simulation.get_target("L5_TTPC1"))[:2]
        syn_type = list(self.ssim.bc_simulation.get_target("L5_TTPC1"))[13]

        params = self.ssim._evaluate_connection_parameters(
            pre_gid,
            post_gid,
            syn_type)

        # checking a few sanity cases

        nt.assert_equal(
            params,
            {'SpontMinis': 0.067000000000000004, 'add_synapse': True,
             'SynapseConfigure':
             ['%s.NMDA_ratio = 0.4 %s.mg = 0.5', '%s.NMDA_ratio = 0.71'],
             'Weight': 2.3500000000000001})

        pre_gid = list(self.ssim.bc_simulation.get_target("L5_MC"))[0]
        syn_type = list(self.ssim.bc_simulation.get_target("L5_MC"))[13]
        params = self.ssim._evaluate_connection_parameters(
            pre_gid,
            post_gid,
            syn_type)
        nt.assert_equal(params,
                        {'SpontMinis': 0.012,
                         'add_synapse': True,
                         'SynapseConfigure': ['%s.e_GABAA = -80.0'],
                         'Weight': 2.0})

        pre_gid = list(self.ssim.bc_simulation.get_target("L5_LBC"))[0]
        syn_type = list(self.ssim.bc_simulation.get_target("L5_LBC"))[13]
        params = self.ssim._evaluate_connection_parameters(
            pre_gid,
            post_gid,
            syn_type)
        nt.assert_equal(params,
                        {'SpontMinis': 0.012,
                         'add_synapse': True,
                         'SynapseConfigure': ['%s.e_GABAA = -80.0'],
                         'Weight': 0.67000000000000004})

        pre_gid = list(self.ssim.bc_simulation.get_target("L1_HAC"))[0]
        syn_type = list(self.ssim.bc_simulation.get_target("L1_HAC"))[13]
        params = self.ssim._evaluate_connection_parameters(
            pre_gid,
            post_gid,
            syn_type)
        nt.assert_equal(
            params, {
                'SpontMinis': 0.012,
                'add_synapse': True,
                'SynapseConfigure': [
                    '%s.e_GABAA = -80.0',
                    '%s.GABAB_ratio = 0.75'],
                'Weight': 2.0})

    def test_add_single_synapse_SynapseConfigure(self):
        """SSim: Check if SynapseConfigure works correctly"""
        gid = list(self.ssim.bc_simulation.get_target("L5_MC"))[0]
        self.ssim.instantiate_gids([gid], synapse_detail=0)
        pre_datas = self.ssim.bc_simulation.circuit.get_presynaptic_data(gid)
        # get second inh synapse (first fails)
        inh_synapses = numpy.nonzero(pre_datas[:, 13] < 100)
        sid = inh_synapses[0][1]
        syn_params = pre_datas[sid, :]
        connection_modifiers = {
            'SynapseConfigure': [
                '%s.e_GABAA = -80.5 %s.e_GABAB = -101.0',
                '%s.tau_d_GABAA = 10.0 %s.tau_r_GABAA = 1.0',
                '%s.e_GABAA = -80.6'],
            'Weight': 2.0}
        self.ssim.add_single_synapse(gid, sid, syn_params, connection_modifiers)

        nt.assert_equal(
            self.ssim.cells[gid].synapses[sid].hsynapse.e_GABAA, -80.6)
        nt.assert_equal(
            self.ssim.cells[gid].synapses[sid].hsynapse.e_GABAB, -101.0)
        nt.assert_equal(
            self.ssim.cells[gid].synapses[sid].hsynapse.tau_d_GABAA,
            10.0)
        nt.assert_equal(
            self.ssim.cells[gid].synapses[sid].hsynapse.tau_r_GABAA,
            1.0)


@attr('bgscratch')
class TestSSimBaseClass_full_run(object):

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(
            "/bgscratch/bbp/l5/projects/proj1/2013.01.14/simulations/"
            "SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/"
            "Control_Mg0p5/BlueConfig",
            record_dt=0.1)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_run(self):
        """SSim: Check if a full replay of a simulation run on BG / P """ \
            """gives the same output trace as on BG / P"""
        gid = 116386
        self.ssim.instantiate_gids(
            [gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True)
        self.ssim.run(500)
        time_bglibpy = self.ssim.get_time()
        voltage_bglibpy = self.ssim.get_voltage_traces()[gid]
        nt.assert_equal(len(time_bglibpy), 5000)
        nt.assert_equal(len(voltage_bglibpy), 5000)
        voltage_bglib = self.ssim.bc_simulation.reports.soma.time_series(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)


@attr('bgscratch')
class TestSSimBaseClass_full_neuronconfigure(object):

    """Class to test SSim with full circuit that uses neuronconfigure"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(
            "/bgscratch/bbp/l5/projects/proj1/2013.01.14/simulations/"
            "SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/"
            "coupled_Ek65_Mg0p25/BlueConfig",
            record_dt=0.1)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_run(self):
        """SSim: Check if a full replay of a simulation with """ \
            """neuronconfigure blocks run gives the same output """ \
            """trace as on BG / P"""
        gid = 116386
        self.ssim.instantiate_gids(
            [gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True)
        self.ssim.run(500)
        time_bglibpy = self.ssim.get_time()
        voltage_bglibpy = self.ssim.get_voltage_traces()[gid]
        nt.assert_equal(len(time_bglibpy), 5000)
        nt.assert_equal(len(voltage_bglibpy), 5000)
        voltage_bglib = self.ssim.bc_simulation.reports.soma.time_series(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 0.01)


@attr('bgscratch')
class TestSSimBaseClass_full_connection_delay(object):

    """Class to test SSim with full circuit that uses a delay field """ \
        """in a connection block"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(
            "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/"
            "SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/"
            "knockout/L4_EXC/BlueConfig",
            record_dt=0.1)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_run(self):
        """SSim: Check if a full replay of a simulation with """ \
            """delayed connection blocks gives the same output """ \
            """trace as on BG / P"""
        gid = 116386
        self.ssim.instantiate_gids(
            [gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True)
        self.ssim.run(500)
        time_bglibpy = self.ssim.get_time()
        voltage_bglibpy = self.ssim.get_voltage_traces()[gid]
        nt.assert_equal(len(time_bglibpy), 5000)
        nt.assert_equal(len(voltage_bglibpy), 5000)
        voltage_bglib = self.ssim.bc_simulation.reports.soma.time_series(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)


@attr('bgscratch')
class TestSSimBaseClass_full_forwardskip(object):

    """Class to test SSim with full circuit that uses a ForwardSkip"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(
            "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/"
            "SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/"
            "k_ca_scan/K5p0/Ca1p3/BlueConfig",
            record_dt=0.1)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_run(self):
        """SSim: Check if a full replay of a simulation with ForwardSkip """ \
            """gives the same output trace as on BG / P"""
        gid = 108849
        self.ssim.instantiate_gids(
            [gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True)
        self.ssim.run(100)
        time_bglibpy = self.ssim.get_time()
        voltage_bglibpy = self.ssim.get_voltage_traces()[gid]
        nt.assert_equal(len(time_bglibpy), 1001)
        nt.assert_equal(len(voltage_bglibpy), 1001)
        voltage_bglib = self.ssim.bc_simulation.reports.soma.time_series(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 0.3)


@attr('bgscratch')
class TestSSimBaseClass_full_realconn(object):

    """Class to test SSim with full circuit and multiple cells """ \
        """instantiate with real connections"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(
            "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/"
            "SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/"
            "knockout/L4_EXC/BlueConfig",
            record_dt=0.1)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_run(self):
        """SSim: Check if a multi - cell full replay of a simulation """ \
            """gives the same output trace as on BG / P"""
        gid = 116390
        # gids = 116390, 116392
        gids = range(gid, gid + 5)
        self.ssim.instantiate_gids(
            gids,
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True,
            interconnect_cells=True)
        self.ssim.run(100)
        time_bglibpy = self.ssim.get_time()
        voltage_bglibpy = self.ssim.get_voltage_traces()[gids[0]]
        nt.assert_equal(len(time_bglibpy), 1001)
        nt.assert_equal(len(voltage_bglibpy), 1001)
        voltage_bglib = self.ssim.bc_simulation.reports.soma.time_series(
            gids[0])[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)


class TestSSimBaseClass_syns(object):

    """Class to test the syns / hsynapses property of Cell"""

    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("examples/sim_twocell_all")
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
        nt.assert_true(isinstance(self.ssim.cells[gid].hsynapses[3].Use, float))
        import warnings
        with warnings.catch_warnings(True) as w:
            nt.assert_true(isinstance(self.ssim.cells[gid].syns[4].Use, float))
            nt.assert_true(len(w) == 1)
            nt.assert_true(issubclass(w[-1].category, DeprecationWarning))
            nt.assert_true("deprecated" in str(w[-1].message))
