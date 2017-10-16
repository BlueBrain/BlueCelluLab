"""Unit tests for SSim"""

# pylint: disable=E1101,W0201,F0401,E0611,W0212

import os
import nose.tools as nt
import numpy
import bglibpy
from nose.plugins.attrib import attr

script_dir = os.path.dirname(__file__)

proj1_path = "/gpfs/bbp.cscs.ch/project/proj1/"
proj64_path = "/gpfs/bbp.cscs.ch/project/proj64/"

# Example ReNCCv2 sim used in BluePy use cases
renccv2_bc_1_path = os.path.join(
    proj1_path,
    "simulations/ReNCCv2/k_ca_scan_dense/K5p0/Ca1p25_synreport/",
    "BlueConfig")

v6_test_bc_1_path = os.path.join(
    proj64_path,
    "circuits/S1HL-200um/20171002/simulations/003",
    "BlueConfig")


@attr('gpfs', 'v5')
class TestSSimBaseClass_full_run(object):

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(renccv2_bc_1_path, record_dt=0.2)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

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

        self.ssim.run(500)

        time_bglibpy = self.ssim.get_time()
        voltage_bglibpy = self.ssim.get_voltage_traces()[gid]
        nt.assert_equal(len(time_bglibpy), 2500)
        nt.assert_equal(len(voltage_bglibpy), 2500)

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        nt.assert_true(rms_error < 2.0)


@attr('gpfs', 'v5')
class TestSSimBaseClass_full_realconn(object):

    """Class to test SSim with full circuit and multiple cells """ \
        """instantiate with real connections"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(renccv2_bc_1_path, record_dt=0.2)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

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
        self.ssim.run(200)
        time_bglibpy = self.ssim.get_time()
        voltage_bglibpy = self.ssim.get_voltage_traces()[gids[0]]
        nt.assert_equal(len(time_bglibpy), 1000)
        nt.assert_equal(len(voltage_bglibpy), 1000)

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gids[0])[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        nt.assert_true(rms_error < 2.0)


@attr('gpfs', 'proj64')
class TestSSimBaseClass_proj64_full_run(object):

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(v6_test_bc_1_path, record_dt=0.1)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_run(self):
        """SSim: Check if a full replay of a simulation run """ \
            """with forwardskip """ \
            """gives the same output trace as on BGQ for proj64"""
        gid = 1
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

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        nt.assert_true(rms_error < 0.001)


'''

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
        l23_btc = pickle.load(
            open(
                'examples/mtype_lists/l23_btc_gids.pkl' %
                script_dir))
        l23_several = pickle.load(
            open('examples/mtype_lists/l23_several_gids.pkl' % script_dir))
        l56 = pickle.load(
            open(
                'examples/mtype_lists/l56_gids.pkl' %
                script_dir))

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
        self.ssim.add_single_synapse(
            gid,
            sid,
            syn_params,
            connection_modifiers)

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

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 0.01)

'''

'''
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

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 1.0)
'''

'''
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

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 0.3)
'''
