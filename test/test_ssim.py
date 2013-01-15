"""Unit tests for SSim"""

# pylint: disable=E1101,W0201

import nose.tools as nt
import numpy
import bglibpy.ssim
import os

class TestSSimBaseClass_twocell(object):
    """Class to test SSim with two cell circuit"""
    def setup(self):
        """Setup"""
        self.prev_cwd = os.getcwd()
        os.chdir("examples/sim_twocell_empty")
        self.ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=0.1)
        self.ssim_bglib = bglibpy.SSim("BlueConfig")
        self.ssim_bglibpy.instantiate_gids([1], 3)
        self.ssim_bglibpy.run()

    def test_compare_traces(self):
        """Compare the output traces of BGLib against those of BGLibPy for two cell circuit"""

        #time_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_range
        voltage_bglib = self.ssim_bglib.bc_simulation.reports.soma.time_series(1)
        nt.assert_equal(len(voltage_bglib), 1000)

        #time_bglibpy = self.ssim_bglibpy.get_time()
        voltage_bglibpy = self.ssim_bglibpy.get_voltage_traces()[1][0:len(voltage_bglib)]

        rms_error = numpy.sqrt(numpy.mean((voltage_bglibpy-voltage_bglib)**2))
        nt.assert_true(rms_error < 1.0)

    def teardown(self):
        """Teardown"""
        del self.ssim_bglibpy
        del self.ssim_bglib
        os.chdir(self.prev_cwd)

class TestSSimBaseClass_full(object):
    """Class to test SSim with full circuit"""
    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim("/bgscratch/bbp/release/19.11.12/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/BlueConfig")
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_evaluate_connection_parameters(self):
        """Check if Connection block parsers yield expected output"""

        # check a TTPC1 pair
        pre_gid, post_gid = list(self.ssim.bc_simulation.get_target("L5_TTPC1"))[:2]

        params = self.ssim._evaluate_connection_parameters(pre_gid, post_gid)

        # checking a few sanity cases

        nt.assert_equal(params, {'SpontMinis': 0.067000000000000004,
            'SynapseConfigure': ['%s.NMDA_ratio = 0.4', '%s.NMDA_ratio = 0.71'], 'Weight': 2.3500000000000001})

        pre_gid = list(self.ssim.bc_simulation.get_target("L5_MC"))[0]
        params = self.ssim._evaluate_connection_parameters(pre_gid, post_gid)
        nt.assert_equal(params, {'SpontMinis': 0.012, 'SynapseConfigure': ['%s.e_GABAA = -80.0'], 'Weight': 2.0})

        pre_gid = list(self.ssim.bc_simulation.get_target("L5_LBC"))[0]
        params = self.ssim._evaluate_connection_parameters(pre_gid, post_gid)
        nt.assert_equal(params, {'SpontMinis': 0.012, 'SynapseConfigure': ['%s.e_GABAA = -80.0'], 'Weight': 0.67000000000000004})

        pre_gid = list(self.ssim.bc_simulation.get_target("L1_HAC"))[0]
        params = self.ssim._evaluate_connection_parameters(pre_gid, post_gid)
        nt.assert_equal(params, {'SpontMinis': 0.012, 'SynapseConfigure': ['%s.e_GABAA = -80.0', '%s.GABAB_ratio = 0.75'], 'Weight': 2.0})

    def test_add_single_synapse_SynapseConfigure(self):
        """Check if SynapseConfigure works correctly"""
        gid = list(self.ssim.bc_simulation.get_target("L5_MC"))[0]
        self.ssim.instantiate_gids([gid], 3)
        pre_datas = self.ssim.bc_simulation.circuit.get_presynaptic_data(gid)
        # get second inh synapse (first fails)
        inh_synapses = numpy.nonzero(pre_datas[:, 13] < 100)
        sid = inh_synapses[0][1]
        syn_params = pre_datas[sid, :]
        #pre_gid = syn_params[0]
        connection_modifiers = {'SynapseConfigure': ['%s.e_GABAA = -80.5 %s.e_GABAB = -101.0', '%s.tau_d_GABAA = 10.0 %s.tau_r_GABAA = 1.0', '%s.e_GABAA = -80.6'], 'Weight':2.0}
        self.ssim.add_single_synapse(gid, sid, syn_params, connection_modifiers)

        nt.assert_equal(self.ssim.cells[gid].syns[sid].e_GABAA, -80.6)
        nt.assert_equal(self.ssim.cells[gid].syns[sid].e_GABAB, -101.0)
        nt.assert_equal(self.ssim.cells[gid].syns[sid].tau_d_GABAA, 10.0)
        nt.assert_equal(self.ssim.cells[gid].syns[sid].tau_r_GABAA, 1.0)
