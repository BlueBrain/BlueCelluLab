"""Unit tests for SSim"""

# pylint: disable=E1101,W0201,F0401,E0611,W0212

import os
import numpy
import itertools

import nose.tools as nt
from nose.plugins.attrib import attr

import bglibpy

script_dir = os.path.dirname(__file__)

proj1_path = "/gpfs/bbp.cscs.ch/project/proj1/"
proj35_path = "/gpfs/bbp.cscs.ch/project/proj35/"
proj42_path = "/gpfs/bbp.cscs.ch/project/proj42/"
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

v6_test_bc_3_path = os.path.join(proj64_path,
                                 "home/king/sim/Basic",
                                 "BlueConfig")


v6_test_bc_rnd123_1_path = os.path.join(proj64_path,
                                        "home/vangeit/simulations/",
                                        "random123_tests/",
                                        "random123_tests_newneurod_rnd123",
                                        "BlueConfig")

hip20180219_1_path = os.path.join(
    proj42_path,
    "circuits/O1/20180219",
    "CircuitConfig")


@attr('gpfs', 'v5', 'debugtest')
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

        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
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
        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gids[0])
        nt.assert_equal(len(time_bglibpy), 1000)
        nt.assert_equal(len(voltage_bglibpy), 1000)

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gids[0])[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        nt.assert_true(rms_error < 2.0)


'''
Reenable this once the MOD files are separated out of neurodamus
@attr('gpfs', 'hip', 'debugtest')
class TestSSimBaseClass_hip_20180219(object):

    """Class to test SSim with full circuit and multiple cells """ \
        """instantiate with real connections"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(hip20180219_1_path)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

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


@attr('gpfs', 'v6')
class TestSSimBaseClass_v6_full_run(object):

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
            """gives the same output trace as on BGQ for v6"""
        gid = 8709
        self.ssim.instantiate_gids(
            [gid],
            synapse_detail=2,
            add_replay=True,
            add_stimuli=True)

        self.ssim.run(500)

        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
        nt.assert_equal(len(time_bglibpy), 5000)
        nt.assert_equal(len(voltage_bglibpy), 5000)

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))

        nt.assert_true(rms_error < 0.5)


@attr('gpfs', 'v6')
class TestSSimBaseClass_v6_mvr_run(object):

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
            self.ssim = bglibpy.ssim.SSim(v6_test_bc_3_path, record_dt=0.1)

            self.ssim.instantiate_gids(
                [gid],
                synapse_detail=2,
                add_replay=True,
                add_stimuli=True)

            # Point check of one synapse
            # Manual examination of nrn.h5 showed it has to have Nrrp == 3
            if gid == 29561:
                one_synapse = self.ssim.cells[gid].synapses[150]
                nt.assert_true(hasattr(one_synapse, 'Nrrp'))
                nt.assert_equal(one_synapse.Nrrp, 3)

            self.ssim.run(500)

            time_bglibpy = self.ssim.get_time_trace()
            voltage_bglibpy = self.ssim.get_voltage_trace(gid)
            nt.assert_equal(len(time_bglibpy), 5000)
            nt.assert_equal(len(voltage_bglibpy), 5000)

            voltage_bglib = self.ssim.get_mainsim_voltage_trace(
                gid)[:len(voltage_bglibpy)]

            '''
            time_bglib = self.ssim.get_mainsim_time_trace()[
                :len(voltage_bglibpy)]

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.plot(time_bglibpy, voltage_bglibpy, 'o', label='bglibpy')
            plt.plot(time_bglib, voltage_bglib, label='neurodamus')
            plt.legend()
            plt.savefig('O1v6a_%d.png' % i)
            '''
            rms_error = numpy.sqrt(
                numpy.mean(
                    (voltage_bglibpy - voltage_bglib) ** 2))

            nt.assert_less(rms_error, 10.0)

            del self.ssim


@attr('gpfs', 'v6')
class TestSSimBaseClass_v6_rnd123_1(object):

    """Class to test SSim with 1000 cell random123 circuit"""

    def setup(self):
        """Setup"""
        self.ssim = None

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
                record_dt=0.1)

            self.ssim.instantiate_gids(
                [gid],
                add_synapses=True,
                add_replay=True,
                add_minis=True,
                add_stimuli=True)

            self.ssim.run(200)

            time_bglibpy = self.ssim.get_time_trace()
            voltage_bglibpy = self.ssim.get_voltage_trace(gid)
            nt.assert_equal(len(time_bglibpy), 2000)
            nt.assert_equal(len(voltage_bglibpy), 2000)

            voltage_bglib = self.ssim.get_mainsim_voltage_trace(
                gid)[:len(voltage_bglibpy)]

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

            rms_error = numpy.sqrt(
                numpy.mean(
                    (voltage_bglibpy - voltage_bglib) ** 2))

            nt.assert_less(rms_error, 10.0)

            del self.ssim


@attr('gpfs', 'v5')
class TestSSimBaseClass_full(object):

    """Class to test SSim with full circuit"""

    def setup(self):
        """Setup"""
        self.ssim = bglibpy.ssim.SSim(renccv2_bc_1_path)
        nt.assert_true(isinstance(self.ssim, bglibpy.SSim))

    def teardown(self):
        """Teardown"""
        del self.ssim

    def test_generate_mtype_list(self):
        """SSim: Test generate_mtype_list"""

        import pickle

        mtypes_list = [
            ['L23_BTC'], ['L23_BTC', 'L23_LBC'], ['L5_TTPC1', 'L6_TPC_L1']]

        for mtypes in mtypes_list:
            mtypes_gids = self.ssim.get_gids_of_mtypes(mtypes=mtypes)

            mtypes_filename = os.path.join(
                script_dir, 'examples/mtype_lists', '%s.%s' %
                ('_'.join(mtypes), 'pkl'))
            # with open(mtypes_filename, 'w') as mtypes_file:
            #    pickle.dump(mtypes_gids, mtypes_file)
            with open(mtypes_filename) as mtypes_file:
                expected_gids = pickle.load(mtypes_file)

            nt.assert_equal(expected_gids, mtypes_gids)

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

            nt.assert_equal(pre_gid, syn_desc[0])
            syn_type = syn_desc[13]

            nt.assert_equal(params, self.ssim._evaluate_connection_parameters(
                pre_gid,
                post_gid,
                syn_type))

    def test_add_single_synapse_SynapseConfigure(self):
        """SSim: Check if SynapseConfigure works correctly"""
        gid = self.ssim.get_gids_of_targets(['L5_MC'])[0]
        self.ssim.instantiate_gids([gid], synapse_detail=0)
        pre_datas = numpy.array(self.ssim.get_syn_descriptions(gid))
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


'''

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
        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
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
        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
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
        time_bglibpy = self.ssim.get_time_trace()
        voltage_bglibpy = self.ssim.get_voltage_trace(gid)
        nt.assert_equal(len(time_bglibpy), 1001)
        nt.assert_equal(len(voltage_bglibpy), 1001)

        voltage_bglib = self.ssim.get_mainsim_voltage_trace(
            gid)[:len(voltage_bglibpy)]

        rms_error = numpy.sqrt(
            numpy.mean(
                (voltage_bglibpy - voltage_bglib) ** 2))
        nt.assert_true(rms_error < 0.3)
'''
