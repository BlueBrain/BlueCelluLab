# -*- coding: utf-8 -*-

"""
Class that represents a synapse in BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

import numpy as np
from bluepy.enums import Synapse as BLPSynapse

import bglibpy
from bglibpy.tools import printv


class Synapse:

    """ Class that represents a synapse in BGLibPy """

    def __init__(
            self, cell, location, syn_id, syn_description,
            connection_parameters, condition_parameters, base_seed, popids=(0, 0),
            extracellular_calcium=None):
        """
        Constructor

        Parameters
        ----------
        cell : Cell
               Cell that contains the synapse
        location : float in [0, 1]
                   Location on the section this synapse is placed
        syn_id : (string,integer)
              Synapse identifier, string being the projection name and
               int the synapse id. Empty string refers to a local connection
        syn_description : list of floats
                          Parameters of the synapse
        connection_parameters : list of floats
                                Parameters of the connection
        condition_parameters: dict representing conditions block of BlueConfig
        base_seed : float
                    Base seed of the simulation, the seeds for this synapse
                    will be derived from this
        popids : tuple of (int, int)
                  Source and target popids used by the random number generation
        extracellular_calcium: float
                               the extracellular calcium concentration
        """
        self.persistent = []

        self.cell = cell
        self.post_gid = cell.gid
        self.syn_id = syn_id
        self.projection, self.sid = self.syn_id
        self.syn_description = syn_description
        self.connection_parameters = connection_parameters
        self.hsynapse = None
        self.extracellular_calcium = extracellular_calcium

        self.source_popid, self.target_popid = popids
        # pylint: disable = C0103

        self.pre_gid = int(syn_description[BLPSynapse.PRE_GID])
        post_sec_id = syn_description[BLPSynapse.POST_SECTION_ID]
        self.isec = int(post_sec_id)
        post_seg_id = syn_description[BLPSynapse.POST_SEGMENT_ID]
        self.ipt = post_seg_id
        if post_seg_id == -1:
            post_seg_distance = syn_description["afferent_section_pos"]
        else:
            post_seg_distance = syn_description[BLPSynapse.POST_SEGMENT_OFFSET]
        self.syn_offset = post_seg_distance

        self.syn_D = syn_description[BLPSynapse.D_SYN]
        self.syn_F = syn_description[BLPSynapse.F_SYN]
        self.syn_DTC = syn_description[BLPSynapse.DTC]
        self.syn_type = int(syn_description[BLPSynapse.TYPE])

        if cell.rng_settings is None:
            self.rng_setting = bglibpy.RNGSettings(
                mode='Compatibility',
                base_seed=base_seed)
        elif base_seed is not None:
            raise Exception('Synapse: base_seed and RNGSettings cant '
                            'be used together')
        else:
            self.rng_settings = cell.rng_settings

        if BLPSynapse.NRRP in syn_description:
            self.Nrrp = int(syn_description[BLPSynapse.NRRP])
        else:
            self.Nrrp = None

        self.u_hill_coefficient = (
            syn_description[BLPSynapse.U_HILL_COEFFICIENT]
            if BLPSynapse.U_HILL_COEFFICIENT in syn_description else None)

        self.conductance_ratio = (
            syn_description[BLPSynapse.CONDUCTANCE_RATIO]
            if BLPSynapse.CONDUCTANCE_RATIO in syn_description else None)

        self.u_scale_factor = self.calc_u_scale_factor(
            self.u_hill_coefficient, self.extracellular_calcium)
        self.syn_U = self.u_scale_factor * syn_description[BLPSynapse.U_SYN]
        self.post_segx = location

        # pylint: enable = C0103

        if self.is_inhibitory():
            # see: https://bbpteam.epfl.ch/
            # wiki/index.php/BlueBuilder_Specifications#NRN,
            # inhibitory synapse
            if "randomize_Gaba_risetime" in condition_parameters:
                randomize_gaba_risetime = condition_parameters["randomize_Gaba_risetime"]
            else:
                randomize_gaba_risetime = "True"

            self.use_gabaab_helper(post_sec_id, randomize_gaba_risetime)
        elif self.is_excitatory():
            if (
                    'ModOverride' in self.connection_parameters
                    and self.connection_parameters['ModOverride'] == "GluSynapse"
            ):
                self.use_glusynapse_helper(
                    syn_description, post_sec_id)
            else:
                self.use_ampanmda_helper(post_sec_id)
        else:
            raise Exception('Synapse not inhibitory or excitatory')

        self.synapseconfigure_cmds = []
        # hoc exec synapse configure blocks
        if 'SynapseConfigure' in self.connection_parameters:
            for cmd in self.connection_parameters['SynapseConfigure']:
                self.synapseconfigure_cmds.append(cmd)
                cmd = cmd.replace('%s', '\n%(syn)s')
                hoc_cmd = cmd % {'syn': self.hsynapse.hname()}
                hoc_cmd = '{%s}' % hoc_cmd
                bglibpy.neuron.h(hoc_cmd)

    def use_gabaab_helper(self, post_sec_id, randomize_gaba_risetime):
        """Python implementation of the GABAAB helper.

        This helper object will encapsulate the hoc actions
         needed to create our typical inhibitory synapse.

        Args:
            post_sec_id (int): post section id of the synapse.
            randomize_gaba_risetime (str): if "True" then RNG code runs
            to set tau_r_GABAA value.

        Raises:
            ValueError: raised when the RNG mode is unknown.
        """
        self.mech_name = 'ProbGABAAB_EMS'

        self.hsynapse = bglibpy.neuron.h.\
            ProbGABAAB_EMS(self.post_segx,
                           sec=self.cell.get_hsection(post_sec_id))

        if randomize_gaba_risetime == "True":
            rng = bglibpy.neuron.h.Random()
            if self.rng_settings.mode == "Compatibility":
                rng.MCellRan4(
                    self.sid * 100000 + 100,
                    self.cell.gid + 250 + self.rng_settings.base_seed)
            elif self.rng_settings.mode == "UpdatedMCell":
                rng.MCellRan4(
                    self.sid * 1000 + 100,
                    self.source_popid *
                    16777216 +
                    self.cell.gid +
                    250 +
                    self.rng_settings.base_seed +
                    self.rng_settings.synapse_seed)
            elif self.rng_settings.mode == "Random123":
                rng.Random123(
                    self.cell.gid +
                    250,
                    self.sid +
                    100,
                    self.source_popid *
                    65536 +
                    self.target_popid +
                    self.rng_settings.synapse_seed +
                    450)
            else:
                raise ValueError(
                    "Synapse: unknown RNG mode: %s" %
                    self.rng_settings.mode)

            rng.lognormal(0.2, 0.1)
            self.hsynapse.tau_r_GABAA = rng.repick()

        if self.conductance_ratio is not None:
            self.hsynapse.GABAB_ratio = self.conductance_ratio

        self.hsynapse.tau_d_GABAA = self.syn_DTC
        self.hsynapse.Use = abs(self.syn_U)
        self.hsynapse.Dep = abs(self.syn_D)
        self.hsynapse.Fac = abs(self.syn_F)

        if self.Nrrp is not None:
            self.hsynapse.Nrrp = self.Nrrp

        self._set_gabaab_ampanmda_rng()

        self.hsynapse.synapseID = self.sid

    def use_ampanmda_helper(self, post_sec_id):
        """Python implementation of the AMPANMDA helper.

        This helper object will encapsulate the hoc actions
         needed to create our typical excitatory synapse.

        Args:
            post_sec_id (int): post section id of the synapse.
        """
        self.mech_name = 'ProbAMPANMDA_EMS'

        self.hsynapse = bglibpy.neuron.h.\
            ProbAMPANMDA_EMS(
                self.post_segx, sec=self.cell.get_hsection(post_sec_id))
        self.hsynapse.tau_d_AMPA = self.syn_DTC
        if self.conductance_ratio is not None:
            self.hsynapse.NMDA_ratio = self.conductance_ratio

        self.hsynapse.Use = abs(self.syn_U)
        self.hsynapse.Dep = abs(self.syn_D)
        self.hsynapse.Fac = abs(self.syn_F)

        if self.Nrrp is not None:
            self.hsynapse.Nrrp = self.Nrrp

        self._set_gabaab_ampanmda_rng()
        self.hsynapse.synapseID = self.sid

    def use_glusynapse_helper(self, syn_description, post_sec_id):
        """Python implementation of the GluSynapse helper.

        This helper object will encapsulate the hoc actions
         needed to create our plastic excitatory synapse.

        Args:
            post_sec_id (int): post section id of the synapse.
        """
        self.mech_name = 'GluSynapse'

        self.hsynapse = bglibpy.neuron.h.\
            GluSynapse(
                self.post_segx, sec=self.cell.get_hsection(post_sec_id))

        self.hsynapse.Use_d = syn_description["Use_d_TM"] * self.u_scale_factor
        self.hsynapse.Use_p = syn_description["Use_p_TM"] * self.u_scale_factor

        self.hsynapse.theta_d_GB = syn_description["theta_d"]
        self.hsynapse.theta_p_GB = syn_description["theta_p"]
        self.hsynapse.rho0_GB = syn_description["rho0_GB"]

        self.hsynapse.volume_CR = syn_description["volume_CR"]

        self.hsynapse.gmax_d_AMPA = syn_description["gmax_d_AMPA"]
        self.hsynapse.gmax_p_AMPA = syn_description["gmax_p_AMPA"]

        if self.hsynapse.rho0_GB > bglibpy.neuron.h.rho_star_GB_GluSynapse:
            self.hsynapse.gmax0_AMPA = self.hsynapse.gmax_p_AMPA
            self.hsynapse.Use = self.hsynapse.Use_p
        else:
            self.hsynapse.gmax0_AMPA = self.hsynapse.gmax_d_AMPA
            self.hsynapse.Use = self.hsynapse.Use_d

        self.hsynapse.tau_d_AMPA = self.syn_DTC
        self.hsynapse.Dep = abs(self.syn_D)
        self.hsynapse.Fac = abs(self.syn_F)

        if self.Nrrp >= 0:
            self.hsynapse.Nrrp = self.Nrrp

        self.randseed1 = self.cell.gid
        self.randseed2 = 100000 + self.sid
        self.randseed3 = self.rng_settings.synapse_seed + 200
        self.hsynapse.setRNG(self.randseed1, self.randseed2, self.randseed3)
        self.hsynapse.synapseID = self.sid

    def _set_gabaab_ampanmda_rng(self):
        """Setup the RNG for the gabaab and ampanmd helpers.

        Raises:
            ValueError: when rng mode is not recognised.
        """
        if self.rng_settings.mode == "Random123":
            self.randseed1 = self.cell.gid + 250
            self.randseed2 = self.sid + 100
            self.randseed3 = self.source_popid * 65536 + self.target_popid + \
                self.rng_settings.synapse_seed + 300
            self.hsynapse.setRNG(
                self.randseed1,
                self.randseed2,
                self.randseed3)
        else:
            rndd = bglibpy.neuron.h.Random()
            if self.rng_settings.mode == "Compatibility":
                self.randseed1 = self.sid * 100000 + 100
                self.randseed2 = self.cell.gid + \
                    250 + self.rng_settings.base_seed
            elif self.rng_settings.mode == "UpdatedMCell":
                self.randseed1 = self.sid * 1000 + 100
                self.randseed2 = self.source_popid * 16777216 + \
                    self.cell.gid + \
                    250 + self.rng_settings.base_seed + \
                    self.rng_settings.synapse_seed
            else:
                raise ValueError(
                    "Synapse: unknown RNG mode: %s" %
                    self.rng_settings.mode)
            self.randseed3 = None  # Not used in this case
            rndd.MCellRan4(self.randseed1, self.randseed2)
            rndd.uniform(0, 1)
            self.hsynapse.setRNG(rndd)
            self.persistent.append(rndd)

    def is_inhibitory(self):
        """
        Check if synapse is inhibitory

        Returns
        -------
        is_inhibitory: Boolean
                       Only True if synapse is inhibitory
        """

        return self.syn_type < 100

    def is_excitatory(self):
        """
        Check if synapse is excitatory

        Returns
        -------
        is_excitatory: Boolean
                       Only True if synapse is excitatory
        """

        return self.syn_type >= 100

    def delete(self):
        """
        Delete the connection
        """
        if hasattr(self, 'persistent'):
            for persistent_object in self.persistent:
                del persistent_object

    @staticmethod
    def calc_u_scale_factor(u_hill_coefficient, extracellular_calcium):
        if extracellular_calcium is None or u_hill_coefficient is None:
            return 1.0

        def hill(extracellular_calcium, y, K_half):
            return y * extracellular_calcium**4 / (
                K_half**4 + extracellular_calcium**4)

        def constrained_hill(K_half):
            y_max = (K_half**4 + 16) / 16
            return lambda x: hill(x, y_max, K_half)

        def f_scale(x, y):
            return constrained_hill(x)(y)

        u_scale_factor = np.vectorize(f_scale)(u_hill_coefficient,
                                               extracellular_calcium)
        printv(
            "Scaling synapse Use with u_hill_coeffient %f, "
            "extra_cellular_calcium %f with a factor of %f" %
            (u_hill_coefficient, extracellular_calcium, u_scale_factor),
            50)

        return u_scale_factor

    @property
    def info_dict(self):
        """
        Convert the synapse info to a dict from which it can be reconstructed
        """

        synapse_dict = {}

        synapse_dict['synapse_id'] = self.syn_id
        synapse_dict['pre_cell_id'] = self.pre_gid
        synapse_dict['post_cell_id'] = self.post_gid
        synapse_dict['post_sec_id'] = self.isec

        # Remove cellname using split
        synapse_dict['post_sec_name'] = bglibpy.neuron.h.secname(
            sec=self.cell.get_hsection(self.isec)).split('.')[1]

        synapse_dict['post_segx'] = self.post_segx
        synapse_dict['mech_name'] = self.mech_name
        synapse_dict['syn_type'] = self.syn_type
        synapse_dict['randseed1'] = self.randseed1
        synapse_dict['randseed2'] = self.randseed2
        synapse_dict['randseed3'] = self.randseed3
        synapse_dict['synapseconfigure_cmds'] = self.synapseconfigure_cmds

        # Parameters of the mod mechanism
        synapse_dict['synapse_parameters'] = {}
        synapse_dict['synapse_parameters']['Use'] = self.hsynapse.Use
        synapse_dict['synapse_parameters']['Dep'] = self.hsynapse.Dep
        synapse_dict['synapse_parameters']['Fac'] = self.hsynapse.Fac
        if synapse_dict['mech_name'] == 'ProbGABAAB_EMS':
            synapse_dict['synapse_parameters']['tau_d_GABAA'] = \
                self.hsynapse.tau_d_GABAA
            synapse_dict['synapse_parameters']['tau_r_GABAA'] = \
                self.hsynapse.tau_r_GABAA
        elif synapse_dict['mech_name'] in ['ProbAMPANMDA_EMS', 'GluSynapse']:
            synapse_dict['synapse_parameters']['tau_d_AMPA'] = \
                self.hsynapse.tau_d_AMPA
        else:
            raise Exception('Encountered unknow mech_name %s in synapse' %
                            synapse_dict['mech_name'])

        synapse_dict['synapse_parameters']['conductance_ratio'] = \
            self.conductance_ratio
        synapse_dict['synapse_parameters']['u_hill_coefficient'] = \
            self.u_hill_coefficient
        synapse_dict['synapse_parameters']['u_scale_factor'] = \
            self.u_scale_factor
        synapse_dict['synapse_parameters']['extracellular_calcium'] = \
            self.extracellular_calcium

        return synapse_dict

    def __del__(self):
        """
        Destructor
        """
        self.delete()
