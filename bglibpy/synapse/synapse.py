# -*- coding: utf-8 -*-

"""
Class that represents a synapse in BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

from bluepy.enums import Synapse as BLPSynapse
import pandas as pd

import bglibpy
from bglibpy.tools import lazy_printv


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
        self.extracellular_calcium = extracellular_calcium
        self.syn_description = self.update_syn_description(syn_description)
        self.connection_parameters = connection_parameters
        self.hsynapse = None

        self.source_popid, self.target_popid = popids

        self.pre_gid = int(self.syn_description[BLPSynapse.PRE_GID])

        self.post_segx = location

        if cell.rng_settings is None:
            self.rng_setting = bglibpy.RNGSettings(
                mode='Compatibility',
                base_seed=base_seed)
        elif base_seed is not None:
            raise Exception('Synapse: base_seed and RNGSettings cant '
                            'be used together')
        else:
            self.rng_settings = cell.rng_settings

        if self.is_inhibitory():
            # see: https://bbpteam.epfl.ch/
            # wiki/index.php/BlueBuilder_Specifications#NRN,
            # inhibitory synapse
            if "randomize_Gaba_risetime" in condition_parameters:
                randomize_gaba_risetime = condition_parameters["randomize_Gaba_risetime"]
            else:
                randomize_gaba_risetime = "True"

            self.use_gabaab_helper(randomize_gaba_risetime)
        elif self.is_excitatory():
            if (
                    'ModOverride' in self.connection_parameters
                    and self.connection_parameters['ModOverride'] == "GluSynapse"
            ):
                self.use_glusynapse_helper()
            else:
                self.use_ampanmda_helper()
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

    def update_syn_description(self, syn_description: pd.Series) -> pd.Series:
        """Change data types, compute more columns needed by the simulator."""
        syn_description[BLPSynapse.POST_SECTION_ID] = int(
            syn_description[BLPSynapse.POST_SECTION_ID])
        syn_description[BLPSynapse.TYPE] = int(
            syn_description[BLPSynapse.TYPE])

        if BLPSynapse.NRRP in syn_description:
            try:
                syn_description[BLPSynapse.NRRP] = int(syn_description[BLPSynapse.NRRP])
            except ValueError:
                # delete BLPSynapse.NRRP from syn_description
                syn_description.pop(BLPSynapse.NRRP)

        if BLPSynapse.U_HILL_COEFFICIENT in syn_description:
            syn_description["u_scale_factor"] = self.calc_u_scale_factor(
                syn_description[BLPSynapse.U_HILL_COEFFICIENT], self.extracellular_calcium)
        else:
            syn_description["u_scale_factor"] = 1.0
        syn_description[BLPSynapse.U_SYN] *= syn_description["u_scale_factor"]
        return syn_description

    def use_gabaab_helper(self, randomize_gaba_risetime):
        """Python implementation of the GABAAB helper.

        This helper object will encapsulate the hoc actions
         needed to create our typical inhibitory synapse.

        Args:
            randomize_gaba_risetime (str): if "True" then RNG code runs
            to set tau_r_GABAA value.

        Raises:
            ValueError: raised when the RNG mode is unknown.
        """
        self.mech_name = 'ProbGABAAB_EMS'

        self.hsynapse = bglibpy.neuron.h.ProbGABAAB_EMS(
            self.post_segx,
            sec=self.cell.get_hsection(
                self.syn_description[BLPSynapse.POST_SECTION_ID]
            ),
        )

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

        if BLPSynapse.CONDUCTANCE_RATIO in self.syn_description:
            self.hsynapse.GABAB_ratio = self.syn_description[BLPSynapse.CONDUCTANCE_RATIO]

        self.hsynapse.tau_d_GABAA = self.syn_description[BLPSynapse.DTC]
        self.hsynapse.Use = abs(self.syn_description[BLPSynapse.U_SYN])
        self.hsynapse.Dep = abs(self.syn_description[BLPSynapse.D_SYN])
        self.hsynapse.Fac = abs(self.syn_description[BLPSynapse.F_SYN])

        if BLPSynapse.NRRP in self.syn_description:
            self.hsynapse.Nrrp = self.syn_description[BLPSynapse.NRRP]

        self._set_gabaab_ampanmda_rng()

        self.hsynapse.synapseID = self.sid

    def use_ampanmda_helper(self):
        """Python implementation of the AMPANMDA helper.

        This helper object will encapsulate the hoc actions
         needed to create our typical excitatory synapse.
        """
        self.mech_name = 'ProbAMPANMDA_EMS'

        self.hsynapse = bglibpy.neuron.h.ProbAMPANMDA_EMS(
            self.post_segx,
            sec=self.cell.get_hsection(
                self.syn_description[BLPSynapse.POST_SECTION_ID]
            ),
        )
        self.hsynapse.tau_d_AMPA = self.syn_description[BLPSynapse.DTC]
        if BLPSynapse.CONDUCTANCE_RATIO in self.syn_description:
            self.hsynapse.NMDA_ratio = self.syn_description[BLPSynapse.CONDUCTANCE_RATIO]

        self.hsynapse.Use = abs(self.syn_description[BLPSynapse.U_SYN])
        self.hsynapse.Dep = abs(self.syn_description[BLPSynapse.D_SYN])
        self.hsynapse.Fac = abs(self.syn_description[BLPSynapse.F_SYN])

        if BLPSynapse.NRRP in self.syn_description:
            self.hsynapse.Nrrp = self.syn_description[BLPSynapse.NRRP]

        self._set_gabaab_ampanmda_rng()
        self.hsynapse.synapseID = self.sid

    def use_glusynapse_helper(self):
        """Python implementation of the GluSynapse helper.

        This helper object will encapsulate the hoc actions
         needed to create our plastic excitatory synapse.
        """
        self.mech_name = 'GluSynapse'

        self.hsynapse = bglibpy.neuron.h.GluSynapse(
            self.post_segx,
            sec=self.cell.get_hsection(
                self.syn_description[BLPSynapse.POST_SECTION_ID]
            ),
        )

        self.hsynapse.Use_d = self.syn_description["Use_d_TM"] * \
            self.syn_description["u_scale_factor"]
        self.hsynapse.Use_p = self.syn_description["Use_p_TM"] * \
            self.syn_description["u_scale_factor"]

        self.hsynapse.theta_d_GB = self.syn_description["theta_d"]
        self.hsynapse.theta_p_GB = self.syn_description["theta_p"]
        self.hsynapse.rho0_GB = self.syn_description["rho0_GB"]

        self.hsynapse.volume_CR = self.syn_description["volume_CR"]

        self.hsynapse.gmax_d_AMPA = self.syn_description["gmax_d_AMPA"]
        self.hsynapse.gmax_p_AMPA = self.syn_description["gmax_p_AMPA"]

        if self.hsynapse.rho0_GB > bglibpy.neuron.h.rho_star_GB_GluSynapse:
            self.hsynapse.gmax0_AMPA = self.hsynapse.gmax_p_AMPA
            self.hsynapse.Use = self.hsynapse.Use_p
        else:
            self.hsynapse.gmax0_AMPA = self.hsynapse.gmax_d_AMPA
            self.hsynapse.Use = self.hsynapse.Use_d

        self.hsynapse.gmax_NMDA = self.hsynapse.gmax0_AMPA * \
            self.syn_description[BLPSynapse.CONDUCTANCE_RATIO]
        self.hsynapse.tau_d_AMPA = self.syn_description[BLPSynapse.DTC]
        self.hsynapse.Dep = abs(self.syn_description[BLPSynapse.D_SYN])
        self.hsynapse.Fac = abs(self.syn_description[BLPSynapse.F_SYN])

        if self.syn_description[BLPSynapse.NRRP] >= 0:
            self.hsynapse.Nrrp = self.syn_description[BLPSynapse.NRRP]

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

        return self.syn_description[BLPSynapse.TYPE] < 100

    def is_excitatory(self):
        """
        Check if synapse is excitatory

        Returns
        -------
        is_excitatory: Boolean
                       Only True if synapse is excitatory
        """

        return self.syn_description[BLPSynapse.TYPE] >= 100

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

        def constrained_hill(K_half, y) -> float:
            """Calculates the constrained hill coefficient."""
            K_half_fourth = K_half**4
            y_fourth = y**4
            return (K_half_fourth + 16) / 16 * y_fourth / (K_half_fourth + y_fourth)

        u_scale_factor = constrained_hill(u_hill_coefficient, extracellular_calcium)
        lazy_printv(
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
        synapse_dict['syn_description'] = self.syn_description.to_dict()
        # if keys are enum make them str
        synapse_dict['syn_description'] = {
            str(k): v for k, v in synapse_dict['syn_description'].items()}

        synapse_dict['post_segx'] = self.post_segx
        synapse_dict['mech_name'] = self.mech_name
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

        synapse_dict['synapse_parameters']['extracellular_calcium'] = \
            self.extracellular_calcium

        return synapse_dict

    def __del__(self):
        """
        Destructor
        """
        self.delete()
