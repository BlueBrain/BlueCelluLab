# -*- coding: utf-8 -*-

"""
Class that represents a synapse in BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

import bglibpy
# from bglibpy.tools import printv


class Synapse(object):

    """ Class that represents a synapse in BGLibPy """

    def __init__(self, cell, location, sid, syn_description,
                 connection_parameters, base_seed):
        """
        Constructor

        Parameters
        ----------
        cell : Cell
               Cell that contains the synapse
        location : float in [0, 1]
                   Location on the section this synapse is placed
        sid : integer
              Synapse identifier
        syn_description : list of floats
                          Parameters of the synapse
        connection_parameters : list of floats
                                Parameters of the connection
        base_seed : float
                    Base seed of the simulation, the seeds for this synapse
                    will be derived from this
        rng_settings: RNGSettings
                      RNGsettings object specifying details about
                      random generator
        """
        self.persistent = []

        self.cell = cell
        self.post_gid = cell.gid
        self.sid = sid
        self.syn_description = syn_description
        self.connection_parameters = connection_parameters
        self.hsynapse = None

        # pylint: disable = C0103

        self.pre_gid = int(syn_description[0])
        # self.delay = syn_description[1]
        post_sec_id = syn_description[2]
        self.isec = int(post_sec_id)
        post_seg_id = syn_description[3]
        self.ipt = post_seg_id
        post_seg_distance = syn_description[4]
        self.syn_offset = post_seg_distance
        # weight = syn_description[8]
        self.syn_U = syn_description[9]
        self.syn_D = syn_description[10]
        self.syn_F = syn_description[11]
        self.syn_DTC = syn_description[12]
        self.syn_type = int(syn_description[13])

        if cell.rng_settings is None:
            self.rng_setting = bglibpy.RNGSettings(
                mode='Compatibility',
                base_seed=base_seed)
        else:
            if base_seed is not None:
                raise Exception('Synapse: base_seed and RNGSettings cant '
                                'be used together')
            self.rng_settings = cell.rng_settings

        if len(syn_description) == 18:
            if syn_description[17] <= 0:
                raise ValueError(
                    'Invalid value for Nrrp found:'
                    ' %s at synapse %d in gid %d' %
                    (syn_description[17], self.sid, self.gid))
            self.Nrrp = int(syn_description[17])

        self.post_segx = location

        # pylint: enable = C0103

        if self.syn_type < 100:
            # see: https://bbpteam.epfl.ch/
            # wiki/index.php/BlueBuilder_Specifications#NRN,
            # inhibitory synapse

            self.mech_name = 'ProbGABAAB_EMS'

            self.hsynapse = bglibpy.neuron.h.\
                ProbGABAAB_EMS(self.post_segx,
                               sec=self.cell.get_hsection(post_sec_id))

            self.hsynapse.tau_d_GABAA = self.syn_DTC
            rng = bglibpy.neuron.h.Random()
            if self.rng_settings.mode == "Compatibility":
                rng.MCellRan4(
                    sid * 100000 + 100,
                    self.cell.gid + 250 + self.rng_settings.base_seed)
            elif self.rng_settings.mode == "UpdatedMCell":
                rng.MCellRan4(
                    sid * 100000 + 100,
                    self.cell.gid + 250 + self.rng_settings.base_seed +
                    self.rng_settings.synapse_seed)
            elif self.rng_settings.mode == "Random123":
                rng.Random123(
                    self.cell.gid + 250,
                    sid + 100,
                    self.rng_settings.synapse_seed + 450)
            else:
                raise ValueError(
                    "Synapse: unknown RNG mode: %s" %
                    self.rng_settings.mode)

            rng.lognormal(0.2, 0.1)
            self.hsynapse.tau_r_GABAA = rng.repick()
        else:
            # else we have excitatory synapse

            self.mech_name = 'ProbAMPANMDA_EMS'

            self.hsynapse = bglibpy.neuron.h.\
                ProbAMPANMDA_EMS(
                    self.post_segx, sec=self.cell.get_hsection(post_sec_id))
            self.hsynapse.tau_d_AMPA = self.syn_DTC

        self.hsynapse.Use = abs(self.syn_U)
        self.hsynapse.Dep = abs(self.syn_D)
        self.hsynapse.Fac = abs(self.syn_F)

        if hasattr(self, 'Nrrp'):
            self.hsynapse.Nrrp = self.Nrrp

        if self.rng_settings.mode == "Random123":
            self.randseed1 = self.cell.gid + 250
            self.randseed2 = self.sid + 100
            self.randseed3 = self.rng_settings.synapse_seed + 300
            self.hsynapse.setRNG(
                self.randseed1,
                self.randseed2,
                self.randseed3)
        else:
            rndd = bglibpy.neuron.h.Random()
            self.randseed1 = sid * 100000 + 100
            if self.rng_settings.mode == "Compatibility":
                self.randseed2 = self.cell.gid + \
                    250 + self.rng_settings.base_seed
            elif self.rng_settings.mode == "UpdatedMCell":
                self.randseed2 = self.cell.gid + 250 + \
                    self.rng_settings.base_seed + \
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

        self.hsynapse.synapseID = sid

        self.synapseconfigure_cmds = []
        # hoc exec synapse configure blocks
        if 'SynapseConfigure' in self.connection_parameters:
            for cmd in self.connection_parameters['SynapseConfigure']:
                self.synapseconfigure_cmds.append(cmd)
                cmd = cmd.replace('%s', '\n%(syn)s')
                hoc_cmd = cmd % {'syn': self.hsynapse.hname()}
                bglibpy.neuron.h(hoc_cmd)
                # printv("Executed hoc command: %s" % hoc_cmd, 50)
                # printv("Use: %f" % self.hsynapse.Use, 50)

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

    @property
    def info_dict(self):
        """
        Convert the synapse info to a dict from which it can be reconstructed
        """

        synapse_dict = {}

        synapse_dict['synapse_id'] = self.sid
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
        elif synapse_dict['mech_name'] == 'ProbAMPANMDA_EMS':
            synapse_dict['synapse_parameters']['tau_d_AMPA'] = \
                self.hsynapse.tau_d_AMPA
        else:
            raise Exception('Encountered unknow mech_name %s in synapse' %
                            synapse_dict['mech_name'])

        return synapse_dict

    def __del__(self):
        """
        Destructor
        """
        self.delete()
