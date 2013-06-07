# -*- coding: utf-8 -*-

"""
Class that represents a synapse in BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

import bglibpy

class Synapse(object):
    """ Class that represents a synapse in BGLibPy """

    def __init__(self, cell, location, sid, syn_description, connection_parameters, base_seed):
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
                    Base seed of the simulation, the seeds for this synapse will be derived from this
        """
        self.persistent = []

        self.cell = cell
        self.sid = sid
        self.syn_description = syn_description
        self.connection_parameters = connection_parameters
        self.hsynapse = None

        pre_gid = int(syn_description[0])
        #delay = syn_description[1]
        post_sec_id = syn_description[2]
        self.isec = post_sec_id
        post_seg_id = syn_description[3]
        self.ipt = post_seg_id
        post_seg_distance = syn_description[4]
        self.syn_offset = post_seg_distance
        #weight = syn_description[8]
        self.syn_U = syn_description[9]
        self.syn_D = syn_description[10]
        self.syn_F = syn_description[11]
        self.syn_DTC = syn_description[12]
        self.syn_type = syn_description[13]

        if self.syn_type < 100:
            ''' see: https://bbpteam.epfl.ch/\
            wiki/index.php/BlueBuilder_Specifications#NRN,
            inhibitory synapse
            '''
            self.hsynapse = bglibpy.neuron.h.\
              ProbGABAAB_EMS(location, \
                             sec=self.cell.get_hsection(post_sec_id))

            self.hsynapse.tau_d_GABAA = self.syn_DTC
            rng = bglibpy.neuron.h.Random()
            rng.MCellRan4(sid * 100000 + 100, self.cell.gid + 250 + base_seed)
            rng.lognormal(0.2, 0.1)
            self.hsynapse.tau_r_GABAA = rng.repick()
        else:
            ''' else we have excitatory synapse '''
            self.hsynapse = bglibpy.neuron.h.\
              ProbAMPANMDA_EMS(location,sec=self.cell.get_hsection(post_sec_id))
            self.hsynapse.tau_d_AMPA = self.syn_DTC

        self.hsynapse.Use = abs( self.syn_U )
        self.hsynapse.Dep = abs( self.syn_D )
        self.hsynapse.Fac = abs( self.syn_F )

        rndd = bglibpy.neuron.h.Random()
        rndd.MCellRan4(sid * 100000 + 100, self.cell.gid + 250 + base_seed )
        rndd.uniform(0, 1)
        self.hsynapse.setRNG(rndd)
        self.persistent.append(rndd)

        self.hsynapse.synapseID = sid

        # hoc exec synapse configure blocks
        if 'SynapseConfigure' in connection_parameters:
            for cmd in connection_parameters['SynapseConfigure']:
                cmd = cmd.replace('%s', '\n%(syn)s')
                bglibpy.neuron.h(cmd % {'syn': self.hsynapse.hname()})

    def is_inhibitory():
        """
        Check if synapse is inhibitory

        Returns
        -------
        is_inhibitory: Boolean
                       Only True if synapse is inhibitory
        """

        return (self.syn_type < 100)

    def is_excitatory():
        """
        Check if synapse is excitatory

        Returns
        -------
        is_excitatory: Boolean
                       Only True if synapse is excitatory
        """

        return (self.syn_type >= 100)

    def delete(self):
        """
        Delete the connection
        """
        for persistent_object in self.persistent:
            del(persistent_object)

    def __del__(self):
        """
        Destructor
        """
        self.delete()
