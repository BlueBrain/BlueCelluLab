# -*- coding: utf-8 -*-

"""
Class that represents a connection between two cells in BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

import bglibpy


class Connection(object):

    """ Class that represents a connection between two cells in BGLibPy """

    def __init__(
            self,
            post_synapse,
            pre_spiketrain=None,
            pre_cell=None,
            stim_dt=None):
        self.persistent = []
        self.delay = post_synapse.syn_description[1]
        self.weight = post_synapse.syn_description[8]
        self.connection_parameters = post_synapse.connection_parameters
        self.pre_cell = pre_cell
        self.pre_spiketrain = pre_spiketrain
        self.post_synapse = post_synapse

        if 'Weight' in self.connection_parameters:
            self.weight_scalar = self.connection_parameters['Weight']
        else:
            self.weight_scalar = 1.0

        self.post_netcon = None
        self.post_netcon_delay = self.delay
        self.post_netcon_weight = self.weight * self.weight_scalar

        if self.pre_spiketrain is not None:
            if any(self.pre_spiketrain < 0):
                raise ValueError("BGLibPy Connection: a spiketrain contains "
                                 "a negative time, this is not supported by "
                                 "NEURON's Vecstim: %s" %
                                 str(self.pre_spiketrain))
            t_vec = bglibpy.neuron.h.Vector(self.pre_spiketrain)
            vecstim = bglibpy.neuron.h.VecStim()
            vecstim.play(t_vec, stim_dt)
            self.post_netcon = bglibpy.neuron.h.NetCon(
                vecstim, self.post_synapse.hsynapse, -30,
                self.post_netcon_delay,
                self.post_netcon_weight)
            self.persistent.append(t_vec)
            self.persistent.append(vecstim)
        elif self.pre_cell is not None:
            self.post_netcon = self.pre_cell.create_netcon_spikedetector(
                self.post_synapse.hsynapse)
            self.post_netcon.weight[0] = self.post_netcon_weight
            self.post_netcon.delay = self.post_netcon_delay

    @property
    def info_dict(self):
        """Return dict that contains information that can restore this conn."""

        connection_dict = {}

        connection_dict['pre_cell_id'] = self.post_synapse.pre_gid
        connection_dict['post_cell_id'] = self.post_synapse.post_gid
        connection_dict['post_synapse_id'] = self.post_synapse.sid

        connection_dict['post_netcon'] = {}
        connection_dict['post_netcon']['weight'] = self.post_netcon_weight
        connection_dict['post_netcon']['delay'] = self.post_netcon_delay

        return connection_dict

    def delete(self):
        """Delete the connection"""
        for persistent_object in self.persistent:
            del persistent_object

    def __del__(self):
        self.delete()
