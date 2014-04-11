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

        if self.pre_spiketrain is None and self.pre_cell is None:
            raise Exception(
                "Connection: trying to create a connection without presynaptic"
                "spiketrain nor cell")

        if 'Weight' in self.connection_parameters:
            self.weight_scalar = self.connection_parameters['Weight']
        else:
            self.weight_scalar = 1.0

        self.post_netcon = None

        if self.pre_spiketrain is not None:
            t_vec = bglibpy.neuron.h.Vector(self.pre_spiketrain)
            vecstim = bglibpy.neuron.h.VecStim()
            vecstim.play(t_vec, stim_dt)
            self.post_netcon = bglibpy.neuron.h.NetCon(
                vecstim, self.post_synapse.hsynapse, -30, self.delay,
                self.weight * self.weight_scalar)
            self.persistent.append(t_vec)
            self.persistent.append(vecstim)
        elif self.pre_cell is not None:
            self.post_netcon = self.pre_cell.create_netcon_spikedetector(
                self.post_synapse.hsynapse)
            self.post_netcon.weight[0] = self.weight * self.weight_scalar
            self.post_netcon.delay = self.delay
        else:
            raise Exception(
                "Connection: trying to instantiated connection without "
                "presynaptic spiketrain nor cell")

    def delete(self):
        """Delete the connection"""
        for persistent_object in self.persistent:
            del persistent_object

    def __del__(self):
        self.delete()
