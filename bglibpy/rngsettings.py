# -*- coding: utf-8 -*-

"""
Class that represents a synapse in BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

import bglibpy
from bglibpy import printv

default_rng_mode = "Compatibility"


class RNGSettings(object):

    """ Class that represents RNG settings in BGLibPy"""

    def __init__(
            self,
            mode=None,
            blueconfig=None,
            base_seed=None,
            base_noise_seed=None):
        """
        Constructor

        Parameters
        ----------
        mode : str
                   String with rng mode, if not specified mode is taken from
                   BlueConfig
        blueconfig: bluepy blueconfig object
                    object received from blueconfig representing the BlueConfig
                    of the sim
        base_seed: float
                   base seed for entire sim, overrides blueconfig value
        base_noise_seed: float
                         base seed for the noise stimuli
        """

        if mode is None:
            # Ugly, but mimicking neurodamus
            if blueconfig and 'Simulator' in blueconfig.Run and \
                    blueconfig.Run['Simulator'] != 'NEURON':
                self.mode = 'Random123'
            elif blueconfig and 'RNGMode' in blueconfig.Run:
                self.mode = blueconfig.Run['RNGMode']
            else:
                self.mode = default_rng_mode
        else:
            self.mode = mode

        accepted_modes = ['UpdatedMCell', 'Compatibility', 'Random123']
        if self.mode not in accepted_modes:
            raise ValueError(
                "SSim: RNG mode %s not in accepted list: %s" %
                (self.mode, accepted_modes))

        printv("Setting rng mode to: %s" % self.mode, 50)

        if base_seed is None:
            if blueconfig and 'BaseSeed' in blueconfig.Run:
                self.base_seed = int(blueconfig.Run['BaseSeed'])
            else:
                self.base_seed = 0  # in case the seed is not set, it's 0
        else:
            self.base_seed = base_seed

        if self.mode == 'Random123':
            rng = bglibpy.neuron.h.Random()
            rng.Random123_globalindex(self.base_seed)

        if blueconfig and 'SynapseSeed' in blueconfig.Run:
            self.synapse_seed = int(blueconfig.Run['SynapseSeed'])
        else:
            self.synapse_seed = 0

        if blueconfig and 'IonChannelSeed' in blueconfig.Run:
            self.ionchannel_seed = int(blueconfig.Run['IonChannelSeed'])
        else:
            self.ionchannel_seed = 0

        if blueconfig and 'StimulusSeed' in blueconfig.Run:
            self.stimulus_seed = int(blueconfig.Run['StimulusSeed'])
        else:
            self.stimulus_seed = 0

        if blueconfig and 'MinisSeed' in blueconfig.Run:
            self.minis_seed = int(blueconfig.Run['MinisSeed'])
        else:
            self.minis_seed = 0

        if base_noise_seed is None:
            self.base_noise_seed = 0
        else:
            self.base_noise_seed = base_noise_seed
