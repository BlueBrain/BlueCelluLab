# -*- coding: utf-8 -*-

"""
Class that represents a synapse in BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

from bglibpy import printv


class RNGSettings(object):

    """ Class that represents RNG settings in BGLibPy"""

    def __init__(
            self,
            rng_mode=None,
            blueconfig=None,
            base_seed=None,
            base_noise_seed=None):
        """
        Constructor

        Parameters
        ----------
        rng_mode : str
                   String with rng mode, if not specified mode is taken from
                   BlueConfig
        blueconfig: bluepy blueconfig object
                    object received from blueconfig representing the BlueConfig
                    of the sim
        """

        if rng_mode is None:
            if 'RNGMode' in blueconfig.Run:
                self.rng_mode = blueconfig.Run['RNGMode']
            else:
                self.rng_mode = "UpdatedMCell"

        else:
            self.rng_mode = rng_mode

        accepted_rng_modes = ['UpdatedMCell', 'Compatibility', 'Random123']
        if self.rng_mode not in accepted_rng_modes:
            raise ValueError(
                "SSim: RNG mode %s not in accepted list: %s" %
                (self.rng_mode, accepted_rng_modes))

        printv("Setting rng mode to: %s" % self.rng_mode, 50)

        if base_seed is None:
            if 'BaseSeed' in blueconfig.Run:
                self.base_seed = int(blueconfig.Run['BaseSeed'])
            else:
                self.base_seed = 0  # in case the seed is not set, it's 0
        else:
            self.base_seed = base_seed

        if base_noise_seed is None:
            self.base_noise_seed = 0
        else:
            self.base_noise_seed = base_noise_seed
