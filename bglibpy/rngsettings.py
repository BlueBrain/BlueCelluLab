# -*- coding: utf-8 -*-

"""
RNG settings of BGLibPy

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

from typing import Optional

import bglibpy
from bglibpy import Singleton, lazy_printv
from bglibpy.circuit.circuit_access import CircuitAccess
from bglibpy.exceptions import UndefinedRNGException


class RNGSettings(metaclass=Singleton):

    """ Class that represents RNG settings in BGLibPy"""

    def __init__(
            self,
            mode=None,
            circuit_access: Optional[CircuitAccess] = None,
            base_seed=None,
            base_noise_seed=None):
        """
        Constructor

        Parameters
        ----------
        mode : str
                   String with rng mode, if not specified mode is taken from
                   BlueConfig
        circuit: circuit access object, if present seeds are read from simulation
        base_seed: float
                   base seed for entire sim, overrides blueconfig value
        base_noise_seed: float
                         base seed for the noise stimuli
        """

        self._mode = ""
        if mode is None:
            self.mode = circuit_access.config.rng_mode if circuit_access else "Compatibility"
        else:
            self.mode = mode

        lazy_printv("Setting rng mode to: {mode}", 50, mode=self._mode)

        if base_seed is None:
            self.base_seed = circuit_access.config.base_seed if circuit_access else 0
        else:
            self.base_seed = base_seed
        bglibpy.neuron.h.globalSeed = self.base_seed

        if self._mode == 'Random123':
            rng = bglibpy.neuron.h.Random()
            rng.Random123_globalindex(self.base_seed)

        self.synapse_seed = circuit_access.config.synapse_seed if circuit_access else 0
        bglibpy.neuron.h.synapseSeed = self.synapse_seed

        self.ionchannel_seed = circuit_access.config.ionchannel_seed if circuit_access else 0
        bglibpy.neuron.h.ionchannelSeed = self.ionchannel_seed

        self.stimulus_seed = circuit_access.config.stimulus_seed if circuit_access else 0
        bglibpy.neuron.h.stimulusSeed = self.stimulus_seed

        self.minis_seed = circuit_access.config.minis_seed if circuit_access else 0
        bglibpy.neuron.h.minisSeed = self.minis_seed

        if base_noise_seed is None:
            self.base_noise_seed = 0
        else:
            self.base_noise_seed = base_noise_seed

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_val):
        """Setter method for the mode."""

        options = {"Compatibility": 0, "Random123": 1, "UpdatedMCell": 2}
        if new_val not in options:
            raise UndefinedRNGException(
                "SSim: RNG mode %s not in accepted list: %s"
                % (self.mode, list(options.keys()))
            )
        else:
            bglibpy.neuron.h.rngMode = options[new_val]
            self._mode = new_val

    def __repr__(self) -> str:
        """Returns a string representation of the object."""
        return "RNGSettings(mode={mode}, base_seed={base_seed}, " \
               "base_noise_seed={base_noise_seed}, " \
               "synapse_seed={synapse_seed}, " \
               "ionchannel_seed={ionchannel_seed}, " \
               "stimulus_seed={stimulus_seed}, " \
               "minis_seed={minis_seed})".format(
                   mode=self.mode,
                   base_seed=self.base_seed,
                   base_noise_seed=self.base_noise_seed,
                   synapse_seed=self.synapse_seed,
                   ionchannel_seed=self.ionchannel_seed,
                   stimulus_seed=self.stimulus_seed,
                   minis_seed=self.minis_seed)
