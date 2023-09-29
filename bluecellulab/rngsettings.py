# Copyright 2012-2023 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RNG settings of bluecellulab."""

import logging

from typing import Optional

import bluecellulab
from bluecellulab import Singleton
from bluecellulab.circuit.circuit_access import CircuitAccess
from bluecellulab.exceptions import UndefinedRNGException

logger = logging.getLogger(__name__)


class RNGSettings(metaclass=Singleton):
    """Class that represents RNG settings in bluecellulab."""

    def __init__(
            self,
            mode: Optional[str] = None,
            circuit_access: Optional[CircuitAccess] = None,
            base_seed: Optional[int] = None):
        """Constructor.

        Parameters
        ----------
        mode : rng mode, if not specified mode is taken from circuit_access
        circuit: circuit access object, if present seeds are read from simulation
        base_seed: base seed for entire sim, overrides config value
        """

        self._mode = ""
        if mode is None:
            if circuit_access is not None:
                self.mode = circuit_access.config.rng_mode if circuit_access else "Compatibility"
            else:
                self.mode = "Random123"
        else:
            self.mode = mode

        logger.debug("Setting rng mode to: %s", self._mode)

        if base_seed is None:
            self.base_seed = circuit_access.config.base_seed if circuit_access else 0
        else:
            self.base_seed = base_seed
        bluecellulab.neuron.h.globalSeed = self.base_seed

        if self._mode == 'Random123':
            rng = bluecellulab.neuron.h.Random()
            rng.Random123_globalindex(self.base_seed)

        self.synapse_seed = circuit_access.config.synapse_seed if circuit_access else 0
        bluecellulab.neuron.h.synapseSeed = self.synapse_seed

        self.ionchannel_seed = circuit_access.config.ionchannel_seed if circuit_access else 0
        bluecellulab.neuron.h.ionchannelSeed = self.ionchannel_seed

        self.stimulus_seed = circuit_access.config.stimulus_seed if circuit_access else 0
        bluecellulab.neuron.h.stimulusSeed = self.stimulus_seed

        self.minis_seed = circuit_access.config.minis_seed if circuit_access else 0
        bluecellulab.neuron.h.minisSeed = self.minis_seed

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
            bluecellulab.neuron.h.rngMode = options[new_val]
            self._mode = new_val

    def __repr__(self) -> str:
        """Returns a string representation of the object."""
        return "RNGSettings(mode={mode}, base_seed={base_seed}, " \
               "synapse_seed={synapse_seed}, " \
               "ionchannel_seed={ionchannel_seed}, " \
               "stimulus_seed={stimulus_seed}, " \
               "minis_seed={minis_seed})".format(
                   mode=self.mode,
                   base_seed=self.base_seed,
                   synapse_seed=self.synapse_seed,
                   ionchannel_seed=self.ionchannel_seed,
                   stimulus_seed=self.stimulus_seed,
                   minis_seed=self.minis_seed)
