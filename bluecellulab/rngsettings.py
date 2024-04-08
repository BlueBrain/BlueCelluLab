# Copyright 2012-2024 Blue Brain Project / EPFL

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

import neuron
from bluecellulab.circuit.config.definition import SimulationConfig
from bluecellulab.exceptions import UndefinedRNGException
from bluecellulab.importer import load_mod_files

logger = logging.getLogger(__name__)


class RNGSettings:
    """Singleton object that represents RNG settings in bluecellulab."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Return the instance of the class."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @load_mod_files
    def __init__(
        self,
        mode="Random123",
        base_seed=0,
        synapse_seed=0,
        ionchannel_seed=0,
        stimulus_seed=0,
        minis_seed=0,
    ) -> None:
        self.mode = mode
        self.base_seed = base_seed
        self.synapse_seed = synapse_seed
        self.ionchannel_seed = ionchannel_seed
        self.stimulus_seed = stimulus_seed
        self.minis_seed = minis_seed

    def set_seeds(
            self,
            mode: Optional[str] = None,
            sim_config: Optional[SimulationConfig] = None,
            base_seed: Optional[int] = None):
        """Constructor.

        Parameters
        ----------
        mode : rng mode, if not specified mode is taken from circuit_access
        sim_config: simulation config object, if present seeds are read from simulation
        base_seed: base seed for entire sim, overrides config value
        """
        self._mode = ""
        if mode is None:
            if sim_config is not None:
                self.mode = sim_config.rng_mode if sim_config else "Compatibility"
            else:
                self.mode = "Random123"
        else:
            self.mode = mode

        logger.debug("Setting rng mode to: %s", self._mode)

        if base_seed is None:
            self.base_seed = sim_config.base_seed if sim_config else 0
        else:
            self.base_seed = base_seed
        neuron.h.globalSeed = self.base_seed

        if self._mode == 'Random123':
            rng = neuron.h.Random()
            rng.Random123_globalindex(self.base_seed)

        if sim_config:
            self.synapse_seed = sim_config.synapse_seed
            self.ionchannel_seed = sim_config.ionchannel_seed
            self.stimulus_seed = sim_config.stimulus_seed
            self.minis_seed = sim_config.minis_seed

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_val):
        """Setter method for the mode."""

        options = {"Compatibility": 0, "Random123": 1, "UpdatedMCell": 2}
        if new_val not in options:
            raise UndefinedRNGException(
                "RNG mode's value %s is not in the accepted list: %s"
                % (self.mode, list(options.keys()))
            )
        else:
            neuron.h.rngMode = options[new_val]
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
