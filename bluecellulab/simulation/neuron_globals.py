# Copyright 2023-2024 Blue Brain Project / EPFL

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module that handles the global NEURON parameters."""

from typing import NamedTuple
import neuron
from bluecellulab.circuit.config.sections import Conditions, MechanismConditions
from bluecellulab.exceptions import error_context


def set_global_condition_parameters(condition_parameters: Conditions) -> None:
    """Sets the global condition parameters in NEURON objects if GluSynapse is
    available."""
    if condition_parameters.extracellular_calcium is not None and hasattr(
        neuron.h, "cao_CR_GluSynapse"
    ):
        cao_cr_glusynapse = condition_parameters.extracellular_calcium
        with error_context("mechanism/s for cao_CR_GluSynapse need to be compiled"):
            neuron.h.cao_CR_GluSynapse = cao_cr_glusynapse

    mechanism_conditions = condition_parameters.mech_conditions
    if mechanism_conditions is not None:
        set_minis_single_vesicle_values(mechanism_conditions)
        set_init_depleted_values(mechanism_conditions)


def set_init_depleted_values(mech_conditions: MechanismConditions) -> None:
    """Set the init_depleted values in NEURON."""
    with error_context("mechanism/s for init_depleted need to be compiled"):
        if mech_conditions.glusynapse and mech_conditions.glusynapse.init_depleted is not None:
            neuron.h.init_depleted_GluSynapse = mech_conditions.glusynapse.init_depleted
        if mech_conditions.ampanmda and mech_conditions.ampanmda.init_depleted is not None:
            neuron.h.init_depleted_ProbAMPANMDA_EMS = mech_conditions.ampanmda.init_depleted
        if mech_conditions.gabaab and mech_conditions.gabaab.init_depleted is not None:
            neuron.h.init_depleted_ProbGABAAB_EMS = mech_conditions.gabaab.init_depleted


def set_minis_single_vesicle_values(mech_conditions: MechanismConditions) -> None:
    """Set the minis_single_vesicle values in NEURON."""
    with error_context("mechanism/s for minis_single_vesicle need to be compiled"):
        if mech_conditions.ampanmda and mech_conditions.ampanmda.minis_single_vesicle is not None:
            neuron.h.minis_single_vesicle_ProbAMPANMDA_EMS = (
                mech_conditions.ampanmda.minis_single_vesicle
            )
        if mech_conditions.gabaab and mech_conditions.gabaab.minis_single_vesicle is not None:
            neuron.h.minis_single_vesicle_ProbGABAAB_EMS = (
                mech_conditions.gabaab.minis_single_vesicle
            )
        if mech_conditions.glusynapse and mech_conditions.glusynapse.minis_single_vesicle is not None:
            neuron.h.minis_single_vesicle_GluSynapse = (
                mech_conditions.glusynapse.minis_single_vesicle
            )


class NeuronGlobalParams(NamedTuple):
    temperature: float
    v_init: float


class NeuronGlobals:
    _instance = None

    def __init__(self):
        raise RuntimeError('Call get_instance() instead')

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            # Initialize default values
            cls._instance._temperature = 34.0  # Default temperature
            cls._instance.v_init = neuron.h.v_init
            # Set the default values in NEURON
            neuron.h.celsius = cls._instance._temperature
        return cls._instance

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value
        neuron.h.celsius = value

    @property
    def v_init(self):
        return self._v_init

    @v_init.setter
    def v_init(self, value):
        self._v_init = value
        neuron.h.v_init = value

    def export_params(self) -> NeuronGlobalParams:
        return NeuronGlobalParams(self.temperature, self.v_init)

    def load_params(self, params: NeuronGlobalParams) -> None:
        self.temperature = params.temperature
        self.v_init = params.v_init
