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
"""Factory that separates Synapse creation from Synapse instances."""

from enum import Enum
from typing import Optional

from bluecellulab.synapse import Synapse, GabaabSynapse, AmpanmdaSynapse, GluSynapse
from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit.synapse_properties import SynapseProperty


SynapseType = Enum('SynapseType', 'GABAAB AMPANMDA GLUSYNAPSE')


class SynapseFactory:
    """Creates different types of Synapses."""

    @classmethod
    def create_synapse(cls, cell, location, syn_id, syn_description, condition_parameters: Conditions,
                       base_seed, popids, extracellular_calcium, connection_modifiers) -> Synapse:
        """Returns a Synapse object."""
        is_inhibitory = int(syn_description[SynapseProperty.TYPE]) < 100

        if "ModOverride" in connection_modifiers:
            mod_override = connection_modifiers["ModOverride"]
        else:
            mod_override = None

        syn_type = cls.determine_synapse_type(is_inhibitory, mod_override)

        synapse: Synapse
        if syn_type == SynapseType.GABAAB:
            if condition_parameters.randomize_gaba_rise_time is not None:
                randomize_gaba_risetime = condition_parameters.randomize_gaba_rise_time
            else:
                randomize_gaba_risetime = True
            synapse = GabaabSynapse(cell, location, syn_id, syn_description,
                                    base_seed, popids, extracellular_calcium, randomize_gaba_risetime)
        elif syn_type == SynapseType.AMPANMDA:
            synapse = AmpanmdaSynapse(cell, location, syn_id, syn_description, base_seed,
                                      popids, extracellular_calcium)
        else:
            synapse = GluSynapse(cell, location, syn_id, syn_description, base_seed,
                                 popids, extracellular_calcium)

        synapse = cls.apply_connection_modifiers(connection_modifiers, synapse)

        return synapse

    @staticmethod
    def apply_connection_modifiers(connection_modifiers, synapse) -> Synapse:
        if "DelayWeights" in connection_modifiers:
            synapse.delay_weights = connection_modifiers["DelayWeights"]
        if "Weight" in connection_modifiers:
            synapse.weight = connection_modifiers["Weight"]
        if "SynapseConfigure" in connection_modifiers:
            synapse.apply_hoc_configuration(connection_modifiers["SynapseConfigure"])
        return synapse

    @staticmethod
    def determine_synapse_type(is_inhibitory: int, mod_override: Optional[str]) -> SynapseType:
        """Returns the type of synapse to be created."""
        if is_inhibitory:
            return SynapseType.GABAAB
        else:
            if mod_override == "GluSynapse":
                return SynapseType.GLUSYNAPSE
            else:
                return SynapseType.AMPANMDA
