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
"""Factory that separates Synapse creation from Synapse instances."""
from __future__ import annotations  # PEP-563 forward reference annotations
from enum import Enum
import logging
import math

import neuron
import numpy as np

import pandas as pd

import bluecellulab
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.synapse import Synapse, GabaabSynapse, AmpanmdaSynapse, GluSynapse
from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit.synapse_properties import SynapseProperties, SynapseProperty
from bluecellulab.synapse.synapse_types import SynapseHocArgs
from bluecellulab.type_aliases import NeuronSection


SynapseType = Enum("SynapseType", "GABAAB AMPANMDA GLUSYNAPSE")

logger = logging.getLogger(__name__)


class SynapseFactory:
    """Creates different types of Synapses."""

    @classmethod
    def create_synapse(
        cls,
        cell: bluecellulab.Cell,
        syn_id: tuple[str, int],
        syn_description: pd.Series,
        condition_parameters: Conditions,
        popids: tuple[int, int],
        extracellular_calcium: float | None,
        connection_modifiers: dict,
    ) -> Synapse:
        """Returns a Synapse object."""
        syn_type = cls.determine_synapse_type(syn_description)
        syn_hoc_args = cls.determine_synapse_location(syn_description, cell)

        synapse: Synapse
        if syn_type == SynapseType.GABAAB:
            if condition_parameters.randomize_gaba_rise_time is not None:
                randomize_gaba_risetime = condition_parameters.randomize_gaba_rise_time
            else:
                randomize_gaba_risetime = True
            synapse = GabaabSynapse(cell.cell_id, syn_hoc_args, syn_id, syn_description,
                                    popids, extracellular_calcium, randomize_gaba_risetime)
        elif syn_type == SynapseType.AMPANMDA:
            synapse = AmpanmdaSynapse(cell.cell_id, syn_hoc_args, syn_id, syn_description,
                                      popids, extracellular_calcium)
        else:
            synapse = GluSynapse(cell.cell_id, syn_hoc_args, syn_id, syn_description,
                                 popids, extracellular_calcium)

        synapse = cls.apply_connection_modifiers(connection_modifiers, synapse)

        return synapse

    @staticmethod
    def apply_connection_modifiers(connection_modifiers: dict, synapse: Synapse) -> Synapse:
        if "DelayWeights" in connection_modifiers:
            synapse.delay_weights = connection_modifiers["DelayWeights"]
        if "Weight" in connection_modifiers:
            synapse.weight = connection_modifiers["Weight"]
        if "SynapseConfigure" in connection_modifiers:
            synapse.apply_hoc_configuration(connection_modifiers["SynapseConfigure"])
        return synapse

    @staticmethod
    def determine_synapse_type(
        syn_description: pd.Series,
    ) -> SynapseType:
        """Returns the type of synapse to be created."""
        is_inhibitory: bool = int(syn_description[SynapseProperty.TYPE]) < 100
        all_plasticity_props_available: bool = all(
            x in syn_description and syn_description[x] is not None and not math.isnan(syn_description[x])
            for x in SynapseProperties.plasticity
        )
        no_plasticity_props_available: bool = all(
            x not in syn_description or syn_description[x] is None or math.isnan(syn_description[x])
            for x in SynapseProperties.plasticity
        )
        if is_inhibitory:
            return SynapseType.GABAAB
        else:
            if all_plasticity_props_available:
                return SynapseType.GLUSYNAPSE
            elif no_plasticity_props_available:
                return SynapseType.AMPANMDA
            else:
                raise BluecellulabError("SynapseFactory: Cannot determine synapse type")

    @classmethod
    def determine_synapse_location(cls, syn_description: pd.Series, cell: bluecellulab.Cell) -> SynapseHocArgs:
        """Returns the location of the synapse."""
        isec = int(syn_description[SynapseProperty.POST_SECTION_ID])  # numpy int to int
        section: NeuronSection = cell.get_psection(section_id=isec).hsection

        # old circuits don't have it, it needs to be computed via synlocation_to_segx
        if (SynapseProperty.AFFERENT_SECTION_POS in syn_description and
                not np.isnan(syn_description[SynapseProperty.AFFERENT_SECTION_POS])):
            # position is pre computed in SONATA
            location = syn_description[SynapseProperty.AFFERENT_SECTION_POS]
            if location == 0.0:
                location = 0.0000001
            elif location >= 1.0:
                location = 0.9999999
        else:
            ipt = syn_description[SynapseProperty.POST_SEGMENT_ID]
            syn_offset = syn_description[SynapseProperty.POST_SEGMENT_OFFSET]
            location = cls.synlocation_to_segx(section, ipt, syn_offset)

        return SynapseHocArgs(location, section)

    @staticmethod
    def synlocation_to_segx(
        section: NeuronSection, ipt: float, syn_offset: float
    ) -> float:
        """Translates a synaptic (secid, ipt, offset) to an x coordinate.

        Args:
            section: Section object.
            ipt: Post segment ID.
            syn_offset: Synaptic offset.

        Returns:
            The x coordinate on the section, where the synapse can be placed.
        """
        if syn_offset < 0.0:
            syn_offset = 0.0

        length = section.L

        # access section to compute the distance
        if neuron.h.section_orientation(sec=section) == 1:
            ipt = neuron.h.n3d(sec=section) - 1 - ipt
            syn_offset = -syn_offset

        distance = 0.5
        if ipt < neuron.h.n3d(sec=section):
            distance = (neuron.h.arc3d(ipt, sec=section) + syn_offset) / length
            if distance == 0.0:
                distance = 0.0000001
            if distance >= 1.0:
                distance = 0.9999999

        if neuron.h.section_orientation(sec=section) == 1:
            distance = 1 - distance

        if distance < 0:
            logger.warning(
                f"synlocation_to_segx found negative distance \
                        at curr_sec({neuron.h.secname(sec=section)}) syn_offset: {syn_offset}"
            )
            return 0.0
        else:
            return distance
