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
"""Hoc compatible synapse parameters representation."""

from __future__ import annotations

from bluecellulab.circuit import CircuitAccess, SynapseProperty


class SynDescription:
    """Retrieve syn descriptions for the defined properties."""

    def __init__(self) -> None:
        self.common_properties = [
            SynapseProperty.PRE_GID,
            SynapseProperty.AXONAL_DELAY,
            SynapseProperty.POST_SECTION_ID,
            SynapseProperty.POST_SEGMENT_ID,
            SynapseProperty.POST_SEGMENT_OFFSET,
            SynapseProperty.G_SYNX,
            SynapseProperty.U_SYN,
            SynapseProperty.D_SYN,
            SynapseProperty.F_SYN,
            SynapseProperty.DTC,
            SynapseProperty.TYPE,
            SynapseProperty.NRRP,
            SynapseProperty.U_HILL_COEFFICIENT,
            SynapseProperty.CONDUCTANCE_RATIO,
        ]

    def gabaab_ampanmda_syn_description(self, circuit: CircuitAccess,
                                        gid, projections=None):
        """Wraps circuit.extract_synapses with ampanmda/gabaab properties."""
        return circuit.extract_synapses(gid, self.common_properties, projections)

    def glusynapse_syn_description(self, circuit: CircuitAccess,
                                   gid, projections=None):
        """Wraps circuit.extract_synapses with glusynapse properties."""
        glusynapse_only_properties = [
            "volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA",
            "gmax_p_AMPA", "theta_d", "theta_p"]
        all_properties = self.common_properties + glusynapse_only_properties
        return circuit.extract_synapses(gid, all_properties, projections)
