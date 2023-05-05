"""Hoc compatible synapse parameters representation."""

from __future__ import annotations

from bglibpy.circuit import CircuitAccess, SynapseProperty


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
