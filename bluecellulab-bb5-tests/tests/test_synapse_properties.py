"""Unit tests for synapse_properties module."""

from bluepy.enums import Synapse as BLPSynapse

from bluecellulab.circuit import SynapseProperty
from bluecellulab.circuit.synapse_properties import (
    properties_from_bluepy,
    properties_to_bluepy,
)


def test_synapse_property():
    """Test SynapseProperty."""
    pre_gid = SynapseProperty.PRE_GID
    bluepy_pre_gid = BLPSynapse.PRE_GID

    assert pre_gid != bluepy_pre_gid
    assert pre_gid == SynapseProperty.from_bluepy(bluepy_pre_gid)
    assert SynapseProperty.to_bluepy(pre_gid) == bluepy_pre_gid
    assert SynapseProperty.from_snap("@source_node") == pre_gid
    assert SynapseProperty.to_snap(pre_gid) == "@source_node"


def test_properties_from_bluepy():
    """Test properties_from_bluepy."""
    bluepy_props = [BLPSynapse.PRE_GID, BLPSynapse.POST_SECTION_ID, "rho0_GB"]
    props = properties_from_bluepy(bluepy_props)
    assert props == [
        SynapseProperty.PRE_GID,
        SynapseProperty.POST_SECTION_ID,
        "rho0_GB",
    ]


def test_properties_to_bluepy():
    """Test properties_to_bluepy."""
    props = [
        SynapseProperty.PRE_GID,
        SynapseProperty.POST_SECTION_ID,
        "rho0_GB",
        SynapseProperty.AFFERENT_SECTION_POS,
    ]
    bluepy_props = properties_to_bluepy(props)
    assert bluepy_props == [
        BLPSynapse.PRE_GID,
        BLPSynapse.POST_SECTION_ID,
        "rho0_GB",
        "afferent_section_pos",
    ]
