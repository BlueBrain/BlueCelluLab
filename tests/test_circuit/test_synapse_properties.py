"""Unit tests for synapse_properties module."""

from bluepy.enums import Synapse as BLPSynapse

from bglibpy.circuit import SynapseProperty
from bglibpy.circuit.synapse_properties import (
    properties_from_bluepy,
    properties_from_snap,
    properties_to_bluepy,
    properties_to_snap,
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
    props = [SynapseProperty.PRE_GID, SynapseProperty.POST_SECTION_ID, "rho0_GB"]
    bluepy_props = properties_to_bluepy(props)
    assert bluepy_props == [BLPSynapse.PRE_GID, BLPSynapse.POST_SECTION_ID, "rho0_GB"]


def test_properties_from_snap():
    """Test properties_from_snap."""
    snap_props = ["@source_node", "afferent_section_id", "rho0_GB"]
    props = properties_from_snap(snap_props)
    assert props == [
        SynapseProperty.PRE_GID,
        SynapseProperty.POST_SECTION_ID,
        "rho0_GB",
    ]


def test_properties_to_snap():
    """Test properties_to_snap."""
    props = [SynapseProperty.PRE_GID, SynapseProperty.POST_SECTION_ID, "rho0_GB"]
    snap_props = properties_to_snap(props)
    assert snap_props == ["@source_node", "afferent_section_id", "rho0_GB"]
