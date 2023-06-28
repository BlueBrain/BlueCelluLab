"""Unit tests for synapse_properties module."""
from bluecellulab.circuit import SynapseProperty
from bluecellulab.circuit.synapse_properties import (
    properties_from_snap,
    properties_to_snap,
)


def test_synapse_property():
    """Test SynapseProperty."""
    pre_gid = SynapseProperty.PRE_GID
    assert SynapseProperty.from_snap("@source_node") == pre_gid
    assert SynapseProperty.to_snap(pre_gid) == "@source_node"


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
