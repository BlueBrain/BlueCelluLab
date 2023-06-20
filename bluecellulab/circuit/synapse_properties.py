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
"""Synapse Properties."""

from __future__ import annotations
from enum import Enum
from types import MappingProxyType

from bluecellulab import BLUEPY_AVAILABLE
from bluecellulab.exceptions import ExtraDependencyMissingError

if BLUEPY_AVAILABLE:
    from bluepy.enums import Synapse as BLPSynapse


class SynapseProperty(Enum):
    PRE_GID = "pre_gid"
    AXONAL_DELAY = "axonal_delay"
    POST_SECTION_ID = "post_section_id"
    POST_SEGMENT_ID = "post_segment_id"
    POST_SEGMENT_OFFSET = "post_segment_offset"
    G_SYNX = "g_synx"
    U_SYN = "u_syn"
    D_SYN = "d_syn"
    F_SYN = "f_syn"
    DTC = "DTC"
    TYPE = "type"
    NRRP = "NRRP"
    U_HILL_COEFFICIENT = "u_hill_coefficient"
    CONDUCTANCE_RATIO = "conductance_scale_factor"

    @classmethod
    def from_bluepy(cls, prop: BLPSynapse) -> SynapseProperty:
        return cls(prop.value)

    def to_bluepy(self) -> BLPSynapse:
        if not BLUEPY_AVAILABLE:
            raise ExtraDependencyMissingError("bluepy")
        return BLPSynapse(self.value)

    @classmethod
    def from_snap(cls, prop: str) -> SynapseProperty:
        return cls(snap_to_synproperty[prop])

    def to_snap(self) -> str:
        return synproperty_to_snap[self]


snap_to_synproperty = MappingProxyType({
    "@source_node": SynapseProperty.PRE_GID,
    "delay": SynapseProperty.AXONAL_DELAY,
    "afferent_section_id": SynapseProperty.POST_SECTION_ID,
    "afferent_segment_id": SynapseProperty.POST_SEGMENT_ID,
    "afferent_segment_offset": SynapseProperty.POST_SEGMENT_OFFSET,
    "conductance": SynapseProperty.G_SYNX,
    "u_syn": SynapseProperty.U_SYN,
    "depression_time": SynapseProperty.D_SYN,
    "facilitation_time": SynapseProperty.F_SYN,
    "decay_time": SynapseProperty.DTC,
    "syn_type_id": SynapseProperty.TYPE,
    "n_rrp_vesicles": SynapseProperty.NRRP,
    "u_hill_coefficient": SynapseProperty.U_HILL_COEFFICIENT,
    "conductance_scale_factor": SynapseProperty.CONDUCTANCE_RATIO,
})


# the inverse of snap_to_synproperty
synproperty_to_snap = MappingProxyType({
    v: k for k, v in snap_to_synproperty.items()
})


def properties_from_snap(
    props: list[str],
) -> list[SynapseProperty | str]:
    """Convert list of SNAP Synapse properties to SynapseProperty, spare
    'str's."""
    return [
        SynapseProperty.from_snap(prop)
        if prop in snap_to_synproperty
        else prop
        for prop in props
    ]


def properties_to_snap(props: list[SynapseProperty | str]) -> list[str]:
    """Convert a list of SynapseProperty to SNAP properties, spare 'str's."""
    return [
        prop.to_snap()
        if isinstance(prop, SynapseProperty)
        else prop
        for prop in props
    ]


def properties_from_bluepy(
    props: list[BLPSynapse | str],
) -> list[SynapseProperty | str]:
    """Convert list of bluepy Synapse properties to SynapseProperty, spare
    'str's."""
    if not BLUEPY_AVAILABLE:
        raise ExtraDependencyMissingError("bluepy")
    return [
        SynapseProperty.from_bluepy(prop)
        if isinstance(prop, BLPSynapse)
        else prop
        for prop in props
    ]


def properties_to_bluepy(props: list[SynapseProperty | str]) -> list[BLPSynapse | str]:
    """Convert a list of SynapseProperty to bluepy Synapse properties, spare
    'str's."""
    return [
        prop.to_bluepy()
        if isinstance(prop, SynapseProperty)
        else prop
        for prop in props
    ]
