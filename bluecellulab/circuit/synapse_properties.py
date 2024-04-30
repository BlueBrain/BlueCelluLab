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
"""Synapse Properties."""

from __future__ import annotations
from enum import Enum
from types import MappingProxyType
from typing import Any

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
    AFFERENT_SECTION_POS = "afferent_section_pos"

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


class SynapseProperties:
    """Synapse Properties to be retrieved from circuit and used by the
    simulator."""
    common = (
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
        SynapseProperty.AFFERENT_SECTION_POS,
    )
    plasticity = (
        "volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA",
        "gmax_p_AMPA", "theta_d", "theta_p"
    )


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
    "afferent_section_pos": SynapseProperty.AFFERENT_SECTION_POS,
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
    res: list[SynapseProperty | str] = []
    for prop in props:
        if isinstance(prop, BLPSynapse):
            res.append(SynapseProperty.from_bluepy(prop))
        elif prop == "afferent_section_pos":  # jira_url/project/issues/browse/NSETM-2313
            res.append(SynapseProperty.AFFERENT_SECTION_POS)
        else:
            res.append(prop)
    return res


def properties_to_bluepy(props: list[SynapseProperty | str]) -> list[BLPSynapse | str]:
    """Convert a list of SynapseProperty to bluepy Synapse properties, spare
    'str's."""
    # bluepy does not have AFFERENT_SECTION_POS atm.
    # jira_url/project/issues/browse/NSETM-2313
    bluepy_recognised_props = props.copy()
    removed_afferent_section_pos = False
    if SynapseProperty.AFFERENT_SECTION_POS in bluepy_recognised_props:
        removed_afferent_section_pos = True
        bluepy_recognised_props.remove(SynapseProperty.AFFERENT_SECTION_POS)
    res = [
        prop.to_bluepy()
        if isinstance(prop, SynapseProperty)
        else prop
        for prop in bluepy_recognised_props
    ]
    if removed_afferent_section_pos:
        res.append("afferent_section_pos")
    return res


def synapse_property_encoder(dct: dict[SynapseProperty | str, Any]) -> dict[str, Any]:
    """Convert SynapseProperty enum keys to strings."""
    return {key.name if isinstance(key, SynapseProperty) else key: value for key, value in dct.items()}


def synapse_property_decoder(dct: dict) -> dict[str | SynapseProperty, Any]:
    """For JSON decoding of dict containing SynapseProperty."""
    transformed_dict: dict[str | SynapseProperty, Any] = {}
    for key, value in dct.items():
        if key in SynapseProperty._member_names_:
            transformed_dict[SynapseProperty[key]] = value
        else:
            transformed_dict[key] = value
    return transformed_dict
