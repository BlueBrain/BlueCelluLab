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
"""Functionality for validation the simulation configuration."""

import re

import pandas as pd

from bluecellulab import BLUEPY_AVAILABLE
from bluecellulab.circuit.circuit_access import CircuitAccess
from bluecellulab.circuit.synapse_properties import SynapseProperty
from bluecellulab.exceptions import ConfigError, TargetDoesNotExist, ExtraDependencyMissingError


class SimulationValidator:
    """Validates the simulation configuration, should be called before
    simulation."""

    def __init__(self, circuit_access: CircuitAccess) -> None:
        if not BLUEPY_AVAILABLE:
            raise ExtraDependencyMissingError("bluepy")
        self.circuit_access = circuit_access

    def validate(self) -> None:
        self.check_connection_entries()
        self.check_cao_cr_glusynapse_value()
        self.check_spike_location()

    def check_connection_entries(self):
        """Check all connection entries at once."""
        gid_pttrn = re.compile("^a[0-9]+")
        for entry in self.circuit_access.config.connection_entries():
            for target in (entry.source, entry.target):
                if not (self.circuit_access.is_valid_group(target) or gid_pttrn.match(target)):
                    raise TargetDoesNotExist("%s target does not exist" % target)

    def check_cao_cr_glusynapse_value(self):
        """Make sure cao_CR_GluSynapse is equal to ExtracellularCalcium."""
        condition_parameters = self.circuit_access.config.condition_parameters()
        if condition_parameters.extracellular_calcium is not None:
            cao_cr_glusynapse = condition_parameters.extracellular_calcium

            if cao_cr_glusynapse != self.circuit_access.config.extracellular_calcium:
                raise ConfigError("cao_CR_GluSynapse is not equal to ExtracellularCalcium")

    def check_spike_location(self):
        """Allow only accepted spike locations."""
        if 'SpikeLocation' in self.circuit_access.config.impl.Run:
            spike_location = self.circuit_access.config.impl.Run['SpikeLocation']
            if spike_location not in ["soma", "AIS"]:
                raise ConfigError(
                    "Possible options for SpikeLocation are 'soma' and 'AIS'")


def check_nrrp_value(synapses: pd.DataFrame) -> None:
    """Assures the nrrp values fits the conditions.

    Args:
        synapses: synapse description

    Raises:
        ValueError: when NRRP is <= 0
        ValueError: when NRRP cannot ve cast to integer
    """
    # remove nan ones, don't check them they're an artifact of pd Join
    nrrp_series = synapses[~synapses[SynapseProperty.NRRP].isna()][SynapseProperty.NRRP]

    if any(nrrp_series <= 0):
        raise ValueError(
            'Value smaller than 0.0 found for Nrrp: '
            f'in {nrrp_series}.'
        )

    if any(nrrp_series != nrrp_series.astype(int)):
        raise ValueError(
            'Non-integer value for Nrrp found: '
            f'{synapses[SynapseProperty.NRRP]} at synapse {synapses}.'
        )
