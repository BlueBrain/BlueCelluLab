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
"""Class that represents a synapse in bluecellulab."""

from __future__ import annotations
from typing import Any, Optional
import pandas as pd
import logging

import bluecellulab
from bluecellulab.circuit import SynapseProperty


NeuronType = Any

logger = logging.getLogger(__name__)


class Synapse:
    """Class that represents a synapse in bluecellulab."""

    def __init__(
            self, cell, location: float, syn_id, syn_description, base_seed, popids=(0, 0),
            extracellular_calcium=None):
        """Constructor.

        Parameters
        ----------
        cell : Cell
               Cell that contains the synapse
        location : float in [0, 1]
                   Location on the section this synapse is placed
        syn_id : (string,integer)
              Synapse identifier, string being the projection name and
               int the synapse id. Empty string refers to a local connection
        syn_description : list of floats
                          Parameters of the synapse
        base_seed : float
                    Base seed of the simulation, the seeds for this synapse
                    will be derived from this
        popids : tuple of (int, int)
                  Source and target popids used by the random number generation
        extracellular_calcium: float
                               the extracellular calcium concentration
        """
        self.persistent: list[NeuronType] = []
        self.synapseconfigure_cmds: list[str] = []
        self._delay_weights: list[tuple[float, float]] = []
        self._weight: Optional[float] = None

        self.cell = cell
        self.post_gid = cell.gid
        self.syn_id = syn_id
        self.projection, self.sid = self.syn_id
        self.extracellular_calcium = extracellular_calcium
        self.syn_description = self.update_syn_description(syn_description)
        self.hsynapse: Optional[NeuronType] = None

        self.source_popid, self.target_popid = popids

        self.pre_gid = int(self.syn_description[SynapseProperty.PRE_GID])

        self.post_segx = location

        if cell.rng_settings is None:
            self.rng_setting = bluecellulab.RNGSettings(
                mode='Compatibility',
                base_seed=base_seed)
        elif base_seed is not None:
            raise Exception('Synapse: base_seed and RNGSettings cant '
                            'be used together')
        else:
            self.rng_settings = cell.rng_settings

    @property
    def delay_weights(self) -> list[tuple[float, float]]:
        """Adjustments to synapse delay and weight."""
        return self._delay_weights

    @delay_weights.setter
    def delay_weights(self, value: list[tuple[float, float]]) -> None:
        self._delay_weights = value

    @property
    def weight(self) -> float | None:
        """The last overridden synapse weight."""
        return self._weight

    @weight.setter
    def weight(self, value: float | None) -> None:
        self._weight = value

    def update_syn_description(self, syn_description: pd.Series) -> pd.Series:
        """Change data types, compute more columns needed by the simulator."""
        # if the optional properties are NaN (that happens due to pandas outer join), then remove them
        for prop in [SynapseProperty.U_HILL_COEFFICIENT, SynapseProperty.NRRP]:
            if prop in syn_description and pd.isna(syn_description[prop]):
                syn_description.pop(prop)

        if SynapseProperty.NRRP in syn_description:
            try:
                int(syn_description[SynapseProperty.NRRP])
            except ValueError:
                # delete SynapseProperty.NRRP from syn_description
                syn_description.pop(SynapseProperty.NRRP)

        if SynapseProperty.U_HILL_COEFFICIENT in syn_description:
            syn_description["u_scale_factor"] = self.calc_u_scale_factor(
                syn_description[SynapseProperty.U_HILL_COEFFICIENT], self.extracellular_calcium)
        else:
            syn_description["u_scale_factor"] = 1.0
        syn_description[SynapseProperty.U_SYN] *= syn_description["u_scale_factor"]
        return syn_description

    def apply_hoc_configuration(self, hoc_configure_params: list[str]) -> None:
        """Apply the list of hoc configuration commands to the synapse."""
        self.synapseconfigure_cmds = []
        # hoc exec synapse configure blocks
        for cmd in hoc_configure_params:
            self.synapseconfigure_cmds.append(cmd)
            cmd = cmd.replace('%s', '\n%(syn)s')
            hoc_cmd = cmd % {'syn': self.hsynapse.hname()}  # type: ignore
            hoc_cmd = '{%s}' % hoc_cmd
            bluecellulab.neuron.h(hoc_cmd)

    def _set_gabaab_ampanmda_rng(self) -> None:
        """Setup the RNG for the gabaab and ampanmd helpers.

        Raises:
            ValueError: when rng mode is not recognised.
        """
        if self.rng_settings.mode == "Random123":
            self.randseed1 = self.cell.gid + 250
            self.randseed2 = self.sid + 100
            self.randseed3 = self.source_popid * 65536 + self.target_popid + \
                self.rng_settings.synapse_seed + 300
            self.hsynapse.setRNG(  # type: ignore
                self.randseed1,
                self.randseed2,
                self.randseed3)
        else:
            rndd = bluecellulab.neuron.h.Random()
            if self.rng_settings.mode == "Compatibility":
                self.randseed1 = self.sid * 100000 + 100
                self.randseed2 = self.cell.gid + \
                    250 + self.rng_settings.base_seed
            elif self.rng_settings.mode == "UpdatedMCell":
                self.randseed1 = self.sid * 1000 + 100
                self.randseed2 = self.source_popid * 16777216 + \
                    self.cell.gid + \
                    250 + self.rng_settings.base_seed + \
                    self.rng_settings.synapse_seed
            else:
                raise ValueError(
                    "Synapse: unknown RNG mode: %s" %
                    self.rng_settings.mode)
            self.randseed3 = None  # Not used in this case
            rndd.MCellRan4(self.randseed1, self.randseed2)
            rndd.uniform(0, 1)
            self.hsynapse.setRNG(rndd)  # type: ignore
            self.persistent.append(rndd)

    def delete(self) -> None:
        """Delete the NEURON objects of the connection."""
        if hasattr(self, 'persistent'):
            for persistent_object in self.persistent:
                del persistent_object

    @staticmethod
    def calc_u_scale_factor(u_hill_coefficient, extracellular_calcium):
        if extracellular_calcium is None or u_hill_coefficient is None:
            return 1.0

        def constrained_hill(K_half, y) -> float:
            """Calculates the constrained hill coefficient."""
            K_half_fourth = K_half**4
            y_fourth = y**4
            return (K_half_fourth + 16) / 16 * y_fourth / (K_half_fourth + y_fourth)

        u_scale_factor = constrained_hill(u_hill_coefficient, extracellular_calcium)
        logger.debug(
            "Scaling synapse Use with u_hill_coeffient %f, "
            "extra_cellular_calcium %f with a factor of %f" %
            (u_hill_coefficient, extracellular_calcium, u_scale_factor))

        return u_scale_factor

    @property
    def info_dict(self):
        """Convert the synapse info to a dict from which it can be
        reconstructed."""
        synapse_dict = {}

        synapse_dict['synapse_id'] = self.syn_id
        synapse_dict['pre_cell_id'] = self.pre_gid
        synapse_dict['post_cell_id'] = self.post_gid
        synapse_dict['syn_description'] = self.syn_description.to_dict()
        # if keys are enum make them str
        synapse_dict['syn_description'] = {
            str(k): v for k, v in synapse_dict['syn_description'].items()}

        synapse_dict['post_segx'] = self.post_segx
        synapse_dict['mech_name'] = self.mech_name
        synapse_dict['randseed1'] = self.randseed1
        synapse_dict['randseed2'] = self.randseed2
        synapse_dict['randseed3'] = self.randseed3
        synapse_dict['synapseconfigure_cmds'] = self.synapseconfigure_cmds

        # Parameters of the mod mechanism
        synapse_dict['synapse_parameters'] = {}
        synapse_dict['synapse_parameters']['Use'] = self.hsynapse.Use
        synapse_dict['synapse_parameters']['Dep'] = self.hsynapse.Dep
        synapse_dict['synapse_parameters']['Fac'] = self.hsynapse.Fac

        synapse_dict['synapse_parameters']['extracellular_calcium'] = \
            self.extracellular_calcium

        return synapse_dict

    def __del__(self) -> None:
        self.delete()


class GluSynapse(Synapse):

    def __init__(self, cell, location, syn_id, syn_description, base_seed, popids, extracellular_calcium):
        super().__init__(cell, location, syn_id, syn_description, base_seed, popids, extracellular_calcium)
        self.use_glusynapse_helper()

    def use_glusynapse_helper(self) -> None:
        """Python implementation of the GluSynapse helper.

        This helper object will encapsulate the hoc actions  needed to
        create our plastic excitatory synapse.
        """
        self.mech_name = 'GluSynapse'

        self.hsynapse = bluecellulab.neuron.h.GluSynapse(
            self.post_segx,
            sec=self.cell.get_hsection(
                self.syn_description[SynapseProperty.POST_SECTION_ID]
            ),
        )

        self.hsynapse.Use_d = self.syn_description["Use_d_TM"] * \
            self.syn_description["u_scale_factor"]
        self.hsynapse.Use_p = self.syn_description["Use_p_TM"] * \
            self.syn_description["u_scale_factor"]

        self.hsynapse.theta_d_GB = self.syn_description["theta_d"]
        self.hsynapse.theta_p_GB = self.syn_description["theta_p"]
        self.hsynapse.rho0_GB = self.syn_description["rho0_GB"]

        self.hsynapse.volume_CR = self.syn_description["volume_CR"]

        self.hsynapse.gmax_d_AMPA = self.syn_description["gmax_d_AMPA"]
        self.hsynapse.gmax_p_AMPA = self.syn_description["gmax_p_AMPA"]

        if self.hsynapse.rho0_GB > bluecellulab.neuron.h.rho_star_GB_GluSynapse:
            self.hsynapse.gmax0_AMPA = self.hsynapse.gmax_p_AMPA
            self.hsynapse.Use = self.hsynapse.Use_p
        else:
            self.hsynapse.gmax0_AMPA = self.hsynapse.gmax_d_AMPA
            self.hsynapse.Use = self.hsynapse.Use_d

        self.hsynapse.gmax_NMDA = self.hsynapse.gmax0_AMPA * \
            self.syn_description[SynapseProperty.CONDUCTANCE_RATIO]
        self.hsynapse.tau_d_AMPA = self.syn_description[SynapseProperty.DTC]
        self.hsynapse.Dep = abs(self.syn_description[SynapseProperty.D_SYN])
        self.hsynapse.Fac = abs(self.syn_description[SynapseProperty.F_SYN])

        if self.syn_description[SynapseProperty.NRRP] >= 0:
            self.hsynapse.Nrrp = self.syn_description[SynapseProperty.NRRP]

        self.randseed1 = self.cell.gid
        self.randseed2 = 100000 + self.sid
        self.randseed3 = self.rng_settings.synapse_seed + 200
        self.hsynapse.setRNG(self.randseed1, self.randseed2, self.randseed3)
        self.hsynapse.synapseID = self.sid

    @property
    def info_dict(self):
        parent_dict = super().info_dict
        parent_dict['synapse_parameters']['tau_d_AMPA'] = self.hsynapse.tau_d_AMPA
        return parent_dict


class GabaabSynapse(Synapse):

    def __init__(self, cell, location, syn_id, syn_description, base_seed, popids, extracellular_calcium, randomize_risetime=True):
        super().__init__(cell, location, syn_id, syn_description, base_seed, popids, extracellular_calcium)
        self.use_gabaab_helper(randomize_risetime)

    def use_gabaab_helper(self, randomize_gaba_risetime: bool) -> None:
        """Python implementation of the GABAAB helper.

        This helper object will encapsulate the hoc actions
         needed to create our typical inhibitory synapse.

        Args:
            randomize_gaba_risetime: if True then RNG code runs
            to set tau_r_GABAA value.

        Raises:
            ValueError: raised when the RNG mode is unknown.
        """
        self.mech_name = 'ProbGABAAB_EMS'

        self.hsynapse = bluecellulab.neuron.h.ProbGABAAB_EMS(
            self.post_segx,
            sec=self.cell.get_hsection(
                self.syn_description[SynapseProperty.POST_SECTION_ID]
            )
        )

        if randomize_gaba_risetime is True:
            rng = bluecellulab.neuron.h.Random()
            if self.rng_settings.mode == "Compatibility":
                rng.MCellRan4(
                    self.sid * 100000 + 100,
                    self.cell.gid + 250 + self.rng_settings.base_seed)
            elif self.rng_settings.mode == "UpdatedMCell":
                rng.MCellRan4(
                    self.sid * 1000 + 100,
                    self.source_popid *
                    16777216 +
                    self.cell.gid +
                    250 +
                    self.rng_settings.base_seed +
                    self.rng_settings.synapse_seed)
            elif self.rng_settings.mode == "Random123":
                rng.Random123(
                    self.cell.gid +
                    250,
                    self.sid +
                    100,
                    self.source_popid *
                    65536 +
                    self.target_popid +
                    self.rng_settings.synapse_seed +
                    450)
            else:
                raise ValueError(
                    "Synapse: unknown RNG mode: %s" %
                    self.rng_settings.mode)

            rng.lognormal(0.2, 0.1)
            self.hsynapse.tau_r_GABAA = rng.repick()

        if SynapseProperty.CONDUCTANCE_RATIO in self.syn_description:
            self.hsynapse.GABAB_ratio = self.syn_description[SynapseProperty.CONDUCTANCE_RATIO]

        self.hsynapse.tau_d_GABAA = self.syn_description[SynapseProperty.DTC]
        self.hsynapse.Use = abs(self.syn_description[SynapseProperty.U_SYN])
        self.hsynapse.Dep = abs(self.syn_description[SynapseProperty.D_SYN])
        self.hsynapse.Fac = abs(self.syn_description[SynapseProperty.F_SYN])

        if SynapseProperty.NRRP in self.syn_description:
            self.hsynapse.Nrrp = self.syn_description[SynapseProperty.NRRP]

        self._set_gabaab_ampanmda_rng()

        self.hsynapse.synapseID = self.sid

    @property
    def info_dict(self):
        parent_dict = super().info_dict
        parent_dict['synapse_parameters']['tau_d_GABAA'] = self.hsynapse.tau_d_GABAA
        parent_dict['synapse_parameters']['tau_r_GABAA'] = self.hsynapse.tau_r_GABAA
        return parent_dict


class AmpanmdaSynapse(Synapse):

    def __init__(self, cell, location, syn_id, syn_description, base_seed, popids, extracellular_calcium):
        super().__init__(cell, location, syn_id, syn_description, base_seed, popids, extracellular_calcium)
        self.use_ampanmda_helper()

    def use_ampanmda_helper(self) -> None:
        """Python implementation of the AMPANMDA helper.

        This helper object will encapsulate the hoc actions  needed to
        create our typical excitatory synapse.
        """
        self.mech_name = 'ProbAMPANMDA_EMS'

        self.hsynapse = bluecellulab.neuron.h.ProbAMPANMDA_EMS(
            self.post_segx,
            sec=self.cell.get_hsection(
                self.syn_description[SynapseProperty.POST_SECTION_ID]
            ),
        )
        self.hsynapse.tau_d_AMPA = self.syn_description[SynapseProperty.DTC]
        if SynapseProperty.CONDUCTANCE_RATIO in self.syn_description:
            self.hsynapse.NMDA_ratio = self.syn_description[SynapseProperty.CONDUCTANCE_RATIO]

        self.hsynapse.Use = abs(self.syn_description[SynapseProperty.U_SYN])
        self.hsynapse.Dep = abs(self.syn_description[SynapseProperty.D_SYN])
        self.hsynapse.Fac = abs(self.syn_description[SynapseProperty.F_SYN])

        if SynapseProperty.NRRP in self.syn_description:
            self.hsynapse.Nrrp = self.syn_description[SynapseProperty.NRRP]

        self._set_gabaab_ampanmda_rng()
        self.hsynapse.synapseID = self.sid

    @property
    def info_dict(self):
        parent_dict = super().info_dict
        parent_dict['synapse_parameters']['tau_d_AMPA'] = self.hsynapse.tau_d_AMPA
        return parent_dict
