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
"""Cell class."""

from __future__ import annotations

import logging

from pathlib import Path
import queue
from typing import Optional
from typing_extensions import deprecated

import neuron
import numpy as np
import pandas as pd

import bluecellulab
from bluecellulab.cell.recording import section_to_voltage_recording_str
from bluecellulab.psection import PSection, init_psections
from bluecellulab.cell.injector import InjectableMixin
from bluecellulab.cell.plotting import PlottableMixin
from bluecellulab.cell.section_distance import EuclideanSectionDistance
from bluecellulab.cell.sonata_proxy import SonataProxy
from bluecellulab.cell.template import NeuronTemplate, TemplateParams, public_hoc_cell
from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit import EmodelProperties, SynapseProperty
from bluecellulab.circuit.node_id import CellId
from bluecellulab.circuit.simulation_access import get_synapse_replay_spikes
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.importer import load_mod_files
from bluecellulab.neuron_interpreter import eval_neuron
from bluecellulab.rngsettings import RNGSettings
from bluecellulab.stimulus.circuit_stimulus_definitions import SynapseReplay
from bluecellulab.synapse import SynapseFactory, Synapse
from bluecellulab.synapse.synapse_types import SynapseID
from bluecellulab.type_aliases import HocObjectType, NeuronSection, SectionMapping

logger = logging.getLogger(__name__)


class Cell(InjectableMixin, PlottableMixin):
    """Represents a Cell object."""

    last_id = 0

    @classmethod
    def from_template_parameters(
        cls, template_params: TemplateParams, cell_id: Optional[CellId] = None,
        record_dt: Optional[float] = None
    ) -> Cell:
        """Create a cell from a TemplateParams object.

        Useful in isolating runs.
        """
        return cls(
            template_path=template_params.template_filepath,
            morphology_path=template_params.morph_filepath,
            cell_id=cell_id,
            record_dt=record_dt,
            template_format=template_params.template_format,
            emodel_properties=template_params.emodel_properties,
        )

    @load_mod_files
    def __init__(self,
                 template_path: str | Path,
                 morphology_path: str | Path,
                 cell_id: Optional[CellId] = None,
                 record_dt: Optional[float] = None,
                 template_format: str = "v5",
                 emodel_properties: Optional[EmodelProperties] = None) -> None:
        """Initializes a Cell object.

        Args:
            template_path: Path to hoc template file.
            morphology_path: Path to morphology file.
            cell_id: ID of the cell, used in RNG seeds.
            record_dt: Timestep for the recordings.
            template_format: Cell template format such as 'v5' or 'v6_air_scaler'.
            emodel_properties: Template specific emodel properties.
        """
        super().__init__()
        self.template_params = TemplateParams(
            template_filepath=template_path,
            morph_filepath=morphology_path,
            template_format=template_format,
            emodel_properties=emodel_properties,
        )
        if cell_id is None:
            cell_id = CellId("", Cell.last_id)
            Cell.last_id += 1
        self.cell_id = cell_id

        # Load the template
        neuron_template = NeuronTemplate(template_path, morphology_path, template_format, emodel_properties)
        self.template_id = neuron_template.template_name  # useful to map NEURON and python objects
        self.cell = neuron_template.get_cell(self.cell_id.id)
        if template_format == 'v6':
            if emodel_properties is None:
                raise BluecellulabError('EmodelProperties must be provided for v6 template')
            self.hypamp: float | None = emodel_properties.holding_current
            self.threshold: float = emodel_properties.threshold_current
        else:
            try:
                self.hypamp = self.cell.getHypAmp()
            except AttributeError:
                self.hypamp = None
            try:
                self.threshold = self.cell.getThreshold()
            except AttributeError:
                self.threshold = 0.0
        self.soma = public_hoc_cell(self.cell).soma[0]
        # WARNING: this finitialize 'must' be here, otherwhise the
        # diameters of the loaded morph are wrong
        neuron.h.finitialize()

        self.recordings: dict[str, HocObjectType] = {}
        self.synapses: dict[SynapseID, Synapse] = {}
        self.connections: dict[SynapseID, bluecellulab.Connection] = {}

        self.ips: dict[SynapseID, HocObjectType] = {}
        self.syn_mini_netcons: dict[SynapseID, HocObjectType] = {}

        # Be careful when removing this,
        # time recording needs this push
        self.soma.push()
        self.hocname = neuron.h.secname(sec=self.soma).split(".")[0]
        self.record_dt = record_dt
        self.add_recordings(['self.soma(0.5)._ref_v', 'neuron.h._ref_t'],
                            dt=self.record_dt)

        self.delayed_weights = queue.PriorityQueue()  # type: ignore
        self.psections: dict[int, PSection] = {}
        self.secname_to_psection: dict[str, PSection] = {}

        # Keep track of when a cell is made passive by make_passive()
        # Used to know when re_init_rng() can be executed
        self.is_made_passive = False

        neuron.h.pop_section()  # Undoing soma push
        self.sonata_proxy: Optional[SonataProxy] = None

        # Persistent objects, like clamps, that exist as long
        # as the object exists
        self.persistent: list[HocObjectType] = []

    def _init_psections(self) -> None:
        """Initialize the psections of the cell."""
        if not self.psections:
            self.psections, self.secname_to_psection = init_psections(public_hoc_cell(self.cell))

    def _extract_sections(self, sections) -> SectionMapping:
        res: SectionMapping = {}
        for section in sections:
            key_name = str(section).split(".")[-1]
            res[key_name] = section
        return res

    @property
    def somatic(self) -> list[NeuronSection]:
        return list(public_hoc_cell(self.cell).somatic)

    @property
    def basal(self) -> list[NeuronSection]:
        return list(public_hoc_cell(self.cell).basal)

    @property
    def apical(self) -> list[NeuronSection]:
        return list(public_hoc_cell(self.cell).apical)

    @property
    def axonal(self) -> list[NeuronSection]:
        return list(public_hoc_cell(self.cell).axonal)

    @property
    def sections(self) -> SectionMapping:
        return self._extract_sections(public_hoc_cell(self.cell).all)

    def __repr__(self) -> str:
        base_info = f"Cell Object: {super().__repr__()}"
        hoc_info = f"NEURON ID: {self.template_id}"
        return f"{base_info}.\n{hoc_info}."

    def connect_to_circuit(self, sonata_proxy: SonataProxy) -> None:
        """Connect this cell to a circuit via sonata proxy."""
        self.sonata_proxy = sonata_proxy

    def re_init_rng(self) -> None:
        """Reinitialize the random number generator for stochastic channels."""
        if not self.is_made_passive:
            self.cell.re_init_rng()

    def get_psection(self, section_id: int | str) -> PSection:
        """Return a python section with the specified section id."""
        self._init_psections()
        if isinstance(section_id, int):
            return self.psections[section_id]
        elif isinstance(section_id, str):
            return self.secname_to_psection[section_id]
        else:
            raise BluecellulabError(
                f"Section id must be an int or a str, not {type(section_id)}"
            )

    def make_passive(self) -> None:
        """Make the cell passive by deactivating all the active channels."""
        for section in self.sections.values():
            mech_names = set()
            for seg in section:
                for mech in seg:
                    mech_names.add(mech.name())
            for mech_name in mech_names:
                if mech_name not in ["k_ion", "na_ion", "ca_ion", "pas",
                                     "ttx_ion"]:
                    neuron.h('uninsert %s' % mech_name, sec=section)
        self.is_made_passive = True

    def enable_ttx(self) -> None:
        """Add TTX to the environment (i.e. block the Na channels).

        Enable TTX by inserting TTXDynamicsSwitch and setting ttxo to
        1.0
        """
        if hasattr(public_hoc_cell(self.cell), 'enable_ttx'):
            public_hoc_cell(self.cell).enable_ttx()
        else:
            self._default_enable_ttx()

    def disable_ttx(self) -> None:
        """Remove TTX from the environment (i.e. unblock the Na channels).

        Disable TTX by inserting TTXDynamicsSwitch and setting ttxo to
        1e-14
        """
        if hasattr(public_hoc_cell(self.cell), 'disable_ttx'):
            public_hoc_cell(self.cell).disable_ttx()
        else:
            self._default_disable_ttx()

    def _default_enable_ttx(self) -> None:
        """Default enable_ttx implementation."""
        for section in self.sections.values():
            if not neuron.h.ismembrane("TTXDynamicsSwitch"):
                section.insert('TTXDynamicsSwitch')
            section.ttxo_level_TTXDynamicsSwitch = 1.0

    def _default_disable_ttx(self) -> None:
        """Default disable_ttx implementation."""
        for section in self.sections.values():
            if not neuron.h.ismembrane("TTXDynamicsSwitch"):
                section.insert('TTXDynamicsSwitch')
            section.ttxo_level_TTXDynamicsSwitch = 1e-14

    def area(self) -> float:
        """The total surface area of the cell."""
        area = 0.0
        for section in self.sections.values():
            x_s = np.arange(1.0 / (2 * section.nseg), 1.0,
                            1.0 / (section.nseg))
            for x in x_s:
                area += neuron.h.area(x, sec=section)
            # for segment in section:
            #    area += neuron.h.area(segment.x, sec=section)
        return area

    def add_recording(self, var_name: str, dt: Optional[float] = None) -> None:
        """Add a recording to the cell.

        Args:
            var_name: Variable to be recorded.
            dt: Recording time step. If not provided, the recording step will
            default to the simulator's time step.
        """
        recording = neuron.h.Vector()
        if dt:
            # This float_epsilon stuff is some magic from M. Hines to make
            # the time points fall exactly on the dts
            recording.record(
                eval_neuron(var_name, self=self, neuron=neuron),
                self.get_precise_record_dt(dt),
            )
        else:
            recording.record(eval_neuron(var_name, self=self, neuron=neuron))
        self.recordings[var_name] = recording

    @staticmethod
    def get_precise_record_dt(dt: float) -> float:
        """Get a more precise record_dt to make time points faill on dts."""
        return (1.0 + neuron.h.float_epsilon) / (1.0 / dt)

    def add_recordings(self, var_names: list[str], dt: Optional[float] = None) -> None:
        """Add a list of recordings to the cell.

        Args:
            var_names: Variables to be recorded.
            dt: Recording time step. If not provided, the recording step will
            default to the simulator's time step.
        """
        for var_name in var_names:
            self.add_recording(var_name, dt)

    def add_ais_recording(self, dt: Optional[float] = None) -> None:
        """Adds recording to AIS."""
        self.add_recording("self.axonal[1](0.5)._ref_v", dt=dt)

    def add_voltage_recording(
        self, section: Optional[NeuronSection] = None, segx: float = 0.5, dt: Optional[float] = None
    ) -> None:
        """Add a voltage recording to a certain section at a given segment
        (segx).

        Args:
            section: Section to record from (Neuron section pointer).
            segx: Segment x coordinate. Specify a value between 0 and 1.
                  0 is typically the end closest to the soma, 1 is the distal end.
            dt: Recording time step. If not provided, the recording step will
                default to the simulator's time step.
        """
        if section is None:
            section = self.soma
        var_name = section_to_voltage_recording_str(section, segx)
        self.add_recording(var_name, dt)

    def get_voltage_recording(
        self, section: Optional[NeuronSection] = None, segx: float = 0.5
    ) -> np.ndarray:
        """Get a voltage recording for a certain section at a given segment
        (segx).

        Args:
            section: Section to record from (Neuron section pointer).
            segx: Segment x coordinate. Specify a value between 0 and 1.
                  0 is typically the end closest to the soma, 1 is the distal end.

        Returns:
            A NumPy array containing the voltage recording values.

        Raises:
            BluecellulabError: If voltage recording was not added previously using add_voltage_recording.
        """
        if section is None:
            section = self.soma
        recording_name = section_to_voltage_recording_str(section, segx)
        if recording_name in self.recordings:
            return self.get_recording(recording_name)
        else:
            raise BluecellulabError(
                f"get_voltage_recording: Voltage recording {recording_name}"
                " was not added previously using add_voltage_recording"
            )

    def add_allsections_voltagerecordings(self):
        """Add a voltage recording to every section of the cell."""
        for section in self.sections.values():
            self.add_voltage_recording(section, dt=self.record_dt)

    def get_allsections_voltagerecordings(self) -> dict[str, np.ndarray]:
        """Get all the voltage recordings from all the sections."""
        all_section_voltages = {}
        for section in self.sections.values():
            recording = self.get_voltage_recording(section)
            all_section_voltages[section.name()] = recording
        return all_section_voltages

    def get_recording(self, var_name: str) -> np.ndarray:
        """Get recorded values."""
        return np.array(self.recordings[var_name].to_python())

    def add_replay_synapse(self,
                           synapse_id: SynapseID,
                           syn_description: pd.Series,
                           connection_modifiers: dict,
                           condition_parameters: Conditions,
                           popids: tuple[int, int],
                           extracellular_calcium: float | None) -> None:
        """Add synapse based on the syn_description to the cell."""
        synapse = SynapseFactory.create_synapse(
            cell=self,
            syn_id=synapse_id,
            syn_description=syn_description,
            condition_parameters=condition_parameters,
            popids=popids,
            extracellular_calcium=extracellular_calcium,
            connection_modifiers=connection_modifiers)

        self.synapses[synapse_id] = synapse

        logger.debug(f'Added synapse to cell {self.cell_id.id}')

    def add_replay_delayed_weight(
        self, sid: tuple[str, int], delay: float, weight: float
    ) -> None:
        """Add a synaptic weight for sid that will be set with a time delay."""
        self.delayed_weights.put((delay, (sid, weight)))

    def pre_gids(self) -> list[int]:
        """Get the list of unique gids of cells that connect to this cell.

        Returns:
            A list of gids of cells that connect to this cell.
        """
        pre_gids = {self.synapses[syn_id].pre_gid for syn_id in self.synapses}
        return list(pre_gids)

    def pre_gid_synapse_ids(self, pre_gid: int) -> list[SynapseID]:
        """List of synapse_ids of synapses a cell uses to connect to this cell.

        Args:
            pre_gid: gid of the presynaptic cell.

        Returns:
            synapse_id's that connect the presynaptic cell with this cell.
        """
        syn_id_list = []
        for syn_id in self.synapses:
            if self.synapses[syn_id].pre_gid == pre_gid:
                syn_id_list.append(syn_id)
        return syn_id_list

    def create_netcon_spikedetector(self, target: HocObjectType, location: str, threshold: float = -30.0) -> HocObjectType:
        """Add and return a spikedetector.

        This is a NetCon that detects spike in the current cell, and that
        connects to target

        Args:
            target: target point process
            location: the spike detection location
            threshold: spike detection threshold

        Returns: Neuron netcon object

        Raises:
            ValueError: If the spike detection location is not 'soma' or 'AIS'.
        """
        if location == "soma":
            sec = public_hoc_cell(self.cell).soma[0]
            source = sec(1)._ref_v
        elif location == "AIS":
            sec = public_hoc_cell(self.cell).axon[1]
            source = sec(0.5)._ref_v
        else:
            raise ValueError("Spike detection location must be soma or AIS")
        netcon = neuron.h.NetCon(source, target, sec=sec)
        netcon.threshold = threshold
        return netcon

    def start_recording_spikes(self, target: HocObjectType, location: str, threshold: float = -30) -> None:
        """Start recording spikes in the current cell.

        Args:
            target: target point process
            location: the spike detection location
            threshold: spike detection threshold
        """
        nc = self.create_netcon_spikedetector(target, location, threshold)
        spike_vec = neuron.h.Vector()
        nc.record(spike_vec)
        self.recordings[f"spike_detector_{location}_{threshold}"] = spike_vec

    def get_recorded_spikes(self, location: str, threshold: float = -30) -> list[float]:
        """Get recorded spikes in the current cell.

        Args:
            location: the spike detection location
            threshold: spike detection threshold

        Returns: recorded spikes
        """
        result = self.recordings[f"spike_detector_{location}_{threshold}"]
        return result.to_python()

    def add_replay_minis(self,
                         synapse_id: SynapseID,
                         syn_description: pd.Series,
                         connection_modifiers: dict,
                         popids: tuple[int, int],
                         mini_frequencies: tuple[float | None, float | None]) -> None:
        """Add minis from the replay."""
        source_popid, target_popid = popids

        sid = synapse_id[1]

        weight = syn_description[SynapseProperty.G_SYNX]
        # numpy int to int
        post_sec_id = int(syn_description[SynapseProperty.POST_SECTION_ID])

        weight_scalar = connection_modifiers.get('Weight', 1.0)
        exc_mini_frequency, inh_mini_frequency = mini_frequencies \
            if mini_frequencies is not None else (None, None)

        synapse = self.synapses[synapse_id]

        # SpontMinis in sim config takes precedence of values in nodes file
        if 'SpontMinis' in connection_modifiers:
            spont_minis_rate = connection_modifiers['SpontMinis']
        elif synapse.mech_name in ["GluSynapse", "ProbAMPANMDA_EMS"]:
            spont_minis_rate = exc_mini_frequency
        else:
            spont_minis_rate = inh_mini_frequency

        if spont_minis_rate is not None and spont_minis_rate > 0:
            synapse_hoc_args = SynapseFactory.determine_synapse_location(
                syn_description, self
            )
            # add the *minis*: spontaneous synaptic events
            self.ips[synapse_id] = neuron.h.\
                InhPoissonStim(synapse_hoc_args.location, sec=synapse_hoc_args.section)

            self.syn_mini_netcons[synapse_id] = neuron.h.\
                NetCon(self.ips[synapse_id], synapse.hsynapse, sec=synapse_hoc_args.section)
            self.syn_mini_netcons[synapse_id].delay = 0.1
            self.syn_mini_netcons[synapse_id].weight[0] = weight * weight_scalar
            # set netcon type
            nc_param_name = f'nc_type_param_{synapse.hsynapse}'.split('[')[0]
            if hasattr(neuron.h, nc_param_name):
                nc_type_param = int(getattr(neuron.h, nc_param_name))
                # NC_SPONTMINI
                self.syn_mini_netcons[synapse_id].weight[nc_type_param] = 1

            rng_settings = RNGSettings.get_instance()
            if rng_settings.mode == 'Random123':
                seed2 = source_popid * 65536 + target_popid \
                    + rng_settings.minis_seed
                self.ips[synapse_id].setRNGs(
                    sid + 200,
                    self.cell_id.id + 250,
                    seed2 + 300,
                    sid + 200,
                    self.cell_id.id + 250,
                    seed2 + 350)
            else:
                exprng = neuron.h.Random()
                self.persistent.append(exprng)

                uniformrng = neuron.h.Random()
                self.persistent.append(uniformrng)

                base_seed = rng_settings.base_seed
                if rng_settings.mode == 'Compatibility':
                    exp_seed1 = sid * 100000 + 200
                    exp_seed2 = self.cell_id.id + 250 + base_seed + \
                        rng_settings.minis_seed
                    uniform_seed1 = sid * 100000 + 300
                    uniform_seed2 = self.cell_id.id + 250 + base_seed + \
                        rng_settings.minis_seed
                elif rng_settings.mode == "UpdatedMCell":
                    exp_seed1 = sid * 1000 + 200
                    exp_seed2 = source_popid * 16777216 + self.cell_id.id + 250 + \
                        base_seed + \
                        rng_settings.minis_seed
                    uniform_seed1 = sid * 1000 + 300
                    uniform_seed2 = source_popid * 16777216 + self.cell_id.id + 250 \
                        + base_seed + \
                        rng_settings.minis_seed
                else:
                    raise ValueError(
                        f"Cell: Unknown rng mode: {rng_settings.mode}")

                exprng.MCellRan4(exp_seed1, exp_seed2)
                exprng.negexp(1.0)

                uniformrng.MCellRan4(uniform_seed1, uniform_seed2)
                uniformrng.uniform(0.0, 1.0)

                self.ips[synapse_id].setRNGs(exprng, uniformrng)

            tbins_vec = neuron.h.Vector(1)
            tbins_vec.x[0] = 0.0
            rate_vec = neuron.h.Vector(1)
            rate_vec.x[0] = spont_minis_rate
            self.persistent.append(tbins_vec)
            self.persistent.append(rate_vec)
            self.ips[synapse_id].setTbins(tbins_vec)
            self.ips[synapse_id].setRate(rate_vec)

    def get_childrensections(self, parentsection: HocObjectType) -> list[HocObjectType]:
        """Get the children section of a neuron section."""
        number_children = neuron.h.SectionRef(sec=parentsection).nchild()
        children = []
        for index in range(int(number_children)):
            children.append(neuron.h.SectionRef(sec=self.soma).child[index])
        return children

    @staticmethod
    def get_parentsection(childsection: HocObjectType) -> HocObjectType:
        """Get the parent section of a neuron section."""
        return neuron.h.SectionRef(sec=childsection).parent

    def addAxialCurrentRecordings(self, section):
        """Record all the axial current flowing in and out of the section."""
        secname = neuron.h.secname(sec=section)
        self.add_recording(secname)
        for child in self.get_childrensections(section):
            self.add_recording(child)
        self.get_parentsection(section)

    def getAxialCurrentRecording(self, section):
        """Return the axial current recording."""
        secname = neuron.h.secname(sec=section)
        for child in self.get_childrensections(section):
            self.get_recording(secname)
            self.get_recording(child)

    def somatic_branches(self) -> None:
        """Show the index numbers."""
        nchild = neuron.h.SectionRef(sec=self.soma).nchild()
        for index in range(int(nchild)):
            secname = neuron.h.secname(sec=neuron.h.SectionRef(
                sec=self.soma).child[index])
            if "axon" not in secname:
                if "dend" in secname:
                    dendnumber = int(
                        secname.split("dend")[1].split("[")[1].split("]")[0])
                    secnumber = int(public_hoc_cell(self.cell).nSecAxonalOrig +
                                    public_hoc_cell(self.cell).nSecSoma + dendnumber)
                elif "apic" in secname:
                    apicnumber = int(secname.split(
                        "apic")[1].split("[")[1].split("]")[0])
                    secnumber = int(public_hoc_cell(self.cell).nSecAxonalOrig +
                                    public_hoc_cell(self.cell).nSecSoma +
                                    public_hoc_cell(self.cell).nSecBasal + apicnumber)
                    logger.info((apicnumber, secnumber))
                else:
                    raise BluecellulabError(
                        f"somaticbranches: No apic or dend found in section {secname}"
                    )

    @staticmethod
    @deprecated("Use bluecellulab.cell.section_distance.EuclideanSectionDistance instead.")
    def euclid_section_distance(
            hsection1=None,
            hsection2=None,
            location1=None,
            location2=None,
            projection=None):
        """Calculate euclidian distance between positions on two sections Uses
        bluecellulab.cell.section_distance.EuclideanSectionDistance.

        Parameters
        ----------

        hsection1 : hoc section
                    First section
        hsection2 : hoc section
                    Second section
        location1 : float
                    range x along hsection1
        location2 : float
                    range x along hsection2
        projection : string
                     planes to project on, e.g. 'xy'
        """
        dist = EuclideanSectionDistance()
        return dist(hsection1, hsection2, location1, location2, projection)

    def apical_trunk(self):
        """Return the apical trunk of the cell."""
        if len(self.apical) == 0:
            return []
        else:
            apicaltrunk = []
            max_diam_section = self.apical[0]
            while True:
                apicaltrunk.append(max_diam_section)

                children = [
                    neuron.h.SectionRef(sec=max_diam_section).child[index]
                    for index in range(int(neuron.h.SectionRef(
                        sec=max_diam_section).nchild()))]
                if len(children) == 0:
                    break
                maxdiam = 0
                for child in children:
                    if child.diam > maxdiam:
                        max_diam_section = child
                        maxdiam = child.diam
            return apicaltrunk

    def get_time(self) -> np.ndarray:
        """Get the time vector."""
        return self.get_recording('neuron.h._ref_t')

    def get_soma_voltage(self) -> np.ndarray:
        """Get a vector of the soma voltage."""
        return self.get_recording('self.soma(0.5)._ref_v')

    def get_ais_voltage(self) -> np.ndarray:
        """Get a vector of AIS voltage."""
        return self.get_recording('self.axonal[1](0.5)._ref_v')

    @property
    def n_segments(self) -> int:
        """Get the number of segments in the cell."""
        return sum(section.nseg for section in self.sections.values())

    def add_synapse_replay(
        self, stimulus: SynapseReplay, spike_threshold: float, spike_location: str
    ) -> None:
        """Adds the synapse spike replay to the cell if the synapse is
        connected to that cell."""
        if self.sonata_proxy is None:
            raise BluecellulabError("Cell: add_synapse_replay requires a sonata proxy.")
        synapse_spikes: dict = get_synapse_replay_spikes(stimulus.spike_file)
        for synapse_id, synapse in self.synapses.items():
            source_population = synapse.syn_description["source_population_name"]
            pre_gid = CellId(
                source_population, int(synapse.syn_description[SynapseProperty.PRE_GID])
            )
            if pre_gid.id in synapse_spikes:
                spikes_of_interest = synapse_spikes[pre_gid.id]
                # filter spikes of interest >=stimulus.delay, <=stimulus.duration
                spikes_of_interest = spikes_of_interest[
                    (spikes_of_interest >= stimulus.delay)
                    & (spikes_of_interest <= stimulus.duration)
                ]
                connection = bluecellulab.Connection(
                    synapse,
                    pre_spiketrain=spikes_of_interest,
                    pre_cell=None,
                    stim_dt=self.record_dt,
                    spike_threshold=spike_threshold,
                    spike_location=spike_location,
                )
                logger.debug(
                    f"Added synapse replay from {pre_gid} to {self.cell_id.id}, {synapse_id}"
                )

                self.connections[synapse_id] = connection

    @property
    def info_dict(self):
        """Return a dictionary with all the information of this cell."""
        return {
            'synapses': {
                sid: synapse.info_dict for sid, synapse in self.synapses.items()
            },
            'connections': {
                sid: connection.info_dict for sid, connection in self.connections.items()
            }
        }

    def delete(self):
        """Delete the cell."""
        self.delete_plottable()
        if hasattr(self, 'cell') and self.cell is not None:
            if public_hoc_cell(self.cell) is not None and hasattr(public_hoc_cell(self.cell), 'clear'):
                public_hoc_cell(self.cell).clear()

            self.connections = None
            self.synapses = None

        if hasattr(self, 'recordings'):
            for recording in self.recordings:
                del recording

        if hasattr(self, 'persistent'):
            for persistent_object in self.persistent:
                del persistent_object

    def __del__(self):
        self.delete()
