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
"""Cell class."""

from __future__ import annotations

import logging

import json
from pathlib import Path
import queue
from typing import Any, Optional

import numpy as np

import bluecellulab
from bluecellulab import neuron, psection, tools
from bluecellulab.cell.injector import InjectableMixin
from bluecellulab.cell.plotting import PlottableMixin
from bluecellulab.cell.section_distance import EuclideanSectionDistance
from bluecellulab.cell.sonata_proxy import SonataProxy
from bluecellulab.cell.serialized_sections import SerializedSections
from bluecellulab.cell.template import NeuronTemplate
from bluecellulab.circuit.config.sections import Conditions
from bluecellulab.circuit import EmodelProperties, SynapseProperty
from bluecellulab.circuit.node_id import CellId
from bluecellulab.circuit.simulation_access import get_synapse_replay_spikes
from bluecellulab.exceptions import BluecellulabError
from bluecellulab.neuron_interpreter import eval_neuron
from bluecellulab.stimuli import SynapseReplay
from bluecellulab.synapse import SynapseFactory, Synapse

logger = logging.getLogger(__name__)

NeuronType = Any


class Cell(InjectableMixin, PlottableMixin):
    """Represents a BGLib Cell object."""

    def __init__(self, template_path: str | Path, morphology_path: str | Path,
                 gid=0, record_dt=None, template_format="v5",
                 emodel_properties: Optional[EmodelProperties] = None,
                 rng_settings=None):
        """Constructor.

        Parameters
        ----------
        template_path : Full path to BGLib template to be loaded
        morphology_path : path to morphology file
        gid : integer
             GID of the instantiated cell (default: 0)
        record_dt : float
                   Force a different timestep for the recordings
                   (default: None)
        template_format: str
                         cell template format such as v6 or v6_air_scaler.
        emodel_properties: properties such as threshold_current, holding_current
        rng_settings: bluecellulab.RNGSettings
                      random number generation setting object used by the Cell.
        """
        super().__init__()
        # Persistent objects, like clamps, that exist as long
        # as the object exists
        self.persistent: list[NeuronType] = []

        self.morphology_path = morphology_path

        # Load the template
        neuron_template = NeuronTemplate(template_path, morphology_path)
        self.cell = neuron_template.get_cell(template_format, gid, emodel_properties)

        self.soma = [x for x in self.cell.getCell().somatic][0]
        # WARNING: this finitialize 'must' be here, otherwhise the
        # diameters of the loaded morph are wrong
        neuron.h.finitialize()

        self.cellname = neuron.h.secname(sec=self.soma).split(".")[0]

        # Set the gid of the cell
        self.cell.getCell().gid = gid
        self.gid = gid

        self.rng_settings = rng_settings

        self.recordings: dict[str, NeuronType] = {}
        self.synapses: dict[int, Synapse] = {}
        self.connections: dict[int, bluecellulab.Connection] = {}

        self.ips: dict[int, NeuronType] = {}
        self.syn_mini_netcons: dict[int, NeuronType] = {}
        self.serialized = None

        # Be careful when removing this,
        # time recording needs this push
        self.soma.push()
        self.hocname = neuron.h.secname(sec=self.soma).split(".")[0]
        self.somatic = [x for x in self.cell.getCell().somatic]
        self.basal = [x for x in self.cell.getCell().basal]
        self.apical = [x for x in self.cell.getCell().apical]
        self.axonal = [x for x in self.cell.getCell().axonal]
        self.all = [x for x in self.cell.getCell().all]
        self.record_dt = record_dt
        self.add_recordings(['self.soma(0.5)._ref_v', 'neuron.h._ref_t'],
                            dt=self.record_dt)

        self.delayed_weights = queue.PriorityQueue()  # type: ignore
        self.secname_to_isec: dict[str, int] = {}
        self.secname_to_hsection: dict[str, NeuronType] = {}
        self.secname_to_psection: dict[str, psection.PSection] = {}

        self.emodel_properties = emodel_properties
        if template_format in ['v6', 'v6_ais_scaler']:
            if self.emodel_properties is None:
                raise BluecellulabError('EmodelProperties must be provided for v6 template')
            self.hypamp: float | None = self.emodel_properties.holding_current
            self.threshold: float | None = self.emodel_properties.threshold_current
        else:
            try:
                self.hypamp = self.cell.getHypAmp()
            except AttributeError:
                self.hypamp = None

            try:
                self.threshold = self.cell.getThreshold()
            except AttributeError:
                self.threshold = None

        # Keep track of when a cell is made passive by make_passive()
        # Used to know when re_init_rng() can be executed
        self.is_made_passive = False

        self.psections: dict[int, psection.PSection] = {}

        neuron.h.pop_section()  # Undoing soma push
        # self.init_psections()
        self.sonata_proxy: Optional[SonataProxy] = None

    def connect_to_circuit(self, sonata_proxy: SonataProxy) -> None:
        """Connect this cell to a circuit via sonata proxy."""
        self.sonata_proxy = sonata_proxy

    def init_psections(self):
        """Initialize the psections list.

        This list contains the Python representation of the psections of
        this morphology.
        """
        for hsection in self.all:
            secname = neuron.h.secname(sec=hsection)
            self.secname_to_hsection[secname] = hsection
            self.secname_to_psection[secname] = psection.PSection(hsection)

        # section are not serialized yet, do it now
        if self.serialized is None:
            self.serialized = SerializedSections(self.cell.getCell())

        for isec in self.serialized.isec2sec:
            hsection = self.get_hsection(isec)
            if hsection:
                secname = neuron.h.secname(sec=hsection)
                self.psections[isec] = self.secname_to_psection[secname]
                self.psections[isec].isec = isec
                self.secname_to_isec[secname] = isec

        # Set the parents and children of all the psections
        for psec in self.psections.values():
            hparent = psec.hparent
            if hparent:
                parentname = neuron.h.secname(sec=hparent)
                psec.pparent = self.get_psection(secname=parentname)
            else:
                psec.pparent = None

            for hchild in psec.hchildren:
                childname = neuron.h.secname(sec=hchild)
                pchild = self.get_psection(secname=childname)
                psec.add_pchild(pchild)

    def get_section_id(self, secname=None):
        """Get section based on section id.

        Returns
        -------
        integer: section id
                 section id of the section with name secname
        """
        return self.secname_to_psection[secname].section_id

    def re_init_rng(self, use_random123_stochkv=None):
        """Reinitialize the random number generator for stochastic channels."""

        if not self.is_made_passive:
            if use_random123_stochkv:
                channel_id = 0
                for section in self.somatic:
                    for seg in section:
                        neuron.h.setdata_StochKv(seg.x, sec=section)
                        neuron.h.setRNG_StochKv(channel_id, self.gid)
                        channel_id += 1
                for section in self.basal:
                    for seg in section:
                        neuron.h.setdata_StochKv(seg.x, sec=section)
                        neuron.h.setRNG_StochKv(channel_id, self.gid)
                        channel_id += 1
                for section in self.apical:
                    for seg in section:
                        neuron.h.setdata_StochKv(seg.x, sec=section)
                        neuron.h.setRNG_StochKv(channel_id, self.gid)
                        channel_id += 1
            else:
                self.cell.re_init_rng()

    def get_psection(self, section_id=None, secname=None):
        """Return a python section with the specified section id or name.

        Parameters
        ----------
        section_id: int
                    Return the PSection object based on section id
        secname: string
                 Return the PSection object based on section name

        Returns
        -------
        psection: PSection
                  PSection object of the specified section id or name
        """
        if section_id is not None:
            return self.psections[section_id]
        elif secname is not None:
            return self.secname_to_psection[secname]
        else:
            raise Exception(
                "Cell: get_psection requires or a section_id or a secname")

    def get_hsection(self, section_id):
        """Use the serialized object to find a hoc section from a section id.

        Parameters
        ----------
        section_id : int
                    Section id

        Returns
        -------
        hsection : nrnSection
                   The requested hoc section
        """

        # section are not serialized yet, do it now
        if self.serialized is None:
            self.serialized = SerializedSections(self.cell.getCell())

        try:
            sec_ref = self.serialized.isec2sec[int(section_id)]
        except IndexError as e:
            raise IndexError(
                "bluecellulab get_hsection: section-id %s not found in %s" %
                (section_id, self.morphology_path)) from e
        if sec_ref is not None:
            return self.serialized.isec2sec[int(section_id)].sec
        else:
            return None

    def make_passive(self):
        """Make the cell passive by deactivating all the active channels."""

        for section in self.all:
            mech_names = set()
            for seg in section:
                for mech in seg:
                    mech_names.add(mech.name())
            for mech_name in mech_names:
                if mech_name not in ["k_ion", "na_ion", "ca_ion", "pas",
                                     "ttx_ion"]:
                    neuron.h('uninsert %s' % mech_name, sec=section)
        self.is_made_passive = True

    def enable_ttx(self):
        """Add TTX to the bath (i.e. block the Na channels)"""

        if hasattr(self.cell.getCell(), 'enable_ttx'):
            self.cell.getCell().enable_ttx()
        else:
            self._default_enable_ttx()

    def disable_ttx(self):
        """Add TTX to the bath (i.e. block the Na channels)"""

        if hasattr(self.cell.getCell(), 'disable_ttx'):
            self.cell.getCell().disable_ttx()
        else:
            self._default_disable_ttx()

    def _default_enable_ttx(self):
        """Default enable_ttx implementation."""

        for section in self.all:
            if not neuron.h.ismembrane("TTXDynamicsSwitch"):
                section.insert('TTXDynamicsSwitch')
            section.ttxo_level_TTXDynamicsSwitch = 1.0

    def _default_disable_ttx(self):
        """Default disable_ttx implementation."""

        for section in self.all:
            if not neuron.h.ismembrane("TTXDynamicsSwitch"):
                section.insert('TTXDynamicsSwitch')
            section.ttxo_level_TTXDynamicsSwitch = 1e-14

    def area(self):
        """Calculate the total area of the cell.

        Parameters
        ----------


        Returns
        -------
        area : float
               Total surface area of the cell
        """
        # pylint: disable=C0103
        area = 0
        for section in self.all:
            x_s = np.arange(1.0 / (2 * section.nseg), 1.0,
                            1.0 / (section.nseg))
            for x in x_s:
                area += bluecellulab.neuron.h.area(x, sec=section)
            # for segment in section:
            #    area += bluecellulab.neuron.h.area(segment.x, sec=section)
        return area

    def synlocation_to_segx(self, isec, ipt, syn_offset) -> float:
        """Translate a synaptic (secid, ipt, offset) to a x coordinate.

        Parameters
        ----------
        isec : integer
               section id
        ipt : float
              ipt
        syn_offset : float
                     Synaptic offset

        Returns
        -------
        x : float
            The x coordinate on section with secid, where the synapse
            can be placed
        """
        if syn_offset < 0.0:
            syn_offset = 0.0

        curr_sec = self.get_hsection(isec)
        if curr_sec is None:
            raise Exception(
                "No section found at isec=%d in gid %d" %
                (isec, self.gid))
        length = curr_sec.L

        # access section to compute the distance
        if neuron.h.section_orientation(sec=self.get_hsection(isec)) == 1:
            ipt = neuron.h.n3d(sec=self.get_hsection(isec)) - 1 - ipt
            syn_offset = -syn_offset

        distance = 0.5
        if ipt < neuron.h.n3d(sec=self.get_hsection(isec)):
            distance = (neuron.h.arc3d(ipt, sec=self.get_hsection(isec)) +
                        syn_offset) / length
            if distance == 0.0:
                distance = 0.0000001
            if distance >= 1.0:
                distance = 0.9999999

        if neuron.h.section_orientation(sec=self.get_hsection(isec)) == 1:
            distance = 1 - distance

        if distance < 0:
            logger.warning(f"synlocation_to_segx found negative distance \
                        at curr_sec({neuron.h.secname(sec=curr_sec)}) syn_offset: {syn_offset}")
            return 0
        else:
            return distance

    # pylint: disable=C0103
    def add_recording(self, var_name, dt=None):
        """Add a recording to the cell.

        Parameters
        ----------
        var_name : string
                   Variable to be recorded
        dt : float
             Recording time step
        """

        recording = neuron.h.Vector()
        if dt:
            # This float_epsilon stuff is some magic from M. Hines to make
            # the time points fall exactly on the dts
            recording.record(
                eval_neuron(var_name, self=self, neuron=bluecellulab.neuron),
                self.get_precise_record_dt(dt),
            )
        else:
            recording.record(eval_neuron(var_name, self=self, neuron=bluecellulab.neuron))
        self.recordings[var_name] = recording

    @staticmethod
    def get_precise_record_dt(dt):
        """Get a more precise record_dt to make time points faill on dts."""
        return (1.0 + neuron.h.float_epsilon) / (1.0 / dt)

    def add_recordings(self, var_names, dt=None):
        """Add a list of recordings to the cell.

        Parameters
        ----------
        var_names : list of strings
                    Variables to be recorded
        dt : float
             Recording time step
        """

        for var_name in var_names:
            self.add_recording(var_name, dt)

    def add_ais_recording(self, dt=None):
        """Adds recording to AIS."""
        self.add_recording("self.axonal[1](0.5)._ref_v", dt=dt)

    def add_voltage_recording(self, section, segx, dt: Optional[float] = None):
        """Add a voltage recording to a certain section(segx)

        Parameters
        ----------
        section : nrnSection
                  Section to record from (Neuron section pointer)
        segx : float
               Segment x coordinate
        dt: recording time step
        """
        var_name = f"neuron.h.{section.name()}({segx})._ref_v"
        self.add_recording(var_name, dt)

    def get_voltage_recording(self, section, segx: float) -> np.ndarray:
        """Get a voltage recording for a certain section(segx)

        Parameters
        ----------
        section : nrnSection
                  Section to record from (Neuron section pointer)
        segx :
               Segment x coordinate
        """

        recording_name = f"neuron.h.{section.name()}({segx})._ref_v"
        if recording_name in self.recordings:
            return self.get_recording(recording_name)
        else:
            raise BluecellulabError('get_voltage_recording: Voltage recording %s'
                                    ' was not added previously using '
                                    'add_voltage_recording' % recording_name)

    def add_allsections_voltagerecordings(self):
        """Add a voltage recording to every section of the cell."""
        all_sections = self.cell.getCell().all
        for section in all_sections:
            self.add_voltage_recording(section, segx=0.5, dt=self.record_dt)

    def get_allsections_voltagerecordings(self) -> dict[str, np.ndarray]:
        """Get all the voltage recordings from all the sections."""
        all_section_voltages = {}
        all_sections = self.cell.getCell().all
        for section in all_sections:
            recording = self.get_voltage_recording(section, segx=0.5)
            all_section_voltages[section.name()] = recording
        return all_section_voltages

    def get_recording(self, var_name: str) -> np.ndarray:
        """Get recorded values."""
        return np.array(self.recordings[var_name].to_python())

    def add_replay_synapse(self, synapse_id, syn_description, connection_modifiers,
                           condition_parameters=None, base_seed=None,
                           popids=(0, 0), extracellular_calcium=None):
        """Add synapse based on the syn_description to the cell.

        This operation can fail.  Returns True on success, otherwise
        False.
        """
        if condition_parameters is None:
            condition_parameters = Conditions.init_empty()
        isec = syn_description[SynapseProperty.POST_SECTION_ID]

        # old circuits don't have it, it needs to be computed via synlocation_to_segx
        if ("afferent_section_pos" in syn_description and
                not np.isnan(syn_description["afferent_section_pos"])):
            # position is pre computed in SONATA
            location = syn_description["afferent_section_pos"]
        else:
            ipt = syn_description[SynapseProperty.POST_SEGMENT_ID]
            syn_offset = syn_description[SynapseProperty.POST_SEGMENT_OFFSET]
            location = self.synlocation_to_segx(isec, ipt, syn_offset)

        if location is None:
            logger.warning("add_single_synapse: skipping a synapse at \
                            isec %d" % (isec))
            return False

        synapse = SynapseFactory.create_synapse(
            cell=self,
            location=location,
            syn_id=synapse_id,
            syn_description=syn_description,
            condition_parameters=condition_parameters,
            base_seed=base_seed,
            popids=popids,
            extracellular_calcium=extracellular_calcium,
            connection_modifiers=connection_modifiers)

        self.synapses[synapse_id] = synapse

        logger.debug('Added synapse to cell {gid}: {s_info_dict}', gid=self.gid,
                     s_info_dict=json.dumps(synapse.info_dict, cls=tools.NumpyEncoder))

        return True

    def add_replay_delayed_weight(self, sid: int, delay: float, weight: float) -> None:
        """Add a synaptic weight for sid that will be set with a time delay."""
        self.delayed_weights.put((delay, (sid, weight)))

    def pre_gids(self):
        """List of unique gids of cells that connect to this cell.

        Returns
        -------
        A list of gids of cells that connect to this cell.
        """
        pre_gids = {self.synapses[syn_id].pre_gid for syn_id in self.synapses}
        return list(pre_gids)

    def pre_gid_synapse_ids(self, pre_gid):
        """List of synapse_ids of synapses a cell uses to connect to this cell.

        Parameters
        ----------
        pre_gid : int
                  gid of the presynaptic cell

        Returns
        -------
        A list of the synapse_id's that connect the presynaptic cell with
        this cell.
        In case there are no such synapses because the cells e.g. are not
        connected, an empty list is returned.
        The synapse_id's can be used in the 'synapse' dictionary of this cell
        to return the Synapse objects
        """

        syn_id_list = []
        for syn_id in self.synapses:
            if self.synapses[syn_id].pre_gid == pre_gid:
                syn_id_list.append(syn_id)

        return syn_id_list

    def create_netcon_spikedetector(self, target, location: str, threshold: float = -30.0):
        """Add and return a spikedetector.

        This is a NetCon that detects spike in the current cell, and that
        connects to target

        Args:
            target: target point process
            location: the spike detection location
            threshold: spike detection threshold

        Returns: Neuron netcon object
        """
        if location == "soma":
            sec = self.cell.getCell().soma[0]
            source = self.cell.getCell().soma[0](1)._ref_v
        elif location == "AIS":
            sec = self.cell.getCell().axon[1]
            source = self.cell.getCell().axon[1](0.5)._ref_v
        else:
            raise Exception("Spike detection location must be soma or AIS")
        netcon = bluecellulab.neuron.h.NetCon(source, target, sec=sec)
        netcon.threshold = threshold

        return netcon

    def start_recording_spikes(self, target, location: str, threshold: float = -30):
        """Start recording spikes in the current cell.

        Args:
            target: target point process
            location: the spike detection location
            threshold: spike detection threshold
        """
        nc = self.create_netcon_spikedetector(target, location, threshold)
        spike_vec = bluecellulab.neuron.h.Vector()
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

    def add_replay_minis(self, syn_id, syn_description, connection_parameters,
                         base_seed=None, popids=(0, 0), mini_frequencies=None):
        """Add minis from the replay."""

        source_popid, target_popid = popids

        sid = syn_id[1]

        if base_seed is None:
            base_seed = self.rng_settings.base_seed
        weight = syn_description[SynapseProperty.G_SYNX]
        post_sec_id = syn_description[SynapseProperty.POST_SECTION_ID]
        if "afferent_section_pos" in syn_description:
            location = syn_description["afferent_section_pos"]  # position is pre computed in SONATA
        else:
            post_seg_distance = syn_description[SynapseProperty.POST_SEGMENT_OFFSET]
            post_seg_id = syn_description[SynapseProperty.POST_SEGMENT_ID]
            location = self.\
                synlocation_to_segx(post_sec_id, post_seg_id,
                                    post_seg_distance)
        # todo: False
        if 'Weight' in connection_parameters:
            weight_scalar = connection_parameters['Weight']
        else:
            weight_scalar = 1.0

        exc_mini_frequency, inh_mini_frequency = mini_frequencies \
            if mini_frequencies is not None else (None, None)

        synapse = self.synapses[syn_id]

        # SpontMinis in sim config takes precedence of values in nodes file
        if 'SpontMinis' in connection_parameters:
            spont_minis_rate = connection_parameters['SpontMinis']
        else:
            if synapse.mech_name in ["GluSynapse", "ProbAMPANMDA_EMS"]:
                spont_minis_rate = exc_mini_frequency
            else:
                spont_minis_rate = inh_mini_frequency

        if spont_minis_rate is not None and spont_minis_rate > 0:
            sec = self.get_hsection(post_sec_id)
            # add the *minis*: spontaneous synaptic events
            self.ips[syn_id] = bluecellulab.neuron.h.\
                InhPoissonStim(location, sec=sec)

            self.syn_mini_netcons[syn_id] = bluecellulab.neuron.h.\
                NetCon(self.ips[syn_id], synapse.hsynapse, sec=sec)
            self.syn_mini_netcons[syn_id].delay = 0.1
            self.syn_mini_netcons[syn_id].weight[0] = weight * weight_scalar
            # set netcon type
            nc_param_name = 'nc_type_param_{}'.format(
                synapse.hsynapse).split('[')[0]
            if hasattr(bluecellulab.neuron.h, nc_param_name):
                nc_type_param = int(getattr(bluecellulab.neuron.h, nc_param_name))
                # NC_SPONTMINI
                self.syn_mini_netcons[syn_id].weight[nc_type_param] = 1

            if self.rng_settings.mode == 'Random123':
                seed2 = source_popid * 65536 + target_popid \
                    + self.rng_settings.minis_seed
                self.ips[syn_id].setRNGs(
                    sid + 200,
                    self.gid + 250,
                    seed2 + 300,
                    sid + 200,
                    self.gid + 250,
                    seed2 + 350)
            else:
                exprng = bluecellulab.neuron.h.Random()
                self.persistent.append(exprng)

                uniformrng = bluecellulab.neuron.h.Random()
                self.persistent.append(uniformrng)

                if self.rng_settings.mode == 'Compatibility':
                    exp_seed1 = sid * 100000 + 200
                    exp_seed2 = self.gid + 250 + base_seed + \
                        self.rng_settings.minis_seed
                    uniform_seed1 = sid * 100000 + 300
                    uniform_seed2 = self.gid + 250 + base_seed + \
                        self.rng_settings.minis_seed
                elif self.rng_settings.mode == "UpdatedMCell":
                    exp_seed1 = sid * 1000 + 200
                    exp_seed2 = source_popid * 16777216 + self.gid + 250 + \
                        base_seed + \
                        self.rng_settings.minis_seed
                    uniform_seed1 = sid * 1000 + 300
                    uniform_seed2 = source_popid * 16777216 + self.gid + 250 \
                        + base_seed + \
                        self.rng_settings.minis_seed
                else:
                    raise ValueError(
                        f"Cell: Unknown rng mode: {self.rng_settings.mode}")

                exprng.MCellRan4(exp_seed1, exp_seed2)
                exprng.negexp(1.0)

                uniformrng.MCellRan4(uniform_seed1, uniform_seed2)
                uniformrng.uniform(0.0, 1.0)

                self.ips[syn_id].setRNGs(exprng, uniformrng)

            tbins_vec = bluecellulab.neuron.h.Vector(1)
            tbins_vec.x[0] = 0.0
            rate_vec = bluecellulab.neuron.h.Vector(1)
            rate_vec.x[0] = spont_minis_rate
            self.persistent.append(tbins_vec)
            self.persistent.append(rate_vec)
            self.ips[syn_id].setTbins(tbins_vec)
            self.ips[syn_id].setRate(rate_vec)

    def initialize_synapses(self):
        """Initialize the synapses."""
        for synapse in self.synapses.values():
            syn = synapse.hsynapse
            syn_type = syn.hname().partition('[')[0]
            # todo: Is there no way to call the mod file's INITIAL block?
            # ... and do away with this brittle mess
            assert syn_type in ['ProbAMPANMDA_EMS', 'ProbGABAAB_EMS']
            if syn_type == 'ProbAMPANMDA_EMS':
                # basically what's in the INITIAL block
                syn.Rstate = 1
                syn.tsyn_fac = bluecellulab.neuron.h.t
                syn.u = syn.u0
                syn.A_AMPA = 0
                syn.B_AMPA = 0
                syn.A_NMDA = 0
                syn.B_NMDA = 0
            elif syn_type == 'ProbGABAAB_EMS':
                syn.Rstate = 1
                syn.tsyn_fac = bluecellulab.neuron.h.t
                syn.u = syn.u0
                syn.A_GABAA = 0
                syn.B_GABAA = 0
                syn.A_GABAB = 0
                syn.B_GABAB = 0
            else:
                assert False, "Problem with initialize_synapse"

    def locate_bapsite(self, seclist_name, distance):
        """Return the location of the BAP site.

        Parameters
        ----------

        seclist_name : str
            SectionList to search in
        distance : float
            Distance from soma

        Returns
        -------

        list of sections at the specified distance from the soma
        """
        return [x for x in self.cell.getCell().locateBAPSite(seclist_name,
                                                             distance)]

    def get_childrensections(self, parentsection):
        """Get the children section of a neuron section.

        Returns
        -------

        list of sections : child sections of the specified parent section
        """
        number_children = neuron.h.SectionRef(sec=parentsection).nchild()
        children = []
        for index in range(0, int(number_children)):
            children.append(neuron.h.SectionRef(sec=self.soma).child[index])
        return children

    @staticmethod
    def get_parentsection(childsection):
        """Get the parent section of a neuron section.

        Returns
        -------

        section : parent section of the specified child section
        """
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

    def somatic_branches(self):
        """Show the index numbers."""
        nchild = neuron.h.SectionRef(sec=self.soma).nchild()
        for index in range(0, int(nchild)):
            secname = neuron.h.secname(sec=neuron.h.SectionRef(
                sec=self.soma).child[index])
            if "axon" not in secname:
                if "dend" in secname:
                    dendnumber = int(
                        secname.split("dend")[1].split("[")[1].split("]")[0])
                    secnumber = int(self.cell.getCell().nSecAxonalOrig +
                                    self.cell.getCell().nSecSoma + dendnumber)
                elif "apic" in secname:
                    apicnumber = int(secname.split(
                        "apic")[1].split("[")[1].split("]")[0])
                    secnumber = int(self.cell.getCell().nSecAxonalOrig +
                                    self.cell.getCell().nSecSoma +
                                    self.cell.getCell().nSecBasal + apicnumber)
                    logger.info((apicnumber, secnumber))
                else:
                    raise Exception(
                        "somaticbranches: No apic or \
                                dend found in section %s" % secname)

    @staticmethod
    @tools.deprecated("bluecellulab.cell.section_distance.EuclideanSectionDistance")
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
                    for index in range(0, int(neuron.h.SectionRef(
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

    def getNumberOfSegments(self) -> int:
        """Get the number of segments in the cell."""
        return sum(section.nseg for section in self.all)

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
            if self.sonata_proxy.circuit_access.target_contains_cell(stimulus.source, pre_gid):
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
                        f"Added synapse replay from {pre_gid} to {self.gid}, {synapse_id}"
                    )

                    self.connections[synapse_id] = connection

    @property
    def info_dict(self):
        """Return a dictionary with all the information of this cell."""

        cell_info = {}

        cell_info['synapses'] = {}
        for sid, synapse in self.synapses.items():
            cell_info['synapses'][sid] = synapse.info_dict

        cell_info['connections'] = {}
        for sid, connection in self.connections.items():
            cell_info['connections'][sid] = connection.info_dict

        return cell_info

    def delete(self):
        """Delete the cell."""
        self.delete_plottable()
        if hasattr(self, 'cell') and self.cell is not None:
            if self.cell.getCell() is not None and hasattr(self.cell.getCell(), 'clear'):
                self.cell.getCell().clear()

            self.connections = None
            self.synapses = None

        if hasattr(self, 'recordings'):
            for recording in self.recordings:
                del recording

        if hasattr(self, 'persistent'):
            for persistent_object in self.persistent:
                del persistent_object

    @property
    def hsynapses(self):
        """Contains a dictionary of all the hoc synapses in the cell with as
        key the gid."""
        return {gid: synapse.hsynapse for (gid, synapse) in self.synapses.items()}

    def __del__(self):
        self.delete()
