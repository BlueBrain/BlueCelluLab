# -*- coding: utf-8 -*- #pylint: disable=C0302, W0123

"""
Cell class

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

# pylint: disable=F0401, R0915, R0914

# WARNING: I am ignoring pylint warnings which don't allow one to use eval()
# This might be a possible security risk, but in this specific case,
# avoiding eval() is not trivial at all, due to Neuron's complex attributes
# Since importing the neuron module is already a big security risk on it's
# own, I'm ignoring this warning for the moment

import os

import queue
import json

import numpy as np

from bluepy.enums import Synapse as BLPSynapse

import bglibpy
from bglibpy import tools
from bglibpy.importer import neuron
from bglibpy import psection
from bglibpy import printv
from bglibpy.cell.template import NeuronTemplate
from bglibpy.cell.section_distance import EuclideanSectionDistance
from bglibpy.cell.injector import InjectableMixin


class Cell(InjectableMixin):
    """Represents a BGLib Cell object."""

    def __init__(self, template_filename, morphology_name,
                 gid=0, record_dt=None, template_format=None, morph_dir=None,
                 extra_values=None, rng_settings=None):
        """ Constructor.

        Parameters
        ----------
        template_filename : string
                        Full path to BGLib template to be loaded
        morphology_name : string
                          Morphology name passed to the BGLib template
                          When the argument ends '.asc', that specific morph
                          will be loaded otherwise this argument is
                          interpreted as the directory containing the
                          morphologies
        gid : integer
             GID of the instantiated cell (default: 0)
        record_dt : float
                   Force a different timestep for the recordings
                   (default: None)
        template_format: str
                         cell template format such as v6 or v6_air_scaler.
        morph_dir: str
                   path to the directory containing morphology file.
        extra_values: dict
                      any extra values such as threshold_current
                       or holding_current.
        rng_settings: bglibpy.RNGSettings
                      random number generation setting object used by the Cell.
        """
        # Persistent objects, like clamps, that exist as long
        # as the object exists
        self.persistent = []

        if not os.path.exists(template_filename):
            raise FileNotFoundError("Couldn't find template file [%s]"
                                    % template_filename)

        # Load the template
        self.template_name = NeuronTemplate.load(template_filename)

        if template_format == 'v6':
            attr_names = getattr(neuron.h, self.template_name + "_NeededAttributes", None)
            if attr_names is not None:
                self.cell = getattr(
                    neuron.h,
                    self.template_name)(
                    gid,
                    morph_dir,
                    morphology_name,
                    *[extra_values[name] for name in attr_names.split(";")])

            self.cell = getattr(
                neuron.h,
                self.template_name)(
                gid,
                morph_dir,
                morphology_name)
        elif template_format == 'v6_ais_scaler':
            self.cell = getattr(
                neuron.h,
                self.template_name)(
                gid,
                morph_dir,
                morphology_name,
                extra_values['AIS_scaler'])
        else:
            self.cell = getattr(
                neuron.h,
                self.template_name)(
                gid,
                morphology_name)

        self.soma = [x for x in self.cell.getCell().somatic][0]
        # WARNING: this finitialize 'must' be here, otherwhise the
        # diameters of the loaded morph are wrong
        neuron.h.finitialize()

        self.morphology_name = morphology_name
        self.cellname = neuron.h.secname(sec=self.soma).split(".")[0]

        # Set the gid of the cell
        self.cell.getCell().gid = gid
        self.gid = gid

        self.rng_settings = rng_settings

        self.recordings = {}  # Recordings in this cell
        self.voltage_recordings = {}  # Voltage recordings in this cell
        self.synapses = {}  # Synapses on this cell
        self.netstims = {}  # Netstims connected to this cell
        self.connections = {}  # Outside connections to this cell

        self.pre_spiketrains = {}
        self.ips = {}
        self.syn_mini_netcons = {}
        self.serialized = None

        self.soma = [x for x in self.cell.getCell().somatic][0]
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
        self.cell_dendrograms = []
        self.plot_windows = []

        self.fih_plots = None
        self.fih_weights = None

        # As long as no PlotWindow or active Dendrogram exist, don't update
        self.plot_callback_necessary = False
        self.delayed_weights = queue.PriorityQueue()
        self.secname_to_isec = {}
        self.secname_to_hsection = {}
        self.secname_to_psection = {}

        self.extra_values = extra_values
        if template_format == 'v6':
            self.hypamp = self.extra_values['holding_current']
            self.threshold = self.extra_values['threshold_current']
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

        self.psections = {}

        neuron.h.pop_section()  # Undoing soma push
        # self.init_psections()

    def init_psections(self):
        """Initialize the psections list.

        This list contains the Python representation of the psections
        of this morphology.

        """
        for hsection in self.all:
            secname = neuron.h.secname(sec=hsection)
            self.secname_to_hsection[secname] = hsection
            self.secname_to_psection[secname] = psection.PSection(hsection)

        max_isec = int(self.cell.getCell().nSecAll)
        for isec in range(0, max_isec):
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
            self.serialized = neuron.h.SerializedSections(self.cell.getCell())

        try:
            sec_ref = self.serialized.isec2sec[int(section_id)]
        except IndexError as e:
            raise IndexError(
                "BGLibPy get_hsection: section-id %s not found in %s" %
                (section_id, self.morphology_name)) from e
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
        """Default enable_ttx implementation"""

        for section in self.all:
            if not neuron.h.ismembrane("TTXDynamicsSwitch"):
                section.insert('TTXDynamicsSwitch')
            section.ttxo_level_TTXDynamicsSwitch = 1.0

    def _default_disable_ttx(self):
        """Default disable_ttx implementation"""

        for section in self.all:
            if not neuron.h.ismembrane("TTXDynamicsSwitch"):
                section.insert('TTXDynamicsSwitch')
            section.ttxo_level_TTXDynamicsSwitch = 1e-14

    def execute_neuronconfigure(self, expression, sections=None):
        """Execute a statement from a BlueConfig NeuronConfigure block.

        Parameters
        ----------
        expression : string
                     Expression to evaluate on this cell object
        sections : string
                   Section group this expression has to be evaluated on
                   Possible values are
                   'axonal', 'basal', 'apical', 'somatic', 'dendritic', None
                   When None is passed, the expression is evaluated on all
                   sections

        """
        sections_map = {'axonal': self.axonal, 'basal': self.basal,
                        'apical': self.apical, 'somatic': self.somatic,
                        'dendritic': self.basal + self.apical + self.somatic,
                        None: self.all}

        for section in sections_map[sections]:
            sec_expression = \
                expression.replace('%s', neuron.h.secname(sec=section))
            if '%g' in expression:
                for segment in section:
                    seg_expression = sec_expression.replace('%g', segment.x)
                    bglibpy.neuron.h('execute1(%s, 0)' % seg_expression)
            else:
                bglibpy.neuron.h('execute1(%s, 0)' % sec_expression)

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
                area += bglibpy.neuron.h.area(x, sec=section)
            # for segment in section:
            #    area += bglibpy.neuron.h.area(segment.x, sec=section)
        return area

    def synlocation_to_segx(self, isec, ipt, syn_offset):
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

        # SONATA has pre-calculated distance field
        if ipt == -1:
            return syn_offset

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
            printv("WARNING: synlocation_to_segx found negative distance \
                    at curr_sec(%s) syn_offset: %f"
                   % (neuron.h.secname(sec=curr_sec), syn_offset), 1)
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
            recording.record(eval(var_name),
                             self.get_precise_record_dt(dt))
        else:
            recording.record(eval(var_name))
        self.recordings[var_name] = recording

    @staticmethod
    def get_precise_record_dt(dt):
        """Get a more precise record_dt to make time points faill on dts"""
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

    def add_voltage_recording(self, section, segx):
        """Add a voltage recording to a certain section(segx)

        Parameters
        ----------
        section : nrnSection
                  Section to record from (Neuron section pointer)
        segx : float
               Segment x coordinate
        """

        recording = neuron.h.Vector()

        recording.record(
            eval(
                'neuron.h.%s(%f)._ref_v' %
                (section.name(), segx)))

        self.voltage_recordings['%s(%f)' % (section.name(), segx)] = recording

    def get_voltage_recording(self, section, segx):
        """Get a voltage recording for a certain section(segx)

        Parameters
        ----------
        section : nrnSection
                  Section to record from (Neuron section pointer)
        segx : float
               Segment x coordinate
        """

        recording_name = '%s(%f)' % (section.name(), segx)
        if recording_name in self.voltage_recordings:
            return self.voltage_recordings[recording_name].to_python()
        else:
            raise Exception('get_voltage_recording: Voltage recording %s'
                            ' was not added previously using '
                            'add_voltage_recording' % recording_name)

    def add_allsections_voltagerecordings(self):
        """Add a voltage recording to every section of the cell."""
        all_sections = self.cell.getCell().all
        for section in all_sections:
            var_name = 'neuron.h.' + section.name() + "(0.5)._ref_v"
            self.add_recording(var_name)

    def get_allsections_voltagerecordings(self):
        """Get all the voltage recordings from all the sections.

        Returns
        -------
        dict of numpy arrays : dict with secname of sections as keys

        """
        allSectionVoltages = {}
        all_sections = self.cell.getCell().all
        for section in all_sections:
            var_name = 'neuron.h.' + section.name() + "(0.5)._ref_v"
            allSectionVoltages[section.name()] = self.get_recording(var_name)
        return allSectionVoltages

    def get_recording(self, var_name):
        """Get recorded values.


        Returns
        -------
        numpy array : array with the recording var_name variable values

        """
        return np.array(self.recordings[var_name])

    def add_replay_synapse(self, synapse_id, syn_description, connection_modifiers,
                           condition_parameters=None, base_seed=None,
                           popids=(0, 0), extracellular_calcium=None):
        """Add synapse based on the syn_description to the cell.

        This operation can fail.  Returns True on success, otherwise False.

        """

        if condition_parameters is None:
            condition_parameters = {}
        isec = syn_description[BLPSynapse.POST_SECTION_ID]
        ipt = syn_description[BLPSynapse.POST_SEGMENT_ID]

        if ipt == -1:
            syn_offset = syn_description["afferent_section_pos"]
        else:
            syn_offset = syn_description[BLPSynapse.POST_SEGMENT_OFFSET]

        location = self.synlocation_to_segx(isec, ipt, syn_offset)
        if location is None:
            printv('WARNING: add_single_synapse: skipping a synapse at \
                        isec %d ipt %f' % (isec, ipt), 1)
            return False

        synapse = bglibpy.Synapse(
            self,
            location,
            synapse_id,
            syn_description,
            connection_parameters=connection_modifiers,
            condition_parameters=condition_parameters,
            base_seed=base_seed,
            popids=popids,
            extracellular_calcium=extracellular_calcium)

        self.synapses[synapse_id] = synapse

        printv(
            'Added synapse to cell %d: %s' %
            (self.gid,
             json.dumps(
                 synapse.info_dict,
                 cls=tools.NumpyEncoder)),
            50)

        return True

    def add_replay_delayed_weight(self, sid, delay, weight):
        """Add a synaptic weight for sid that will be set with a time delay.

        Parameters
        ----------
        sid : int
              synapse id
        delay : float
                synaptic delay
        weight : float
                 synaptic weight
        """

        self.delayed_weights.put((delay, (sid, weight)))

    def pre_gids(self):
        """List of gids of cells that connect to this cell.

        Returns
        -------
        A list of gids of cells that connect to this cell.
        """

        pre_gid_list = set()
        for syn_id in self.synapses:
            pre_gid_list.add(self.synapses[syn_id].pre_gid)

        return list(pre_gid_list)

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

    def create_netcon_spikedetector(self, target, location, threshold=-30):
        """Add and return a spikedetector.

        This is a NetCon that detects spike in the current cell, and that
        connects to target
        Args:
            target: target point process
            location (str): the spike detection location
            threshold (float): spike detection threshold

        Returns
        -------

        NetCon : Neuron netcon object

        """
        if location == "soma":
            sec = self.cell.getCell().soma[0]
            source = self.cell.getCell().soma[0](1)._ref_v
        elif location == "AIS":
            sec = self.cell.getCell().axon[1]
            source = self.cell.getCell().axon[1](0.5)._ref_v
        else:
            raise Exception("Spike detection location must be soma or AIS")
        netcon = bglibpy.neuron.h.NetCon(source, target, sec=sec)
        netcon.threshold = threshold

        return netcon

    def add_replay_minis(self, syn_id, syn_description, connection_parameters,
                         base_seed=None, popids=(0, 0), mini_frequencies=None):
        """Add minis from the replay."""

        source_popid, target_popid = popids

        sid = syn_id[1]

        if base_seed is None:
            base_seed = self.rng_settings.base_seed
        weight = syn_description[BLPSynapse.G_SYNX]
        post_sec_id = syn_description[BLPSynapse.POST_SECTION_ID]
        post_seg_id = syn_description[BLPSynapse.POST_SEGMENT_ID]
        if post_seg_id == -1:
            post_seg_distance = syn_description["afferent_section_pos"]
        else:
            post_seg_distance = syn_description[BLPSynapse.POST_SEGMENT_OFFSET]
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

        # SpontMinis in BlueConfig takes precedence of values in nodes file
        if 'SpontMinis' in connection_parameters:
            spont_minis_rate = connection_parameters['SpontMinis']
        else:
            if synapse.is_excitatory():
                spont_minis_rate = exc_mini_frequency
            elif synapse.is_inhibitory():
                spont_minis_rate = inh_mini_frequency
            else:
                raise Exception('Synapse not inhibitory nor excitatory, '
                                'can not set minis frequency')

        if spont_minis_rate is not None:
            # add the *minis*: spontaneous synaptic events
            self.ips[syn_id] = bglibpy.neuron.h.\
                InhPoissonStim(location,
                               sec=self.get_hsection(post_sec_id))

            delay = 0.1
            self.syn_mini_netcons[syn_id] = bglibpy.neuron.h.\
                NetCon(self.ips[syn_id], synapse.hsynapse,
                       -30, delay, weight * weight_scalar)
            # set netcon type
            nc_param_name = 'nc_type_param_{}'.format(
                synapse.hsynapse).split('[')[0]
            if hasattr(bglibpy.neuron.h, nc_param_name):
                nc_type_param = int(getattr(bglibpy.neuron.h, nc_param_name))
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
                exprng = bglibpy.neuron.h.Random()
                self.persistent.append(exprng)

                uniformrng = bglibpy.neuron.h.Random()
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
                        "Cell: Unknown rng mode: %s" %
                        self.rng_settings.mode)

                exprng.MCellRan4(exp_seed1, exp_seed2)
                exprng.negexp(1.0)

                uniformrng.MCellRan4(uniform_seed1, uniform_seed2)
                uniformrng.uniform(0.0, 1.0)

                self.ips[syn_id].setRNGs(exprng, uniformrng)

            tbins_vec = bglibpy.neuron.h.Vector(1)
            tbins_vec.x[0] = 0.0
            rate_vec = bglibpy.neuron.h.Vector(1)
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
                syn.tsyn_fac = bglibpy.neuron.h.t
                syn.u = syn.u0
                syn.A_AMPA = 0
                syn.B_AMPA = 0
                syn.A_NMDA = 0
                syn.B_NMDA = 0
            elif syn_type == 'ProbGABAAB_EMS':
                syn.Rstate = 1
                syn.tsyn_fac = bglibpy.neuron.h.t
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
                    printv(apicnumber, secnumber, 1)
                else:
                    raise Exception(
                        "somaticbranches: No apic or \
                                dend found in section %s" % secname)

    @staticmethod
    @tools.deprecated("bglibpy.cell.section_distance.EuclideanSectionDistance")
    def euclid_section_distance(
            hsection1=None,
            hsection2=None,
            location1=None,
            location2=None,
            projection=None):
        """Calculate euclidian distance between positions on two sections
           Uses bglibpy.cell.section_distance.EuclideanSectionDistance

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

    # Disable unused argument warning for dt. This is there for backward
    # compatibility
    # pylint: disable=W0613

    # pylint: enable=W0613

    def get_time(self):
        """Get the time vector."""
        return self.get_recording('neuron.h._ref_t')

    def get_soma_voltage(self):
        """Get a vector of the soma voltage."""
        return self.get_recording('self.soma(0.5)._ref_v')

    def get_ais_voltage(self):
        """Get a vector of AIS voltage."""
        return self.get_recording('self.axonal[1](0.5)._ref_v')

    def getNumberOfSegments(self):
        """Get the number of segments in the cell."""
        return sum(section.nseg for section in self.all)

    def add_plot_window(self, var_list, xlim=None, ylim=None, title=""):
        """Add a window to plot a variable."""
        xlim = [0, 1000] if xlim is None else xlim
        ylim = [-100, 100] if ylim is None else ylim
        for var_name in var_list:
            if var_name not in self.recordings:
                self.add_recording(var_name)
        self.plot_windows.append(bglibpy.PlotWindow(
            var_list, self, xlim, ylim, title))
        self.plot_callback_necessary = True

    def add_dendrogram(
            self,
            variable=None,
            active=False,
            save_fig_path=None,
            interactive=False,
            scale_bar=True,
            scale_bar_size=10.0,
            fig_title=None):
        """Show a dendrogram of the cell."""
        self.init_psections()
        cell_dendrogram = bglibpy.Dendrogram(
            self.psections,
            variable=variable,
            active=active,
            save_fig_path=save_fig_path,
            interactive=interactive,
            scale_bar=scale_bar,
            scale_bar_size=scale_bar_size,
            fig_title=fig_title)
        cell_dendrogram.redraw()
        self.cell_dendrograms.append(cell_dendrogram)
        if active:
            self.plot_callback_necessary = True

    def init_callbacks(self):
        """Initialize the callback function (if necessary)."""
        if not self.delayed_weights.empty():
            self.fih_weights = neuron.h.FInitializeHandler(
                1, self.weights_callback)

        if self.plot_callback_necessary:
            self.fih_plots = neuron.h.FInitializeHandler(1, self.plot_callback)

    def weights_callback(self):
        """Callback function that updates the delayed weights,
        when a certain delay has been reached"""
        while not self.delayed_weights.empty() and \
                abs(self.delayed_weights.queue[0][0] - neuron.h.t) < \
                neuron.h.dt:
            (_, (sid, weight)) = self.delayed_weights.get()
            if sid in self.connections:
                if self.connections[sid].post_netcon is not None:
                    self.connections[sid].post_netcon.weight[0] = weight

        if not self.delayed_weights.empty():
            neuron.h.cvode.event(self.delayed_weights.queue[0][0],
                                 self.weights_callback)

    def plot_callback(self):
        """Update all the windows."""
        for window in self.plot_windows:
            window.redraw()
        for cell_dendrogram in self.cell_dendrograms:
            cell_dendrogram.redraw()

        neuron.h.cvode.event(neuron.h.t + 1, self.plot_callback)

    @property
    def info_dict(self):
        """Return a dictionary with all the information of this cell"""

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
        if hasattr(self, 'cell') and self.cell is not None:
            if self.cell.getCell() is not None:
                self.cell.getCell().clear()

            self.fih_plots = None
            self.fih_weights = None
            self.connections = None
            self.synapses = None

        if hasattr(self, 'recordings'):
            for recording in self.recordings:
                del recording

        if hasattr(self, 'voltage_recordings'):
            for voltage_recording in self.voltage_recordings:
                del voltage_recording

        if hasattr(self, 'persistent'):
            for persistent_object in self.persistent:
                del persistent_object

    @property
    def hsynapses(self):
        """Contains a dictionary of all the hoc synapses
        in the cell with as key the gid"""
        return dict((gid, synapse.hsynapse) for (gid, synapse)
                    in self.synapses.items())

    def __del__(self):
        self.delete()
