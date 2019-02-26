#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SSim Class

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

# pylint: disable=C0103, R0914, R0912, F0401, R0101

import collections
import os

import numpy

from bglibpy import bluepy
import bglibpy
from bglibpy import printv
from bglibpy import tools

from bluepy.v2.enums import Synapse as BLPSynapse


class SSim(object):

    """Class that can load a BGLib BlueConfig,
               and instantiate the simulation"""

    # pylint: disable=R0913

    def __init__(self, blueconfig_filename, dt=0.025, record_dt=None,
                 base_seed=None, base_noise_seed=None, rng_mode=None):
        """Object dealing with BlueConfig configured Small Simulations

        Parameters
        ----------
        blueconfig_filename : string
                              Absolute filename of the Blueconfig
        dt : float
             Timestep of the simulation
        record_dt : float
                    Sampling interval of the recordings
        base_seed : int
                    Base seed used for this simulation. Setting this
                    will override the value set in the BlueConfig.
                    Has to positive integer.
                    When this is not set, and no seed is set in the
                    BlueConfig, the seed will be 0.
        base_noise_seed : int
                    Base seed used for the noise stimuli in the simulation.
                    Not setting this will result in the default Neurodamus
                    behavior (i.e. seed=0)
                    Has to positive integer.
        """
        self.dt = dt
        self.record_dt = record_dt
        self.blueconfig_filename = blueconfig_filename
        self.bc_simulation = bluepy.Simulation(blueconfig_filename).v2
        self.bc_circuit = self.bc_simulation.circuit
        self.bc = self.bc_simulation.config

        if 'MEComboInfoFile' in self.bc.Run:
            self.use_mecombotsv = True
            self.mecombo_emodels, \
                self.mecombo_thresholds, \
                self.mecombo_hypamps = self.get_mecombo_emodels()
        else:
            self.use_mecombotsv = False
            self.mecombo_emodels = None
            self.mecombo_thresholds = None
            self.mecombo_hypamps = None

        self.rng_settings = bglibpy.RNGSettings(
            rng_mode,
            self.bc,
            base_seed=base_seed,
            base_noise_seed=base_noise_seed)

        self.connection_entries = self.bc.typed_sections('Connection')
        self.all_targets = self.bc_circuit.cells.targets
        self.all_targets_dict = {}
        for target in self.all_targets:
            self.all_targets_dict[target] = \
                self.bc_circuit.cells.ids(target)

        self.gids = []
        self.cells = {}

        self.neuronconfigure_entries = \
            self.bc.typed_sections("NeuronConfigure")
        self.neuronconfigure_expressions = {}
        for entry in self.neuronconfigure_entries:
            for gid in self.all_targets_dict[entry.Target]:
                conf = entry.Configure
                self.neuronconfigure_expressions.\
                    setdefault(gid, []).append(conf)

        self.gids_instantiated = False
        self.connections = \
            collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: None))

        self.emodels_dir = self.bc.Run['METypePath']
        self.morph_dir = self.bc.Run['MorphologyPath']

        # Make sure tstop is set correctly, because it is used by the
        # TStim noise stimulus
        if 'Duration' in self.bc.Run:
            bglibpy.neuron.h.tstop = float(self.bc.Run['Duration'])

        # backwards compatible
        if self.morph_dir[-3:] == "/h5":
            self.morph_dir = self.morph_dir[:-3]

        self.morph_dir = os.path.join(self.morph_dir, 'ascii')

    @property
    def base_seed(self):
        """Baseseed of sim"""

        return self.rng_settings.base_seed

    @property
    def base_noise_seed(self):
        """Baseseed of noise stimuli in sim"""

        return self.rng_settings.base_noise_seed

    # pylint: disable=R0913
    def instantiate_gids(self, gids, synapse_detail=None,
                         add_replay=False,
                         add_stimuli=False,
                         add_synapses=None,
                         add_minis=None,
                         add_noise_stimuli=False,
                         add_hyperpolarizing_stimuli=False,
                         add_relativelinear_stimuli=False,
                         add_pulse_stimuli=False,
                         intersect_pre_gids=None,
                         interconnect_cells=True,
                         pre_spike_trains=None,
                         projection=None):
        """ Instantiate a list of GIDs

        Parameters
        ----------
        gids : list of integers
               List of GIDs. Must be a list,
               even in case of instantiation of a single GID.
        synapse_detail : {0 , 1, 2}
                         Level of detail. If chosen, all settings are taken
                         from the "large" cortical simulation.
                         Possible values:

                         * 0 No synapses

                         * 1 Add synapse of the correct type at the
                            simulated locations with all settings
                            as in the "large" simulation

                         * 2 As 1 but with minis

        add_replay : Boolean
                     Add presynaptic spiketrains from the large simulation
                     throws an exception if this is set when synapse_detail < 1
                     If pre_spike_trains is combined with this option the
                     spiketrains will be merged
        add_stimuli : Boolean
                      Add the same stimuli as in the large simulation
        add_synapses : Boolean
                       Add the touch-detected synapses, as described by the
                       circuit to the cell
                       (This option only influence the 'creation' of synapses,
                       it doesn't add any connections)
                       Default value is False
        add_minis : Boolean
                    Add synaptic minis to the synapses
                    (this requires add_synapses=True)
                    Default value is False
        add_noise_stimuli : Boolean
                            Process the 'noise' stimuli blocks of the
                            BlueConfig,
                            Setting add_stimuli=True,
                            will automatically set this option to True.
        add_hyperpolarizing_stimuli : Boolean
                                      Process the 'hyperpolarizing' stimuli
                                      blocks of the BlueConfig.
                                      Setting add_stimuli=True,
                                      will automatically set this option to
                                      True.
        add_relativelinear_stimuli : Boolean
                                      Process the 'relativelinear' stimuli
                                      blocks of the BlueConfig.
                                      Setting add_stimuli=True,
                                      will automatically set this option to
                                      True.
        add_pulse_stimuli : Boolean
                                      Process the 'pulse' stimuli
                                      blocks of the BlueConfig.
                                      Setting add_stimuli=True,
                                      will automatically set this option to
                                      True.
        intersect_pre_gids : list of gids
                             Only add synapses to the cells if their
                             presynaptic gid is in this list
        interconnect_cells : Boolean
                             When multiple gids are instantiated,
                             interconnect the cells with real (non-replay)
                             synapses. When this option is combined with
                             add_replay, replay spiketrains will only be added
                             for those presynaptic cells that are not in the
                             network that's instantiated.
                             This option requires add_synapses=True
        pre_spike_trains : dict
                           A dictionary with keys the presynaptic gids, and
                           values the list of spike timings of the
                           presynaptic cells with the given gids.
                           If this option is used in combination with
                           add_replay=True, the spike trains for the same
                           gids will be automatically merged
        projection: string
                    Name of the projection where all information about synapses
                    come from. Mixing different projections is not possible
                    at the moment.
                    Beware, this option might disappear in the future if
                    BluePy unifies the API to get the synapse information for
                    a certain gid.
        """

        if synapse_detail is not None:
            printv(
                'WARNING: SSim: synapse_detail is deprecated and will '
                'removed from future release of BGLibPy', 2)
            if synapse_detail > 0:
                if add_minis is False:
                    raise Exception('SSim: synapse_detail >= 1 cannot be used'
                                    ' with add_synapses == False')
                add_synapses = True
            if synapse_detail > 1:
                if add_minis is False:
                    raise Exception('SSim: synapse_detail >= 2 cannot be used'
                                    ' with add_minis == False')
                add_minis = True

        if add_minis is None:
            add_minis = False

        if self.gids_instantiated:
            raise Exception("SSim: instantiate_gids() called twice on the \
                    same SSim, this is not supported yet")
        else:
            self.gids_instantiated = True

        if pre_spike_trains or add_replay:
            if add_synapses is not None and add_synapses is False:
                raise Exception("SSim: you need to set add_synapses to True "
                                "if you want to specify use add_replay or "
                                "pre_spike_trains")
            add_synapses = True
        else:
            if add_synapses is None:
                add_synapses = False

        self._add_cells(gids)
        if add_stimuli:
            add_noise_stimuli = True
            add_hyperpolarizing_stimuli = True
            add_relativelinear_stimuli = True
            add_pulse_stimuli = True

        if add_noise_stimuli or \
                add_hyperpolarizing_stimuli or \
                add_pulse_stimuli or \
                add_relativelinear_stimuli:
            self._add_stimuli(
                add_noise_stimuli=add_noise_stimuli,
                add_hyperpolarizing_stimuli=add_hyperpolarizing_stimuli,
                add_relativelinear_stimuli=add_relativelinear_stimuli,
                add_pulse_stimuli=add_pulse_stimuli)
        if add_synapses:
            self._add_synapses(intersect_pre_gids=intersect_pre_gids,
                               add_minis=add_minis, projection=projection)
        if add_replay or interconnect_cells or pre_spike_trains:
            if add_replay and not add_synapses:
                raise Exception("SSim: add_replay option can not be used if "
                                "add_synapses is False")
            self._add_connections(add_replay=add_replay,
                                  interconnect_cells=interconnect_cells,
                                  user_pre_spike_trains=pre_spike_trains)

    # pylint: enable=R0913

    def _add_stimuli(self, add_noise_stimuli=False,
                     add_hyperpolarizing_stimuli=False,
                     add_relativelinear_stimuli=False,
                     add_pulse_stimuli=False):
        """Instantiate all the stimuli"""
        for gid in self.gids:
            # Also add the injections / stimulations as in the cortical model
            self._add_stimuli_gid(
                gid,
                add_noise_stimuli=add_noise_stimuli,
                add_hyperpolarizing_stimuli=add_hyperpolarizing_stimuli,
                add_relativelinear_stimuli=add_relativelinear_stimuli,
                add_pulse_stimuli=add_pulse_stimuli)
            printv("Added stimuli for gid %d" % gid, 2)

    def _add_synapses(self, intersect_pre_gids=None, add_minis=None,
                      projection=None):
        """Instantiate all the synapses"""
        for gid in self.gids:
            syn_descriptions = self.get_syn_descriptions_dict(
                gid,
                projection=projection)

            if intersect_pre_gids is not None:
                syn_descriptions = {syn_id: syn_description
                                    for syn_id, syn_description in
                                    syn_descriptions.items()
                                    if syn_description[0] in
                                    intersect_pre_gids}

            # Check if there are any presynaptic cells, otherwise skip adding
            # synapses
            if syn_descriptions is None:
                printv(
                    "Warning: No presynaptic cells found for gid %d, "
                    "no synapses added" % gid, 2)
            else:
                for syn_id, syn_description in syn_descriptions.items():
                    self._instantiate_synapse(gid, syn_id, syn_description,
                                              add_minis=add_minis)
                printv("Added %d synapses for gid %d" %
                       (len(syn_descriptions), gid), 2)
                if add_minis:
                    printv("Added minis for gid %d" % gid, 2)

    @tools.deprecated("get_syn_descriptions_dict")
    def get_syn_descriptions(self, gid, projection=None):
        """Get synapse description arrays from bluepy"""

        syn_descriptions = [
            v for _,
            v in sorted(
                self.get_syn_descriptions_dict(
                    gid,
                    projection=projection).items())]

        return syn_descriptions

    def get_syn_descriptions_dict(self, gid, projection=None):
        """Get synapse descriptions dict from bluepy, Keys are synapse ids"""
        syn_descriptions_dict = {}

        all_properties = [
            BLPSynapse.PRE_GID,
            BLPSynapse.AXONAL_DELAY,
            BLPSynapse.POST_SECTION_ID,
            '_POST_SEGMENT_ID',
            '_POST_DISTANCE',
            BLPSynapse.PRE_SECTION_ID,
            '_PRE_SEGMENT_ID',
            '_PRE_DISTANCE',
            BLPSynapse.G_SYNX,
            BLPSynapse.U_SYN,
            BLPSynapse.D_SYN,
            BLPSynapse.F_SYN,
            BLPSynapse.DTC,
            BLPSynapse.TYPE,
            BLPSynapse.NRRP]

        if projection is None:
            connectome = self.bc_circuit.connectome
        else:
            connectome = self.bc_circuit.projection(projection)

        nrn_h5_path = connectome._impl._prefix + 'nrn.h5'
        nrn_h5_version = self.get_nrn_h5_version(nrn_h5_path)

        if nrn_h5_version == 5:
            # Get properties with Nrrp
            synapses = connectome.afferent_synapses(
                gid,
                properties=all_properties)
            nrrp_defined = True
            printv('Version of nrn.h5 file is 5, enabling multivesicular '
                   'release', 50)
        elif nrn_h5_version is None or nrn_h5_version < 5:
            # Get properties without Nrrp
            all_properties = all_properties[:-1]
            synapses = connectome.afferent_synapses(
                gid,
                properties=all_properties)
            nrrp_defined = False
            printv('Version of nrn.h5 file not specified or smaller than 5, '
                   'DISABLING multivesicular release', 50)
        else:
            raise ValueError(
                'Unknown nrn.h5 version "%s" for %s' %
                (nrn_h5_version, nrn_h5_path))

        for (syn_gid, syn_id), synapse in synapses.iterrows():
            if gid != gid:
                raise Exception(
                    "BGLibPy SSim: synapse gid doesnt match with cell gid !")
            if syn_id in syn_descriptions_dict:
                raise Exception(
                    "BGLibPy SSim: trying to synapse id %d twice !" %
                    syn_descriptions_dict)
            if nrrp_defined:
                old_syn_description = synapse[all_properties].values[:14]
                nrrp = synapse[all_properties].values[14]
                ext_syn_description = numpy.array([-1, -1, -1, nrrp])
                # 14 - 16 are dummy values, 17 is Nrrp
                syn_description = numpy.append(
                    old_syn_description,
                    ext_syn_description)
            else:
                # old behavior
                syn_description = synapse[all_properties].values

            syn_descriptions_dict[syn_id] = syn_description

        return syn_descriptions_dict

    @staticmethod
    def merge_pre_spike_trains(*train_dicts):
        """Merge presynaptic spike train dicts"""

        ret_dict = None

        for train_dict in train_dicts:
            if train_dict is not None:
                if ret_dict is None:
                    ret_dict = train_dict.copy()
                    continue
                else:
                    # Not super efficient, but out.dats to tend not to be
                    # very large
                    for pre_gid, train in train_dict.items():
                        ret_dict.setdefault(pre_gid, []).extend(train)

        if ret_dict is not None:
            for pre_gid, train in ret_dict.items():
                if train is not None:
                    ret_dict[pre_gid] = numpy.array(sorted(train))

        return ret_dict

    # pylint: disable=R0913
    def _add_connections(
            self,
            add_replay=None,
            interconnect_cells=None,
            outdat_path=None,
            source=None,
            dest=None,
            user_pre_spike_trains=None):
        """Instantiate the (replay and real) connections in the network"""
        if add_replay:
            if outdat_path is None:
                outdat_path = os.path.join(
                    self.bc.Run['OutputRoot'],
                    'out.dat')
            pre_spike_trains = _parse_outdat2(outdat_path)
        else:
            pre_spike_trains = None

        pre_spike_trains = self.merge_pre_spike_trains(
            pre_spike_trains,
            user_pre_spike_trains)

        for post_gid in self.gids:
            if dest and post_gid not in dest:
                continue
            for syn_id in self.cells[post_gid].synapses:
                synapse = self.cells[post_gid].synapses[syn_id]
                syn_description = synapse.syn_description
                connection_parameters = synapse.connection_parameters
                pre_gid = syn_description[0]
                if source and pre_gid not in source:
                    continue
                real_synapse_connection = pre_gid in self.gids \
                    and interconnect_cells

                connection = None
                if real_synapse_connection:
                    connection = bglibpy.Connection(
                        self.cells[post_gid].synapses[syn_id],
                        pre_spiketrain=None,
                        pre_cell=self.cells[pre_gid],
                        stim_dt=self.dt)
                    printv("Added real connection between pre_gid %d and \
                            post_gid %d, syn_id %d" % (pre_gid,
                                                       post_gid,
                                                       syn_id), 5)
                elif pre_spike_trains is not None:
                    pre_spiketrain = pre_spike_trains.setdefault(pre_gid, None)
                    connection = bglibpy.Connection(
                        self.cells[post_gid].synapses[syn_id],
                        pre_spiketrain=pre_spiketrain,
                        pre_cell=None,
                        stim_dt=self.dt)
                    printv(
                        "Added replay connection from pre_gid %d to "
                        "post_gid %d, syn_id %d" %
                        (pre_gid, post_gid, syn_id), 5)

                if connection is not None:
                    self.cells[post_gid].connections[syn_id] = connection
                    if "DelayWeights" in connection_parameters:
                        for delay, weight in \
                                connection_parameters['DelayWeights']:
                            self.cells[post_gid].add_replay_delayed_weight(
                                syn_id, delay, weight)

            if len(self.cells[post_gid].connections) > 0:
                printv("Added synaptic connections for target post_gid %d" %
                       post_gid, 2)

    def _add_cells(self, gids):
        """Instantiate cells from a gid list"""
        self.gids = gids
        self.cells = {}

        for gid in self.gids:
            printv(
                'Adding gid %d from emodel %s and morph %s' %
                (gid,
                 self.fetch_emodel_name(gid),
                 self.fetch_morph_name(gid)),
                1)

            self.cells[gid] = bglibpy.Cell(**self.fetch_cell_kwargs(gid))

            if gid in self.neuronconfigure_expressions:
                for expression in self.neuronconfigure_expressions[gid]:
                    self.cells[gid].execute_neuronconfigure(expression)

    def _instantiate_synapse(self, gid, syn_id, syn_description,
                             add_minis=None):
        """Instantiate one synapse for a given gid, syn_id and
        syn_description"""

        syn_type = syn_description[13]

        connection_parameters = self. _evaluate_connection_parameters(
            syn_description[0],
            gid,
            syn_type)

        if connection_parameters["add_synapse"]:
            self.add_single_synapse(gid, syn_id, syn_description,
                                    connection_parameters)
            if add_minis:
                self.add_replay_minis(gid, syn_id, syn_description,
                                      connection_parameters)

    def _add_stimuli_gid(self, gid,
                         add_noise_stimuli=False,
                         add_hyperpolarizing_stimuli=False,
                         add_relativelinear_stimuli=False,
                         add_pulse_stimuli=False):
        """ Adds indeitical stimuli to the simulated cell as in the 'large'
            model

        Parameters:
        -----------
        gid: gid of the simulated cell
        """
        # check in which StimulusInjects the gid is a target
        # Every noise stimulus gets a new seed
        noisestim_count = 0

        for entry in self.bc.values():
            if entry.section_type == 'StimulusInject':
                destination = entry.Target
                gids_of_target = self.bc_circuit.cells.ids(destination)
                if gid in gids_of_target:
                    # retrieve the stimulus to apply
                    stimulus_name = entry.Stimulus
                    # bluepy magic to add underscore Stimulus underscore
                    # stimulus_name
                    stimulus = self.bc['Stimulus_%s' % stimulus_name]
                    if stimulus.Pattern == 'Noise':
                        if add_noise_stimuli:
                            self._add_replay_noise(
                                gid,
                                stimulus,
                                noisestim_count=noisestim_count)
                        noisestim_count += 1
                    elif stimulus.Pattern == 'Hyperpolarizing':
                        if add_hyperpolarizing_stimuli:
                            self._add_replay_hypamp_injection(
                                gid,
                                stimulus)
                    elif stimulus.Pattern == 'Pulse':
                        if add_pulse_stimuli:
                            self._add_pulse(gid, stimulus)
                    elif stimulus.Pattern == 'RelativeLinear':
                        if add_relativelinear_stimuli:
                            self._add_relativelinear(gid, stimulus)
                    elif stimulus.Pattern == 'SynapseReplay':
                        printv("Found stimulus with pattern %s, ignoring" %
                               stimulus['Pattern'], 1)
                    else:
                        raise Exception("Found stimulus with pattern %s, "
                                        "not supported" %
                                        stimulus.Pattern)

    def _add_replay_hypamp_injection(self, gid, stimulus):
        """Add injections from the replay"""
        self.cells[gid].add_replay_hypamp(stimulus)

    def _add_relativelinear(self, gid, stimulus):
        """Add relative linear injections from the replay"""
        self.cells[gid].add_replay_relativelinear(stimulus)

    def _add_pulse(self, gid, stimulus):
        """Add injections from the replay"""
        self.cells[gid].add_pulse(stimulus)

    def _add_replay_noise(
            self,
            gid,
            stimulus,
            noisestim_count=None):
        """Add noise injection from the replay"""
        self.cells[gid].add_replay_noise(
            stimulus,
            noisestim_count=noisestim_count)

    def add_replay_minis(self, gid, syn_id, syn_description, syn_parameters):
        """Add minis from the replay"""
        self.cells[gid].add_replay_minis(syn_id,
                                         syn_description,
                                         syn_parameters)

    def add_single_synapse(self, gid, syn_id,
                           syn_description, connection_modifiers):
        """Add a replay synapse on the cell

        Parameters
        ----------
        gid : int
              GID of the cell
        syn_id: int
              Synapse ID of the synapse
        syn_description: dict
              Description of the synapse
        connection_modifiers: dict
              Connection modifiers for the synapse
        """
        return self.cells[gid].add_replay_synapse(syn_id,
                                                  syn_description,
                                                  connection_modifiers)

    def check_connection_contents(self, contents):
        """Check the contents of a connection block,
           to see if we support all the fields"""

        allowed_keys = set(['Weight', 'SynapseID', 'SpontMinis',
                            'SynapseConfigure', 'Source',
                            'Destination', 'Delay', 'CreateMode'])
        for key in contents.keys():
            if key not in allowed_keys:
                raise Exception(
                    "Key %s in Connection blocks not supported by BGLibPy"
                    % key)

    def _evaluate_connection_parameters(self, pre_gid, post_gid, syn_type):
        """ Apply connection blocks in order for pre_gid, post_gid to
            determine a final connection override for this pair
            (pre_gid, post_gid)

        Parameters:
        -----------
        gid : int
              gid of the post-synaptic cell

        """
        parameters = {}
        parameters['add_synapse'] = True
        spontminis_set = False

        for entry in self.connection_entries:
            entry_name = entry.name
            self.check_connection_contents(entry)
            src = entry['Source']
            dest = entry['Destination']

            if src in self.all_targets_dict and dest in self.all_targets_dict:
                if pre_gid in self.all_targets_dict[src] and \
                        post_gid in self.all_targets_dict[dest]:
                    # whatever specified in this block, is applied to gid
                    apply_parameters = True
                    keys = set(entry.keys())

                    if 'SynapseID' in keys:
                        if int(entry['SynapseID']) != syn_type:
                            apply_parameters = False

                    if 'Delay' in keys:
                        parameters.setdefault('DelayWeights', []).append((
                            float(entry['Delay']),
                            float(entry['Weight'])))
                        apply_parameters = False

                    if apply_parameters:
                        if 'CreateMode' in keys:
                            if entry['CreateMode'] == 'NoCreate':
                                parameters['add_synapse'] = False
                            else:
                                raise Exception('Connection %s: Unknown '
                                                'CreateMode option %s'
                                                % (entry_name,
                                                   entry['CreateMode']))
                        if 'Weight' in keys:
                            parameters['Weight'] = float(entry['Weight'])
                        if not spontminis_set:
                            if 'SpontMinis' in keys:
                                parameters['SpontMinis'] = float(
                                    entry['SpontMinis'])
                                spontminis_set = True
                            else:
                                parameters['SpontMinis'] = 0.0
                                spontminis_set = True
                        elif 'SpontMinis' in keys:
                            import warnings
                            warnings.warn(
                                "Connection '%s': SpontMinis was already set "
                                "in previous block, IGNORING" % entry_name)

                        if 'SynapseConfigure' in keys:
                            conf = entry['SynapseConfigure']
                            # collect list of applicable configure blocks to be
                            # applied with a "hoc exec" statement
                            parameters.setdefault(
                                'SynapseConfigure', []).append(conf)

        return parameters

    def initialize_synapses(self):
        """ Resets the state of all synapses of all cells to initial values """
        for cell in self.cells.itervalues():
            cell.initialize_synapses()

    def run(self, t_stop=None, v_init=-65, celsius=34, dt=None,
            forward_skip=True, forward_skip_value=None,
            cvode=False, show_progress=False):
        """Simulate the SSim

        Parameters
        ----------
        t_stop : int
                 This function will run the simulation until t_stop
        v_init : float
                 Voltage initial value when the simulation starts
        celsius : float
                  Temperature at which the simulation runs
        dt : float
             Timestep (delta-t) for the simulation
        forward_skip : boolean
                       Enable/disable ForwardSkip (default=True, when
                       forward_skip_value is None, forward skip will only be
                       enabled if BlueConfig has a ForwardSkip value)
        forward_skip_value : float
                       Overwrite the ForwardSkip value in the BlueConfig. If
                       this is set to None, the value in the BlueConfig is
                       used.
        cvode : boolean
                Force the simulation to run in variable timestep. Not possible
                when there are stochastic channels in the neuron model. When
                enabled results from a large network simulation will not be
                exactly reproduced.
        show_progress: boolean
                       Show a progress bar during simulations. When
                       enabled results from a large network simulation
                       will not be exactly reproduced.
        """
        if t_stop is None:
            t_stop = float(self.bc.Run['Duration'])
        if dt is None:
            dt = float(self.bc.Run['Dt'])
        if forward_skip_value is None:
            if 'ForwardSkip' in self.bc.Run:
                forward_skip_value = float(
                    self.bc.Run['ForwardSkip'])

        sim = bglibpy.Simulation()
        for gid in self.gids:
            sim.add_cell(self.cells[gid])

        if show_progress:
            printv("Warning: show_progress enabled, this will very likely"
                   "break the exact reproducibility of large network"
                   "simulations", 2)

        sim.run(
            t_stop,
            cvode=cvode,
            dt=dt,
            celsius=celsius,
            v_init=v_init,
            forward_skip=forward_skip,
            forward_skip_value=forward_skip_value,
            show_progress=show_progress)

    def get_mainsim_voltage_trace(self, gid=None):
        """Get the voltage trace from a cell from the main simulation"""

        voltage = self.bc_simulation.report('soma').get_gid(gid).values
        return voltage

    def get_mainsim_time_trace(self):
        """Get the time trace from the main simulation"""

        time = self.bc_simulation.report('soma').get().index
        return time

    @tools.deprecated("get_voltage_trace")
    def get_voltage_traces(self):
        """Get the voltage traces from all the cells as a dictionary
           based on gid"""
        vm = {}
        for gid in self.gids:
            vm[gid] = self.cells[gid].get_soma_voltage()
        return vm

    @tools.deprecated("get_time_trace")
    def get_time(self):
        """Get the time vector for the recordings"""
        return self.cells[self.gids[0]].get_time()

    def get_time_trace(self):
        """Get the time vector for the recordings, negative times removed"""

        time = self.cells[self.gids[0]].get_time()
        pos_time = time[numpy.where(time >= 0.0)]
        return pos_time

    def get_voltage_trace(self, gid):
        """Get the voltage vector for the gid, negative times removed"""

        time = self.get_time_trace()
        voltage = self.cells[gid].get_soma_voltage()
        pos_voltage = voltage[numpy.where(time >= 0.0)]
        return pos_voltage

    def __del__(self):
        """Destructor"""
        if hasattr(self, 'cells'):
            for gid in self.cells:
                self.cells[gid].delete()

    # Auxialliary methods ###

    def get_mecombo_emodels(self):
        """Create a dict matching me_combo names to template_names"""

        mecombo_filename = self.bc.Run['MEComboInfoFile']

        with open(mecombo_filename) as mecombo_file:
            mecombo_content = mecombo_file.read()

        mecombo_emodels = {}
        mecombo_thresholds = {}
        mecombo_hypamps = {}

        for line in mecombo_content.split('\n')[1:-1]:
            mecombo_info = line.split('\t')
            emodel = mecombo_info[4]
            me_combo = mecombo_info[5]
            try:
                threshold = float(mecombo_info[6])
            except (ValueError, IndexError):
                threshold = 0.0
                printv('WARNING: No threshold found for me-model %s, '
                       'replacing with 0.0!' % me_combo, 2)
            try:
                hypamp = float(mecombo_info[7])
            except (ValueError, IndexError):
                hypamp = 0.0
                printv('WARNING: No hypamp found for me-model %s, '
                       'replacing with 0.0!' % me_combo, 2)
            mecombo_emodels[me_combo] = emodel
            mecombo_thresholds[me_combo] = threshold
            mecombo_hypamps[me_combo] = hypamp

        return mecombo_emodels, mecombo_thresholds, mecombo_hypamps

    def fetch_gid_cell_info(self, gid):
        """Fetch bluepy cell info of a gid"""
        if gid in self.bc_circuit.cells.ids():
            cell_info = self.bc_circuit.cells.get(gid)
        else:
            raise Exception("Gid %d not found in circuit" % gid)

        return cell_info

    def fetch_mecombo_name(self, gid):
        """Fetch mecombo name for a certain gid"""

        cell_info = self.fetch_gid_cell_info(gid)

        me_combo = str(cell_info['me_combo'])

        return me_combo

    def fetch_emodel_name(self, gid):
        """Get the emodel path of a gid"""

        me_combo = self.fetch_mecombo_name(gid)

        if self.use_mecombotsv:
            emodel_name = self.mecombo_emodels[me_combo]
        else:
            emodel_name = me_combo

        return emodel_name

    def fetch_morph_name(self, gid):
        """Get the morph name of a gid"""

        cell_info = self.fetch_gid_cell_info(gid)

        morph_name = str(cell_info['morphology'])

        return morph_name

    def fetch_cell_kwargs(self, gid):
        """Get the kwargs to instantiate a gid's Cell object"""
        emodel_path = os.path.join(
            self.emodels_dir,
            self.fetch_emodel_name(gid) +
            '.hoc')

        morph_filename = '%s.%s' % \
            (self.fetch_morph_name(gid), 'asc')

        if self.use_mecombotsv:
            me_combo = self.fetch_mecombo_name(gid)
            extra_values = {
                'threshold_current': self.mecombo_thresholds[me_combo],
                'holding_current': self.mecombo_hypamps[me_combo]
            }
            cell_kwargs = {
                'template_filename': emodel_path,
                'morphology_name': morph_filename,
                'gid': gid,
                'record_dt': self.record_dt,
                'morph_dir': self.morph_dir,
                'template_format': 'v6',
                'extra_values': extra_values,
                'rng_settings': self.rng_settings
            }
        else:
            cell_kwargs = {
                'template_filename': emodel_path,
                'morphology_name': self.morph_dir,
                'gid': gid,
                'record_dt': self.record_dt,
                'rng_settings': self.rng_settings
            }

        return cell_kwargs

    def get_gids_of_targets(self, targets=None):

        gids = []
        for target in targets:
            gids.extend(self.bc_circuit.cells.ids(target))

        return gids

    def get_gids_of_mtypes(self, mtypes=None):
        """
        Helper function that, provided a BlueConfig, returns all the GIDs \
        associated with a specified M-type. (For instance, when you only want \
        to insert synapses of a specific pathway)


        Parameters
        ----------
        mtypes : list
            List of M-types (each as a string). Wildcards are *not* allowed, \
            the strings must represent the true M-type names. A list with \
            names can be found here: \
            bbpteam.epfl.ch/projects/spaces/display/MEETMORPH/m-types

        Returns
        -------
        gids : list
            List of all GIDs associated with one of the specified M-types

        """
        gids = []
        for mtype in mtypes:
            gids.extend(
                self.bc_circuit.cells.get({bluepy.v2.Cell.MTYPE: mtype}).
                index.values)

        return gids

    @staticmethod
    def get_nrn_h5_version(nrn_h5_path):
        """Get version of nrn.h5 file"""

        import h5py
        with h5py.File(nrn_h5_path, 'r') as nrn_h5:
            if 'info' not in nrn_h5 or 'version' not in nrn_h5['info'].attrs:
                version = None
            else:
                version_value = nrn_h5['info'].attrs['version']

                if version_value == 5 or version_value == [5]:
                    version = 5
                elif version_value == 3 or version_value == [3]:
                    version = 3
                elif version_value == 4 or version_value == [4]:
                    version = 4
                else:
                    raise ValueError(
                        'Unknown version in nrn.h5: %s' % str(version_value)
                    )

        return version


def _parse_outdat2(path):
    """Parse the replay spiketrains in a out.dat formatted file
       pointed to by path"""

    import bluepy.v2.impl.spike_report
    spikes = bluepy.v2.impl.spike_report.SpikeReport(path)

    outdat = {}

    for gid in spikes.gids:
        spike_times = spikes.get_gid(gid)
        if any(spike_times < 0):
            printv(
                'WARNING: SSim: Found negative spike times in out.dat ! '
                'Clipping them to 0', 2)
            spike_times = spike_times.clip(min=0.0)

        outdat[gid] = spike_times

    return outdat


def _parse_outdat(path, outdat_name='out.dat'):
    """Parse the replay spiketrains in out.dat"""
    full_outdat_name = os.path.join(path, outdat_name)
    return _parse_outdat2(full_outdat_name)
