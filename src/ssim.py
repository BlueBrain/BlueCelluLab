#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SSim Class

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

from bglibpy import bluepy
import bglibpy
import collections
import os
from bglibpy import printv
from bglibpy import printv_err

class SSim(object):
    """Class that can load a BGLib BlueConfig,
               and instantiate the simulation"""

    def __init__(self, blueconfig_filename, dt=0.025, record_dt=None):
        """Object dealing with BlueConfig configured Small Simulations

        To relieve from an empty stomach, eat spam and eggs

        Parameters
        ----------
        blueconfig_filename : string
                              Absolute filename of the Blueconfig

        """
        self.dt = dt
        self.record_dt = record_dt
        self.blueconfig_filename = blueconfig_filename
        self.bc_simulation = bluepy.Simulation(blueconfig_filename)
        self.bc = self.bc_simulation.config
        try:
            self.base_seed  = \
                int(self.bc.entry_map['Default'].CONTENTS.BaseSeed)
        except AttributeError:
            self.base_seed = 0 # in case the seed is not set, it's 0

        self.connection_entries = \
                self.bc_simulation.config.typed_entries("Connection")
        self.all_targets = self.bc_simulation.TARGETS.available_targets()
        self.all_targets_dict = {}
        for target in self.all_targets:
            self.all_targets_dict[target] = \
                self.bc_simulation.get_target(target)

        self.gids = []
        self.templates = []
        self.cells = {}

        self.neuronconfigure_entries = \
                self.bc_simulation.config.typed_entries("NeuronConfigure")
        self.neuronconfigure_expressions = {}
        for entry in self.neuronconfigure_entries:
            for gid in self.all_targets_dict[entry.CONTENTS.Target]:
                conf = entry.CONTENTS.Configure
                self.neuronconfigure_expressions.\
                        setdefault(gid, []).append(conf)

        self.gids_instantiated = False
        self.connections = \
                collections.defaultdict(\
                        lambda: collections.defaultdict(\
                                lambda: None))

    def instantiate_gids(self, gids, synapse_detail=None,
            add_replay=False,
            add_stimuli=False,
            add_synapses=False,
            add_minis=False,
            intersect_pre_gids=None,
            interconnect_cells=True):
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
        add_stimuli : Boolean
                      Add the same stimuli as in the large simulation
        add_synapses : Boolean
                       Add the touch-detected synapses, as described by the
                       circuit to the cell
                       (This option only influence the 'creation' of synapses,
                       it doesn't add any connections)
        add_minis : Boolean
                    Add synaptic minis to the synapses
                    (this requires add_synapses=True)
        intersect_pre_gids : Boolean
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
        """

        if synapse_detail != None:
            if synapse_detail > 0:
                add_synapses = True
            if synapse_detail > 1:
                add_minis = True

        """
        if not add_synapses:
            if interconnect_cells or add_replay or add_minis or
                        intersect_pre_gids != None:
                raise Exception("SSim: instantiate_gids() received an \
                        interconnect_cells, add_replay, add_minis, \
                        intersect_pre_gids request while add_synapses is \
                        False, this is not allowed")
        """

        if self.gids_instantiated:
            raise Exception("SSim: instantiate_gids() called twice on the \
                    same SSim, this is not supported yet")
        else:
            self.gids_instantiated = True

        self._add_cells(gids)
        if add_stimuli:
            self._add_stimuli()
        if add_synapses:
            self._add_synapses(intersect_pre_gids=intersect_pre_gids,
                    add_minis=add_minis)
        if add_replay or interconnect_cells:
            self._add_connections(add_replay=add_replay,
                    interconnect_cells=interconnect_cells)


    def _add_stimuli(self):
        """Instantiate all the stimuli"""
        for gid in self.gids:
            ''' Also add the injections / stimulations as in the cortical model '''
            self._add_stimuli_gid(gid)
            printv("Added stimuli for gid %d" % gid, 2)

    def _add_synapses(self, intersect_pre_gids=None, add_minis=None):
        """Instantiate all the synapses"""
        for gid in self.gids:
            syn_descriptions = self.bc_simulation.circuit.get_presynaptic_data(gid)

            if intersect_pre_gids != None:
                syn_descriptions = [syn_description for syn_description in syn_descriptions
                        if syn_description[0] in intersect_pre_gids]

            # Check if there are any presynaptic cells, otherwise skip adding
            # synapses
            if syn_descriptions is None:
                printv("Warning: No presynaptic cells found for gid %d, no synapses added" % gid, 2)
            else:
                for syn_id, syn_description in enumerate(syn_descriptions):
                    self._instantiate_synapse(gid, syn_id, syn_description,
                                                        add_minis=add_minis)
                printv("Added synapses for gid %d" % gid, 2)
                if add_minis:
                    printv("Added minis for gid %d" % gid, 2)


    def _add_connections(self, add_replay=None, interconnect_cells=None):
        """Instantiate the (replay and real) connections in the network"""
        pre_spike_trains = _parse_outdat(\
                            self.bc.entry_map['Default'].CONTENTS.OutputRoot,\
                            'out.dat')

        for post_gid in self.gids:
            for syn_id in self.cells[post_gid].synapses:
                synapse = self.cells[post_gid].synapses[syn_id]
                syn_description = synapse.syn_description
                connection_parameters = synapse.connection_parameters
                pre_gid = syn_description[0]
                if pre_gid in self.gids and interconnect_cells:
                    real_synapse_connection = True
                else:
                    real_synapse_connection = False

                connection = None
                if real_synapse_connection:
                    connection = bglibpy.Connection(\
                            self.cells[post_gid].synapses[syn_id],
                            pre_spiketrain=None,
                            pre_cell=self.cells[pre_gid],
                            stim_dt=self.dt)
                    printv("Added real connection between pre_gid %d and \
                            post_gid %d, syn_id %d" % (pre_gid, \
                                                        post_gid, \
                                                        syn_id), 5)
                elif add_replay:
                    pre_spiketrain = pre_spike_trains.setdefault(pre_gid, None)
                    if pre_spiketrain:
                        connection = bglibpy.Connection(self.cells[post_gid].synapses[syn_id],
                                                                pre_spiketrain=pre_spiketrain,
                                                                pre_cell=None,
                                                                stim_dt=self.dt)
                        printv("Added replay connection to post_gid %d, syn_id %d" % (post_gid, syn_id), 5)

                if connection != None:
                    self.cells[post_gid].connections[syn_id] = connection
                    if "DelayWeights" in connection_parameters:
                        for delay, weight in connection_parameters['DelayWeights']:
                            self.cells[post_gid].add_replay_delayed_weight(syn_id, delay, weight)

            if len(self.cells[post_gid].connections) > 0:
                printv("Added synaptic connections for target post_gid %d" % post_gid, 2)

    def _add_cells(self, gids):
        """Instantiate cells from a gid list"""
        self.gids = gids
        self.templates = []
        self.cells = {}

        bgc_morph_path = self.bc.entry_map['Default'].CONTENTS.MorphologyPath
        # backwards compatible
        if bgc_morph_path[-3:] == "/h5":
            bgc_morph_path = bgc_morph_path[:-3]

        path_of_morphology = bgc_morph_path+'/ascii'

        for gid in self.gids:
            ''' Fetch the template for this GID '''
            template_name_of_gid = self._fetch_template_name(gid)
            full_template_name_of_gid = self.bc.entry_map['Default'].CONTENTS.\
              METypePath+'/'+template_name_of_gid+'.hoc'
            printv('Added gid %d from template %s' % (gid, full_template_name_of_gid), 1)

            self.cells[gid] = bglibpy.Cell(full_template_name_of_gid, \
                                     path_of_morphology, gid=gid, \
                                     record_dt=self.record_dt)

            if gid in self.neuronconfigure_expressions:
                for expression in self.neuronconfigure_expressions[gid]:
                    self.cells[gid].execute_neuronconfigure(expression)


    def _instantiate_synapse(self, gid, syn_id, syn_description, add_minis=None):
        """Instantiate one synapse for a given gid, syn_id and syn_description"""
        syn_type = syn_description[13]

        connection_parameters = self.\
          _evaluate_connection_parameters(syn_description[0], gid, syn_type)

        if connection_parameters["add_synapse"]:
            self.add_single_synapse(gid, syn_id, syn_description, \
                                    connection_parameters)
            if add_minis:
                self.add_replay_minis(gid, syn_id, syn_description, \
                                    connection_parameters)

    def _add_stimuli_gid(self, gid):
        """ Adds indeitical stimuli to the simulated cell as in the 'large' model

        Parameters:
        -----------
        gid: gid of the simulated cell
        """
        # check in which StimulusInjects the gid is a target
        noise_seed = 0 # Every noise stimulus gets a new seed
        for entry in self.bc.entries:
            if entry.TYPE == 'StimulusInject':
                destination = entry.CONTENTS.Target
                gids_of_target = self.bc_simulation.get_target(destination)
                if gid in gids_of_target:
                    # retrieve the stimulus to apply
                    stimulus_name = entry.CONTENTS.Stimulus
                    stimulus = self.bc.entry_map[stimulus_name]
                    if stimulus.CONTENTS.Pattern == 'Noise':
                        self._add_replay_noise(gid, stimulus, noise_seed=noise_seed)
                        noise_seed += 1
                    elif stimulus.CONTENTS.Pattern == 'Hyperpolarizing':
                        self._add_replay_hypamp_injection(gid, stimulus)
                    elif stimulus.CONTENTS.Pattern == 'SynapseReplay':
                        printv("Found stimulus with pattern %s, ignoring" % stimulus.CONTENTS.Pattern, 1)
                    else:
                        printv_err("Found stimulus with pattern %s, not supported" % stimulus.CONTENTS.Pattern, 1)
                        exit(1)

    def _add_replay_hypamp_injection(self, gid, stimulus):
        """Add injections from the replay"""
        self.cells[gid].add_replay_hypamp(stimulus)

    def _add_replay_noise(self, gid, stimulus, noise_seed=0):
        """Add noise injection from the replay"""
        self.cells[gid].add_replay_noise(stimulus, noise_seed=noise_seed)

    def add_replay_minis(self, gid, syn_id, syn_description, syn_parameters):
        """Add minis from the replay"""
        self.cells[gid].add_replay_minis(syn_id,
                                                syn_description,
                                                syn_parameters,
                                                self.base_seed)

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
                                                      connection_modifiers,
                                                      self.base_seed)

    @staticmethod
    def check_connection_contents(contents):
        """Check the contents of a connection block, to see if we support all the fields"""
        for key in contents.keys:
            if key not in ['Weight', 'SynapseID', 'SpontMinis', 'SynapseConfigure', 'Source', 'Destination', 'Delay']:
                raise Exception("Key %s in Connection blocks not supported by BGLibPy" % key)

    def _evaluate_connection_parameters(self, pre_gid, post_gid, syn_type):
        """ Apply connection blocks in order for pre_gid, post_gid to determine a final connection override for this pair (pre_gid, post_gid)

        Parameters:
        -----------
        gid : int
              gid of the post-synaptic cell

        """
        parameters = {}
        parameters['add_synapse'] = False
        spontminis_set = False

        for entry in self.connection_entries:
            self.check_connection_contents(entry.CONTENTS)
            src = entry.CONTENTS.Source
            dest = entry.CONTENTS.Destination

            if pre_gid in self.all_targets_dict[src]:
                if post_gid in self.all_targets_dict[dest]:
                    ''' whatever specified in this block, is applied to gid '''
                    apply_parameters = True

                    if 'SynapseID' in entry.CONTENTS.keys:
                        if int(entry.CONTENTS.SynapseID) != syn_type:
                            apply_parameters = False

                    if 'Delay' in entry.CONTENTS.keys:
                        parameters.setdefault('DelayWeights', []).append((float(entry.CONTENTS.Delay), float(entry.CONTENTS.Weight)))
                        apply_parameters = False

                    if apply_parameters:
                        parameters['add_synapse'] = True
                        if 'Weight' in entry.CONTENTS.keys:
                            parameters['Weight'] = float(entry.CONTENTS.Weight)
                        if not spontminis_set:
                            if 'SpontMinis' in entry.CONTENTS.keys:
                                parameters['SpontMinis'] = float(entry.CONTENTS.SpontMinis)
                                spontminis_set = True
                            else:
                                parameters['SpontMinis'] = 0.0
                                spontminis_set = True
                        elif 'SpontMinis' in entry.CONTENTS.keys:
                            import warnings
                            warnings.warn("Connection '%s': SpontMinis was already set in previous block, IGNORING" % entry.NAME)

                        if 'SynapseConfigure' in entry.CONTENTS.keys:
                            conf = entry.CONTENTS.SynapseConfigure
                            # collect list of applicable configure blocks to be applied with a "hoc exec" statement
                            parameters.setdefault('SynapseConfigure', []).append(conf)

        return parameters

    def initialize_synapses(self):
        """ Resets the state of all synapses of all cells to initial values """
        for cell in self.cells.itervalues():
            cell.initialize_synapses()


    def run(self, t_stop=None, v_init=-65, celsius=34, dt=None):
        """Simulate the SSim"""
        if t_stop is None:
            t_stop = float(self.bc.entry_map['Default'].CONTENTS.Duration)
        if dt is None:
            dt = float(self.bc.entry_map['Default'].CONTENTS.Dt)

        sim = bglibpy.Simulation()
        for gid in self.gids:
            sim.add_cell(self.cells[gid])
        sim.run(t_stop, cvode=False, dt=dt, celsius=celsius, v_init=v_init)

    def get_voltage_traces(self):
        """Get the voltage traces from all the cells as a dictionary based on gid"""
        vm = {}
        for gid in self.gids:
            vm[gid] = self.cells[gid].get_soma_voltage()
        return vm

    def get_time(self):
        """Get the time vector for the recordings"""
        return self.cells[self.gids[0]].get_time()

    def __del__(self):
        """Destructor"""
        for gid in self.cells:
            self.cells[gid].delete()

    """
    Auxialliary methods
    """

    def _fetch_template_name(self, gid):
        """Get the template name of a gid"""
        neurons = self.bc_simulation.circuit.mvddb.load_gids([gid], pbar=False)
        if neurons[0]:
            template_name = str(neurons[0].METype)
        else:
            raise Exception("Gid %d not found in circuit" % gid)

        return template_name


    def get_gids_of_mtypes(self, mtypes=None):
        """
        Helper function that, provided a BlueConfig, returns all the GIDs \
        associated with a specified M-type. (For instance, when you only want \
        to insert synapses of a specific pathway)


        Parameters
        ----------
        mtypes : list
            List of M-types (each as a string). Wildcards are *not* allowed, \
            the strings must represent the true M-type names. A list with names \
            can be found here: \
            bbpteam.epfl.ch/projects/spaces/display/MEETMORPH/m-types

        Returns
        -------
        gids : list
            List of all GIDs associated with one of the specified M-types

        """
        # pylint: disable=W0511, E1101
        # TODO: this functionality doesn't belong here, and should over time
        # be moved to BluePy
        gids = []
        for mtype in mtypes :
            gids += self.bc_simulation.circuit.mvddb.select_gids( \
                             bluepy.targets.mvddb.MType.name == mtype)
        # pylint: enable=W0511, E1101
        return gids

def _parse_outdat(path, outdat_name='out.dat'):
    """Parse the replay spiketrains in out.dat"""

    gid_spiketimes_dict = collections.defaultdict(list)
    full_outdat_name = os.path.join(path, outdat_name)

    if not os.path.exists(full_outdat_name):
        raise IOError("Could not find presynaptic spike file at %s" \
                                               % full_outdat_name)
    # read out.dat lines like 'spiketime, gid', ignore the first line, and the
    # last newline
    with open(full_outdat_name, "r") as full_outdat_file:
        for line in full_outdat_file.read().split("\n")[1:-1]:
            splits = line.split("\t")
            gid = int(splits[1])
            spiketime = float(splits[0])
            gid_spiketimes_dict[gid].append(spiketime)
    return gid_spiketimes_dict
