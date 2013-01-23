#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class that represent can load BGLib BlueConfig, and instantiate the simulation

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

from bglibpy import bluepy
import bglibpy
import collections
import os
from bglibpy import printv
from bglibpy import printv_err

def parse_and_store_GID_spiketrains(path, outdat_name='out.dat'):
    """Parse and store the gid spiketrains"""

    gid_spiketimes_dict = collections.defaultdict(list)
    full_outdat_name = "%s/%s" % (path, outdat_name)

    if not os.path.exists(full_outdat_name):
        raise Exception("Could not find presynaptic spike file %s in %s" % (outdat_name, path))
    # read out.dat lines like 'spiketime, gid', ignore the first line, and the
    # last newline
    with open(full_outdat_name, "r") as full_outdat_file:
        for line in full_outdat_file.read().split("\n")[1:-1]:
            splits = line.split("\t")
            gid = int(splits[1])
            spiketime = float(splits[0])
            gid_spiketimes_dict[gid].append(spiketime)
    return gid_spiketimes_dict

class SSim(object):
    """SSim class"""

    def __init__(self, blueconfig_filename, dt=0.025, record_dt=None):
        """ Object dealing with BlueConfig configured Small Simulations

        To relieve from an empty stomach, eat spam and eggs

        Paramters
        ---------
        blueconfig_filename: Absolute filename of the Blueconfig to be used
        """
        self.dt = dt
        self.record_dt = record_dt
        self.blueconfig_filename = blueconfig_filename
        self.bc_simulation = bluepy.Simulation(blueconfig_filename)
        self.bc = self.bc_simulation.config
        try:
            self.base_seed  = int(self.bc.entry_map['Default'].CONTENTS.BaseSeed)
        except AttributeError:
            self.base_seed = 0 # in case the seed is not set, it's 0

        self.connection_entries = self.bc_simulation.config.typed_entries("Connection")
        self.all_targets = self.bc_simulation.TARGETS.available_targets()
        self.all_targets_dict = {}
        # if pre_gid in self.all_targets_dict[src]: #self.bc_simulation.get_target(src)
        for target in self.all_targets:
            self.all_targets_dict[target] = self.bc_simulation.get_target(target)

        self.gids = []
        self.templates = []
        self.cells = {}

        self.neuronconfigure_entries = self.bc_simulation.config.typed_entries("NeuronConfigure")
        self.neuronconfigure_expressions = {}
        for entry in self.neuronconfigure_entries:
            for gid in self.all_targets_dict[entry.CONTENTS.Target]:
                conf = entry.CONTENTS.Configure
                self.neuronconfigure_expressions.setdefault(gid, []).append(conf)

    def instantiate_gids(self, gids, synapse_detail=0, add_replay=False, add_stimuli=False):
        """ Instantiate a list of GIDs

        Parameters
        ----------
        gids: list of GIDs. Must be a list; even in case of instantiation of
        a single GID
        synapse_detail: Level of detail; if chosen, all settings are taken
         from the "large" cortical simulation. Possible values:
            0: No synapses
            1: Add synapse of the correct type at the simulated locations
               with all settings as in the "large" simulation
            2: As 1 but with minis
        add_replay: Add presynaptic spiketrains from the large simulation
            throws an exception if this is set when synapse_detail < 1
        add_stimuli: Add the same stimuli as in the large simulation
        """
        bgc_morph_path = self.bc.entry_map['Default'].CONTENTS.MorphologyPath
        # backwards compatible
        if bgc_morph_path[-3:] == "/h5":
            bgc_morph_path = bgc_morph_path[:-3]

        path_of_morphology = bgc_morph_path+'/ascii'

        self.gids = gids
        self.templates = []
        self.cells = {}

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

            pre_datas = self.bc_simulation.circuit.get_presynaptic_data(gid)

            if add_replay :
                pre_spike_trains = parse_and_store_GID_spiketrains(\
                                    self.bc.entry_map['Default'].CONTENTS.OutputRoot,\
                                    'out.dat')

            # Check if there are any presynaptic cells, otherwise skip adding
            # synapses
            if pre_datas is None:
                printv("No presynaptic cells found for gid %d, no synapses added" % gid, 1)
            else:
                for sid, syn_description in enumerate(pre_datas):
                    connection_parameters = self.\
                      _evaluate_connection_parameters(syn_description[0],gid)
                    if synapse_detail > 0:
                        self.add_single_synapse(gid, sid, syn_description, \
                                                connection_parameters)
                    if synapse_detail > 1:
                        self.add_replay_minis(gid, sid, syn_description, \
                                              connection_parameters)
                    if add_replay:
                        if synapse_detail < 1:
                            raise Exception("Cannot add replay stimulus if synapse_detail < 1")


                        self.charge_replay_synapse(gid, sid, syn_description, \
                                                connection_parameters, \
                                                pre_spike_trains)

                if synapse_detail > 0:
                    printv("Added synapses for gid %d" % gid, 1)
                if synapse_detail > 1:
                    printv("Added minis for gid %d" % gid, 1)
                if add_replay:
                    printv("Added presynaptic spiketrains for gid %d" % gid, 1)

            if add_stimuli:
                ''' Also add the injections / stimulations as in the cortical model '''
                self._add_replay_stimuli(gid)
                printv("Added stimuli for gid %d" % gid, 1)


    def _add_replay_stimuli(self, gid):
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
                    #print 'found stimulus: ', stimulus
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
        #hypamp_i = self.cells[gid].hypamp
        #self.cells[gid].addRamp(float(stimulus.CONTENTS.Delay), float(stimulus.CONTENTS.Delay)+float(stimulus.CONTENTS.Duration), hypamp_i, hypamp_i, dt=self.dt)
        self.cells[gid].add_replay_hypamp(stimulus)

    def _add_replay_noise(self, gid, stimulus, noise_seed=0):
        """Add noise injection from the replay"""
        self.cells[gid].add_replay_noise(stimulus, noise_seed=noise_seed)

    def add_replay_minis(self, gid, sid, syn_description, syn_parameters):
        """Add minis from the replay"""
        self.cells[gid].add_replay_minis(sid, syn_description, syn_parameters, self.base_seed)

    def charge_replay_synapse(self, gid, sid, syn_description, syn_parameters, pre_spike_trains):
        """Put the pre-spike-trains on the synapses"""
        pre_gid = int(syn_description[0])
        spike_train = pre_spike_trains.setdefault(pre_gid, None)
        if spike_train:
            self.cells[gid].charge_replay_synapse(sid, syn_description, syn_parameters, spike_train, stim_dt=self.dt)
        #else:
        #    print "Warning: presynaptic gid %d has no spikes, no netcon created" % pre_gid

    def add_single_synapse(self, gid, sid, syn_description, connection_modifiers):
        """Add a replay synapse on the cell"""
        self.cells[gid].add_replay_synapse(sid, syn_description, connection_modifiers, self.base_seed)

    def _evaluate_connection_parameters(self, pre_gid, post_gid):
        """ Apply connection blocks in order for pre_gid, post_gid to determine a final connection override for this pair (pre_gid, post_gid)
        Parameters:
        ----------
        gid: gid of the post-synaptic cell
        """
        parameters = {}

        for entry in self.connection_entries:
            src = entry.CONTENTS.Source
            dest = entry.CONTENTS.Destination

            # todo: this check should be done once up front for all connection blocks
            # and thus moved out of here

            # if (dest not in all_targets):
            #     raise ValueError, "Connection '%s' Destination target '%s' not found in start.target or user.target" % (entry.NAME, dest)
            # if (src not in all_targets):
            #     raise ValueError, "Connection '%s' Source target '%s' not found in start.target or user.target" % (entry.NAME, src)

            if pre_gid in self.all_targets_dict[src]: #self.bc_simulation.get_target(src):
                if post_gid in self.all_targets_dict[dest]:#self.bc_simulation.get_target(dest):
                    ''' whatever specified in this block, is applied to gid '''
                    if('Weight' in entry.CONTENTS.keys):
                        parameters['Weight'] = float(entry.CONTENTS.Weight)
                        #print 'found weight: ', entry.CONTENTS.Weight
                    if('SpontMinis' in entry.CONTENTS.keys):
                        parameters['SpontMinis'] = float(entry.CONTENTS.SpontMinis)
                        #print 'found SpontMinis: ', entry.CONTENTS.SpontMinis
                    if('SynapseConfigure' in entry.CONTENTS.keys):
                        conf = entry.CONTENTS.SynapseConfigure
                        # collect list of applicable configure blocks to be applied with a "hoc exec" statement
                        parameters.setdefault('SynapseConfigure', []).append(conf)
                    if('Delay' in entry.CONTENTS.keys):
                        import warnings
                        warnings.warn("Connection '%s': BlueConfig Delay keyword for connection blocks unsupported." % entry.NAME)

        #print 'params:\n', parameters
        return parameters

    def run(self, t_stop=None, v_init=-65, celsius=34, dt=None):
        """Simulate the SSim"""
        if t_stop is None:
            t_stop = float(self.bc.entry_map['Default'].CONTENTS.Duration)
        if dt is None:
            dt = float(self.bc.entry_map['Default'].CONTENTS.Dt)

        sim = bglibpy.Simulation()
        for gid in self.gids:
            sim.addCell(self.cells[gid])
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
