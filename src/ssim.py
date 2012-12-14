#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import bluepy
from bluepy.targets.mvddb import Neuron
import bglibpy

def parse_and_store_GID_spiketrains(path,fName='out.dat') :
    xhashed = {}
    inFile = open(path + '/' + fName,'r')
    inLine = inFile.readline()
    inLine = inFile.readline() # avoid the trailing "/scatter"
    counter = 0
    all_gids = []
    while(inLine != None) :
        try :
            # format: time gid
            t,gid = inLine.split() # returns 2 strings
            t = float(t)
            igid = int(gid)
            all_gids.append(igid)
            if(igid in xhashed) :
                xhashed[igid].append(t)
            else :
                xhashed[igid] = []
                xhashed[igid].append(t)
            counter = counter + 1
            inLine = inFile.readline()
        except :
            print 'no issue, most likely the last entry'
            print 'current inLine: ', inLine
            print 'current counter: ', counter
            inLine = inFile.readline()
            break

    print 'read and stored all spikes. now going to write the output'
    return xhashed

class SSim(object) :
    def __init__(self,blueconfig_filename, dt=0.025, record_dt=None) :
        """ Object dealing with BlueConfig configured Small Simulations

        To relieve from an empty stomach, eat spam and eggs

        Paramters
        ---------
        blueconfig_filename : Absolute filename of the Blueconfig to be used
        """
        self.dt = dt
        self.record_dt = record_dt
        self.blueconfig_filename = blueconfig_filename
        self.bc_simulation = bluepy.Simulation(blueconfig_filename)
        self.bc = self.bc_simulation.config
        try :
            self.base_seed  = self.bc.entry_map['Default'].CONTENTS.baseSeed
        except AttributeError:
            self.base_seed = 0 # in case the seed is not set, it's 0

        self.connection_entries = self.bc_simulation.config.typed_entries("Connection")
        self.all_targets = self.bc_simulation.TARGETS.available_targets()
        self.all_targets_dict = {}
        # if pre_gid in self.all_targets_dict[src]: #self.bc_simulation.get_target(src)
        for target in self.all_targets :
            self.all_targets_dict[target] = self.bc_simulation.get_target(target)

    def instantiate_gids(self,gids,synapse_detail,full=True) :
        """ Instantiate a list of GIDs

        Parameters
        ----------
        gids : list of GIDs. Must be a list; even in case of instantiation of\
        a single GID
        synapse_detail : Level of detail; if chosen, all settings are taken\
         from the "large" cortical simulation. Possible values:
            0: To be defined...
            1: Add synapse of the correct type at the simulated locations.\
            Preserves only the location and the type
            2: As one but with all settings as in the "large" simulation
            3: As 2 but with minis and all, as well as the real pre-synaptic\
            spiketrains.
        """
        bgc_morph_path = self.bc.entry_map['Default'].CONTENTS.MorphologyPath
        # backwards compatible
        if bgc_morph_path[-3:] == "/h5":
            bgc_morph_path=bgc_morph_path[:-3]

        path_of_morphology = bgc_morph_path+'/ascii'

        self.gids = gids
        self.templates = []
        self.cells = {}
        self.mechanisms = {}
        #self.syns = {}
        self.syn_vecs = {}
        self.syn_vecstims = {}
        self.syn_ncs = {}
        self.ips = {}
        self.syn_mini_ncs = {}

        for gid in self.gids :
            print 'setting up gid=%i' % (gid)
            ''' Fetch the template for this GID '''
            template_name_of_gid = self._fetch_template_name(gid)
            full_template_name_of_gid = self.bc.entry_map['Default'].CONTENTS.\
              METypePath+'/'+template_name_of_gid+'.hoc'
            print 'full_template_name_of_gid: ', full_template_name_of_gid

            temp_cell = bglibpy.Cell(full_template_name_of_gid,\
                                     path_of_morphology, gid=gid, \
                                     record_dt=self.record_dt)
            self.cells[gid] = temp_cell
            self.mechanisms[gid] = []
            #self.syns[gid] = {}
            self.syn_vecs[gid] = {}
            self.syn_vecstims[gid] = {}
            self.syn_ncs[gid] = {}
            self.ips[gid] = {}
            self.syn_mini_ncs[gid] = {}

            # self._add_replay_synapses(gid,gids,synapse_detail=synapse_detail)
            # self._add_replay_stimuli(gid)
            # self._charge_replay_synapses(gid,gids)

            pre_datas = self.bc_simulation.circuit.get_presynaptic_data(gid)
            pre_spike_trains = \
                      parse_and_store_GID_spiketrains(\
                        self.bc.entry_map['Default'].CONTENTS.OutputRoot,\
                        'out.dat')
            for syn_description, SID in zip(pre_datas, \
                                           range(len(pre_datas))) :
                syn_parameters = self.\
                  _evaluate_connection_parameters(syn_description[0],gid)
                if synapse_detail > 0 :
                    self.add_single_synapse(gid,SID,syn_description, \
                                            syn_parameters)
                if synapse_detail > 1 :
                    self.add_replay_minis(gid,SID,syn_description, \
                                          syn_parameters)
                if synapse_detail > 2 :
                    self.charge_replay_synapses(gid,SID,syn_description, \
                                                syn_parameters,\
                                                pre_spike_trains)
            print 'syn_parameters: ', syn_parameters

        if full :
            ''' Also add the injections / stimulations as in the cortical model '''
            self._add_replay_stimuli(gid)


    def _add_replay_stimuli(self,gid) :
        """ Adds indeitical stimuli to the simulated cell as in the 'large' model

        Parameters:
        -----------
        gid : gid of the simulated cell
        """
        # check in which StimulusInjects the gid is a target
        for entry in self.bc.entries :
            if entry.TYPE == 'StimulusInject' :
                destination = entry.CONTENTS.Target
                gids_of_target = self.bc_simulation.circuit.get_target(destination)
                if gid in gids_of_target :
                    # retrieve the stimulus to apply
                    stimulus_name = entry.CONTENTS.Stimulus
                    stimulus = self.bc.entry_map[stimulus_name]
                    print 'found stimulus: ', stimulus
                    if stimulus.CONTENTS.Pattern == 'Noise' :
                        self._add_replay_noise(gid,stimulus)
                    else :
                        self._add_replay_injection(gid,stimulus)

    def _add_replay_injection(self,gid,stimulus) :
        hypamp_i = 1.0 * self.cells[gid].hypamp
        self.cells[gid].addRamp(0,10000,hypamp_i,hypamp_i,dt=self.dt)
        print 'hypamp injected<--------'

    def _add_replay_noise(self,gid,stimulus) :
        noise_seed = 0
        delay= float(stimulus.CONTENTS.Delay)
        dur= float(stimulus.CONTENTS.Duration)
        mean= float(stimulus.CONTENTS.MeanPercent)/100.0 * self.cells[gid].threshold
        variance= float(stimulus.CONTENTS.Variance) * self.cells[gid].threshold
        rand = bglibpy.neuron.h.Random(gid+noise_seed)
        tstim = bglibpy.neuron.h.TStim(0.5, rand, sec=self.cells[gid].soma)
        tstim.noise(delay, dur, mean, variance)
        self.mechanisms[gid].append(rand)
        self.mechanisms[gid].append(tstim)
        print '----------->noise injected<--------'

    def add_replay_minis(self,gid,SID,syn_description,syn_parameters) :
        gsyn = syn_description[8]
        post_sec_id = syn_description[2]
        post_seg_id = syn_description[3]
        post_seg_distance = syn_description[4]
        location = self.cells[gid].\
          synlocation_to_segx(post_sec_id, post_seg_id, \
                              post_seg_distance, test=False)
        ''' TODO: False'''
        if('Weight' in syn_parameters) :
            weight_scalar = syn_parameters['Weight']
        else :
            weight_scalar = 1.0

        if('SpontMinis' in syn_parameters) :
            spont_minis = syn_parameters['SpontMinis']
        else :
            spont_minis = 0.0

        ''' add the *minis*: spontaneous synaptic events '''
        if spont_minis > 0.0 :
            print 'adding minis!'
            self.cells[gid].ips[SID] = bglibpy.neuron.h.\
              InhPoissonStim(location, \
                             sec=self.cells[gid].get_section(post_sec_id))

            self.cells[gid].syn_mini_netcons[SID] = bglibpy.neuron.h.\
              NetCon(self.cells[gid].ips[SID], self.cells[gid].syns[SID], \
                     -30, 0.1,gsyn*weight_scalar)#0.1, fixed in Connection.hoc

            exprng = bglibpy.neuron.h.Random()
            exprng.MCellRan4( SID*100000+200, gid+250+self.base_seed )
            exprng.negexp(1)
            self.mechanisms[gid].append(exprng)
            uniformrng = bglibpy.neuron.h.Random()
            uniformrng.MCellRan4( SID*100000+300, gid+250+self.base_seed )
            uniformrng.uniform(0.0, 1.0)
            self.mechanisms[gid].append(uniformrng)
            self.cells[gid].ips[SID].setRNGs(exprng, uniformrng)
            tbins_vec = bglibpy.neuron.h.Vector(1)
            tbins_vec.x[0] = 0.0
            rate_vec = bglibpy.neuron.h.Vector(1)
            rate_vec.x[0] = spont_minis
            self.mechanisms[gid].append(tbins_vec)
            self.mechanisms[gid].append(rate_vec)
            self.cells[gid].ips[SID].setTbins(tbins_vec)
            self.cells[gid].ips[SID].setRate(rate_vec)

    def charge_replay_synapses(self,gid,SID,syn_description,syn_parameters,pre_spike_trains) :
        pre_gid = int(syn_description[0])
        delay = syn_description[1]
        #post_sec_id = syn_description[2]
        #post_seg_id = syn_description[3]
        #post_seg_distance = syn_description[4]
        gsyn = syn_description[8]

        spike_train = [] # fetch with bluepy
        no_pre_spikes = 0
        try :
            spike_train = pre_spike_trains[pre_gid]
        except :
            no_pre_spikes = no_pre_spikes + 1
        print 'no_pre_spikes: ', no_pre_spikes
        t_vec = bglibpy.neuron.h.Vector(spike_train)
        t_vec_stim = bglibpy.neuron.h.VecStim()
        self.cells[gid].syn_vecs[SID] = t_vec
        self.cells[gid].syn_vecstims[SID] = t_vec_stim
        self.cells[gid].syn_vecstims[SID].play(self.cells[gid].syn_vecs[SID], self.dt)

        if('Weight' in syn_parameters) :
            weight_scalar = syn_parameters['Weight']
        else :
            weight_scalar = 1.0

        self.cells[gid].syn_netcons[SID] = bglibpy.neuron.h.NetCon(self.cells[gid].syn_vecstims[SID], self.cells[gid].syns[SID], -30, delay, gsyn*weight_scalar) # ...,threshold,delay,weight

    def add_single_synapse(self, gid, sid, syn_description, connection_modifiers, synapse_level=0):
        self.cells[gid].add_replay_synapse(sid, syn_description, connection_modifiers, self.base_seed, synapse_level=0)

    def _evaluate_connection_parameters(self, pre_gid, post_gid) :
        """ Apply connection blocks in order for pre_gid, post_gid to determine a final connection override for this pair (pre_gid, post_gid)
        Parameters:
        ----------
        gid : gid of the post-synaptic cell
        """
        parameters = {}

        for entry in self.connection_entries:
            src = entry.CONTENTS.Source
            dest = entry.CONTENTS.Destination

            # TODO: this check should be done once up front for all connection blocks
            # and thus moved out of here

            # if (dest not in all_targets):
            #     raise ValueError, "Connection '%s' Destination target '%s' not found in start.target or user.target" % (entry.NAME, dest)
            # if (src not in all_targets):
            #     raise ValueError, "Connection '%s' Source target '%s' not found in start.target or user.target" % (entry.NAME, src)

            if pre_gid in self.all_targets_dict[src]: #self.bc_simulation.get_target(src) :
                if post_gid in self.all_targets_dict[dest]:#self.bc_simulation.get_target(dest):
                    ''' whatever specified in this block, is applied to gid '''
                    if('Weight' in entry.CONTENTS.keys) :
                        parameters['Weight'] = float(entry.CONTENTS.Weight)
                        #print 'found weight: ', entry.CONTENTS.Weight
                    if('SpontMinis' in entry.CONTENTS.keys) :
                        parameters['SpontMinis'] = float(entry.CONTENTS.SpontMinis)
                        #print 'found SpontMinis: ', entry.CONTENTS.SpontMinis
                    if('SynapseConfigure' in entry.CONTENTS.keys) :
                        conf = entry.CONTENTS.SynapseConfigure
                        # collect list of applicable configure blocks to be applied with a "hoc exec" statement
                        parameters.setdefault('SynapseConfigure', []).append(conf)
                    if('Delay' in entry.CONTENTS.keys):
                        import warnings
                        warnings.warn("Connection '%s': BlueConfig Delay keyword for connection blocks unsupported." % entry.NAME)

        #print 'params:\n', parameters
        return parameters

    def _get_section(self, gid,raw_section_id) :
        ''' use the serialized object to find your section'''
        return self.cells[gid].get_section(raw_section_id)

    def simulate(self,t_stop=100,v_init=-65,celsius=34) :
        sim = bglibpy.Simulation()
        for gid in self.gids:
            sim.addCell(self.cells[gid])
        sim.run(t_stop, cvode=False, dt=0.025, celsius=celsius, v_init=v_init)

    def get_voltage_traces(self) :
        vm = {}
        for gid in self.gids :
            vm[gid] = self.cells[gid].getSomaVoltage()
        return vm

    def get_time(self) :
        return self.cells[self.gids[0]].getTime()



    """
    Auxialliary methods
    """
    def _fetch_template_name(self,gid) :
        neurons = self.bc_simulation.circuit.mvddb.load_gids([gid], pbar=False)
        template_name2 = str(neurons[0].METype)
        print 'parsed nam2: >', template_name2,'<'
        return template_name2
