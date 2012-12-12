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
    def __init__(self,blueconfig_filename,dt=0.025) :
        """ Object dealing with BlueConfig configured Small Simulations

        To relieve from an empty stomach, eat spam and eggs

        Paramters
        ---------
        blueconfig_filename : Absolute filename of the Blueconfig to be used
        """
        self.dt = dt
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

    def instantiate_gids(self,gids,synapse_detail) :
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
        self.syns = {}
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
                                     path_of_morphology, gid=gid)
            ''' Setting up some internals / administration for later use '''
            self.cells[gid] = temp_cell
            self.mechanisms[gid] = []
            self.syns[gid] = {}
            self.syn_vecs[gid] = {}
            self.syn_vecstims[gid] = {}
            self.syn_ncs[gid] = {}
            self.ips[gid] = {}
            self.syn_mini_ncs[gid] = {}

            # self._add_replay_synapses(gid,gids,synapse_detail=synapse_detail)
            # self._add_replay_stimuli(gid)
            # self._charge_replay_synapses(gid,gids)

            if synapse_detail > 0 :
                # syn_parameters = self._parse_connection_parameters(gid)
                pre_datas = self.bc_simulation.circuit.\
                  get_presynaptic_data(gid)
                for syn_description, SID in zip(pre_datas, \
                                               range(len(pre_datas))) :
                    syn_parameters = self._evaluate_connection_parameters(syn_description[0],gid)
                    self.add_single_synapse(gid,SID,syn_description, \
                                            syn_parameters)
                print 'syn_parameters: ', syn_parameters

            if synapse_detail > 1 :
                ''' 2 or higher: add minis '''
                pass

            if synapse_detail >3:
                ''' 3 or higher: charge the synapses'''
                self.charge_replay_synapses()

    def setup_replay(self,gid) :
        syn_parameters = self._parse_connection_parameters(post_gid)
        pre_datas = ssim.bc_simulation.get_presynaptic_data(post_gid)


    def charge_replay_synapses():
        pass

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
        hypamp_i = 1.0 * self.cells[gid].getHypAmp()
        self.cells[gid].addRamp(0,10000,hypamp_i,hypamp_i,dt=self.dt)
        print 'hypamp injected<--------'

    def _add_replay_noise(self,gid,stimulus) :
        noise_seed = 0
        delay= float(stimulus.CONTENTS.Delay)
        dur= float(stimulus.CONTENTS.Duration)
        mean= float(stimulus.CONTENTS.MeanPercent)/100.0 * self.cells[gid].getThreshold()
        variance= float(stimulus.CONTENTS.Variance) * self.cells[gid].getThreshold()
        rand = bglibpy.neuron.h.Random(gid+noise_seed)
        tstim = bglibpy.neuron.h.TStim(0.5, rand, sec=self.cells[gid].soma)
        tstim.noise(delay, dur, mean, variance)
        self.mechanisms[gid].append(rand)
        self.mechanisms[gid].append(tstim)
        print '----------->noise injected<--------'

    def add_single_synapse(self,gid,SID,syn_description,connection_modifiers,synapse_level=0) :
        pre_gid = int(syn_description[0])
        delay = syn_description[1]
        post_sec_id = syn_description[2]
        #gsyn = syn_description[8]
        syn_U = syn_description[9]
        syn_D = syn_description[10]
        syn_F = syn_description[11]
        syn_DTC = syn_description[12]
        syn_type = syn_description[13]
        ''' --- TODO: what happens with -1 in location_to_point --- '''
        location = self._location_to_point(gid,syn_description)
        if location == None :
            print 'add_single_synapse: going to skip this synapse'
            return -1

        distance =  bglibpy.\
          neuron.h.distance(location,sec=self._get_section(gid,post_sec_id))

        if(syn_type < 100):
            ''' see: https://bbpteam.epfl.ch/\
            wiki/index.php/BlueBuilder_Specifications#NRN,
            inhibitory synapse
            '''
            syn = bglibpy.neuron.h.\
              ProbGABAAB_EMS(location, \
                             sec=self._get_section(gid,post_sec_id))

            syn.tau_d_GABAA = syn_DTC
            rng = bglibpy.neuron.h.Random()
            rng.MCellRan4(SID *100000+100, gid+250+self.base_seed)
            rng.lognormal(0.2, 0.1)
            syn.tau_r_GABAA = rng.repick()
        else:
            ''' else we have excitatory synapse '''
            syn = bglibpy.neuron.h.\
              ProbAMPANMDA_EMS(location,sec=self._get_section(gid,post_sec_id))
            syn.tau_d_AMPA = syn_DTC

        # hoc exec synapse configure blocks
        for cmd in connection_modifiers['SynapseConfigure']:
            cmd = cmd.replace('%s', '\n%(syn)s')
            bglibpy.neuron.h(cmd % {'syn': syn.hname()})

        syn.Use = abs( syn_U )
        syn.Dep = abs( syn_D )
        syn.Fac = abs( syn_F )
        syn.synapseID = SID

        rndd = bglibpy.neuron.h.Random()
        rndd.MCellRan4(SID*100000+100, gid+250+self.base_seed )
        rndd.uniform(0, 1)
        syn.setRNG(rndd)
        self.mechanisms[gid].append(rndd)

        # self.mechanisms.append(syn)
        self.syns[gid][SID] = syn
        return 1


    def _add_replay_synapses(self,gid,gids,synapse_detail=0,\
                             test=False) :
        """ Add to a post-synaptic cell as in the "large" cortical model. \
        Distinct levels of synapse detail can be specified. (Normally passed \
        from the instantiate method)

        Parameters:
        -----------

        """
        # fetch the synapse description
        pre_datas = self.bc_simulation.circuit.get_presynaptic_data(gid)

        # parse the synapse parameters
        syn_parameters = self._parse_connection_parameters(gid)

        # fetch the presynaptic spiketrains
        #pre_spike_trains = parse_and_store_GID_spiketrains(bg_dict['Run']['Default']['OutputRoot'],'out.dat')
        pre_spike_trains = parse_and_store_GID_spiketrains(self.bc.entry_map['Default'].CONTENTS.OutputRoot,'out.dat')

        # add the synapses to the model
        for syn_description,SID in zip(pre_datas,range(len(pre_datas))) :
            pre_gid = int(syn_description[0])
            delay = syn_description[1]
            post_sec_id = syn_description[2]
            #post_seg_id = syn_description[3]
            #post_seg_distance = syn_description[4]
            gsyn = syn_description[8]
            syn_U = syn_description[9]
            syn_D = syn_description[10]
            syn_F = syn_description[11]
            syn_DTC = syn_description[12]
            syn_type = syn_description[13]
            location = self._location_to_point(gid,syn_description,test=test)
            if location == None :
                print 'going to skip this synapse'
                raw_input('Press ENTER')
                continue
            distance =  bglibpy.\
              neuron.h.distance(location,sec=self._get_section(gid,post_sec_id))

            if(syn_type < 100):
                ''' see: https://bbpteam.epfl.ch/\
                wiki/index.php/BlueBuilder_Specifications#NRN,
                inhibitory synapse
                '''
                syn = bglibpy.neuron.h.\
                  ProbGABAAB_EMS(location, \
                                 sec=self._get_section(gid,post_sec_id))

                if('e_GABAA' in syn_parameters['SynapseConfigure'].keys()) :
                    syn.e_GABAA = syn_parameters['SynapseConfigure']['e_GABAA']
                syn.tau_d_GABAA = syn_DTC
                rng = bglibpy.neuron.h.Random()
                rng.MCellRan4(SID *100000+100, gid+250+self.base_seed)
                rng.lognormal(0.2, 0.1)
                syn.tau_r_GABAA = rng.repick()
            else:
                ''' else we have excitatory synapse '''
                syn = bglibpy.neuron.h.\
                  ProbAMPANMDA_EMS(location,sec=self._get_section(gid,post_sec_id))
                syn.tau_d_AMPA = syn_DTC
                if('NMDA_ratio' in syn_parameters['SynapseConfigure'].keys()) :
                    syn.NMDA_ratio = syn_parameters['SynapseConfigure']['NMDA_ratio']

            syn.Use = abs( syn_U )
            syn.Dep = abs( syn_D )
            syn.Fac = abs( syn_F )
            syn.synapseID = SID

            rndd = bglibpy.neuron.h.Random()
            rndd.MCellRan4(SID*100000+100, gid+250+self.base_seed )
            rndd.uniform(0, 1)
            syn.setRNG(rndd)
            self.mechanisms[gid].append(rndd)

            # self.mechanisms.append(syn)
            self.syns[gid][SID] = syn

            spike_train = [] # fetch with bluepy
            # spike_train = sorted(np.random.random_integers(low=0,high=1000,size=10))
            no_pre_spikes = 0
            try :
                spike_train = pre_spike_trains[pre_gid]
            except :
                no_pre_spikes = no_pre_spikes + 1
            #print 'no_pre_spikes: ', no_pre_spikes
            t_vec = bglibpy.neuron.h.Vector(spike_train)
            t_vec_stim = bglibpy.neuron.h.VecStim()
            self.syn_vecs[gid][SID] = t_vec
            self.syn_vecstims[gid][SID] = t_vec_stim
            self.syn_vecstims[gid][SID].play(self.syn_vecs[gid][SID], self.dt)

            if('Weight' in syn_parameters) :
                weight_scalar = syn_parameters['Weight']
            else :
                weight_scalar = 1.0

            self.syn_ncs[gid][SID] = bglibpy.neuron.h.NetCon(self.syn_vecstims[gid][SID], self.syns[gid][SID], -30, delay, gsyn*weight_scalar) # ...,threshold,delay,weight

            if('SpontMinis' in syn_parameters) :
                spont_minis = syn_parameters['SpontMinis']
            else :
                spont_minis = 0.0

            ''' add the *minis*: spontaneous synaptic events '''
            if(spont_minis > 0.0) :
                self.ips[gid][SID] = bglibpy.neuron.h.InhPoissonStim(location, sec=self._get_section(gid,post_sec_id))

                self.syn_mini_ncs[gid][SID] = bglibpy.neuron.h.NetCon(self.ips[gid][SID], self.syns[gid][SID], -30, 0.1, gsyn*weight_scalar) # delay=0.1, fixed in Connection.hoc

                exprng = bglibpy.neuron.h.Random()
                exprng.MCellRan4( SID*100000+200, gid+250+self.base_seed )
                exprng.negexp(1)
                self.mechanisms[gid].append(exprng)
                uniformrng = bglibpy.neuron.h.Random()
                uniformrng.MCellRan4( SID*100000+300, gid+250+self.base_seed )
                uniformrng.uniform(0.0, 1.0)
                self.mechanisms[gid].append(uniformrng)
                self.ips[gid][SID].setRNGs(exprng, uniformrng)
                tbins_vec = bglibpy.neuron.h.Vector(1)
                tbins_vec.x[0] = 0.0
                rate_vec = bglibpy.neuron.h.Vector(1)
                rate_vec.x[0] = spont_minis #spontMiniRate if(syn_type >=100) else  0.012 # according to Blueconfig, ConInh-uni rule
                self.mechanisms[gid].append(tbins_vec)
                self.mechanisms[gid].append(rate_vec)
                self.ips[gid][SID].setTbins(tbins_vec)
                self.ips[gid][SID].setRate(rate_vec)


    # def _charge_replay_synapses(self,gid,gids) :
    #     """ Connect pre-synaptic spikes
    #     Parameters:
    #     ----------
    #     """
    #     # fetch the synapse description
    #     pre_datas = self.bc_simulation.circuit.get_presynaptic_data(gid)

    #     # parse the synapse parameters
    #     syn_parameters = self._parse_connection_parameters(gid)

    #     for syn_description,SID in zip(pre_datas,range(len(pre_datas))) :
    #         spike_train = [] # fetch with bluepy
    #         spike_train = sorted(np.random.random_integers(low=0,high=1000,size=10))
    #         t_vec = bglibpy.neuron.h.Vector(spike_train)
    #         t_vec_stim = bglibpy.neuron.h.VecStim()
    #         self.syn_vecs[gid][SID] = t_vec
    #         self.syn_vecstims[gid][SID] = t_vec_stim
    #         self.syn_vecstims[gid][SID].play(self.syn_vecs[gid][SID], self.dt)

    #         self.syn_ncs[gid][SID] = bglibpy.neuron.h.NetCon(self.syn_vecstims[gid][SID], self.syns[gid][SID], -30, delay, gsyn*weightScalar) # ...,threshold,delay,weight

    def charge_replay_synapses(self,gid,gids) :
        """ Connect pre-synaptic spikes
        Parameters:
        ----------
        """
        # fetch the synapse description
        pre_datas = self.bc_simulation.circuit.get_presynaptic_data(gid)

        # parse the synapse parameters
        syn_parameters = self._parse_connection_parameters(gid)

        for syn_description,SID in zip(pre_datas,range(len(pre_datas))) :
            spike_train = [] # fetch with bluepy
            spike_train = sorted(np.random.random_integers(low=0,high=1000,size=10))
            t_vec = bglibpy.neuron.h.Vector(spike_train)
            t_vec_stim = bglibpy.neuron.h.VecStim()
            self.syn_vecs[gid][SID] = t_vec
            self.syn_vecstims[gid][SID] = t_vec_stim
            self.syn_vecstims[gid][SID].play(self.syn_vecs[gid][SID], self.dt)

            self.syn_ncs[gid][SID] = bglibpy.neuron.h.NetCon(self.syn_vecstims[gid][SID], self.syns[gid][SID], -30, delay, gsyn*weightScalar) # ...,threshold,delay,weight


    def _parse_connection_parameters(self,gid) :
        """ Parse the BlueConfig to find out the NMDA_ratio, etc. for the synapse
        Parameters:
        ----------
        gid : gid of the post-synaptic cell
        """
        parameters = {}
        parameters['SynapseConfigure'] = {}
        neurons = self.bc_simulation.circuit.mvddb.load_gids([gid], pbar=False)
        layer_of_gid = neurons[0].layer
        entries = self.bc_simulation.config.entries
        all_targets = self.bc_simulation.TARGETS.available_targets()
        for entry in entries :
            if(entry.TYPE == 'Connection') :
                ''' i assume desitnation is a "target"... '''
                destination = entry.CONTENTS.Destination
                if(destination in all_targets) :
                    if(gid in self.bc_simulation.get_target(destination)):
                        ''' whatever specified in this block, is applied to gid '''
                        if('Weight' in entry.CONTENTS.keys) :
                            parameters['Weight'] = float(entry.CONTENTS.Weight)
                            #print 'found weight: ', entry.CONTENTS.Weight
                        if('SpontMinis' in entry.CONTENTS.keys) :
                            parameters['SpontMinis'] = float(entry.CONTENTS.SpontMinis)
                            #print 'found SpontMinis: ', entry.CONTENTS.SpontMinis
                        if('SynapseConfigure' in entry.CONTENTS.keys) :
                            conf = entry.CONTENTS.SynapseConfigure
                            #print 'conf: ', conf
                            #key = conf[3:].split()[0]
                            key = conf.split()[0]
                            value = float(conf[3:].split()[2])
                            parameters['SynapseConfigure'].update({key:value})
                else:
                    raise ValueError, "Target '%s' not found in start.target or user.target" % destination

        #print 'params:\n', parameters
        return parameters


# parameters['SynapseConfiguration'].append(entry.CONTENTS.SynapseConfiguration)


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


# parameters['SynapseConfiguration'].append(entry.CONTENTS.SynapseConfiguration)


    def _get_section(self, gid,raw_section_id) :
        ''' use the serialized object to find your section'''
        return self.cells[gid].serialized.isec2sec[int(raw_section_id)].sec

    def _location_to_point(self, gid, syn_description, test=False):
        """need to put  description"""
        #pre_gid =  syn_description[0]
        post_sec_id = syn_description[2]
        isec = post_sec_id
        post_seg_id = syn_description[3]
        ipt = post_seg_id
        post_seg_distance = syn_description[4]
        syn_offset = post_seg_distance

        curr_sec = self._get_section(gid,post_sec_id)
        L = curr_sec.L

        debug_too_large = 0
        debug_too_small = 0
        # access section to compute the distance
        if(bglibpy. neuron.h.section_orientation(sec=self._get_section(gid,isec)) == 1) :
            ipt = bglibpy.neuron.h.n3d(sec=self._get_section(gid,isec)) -1 - ipt

        distance = -1
        if(ipt < bglibpy.neuron.h.n3d(sec=self._get_section(gid,isec)) ) :
            distance = ( bglibpy.neuron.h.arc3d(ipt,sec=self._get_section(gid,isec))+syn_offset)/L
            if(distance >= 1.0) :
                distance = 1.0
                debug_too_large = debug_too_large + 1

        if( bglibpy.neuron.h.section_orientation(sec=self._get_section(gid,isec)) == 1  ) :
            distance = 1 - distance

        if(distance <=0 ) :
            distance = None
            debug_too_small = debug_too_small + 1

        if(test) :
            print 'location_to_point:: %i <=0 and %i >= 1' % (debug_too_small, debug_too_large)

        return distance


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
        # ncs_file = open(self.bc.entry_map['Default'].CONTENTS.nrnPath +\
        #                 '/start.ncs')
        # for line in ncs_file.readlines() :
        #     stripped_line = line.strip()
        #     #print 'stripped line >%s<' % (stripped_line)
        #     #print stripped_line.split()
        #     try :
        #         found_gid = int( stripped_line.split()[0][1:])
        #         #print 'found_gid=', found_gid
        #         if(found_gid == gid) :
        #         #print 'found!'
        #             template_name = (stripped_line.split()[4]).strip()
        #             #print 'template_name: ', template_name
        #             #raw_input('ENTER')
        #     except:
        #         pass #print 'something went wrong, line: ', stripped_line
        # print 'parsed name: >', template_name,'<'
        neurons = self.bc_simulation.circuit.mvddb.load_gids([gid], pbar=False)
        template_name2 = str(neurons[0].METype)
        print 'parsed nam2: >', template_name2,'<'
        return template_name2

