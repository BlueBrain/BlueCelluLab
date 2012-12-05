import numpy
import re
import pylab
import math
import bglibpy
from bglibpy import neuron


class Cell:
    """Represents a bglib cell"""
    class persistent:
        """The objects that need to stay persistent in python"""
        objects = []

    def __init__(self, template_name, morphology_name):
        neuron.h.load_file(template_name)
        template_content = open(template_name, "r").read()
        match = re.search("begintemplate\s*(\S*)", template_content)
        cell_name = match.group(1)
        self.cell = eval("neuron.h." + cell_name + "(0, morphology_name)")
        self.recordings = {}
        self.synapses = {}
        self.netstims = {}
        self.connections = {}

        self.mechanisms = []  # BTN: all additional mechanism stored in one list. easy to delete...

        self.synapse_number = 0
        self.syn_vecs = {}
        self.syn_vecstims = {}
        self.syns = {}
        self.syn_netcons = {}
        self.ips = {}
        self.syn_mini_netcons = {}
        self.serialized = neuron.h.SerializedSections(self.cell.getCell())
        neuron.h.finitialize()

        self.soma = [x for x in self.cell.getCell().somatic][0]
        self.somatic = [x for x in self.cell.getCell().somatic]
        self.basal = [x for x in self.cell.getCell().basal]
        self.apical = [x for x in self.cell.getCell().apical]
        self.axonal = [x for x in self.cell.getCell().axonal]
        self.all = [x for x in self.cell.getCell().all]

        self.addRecordings(['self.soma(0.5)._ref_v', 'neuron.h._ref_t'])
        self.cell_dendrograms = []
        self.plotWindows = []

    def re_init_rng(self):
        """Reinitialize the random number generator for the stochastic channels"""
        self.cell.re_init_rng()
        neuron.h.finitialize()

    def get_section(self, raw_section_id):
        ''' use the serialized object to find your section'''
        return self.serialized.isec2sec[int(raw_section_id)].sec

    def location_to_point(self, syn_description, test=False):
        """need to put  description"""
        #pre_gid =  syn_description[0]
        post_sec_id = syn_description[2]
        isec = post_sec_id
        post_seg_id = syn_description[3]
        ipt = post_seg_id
        post_seg_distance = syn_description[4]
        syn_offset = post_seg_distance

        curr_sec = self.get_section(post_sec_id)
        L = curr_sec.L

        debug_too_large = 0
        debug_too_small = 0
        # access section to compute the distance
        if neuron.h.section_orientation(sec=self.get_section(isec)) == 1:
            ipt = neuron.h.n3d(sec=self.get_section(isec)) - 1 - ipt

        if ipt < neuron.h.n3d(sec=self.get_section(isec)):
            distance = (ipt + syn_offset) / L
            if distance >= 1.0:
                distance = 1
                debug_too_large = debug_too_large + 1

        if neuron.h.section_orientation(sec=self.get_section(isec)) == 1:
            distance = 1 - distance

        if distance <= 0:
            distance = 0
            debug_too_small = debug_too_small + 1

        if(test):
            print 'location_to_point:: %i <=0 and %i >= 1' % (debug_too_small, debug_too_large)

        return distance

    def getThreshold(self):
        """Get the spiking threshold of the cell"""
        return self.cell.getThreshold()

    def getHypAmp(self):
        """Get the current level necessary to bring the cell to -85 mV"""
        return self.cell.getHypAmp()

    def showDendDiam(self):
        """Show a dendrogram plot"""
        diamlist = [section.diam for section in self.apical]
        pylab.hist(diamlist, bins=int((max(diamlist) - min(diamlist)) / .1))
        pylab.show()

    def addRecording(self, var_name):
        """Add a recording to the cell"""
        #soma = [x for x in self.cell.getCell().somatic][0]
        #soma = soma
        recording = neuron.h.Vector()
        recording.record(eval(var_name))
        self.recordings[var_name] = recording

    def addRecordings(self, var_names):
        """Add a set of recordings to the cell"""
        for var_name in var_names:
            self.addRecording(var_name)

    def addAllSectionsVoltageRecordings(self):
        """Add a voltage recording to every section of the cell"""
        all_sections = self.cell.getCell().all
        for section in all_sections:
            var_name = 'neuron.h.' + section.name() + "(0.5)._ref_v"
            self.addRecording(var_name)

    def getAllSectionsVoltageRecordings(self):
        """Get all the voltage recordings from all the sections"""
        allSectionVoltages = {}
        all_sections = self.cell.getCell().all
        for section in all_sections:
            var_name = 'neuron.h.' + section.name() + "(0.5)._ref_v"
            allSectionVoltages[section.name()] = self.getRecording(var_name)
        return allSectionVoltages

    def getRecording(self, var_name):
        """Get recorded values"""
        return self.recordings[var_name].to_python()

    def add_replay_noise(self, gid, mean, variance, noise_seed, delay=0, dur=10000):
        """need to put  description"""
        rand = neuron.h.Random(gid + noise_seed)
        tstim = neuron.h.TStim(0.5, rand, sec=self.soma)  # self.get_section(0)) # assuming that section 0 is the soma
        tstim.noise(delay, dur, mean, variance)
        self.mechanisms.append(rand)
        self.mechanisms.append(tstim)

    def add_replay_synapse(self, syn_description, spike_train, SID=1, TGID=1, baseSeed=0, weightScalar=1.0,
                           spontMiniRate=0.0, nmda_ratio=-66, e_gaba=-80, dt=0.025, test=False):
        ''' Adds synapses the way BGLib does is. Supposedly.'''
        #pre_gid = syn_description[0]
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
        #syn_pre_m_type = syn_description[14]
        #syn_ASE = syn_description[17]

        ''' set up the vecevent thing '''
        t_vec = neuron.h.Vector(spike_train)
        t_vec_stim = neuron.h.VecStim()
        self.syn_vecs[SID] = t_vec
        self.syn_vecstims[SID] = t_vec_stim
        self.syn_vecstims[SID].play(self.syn_vecs[SID], dt)

        location = self.location_to_point(syn_description, test=test)
        distance = neuron.h.distance(location, sec=self.get_section(post_sec_id))
        if(syn_type < 100):
            ''' see: https://bbpteam.epfl.ch/wiki/index.php/BlueBuilder_Specifications#NRN,'''
            ''' inhibitory synapse '''
            if(test):
                print 'Inserting synapse in %i(%f)' % (post_sec_id, location)
            syn = neuron.h.ProbGABAAB_EMS(location, sec=self.get_section(post_sec_id))
            syn.e_GABAA = e_gaba
            syn.tau_d_GABAA = syn_DTC
            rng = neuron.h.Random()
            rng.MCellRan4(SID * 100000 + 100, TGID + 250 + baseSeed)  # +0 is the baseSeed, what's baseSeed?
            rng.lognormal(0.2, 0.1)
            syn.tau_r_GABAA = rng.repick()
        else:
            ''' else we have excitatory synapse '''
            syn = neuron.h.ProbAMPANMDA_EMS(location, sec=self.get_section(post_sec_id))
            syn.tau_d_AMPA = syn_DTC
            if (nmda_ratio != -66):
                syn.NMDA_ratio = nmda_ratio

        syn.Use = abs(syn_U)
        syn.Dep = abs(syn_D)
        syn.Fac = abs(syn_F)
        syn.synapseID = SID

        rndd = neuron.h.Random()
        rndd.MCellRan4(SID * 100000 + 100, TGID + 250 + baseSeed)
        rndd.uniform(0, 1)
        syn.setRNG(rndd)
        self.mechanisms.append(rndd)

        # self.mechanisms.append(syn)
        self.syns[SID] = syn

        self.syn_netcons[SID] = neuron.h.NetCon(self.syn_vecstims[SID], self.syns[SID], -30, delay, gsyn * weightScalar)  # ...,threshold,delay,weight

        ''' add the *minis*: spontaneous synaptic events '''
        if(spontMiniRate > 0.0):
            self.ips[SID] = neuron.h.InhPoissonStim(location, sec=self.get_section(post_sec_id))

            self.syn_mini_netcons[SID] = neuron.h.NetCon(self.ips[SID], self.syns[SID], -30, 0.1, gsyn * weightScalar)  # delay=0.1, fixed in Connection.hoc

            exprng = neuron.h.Random()
            exprng.MCellRan4(SID * 100000 + 200, TGID + 250 + baseSeed)
            exprng.negexp(1)
            self.mechanisms.append(exprng)
            uniformrng = neuron.h.Random()
            uniformrng.MCellRan4(SID * 100000 + 300, TGID + 250 + baseSeed)
            uniformrng.uniform(0.0, 1.0)
            self.mechanisms.append(uniformrng)
            self.ips[SID].setRNGs(exprng, uniformrng)
            tbins_vec = neuron.h.Vector(1)
            tbins_vec.x[0] = 0.0
            rate_vec = neuron.h.Vector(1)
            if(spontMiniRate == 0):
                rate_vec.x[0] = spontMiniRate  # hack to set ALL the spont mini rates to 0
            else:
                rate_vec.x[0] = spontMiniRate if (syn_type >= 100) else 0.012  # according to Blueconfig, ConInh-uni rule
            self.mechanisms.append(tbins_vec)
            self.mechanisms.append(rate_vec)
            self.ips[SID].setTbins(tbins_vec)
            self.ips[SID].setRate(rate_vec)

        return distance

    def addSynapticStimulus(self, section, location, delay=150, gmax=.000000002):
        """Add a synaptic stimulus to a certain section"""
        segname = section.name() + "(" + str(location) + ")"
        synapse = neuron.h.tmgExSyn(location, sec=section)
        synapse.Use = 0.5
        synapse.Fac = 21

        netstim = neuron.h.NetStim(sec=section)
        stimfreq = 70
        netstim.interval = 1000 / stimfreq
        netstim.number = 1
        netstim.start = delay
        netstim.noise = 0
        connection = neuron.h.NetCon(netstim, synapse, 10, 0, 700, sec=section)
        connection.weight[0] = 1.0
        self.synapses[segname] = synapse
        self.netstims[segname] = netstim
        self.connections[segname] = connection

    def locateBAPSite(self, seclistName, distance):
        """Return the location of the BAP site"""
        return [x for x in self.cell.getCell().locateBAPSite(seclistName, distance)]

    def removeSynapticStimulus(self, segname):
        """Removed a synaptic stimulus"""
        self.synapses[segname] = None
        self.netstims[segname] = None
        self.connections[segname] = None

    def addAllSynapses(self):
        """Add synapses to all dendritic sections"""
        dendritic_sections = [x for x in self.cell.getCell().basal] + [x for x in self.cell.getCell().apical]
        for section in dendritic_sections:
            self.addSynapticStimulus(section, 0)
            self.addSynapticStimulus(section, 0.5)
            self.addSynapticStimulus(section, 1)

    def injectCurrentWaveform(self, t_content, i_content):
        """Inject a current in the cell"""
        start_time = t_content[0]
        stop_time = t_content[-1]
        time = neuron.h.Vector()
        currents = neuron.h.Vector()
        time = time.from_python(t_content)
        currents = currents.from_python(i_content)

        pulse = neuron.h.new_IClamp(0.5, sec=self.soma)
        self.persistent.objects.append(pulse)
        self.persistent.objects.append(time)
        self.persistent.objects.append(currents)
        setattr(pulse, 'del', start_time)
        pulse.dur = stop_time - start_time
        currents.play(pulse._ref_amp, time)

    def get_childrensections(self, parentsection):
        """Get the children section of a neuron section"""
        number_children = neuron.h.SectionRef(sec=parentsection).nchild()
        children = []
        for index in range(0, int(number_children)):
            children.append(neuron.h.SectionRef(sec=self.soma).child[index])
        return children

    def get_parentsection(self, childsection):
        """Get the parent section of a neuron section"""
        print self.soma
        return neuron.h.SectionRef(sec=childsection).parent

    def addAxialCurrentRecordings(self, section):
        """Create a recording that will contain all the axial current flowing in and out of the section"""
        secname = neuron.h.secname(sec=section)
        self.addRecording(secname)
        for child in self.get_childrensections(section):
            self.addRecording(child)
        self.get_parentsection(section)

    def getAxialCurrentRecording(self, section):
        """Return the axial current recording"""
        secname = neuron.h.secname(sec=section)
        for child in self.get_childrensections(section):
            self.getRecording(secname)
            self.getRecording(child)

    def somatic_branches(self):
        """Show the index numbers """
        nchild = neuron.h.SectionRef(sec=self.soma).nchild()
        for index in range(0, int(nchild)):
            secname = neuron.h.secname(sec=neuron.h.SectionRef(sec=self.soma).child[index])
            if not "axon" in secname:
                if "dend" in secname:
                    dendnumber = int(secname.split("dend")[1].split("[")[1].split("]")[0])
                    secnumber = int(self.cell.getCell().nSecAxonalOrig + self.cell.getCell().nSecSoma + dendnumber)
                    print dendnumber, secnumber
                elif "apic" in secname:
                    apicnumber = int(secname.split("apic")[1].split("[")[1].split("]")[0])
                    secnumber = int(self.cell.getCell().nSecAxonalOrig + self.cell.getCell().nSecSoma + self.cell.getCell().nSecBasal + apicnumber)
                    print apicnumber, secnumber
                else:
                    raise Exception("somaticbranches: No apic or dend found in section %s" % secname)

    def apical_trunk(self):
        """Return the apical trunk of the cell"""
        if len(self.apical) is 0:
            return []
        else:
            apicaltrunk = []
            apicaltrunk.append(self.apical[0])
            currentsection = self.apical[0]
            while True:
                children = [neuron.h.SectionRef(sec=currentsection).child[index] for index in range(0, int(neuron.h.SectionRef(sec=currentsection).nchild()))]
                if len(children) is 0:
                    break
                maxdiam = 0
                for child in children:
                    if child.diam > maxdiam:
                        currentsection = child
                        maxdiam = child.diam
                apicaltrunk.append(child)
            return apicaltrunk

    def addRamp(self, start_time, stop_time, start_level, stop_level, dt=0.1):
        """Add a ramp current injection"""
        t_content = numpy.arange(start_time, stop_time, dt)
        i_content = [((stop_level - start_level) / (stop_time - start_time)) * (x - start_time) + start_level for x in t_content]
        self.injectCurrentWaveform(t_content, i_content)

    def addVClamp(self, stop_time, level):
        """Add a voltage clamp"""
        vclamp = neuron.h.SEClamp(0.5, sec=self.soma)
        vclamp.amp1 = level
        vclamp.dur1 = stop_time
        vclamp.dur2 = 0
        vclamp.dur3 = 0
        self.persistent.objects.append(vclamp)

    def addSineCurrentInject(self, start_time, stop_time, freq, amplitude, mid_level, dt=1.0):
        """Add a sinusoidal current injection"""
        t_content = numpy.arange(start_time, stop_time, dt)
        i_content = [amplitude * math.sin(freq * (x - start_time) * (2 * math.pi)) + mid_level for x in t_content]
        self.injectCurrentWaveform(t_content, i_content)
        return (t_content, i_content)

    def getTime(self):
        """Get the time vector"""
        return numpy.array(self.getRecording('neuron.h._ref_t'))

    def getSomaVoltage(self):
        """Get a vector of the soma voltage"""
        return numpy.array(self.getRecording('self.soma(0.5)._ref_v'))

    def getNumberOfSegments(self):
        """Get the number of segments in the cell"""
        totalnseg = 0
        for section in self.all:
            totalnseg += section.nseg
        return totalnseg

    def addPlotWindow(self, var_list, xlim=None, ylim=None, title=""):
        """Add a window to plot a variable"""
        xlim = [0, 1000] if xlim is None else xlim
        ylim = [-100, 100] if ylim is None else ylim
        for var_name in var_list:
            if var_name not in self.recordings:
                self.addRecording(var_name)
        self.plotWindows.append(bglibpy.PlotWindow(var_list, self, xlim, ylim, title))

    def showDendrogram(self, variable=None, active=False):
        """Show a dendrogram of the cell"""
        cell_dendrogram = bglibpy.dendrogram([x for x in self.cell.getCell().all], variable=variable, active=active)
        cell_dendrogram.redraw()
        self.cell_dendrograms.append(cell_dendrogram)

    def update(self):
        """Update all the windows"""
        for window in self.plotWindows:
            window.redraw()
        for cell_dendrogram in self.cell_dendrograms:
            cell_dendrogram.redraw()

    def delete(self):
        """Delete the cell"""
        if self.cell:
            if self.cell.getCell():
                self.cell.getCell().clear()
        #for window in self.plotWindows:
        #    window.process.join()
        ''' BTN, clear mechanisms as well '''
        self.mechanisms = []
        del(self.mechanisms)

    def __del__(self):
        self.delete()
