# -*- coding: utf-8 -*-

"""
Cell class

@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.

"""

# pylint: disable=F0401, R0915

import numpy
import re
import math
import bglibpy
import os
from bglibpy import tools
from bglibpy.importer import neuron
from bglibpy import psection
import Queue


class Cell(object):

    """Represents a BGLib Cell object."""

    def __init__(self, template_name, morphology_name, gid=0, record_dt=None):
        """ Constructor.

        Parameters
        ----------
        template_name : string
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

        """

        # Persistent objects, like clamps, that exist as long
        # as the object exists
        self.persistent = []

        if not os.path.exists(template_name):
            raise Exception("Couldn't find template file [%s]" % template_name)

        # Load the template
        neuron.h.load_file(template_name)
        template_content = open(template_name, "r").read()
        match = re.search(r"begintemplate\s*(\S*)", template_content)
        cell_name = match.group(1)
        self.cell = eval("neuron.h." + cell_name + "(0, morphology_name)")
        self.soma = [x for x in self.cell.getCell().somatic][0]
        # WARNING: this finitialize 'must' be here, otherwhise the
        # diameters of the loaded morph are wrong
        neuron.h.finitialize()

        self.morphology_name = morphology_name
        self.template_name = template_name
        self.cellname = neuron.h.secname(sec=self.soma).split(".")[0]

        # Set the gid of the cell
        self.cell.getCell().gid = gid
        self.gid = gid

        self.recordings = {}  # Recordings in this cell
        self.synapses = {}  # Synapses on this cell
        self.netstims = {}  # Netstims connected to this cell
        self.connections = {}  # Outside connections to this cell

        self.pre_spiketrains = {}
        self.ips = {}
        self.syn_mini_netcons = {}
        self.serialized = neuron.h.SerializedSections(self.cell.getCell())

        self.soma = [x for x in self.cell.getCell().somatic][0]
        self.hocname = neuron.h.secname(sec=self.soma).split(".")[0]
        self.somatic = [x for x in self.cell.getCell().somatic]
        self.basal = [x for x in self.cell.getCell().basal]
        self.apical = [x for x in self.cell.getCell().apical]
        self.axonal = [x for x in self.cell.getCell().axonal]
        self.all = [x for x in self.cell.getCell().all]
        self.add_recordings(['self.soma(0.5)._ref_v', 'neuron.h._ref_t'],
                            dt=record_dt)
        self.cell_dendrograms = []
        self.plot_windows = []

        self.fih_plots = None
        self.fih_weights = None

        # As long as no PlotWindow or active Dendrogram exist, don't update
        self.plot_callback_necessary = False
        self.delayed_weights = Queue.PriorityQueue()
        self.secname_to_isec = {}
        self.secname_to_hsection = {}
        self.secname_to_psection = {}

        try:
            self.hypamp = self.cell.getHypAmp()
        except AttributeError:
            self.hypamp = None

        try:
            self.threshold = self.cell.getThreshold()
        except AttributeError:
            self.threshold = None

        self.psections = {}
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
        for psec in self.psections.itervalues():
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

    def re_init_rng(self):
        """Reinitialize the random number generator for stochastic channels."""
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
                "SSim: get_psection requires or a section_id or a secname")

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
        sec_ref = self.serialized.isec2sec[int(section_id)]
        if sec_ref:
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
                if mech_name not in ["k_ion", "na_ion", "ca_ion", "pas"]:
                    neuron.h('uninsert %s' % mech_name, sec=section)

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
            x_s = numpy.arange(1.0 / (2 * section.nseg), 1.0,
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
        length = curr_sec.L

        # access section to compute the distance
        if neuron.h.section_orientation(sec=self.get_hsection(isec)) == 1:
            ipt = neuron.h.n3d(sec=self.get_hsection(isec)) - 1 - ipt
            syn_offset = -syn_offset

        distance = 0.5
        if ipt < neuron.h.n3d(sec=self.get_hsection(isec)):
            distance = (neuron.h.arc3d(ipt, sec=self.get_hsection(isec)) +
                        syn_offset) / length
            if distance >= 1.0:
                distance = 1.0

        if neuron.h.section_orientation(sec=self.get_hsection(isec)) == 1:
            distance = 1 - distance

        if distance < 0:
            print "WARNING: synlocation_to_segx found negative distance \
                    at curr_sec(%s) syn_offset: %f" \
                        % (neuron.h.secname(sec=curr_sec), syn_offset)
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
            # recording.record(eval(var_name),
            # (1.0+neuron.h.float_epsilon)/(1.0/dt))
            recording.record(eval(var_name), dt)
        else:
            recording.record(eval(var_name))
        self.recordings[var_name] = recording

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
            allSectionVoltages[section.name()] = self.getRecording(var_name)
        return allSectionVoltages

    def get_recording(self, var_name):
        """Get recorded values.


        Returns
        -------
        numpy array : array with the recording var_name variable values

        """
        return self.recordings[var_name].to_python()

    def add_replay_hypamp(self, stimulus):
        """Inject hypamp for the replay."""
        tstim = bglibpy.neuron.h.TStim(0.5, sec=self.soma)
        tstim.pulse(float(stimulus.CONTENTS.Delay),
                    float(stimulus.CONTENTS.Duration), self.hypamp)
        self.persistent.append(tstim)

    def add_replay_noise(self, stimulus, noise_seed=0):
        """Add a replay noise stimulus."""
        mean = (float(stimulus.CONTENTS.MeanPercent) * self.threshold) / 100.0
        variance = (float(stimulus.CONTENTS.Variance) * self.threshold) / 100.0
        delay = float(stimulus.CONTENTS.Delay)
        duration = float(stimulus.CONTENTS.Duration)
        self.add_noise_step(self.soma, 0.5, mean, variance, delay, duration,
                            self.gid + noise_seed)

    def add_noise_step(self, section,
                       segx,
                       mean, variance,
                       delay,
                       duration, seed):
        """Inject a step current with noise on top."""
        rand = bglibpy.neuron.h.Random(seed)
        tstim = bglibpy.neuron.h.TStim(segx, rand, sec=section)
        tstim.noise(delay, duration, mean, variance)
        self.persistent.append(rand)
        self.persistent.append(tstim)

    def add_replay_synapse(self, synapse_id, syn_description,
                           connection_modifiers, base_seed):
        """Add synapse based on the syn_description to the cell.

        This operation can fail.  Returns True on success, otherwise False.

        """

        isec = syn_description[2]
        ipt = syn_description[3]
        syn_offset = syn_description[4]

        location = self.synlocation_to_segx(isec, ipt, syn_offset)
        if location is None:
            print 'WARNING: add_single_synapse: skipping a synapse at \
                        isec %d ipt %f' % (isec, ipt)
            return False

        self.synapses[synapse_id] = bglibpy.Synapse(
            self, location, synapse_id, syn_description,
            connection_modifiers, base_seed)
        return True

    def add_replay_delayed_weight(self, sid, delay, weight):
        """Add a synaptic weight for sid that will be set with a time delay."""
        self.delayed_weights.put((delay, (sid, weight)))

    def pre_gids(self):
        """List of gids of cells that connect to this cell

        Returns
        -------
        A list of gids of cell that connect to this cell.
        """

        pre_gid_list = []
        for syn_id in self.synapses:
            pre_gid_list.append(self.synapses[syn_id].pre_gid)

        return pre_gid_list

    def pre_gid_synapse_id(self, pre_gid):
        """List of synapse_id's of synapses a cell uses to connect to this cell

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

    def create_netcon_spikedetector(self, target):
        """Add and return a spikedetector.

        This is a NetCon that detects spike in the current cell, and that
        connects to target

        Returns
        -------

        NetCon : Neuron netcon object

        """

        # M. Hines magic to return a variable by reference to a python function
        netcon = neuron.h.ref(None)
        self.cell.getCell().connect2target(target, netcon)
        netcon = netcon[0]

        return netcon

    def add_replay_minis(self, sid, syn_description, connection_parameters,
                         base_seed):
        """Add minis from the replay."""
        weight = syn_description[8]
        post_sec_id = syn_description[2]
        post_seg_id = syn_description[3]
        post_seg_distance = syn_description[4]
        location = self.\
            synlocation_to_segx(post_sec_id, post_seg_id,
                                post_seg_distance)
        # todo: False
        if('Weight' in connection_parameters):
            weight_scalar = connection_parameters['Weight']
        else:
            weight_scalar = 1.0

        if('SpontMinis' in connection_parameters):
            # add the *minis*: spontaneous synaptic events
            spont_minis_rate = connection_parameters['SpontMinis']
            self.ips[sid] = bglibpy.neuron.h.\
                InhPoissonStim(location,
                               sec=self.get_hsection(post_sec_id))

            delay = 0.1
            self.syn_mini_netcons[sid] = bglibpy.neuron.h.\
                NetCon(self.ips[sid], self.synapses[sid].hsynapse,
                       -30, delay, weight * weight_scalar)

            exprng = bglibpy.neuron.h.Random()
            exp_seed1 = sid * 100000 + 200
            exp_seed2 = self.gid + 250 + base_seed
            exprng.MCellRan4(exp_seed1, exp_seed2)
            exprng.negexp(1.0)
            self.persistent.append(exprng)
            uniformrng = bglibpy.neuron.h.Random()
            uniform_seed1 = sid * 100000 + 300
            uniform_seed2 = self.gid + 250 + base_seed
            uniformrng.MCellRan4(uniform_seed1, uniform_seed2)
            uniformrng.uniform(0.0, 1.0)
            self.persistent.append(uniformrng)
            self.ips[sid].setRNGs(exprng, uniformrng)
            tbins_vec = bglibpy.neuron.h.Vector(1)
            tbins_vec.x[0] = 0.0
            rate_vec = bglibpy.neuron.h.Vector(1)
            rate_vec.x[0] = spont_minis_rate
            self.persistent.append(tbins_vec)
            self.persistent.append(rate_vec)
            self.ips[sid].setTbins(tbins_vec)
            self.ips[sid].setRate(rate_vec)
            # print "Added minis gid:%d, sid:%d, rate:%f, seed:%d,%d/%d,%d" % \
            # (self.gid, sid, spont_minis_rate, exp_seed1, exp_seed2, \
            # uniform_seed1, uniform_seed2)

    def charge_replay_synapse(self, sid,
                              syn_description,
                              connection_parameters,
                              pre_spiketrain,
                              stim_dt=None):
        """Put the replay spiketrains from out.dat on the synapses."""

        if sid in self.connections:
            raise Exception("Cell: trying to add a connection twice to the \
                                        same synapse id: %d" % sid)
        else:
            self.connections[sid] = \
                bglibpy.Connection(self.synapses[sid].hsynapse,
                                   syn_description,
                                   connection_parameters,
                                   pre_spiketrain=pre_spiketrain,
                                   pre_cell=None,
                                   stim_dt=stim_dt)

    def initialize_synapses(self):
        """Initialize the synapses."""
        for synapse in self.synapses.itervalues():
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
        """Return the location of the BAP site."""
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

    def get_parentsection(self, childsection):
        """Get the parent section of a neuron section.

        Returns
        -------

        section : parent section of the specified child section

        """
        print self.soma
        return neuron.h.SectionRef(sec=childsection).parent

    def addAxialCurrentRecordings(self, section):
        """Record all the axial current flowing in and out of the section."""
        secname = neuron.h.secname(sec=section)
        self.addRecording(secname)
        for child in self.get_childrensections(section):
            self.addRecording(child)
        self.get_parentsection(section)

    def getAxialCurrentRecording(self, section):
        """Return the axial current recording."""
        secname = neuron.h.secname(sec=section)
        for child in self.get_childrensections(section):
            self.getRecording(secname)
            self.getRecording(child)

    def somatic_branches(self):
        """Show the index numbers."""
        nchild = neuron.h.SectionRef(sec=self.soma).nchild()
        for index in range(0, int(nchild)):
            secname = neuron.h.secname(sec=neuron.h.SectionRef(
                sec=self.soma).child[index])
            if not "axon" in secname:
                if "dend" in secname:
                    dendnumber = int(secname.split("dend")[
                                     1].split("[")[1].split("]")[0])
                    secnumber = int(
                        self.cell.getCell().nSecAxonalOrig +
                        self.cell.getCell().nSecSoma + dendnumber)
                elif "apic" in secname:
                    apicnumber = int(secname.split(
                        "apic")[1].split("[")[1].split("]")[0])
                    secnumber = int(self.cell.getCell().nSecAxonalOrig
                                    + self.cell.getCell().nSecSoma +
                                    self.cell.getCell().nSecBasal + apicnumber)
                    print apicnumber, secnumber
                else:
                    raise Exception(
                        "somaticbranches: No apic or \
                                dend found in section %s" % secname)

    def apical_trunk(self):
        """Return the apical trunk of the cell."""
        if len(self.apical) is 0:
            return []
        else:
            apicaltrunk = []
            apicaltrunk.append(self.apical[0])
            currentsection = self.apical[0]
            while True:
                children = \
                    [neuron.h.SectionRef(sec=currentsection).child[index]
                     for index in range(0,
                                        int(neuron.h.SectionRef(sec=
                                            currentsection).nchild()))]
                if len(children) is 0:
                    break
                maxdiam = 0
                for child in children:
                    if child.diam > maxdiam:
                        currentsection = child
                        maxdiam = child.diam
                        apicaltrunk.append(child)
            return apicaltrunk

    def add_ramp(self, start_time, stop_time, start_level, stop_level,
                 dt=0.1, location=None):
        """Add a ramp current injection."""
        t_content = numpy.arange(start_time, stop_time, dt)
        i_content = [((stop_level - start_level)
                      / (stop_time - start_time)) * (
            x - start_time) + start_level for x in t_content]
        self.injectCurrentWaveform(t_content, i_content, location=location)

    def addVClamp(self, stop_time, level):
        """Add a voltage clamp."""
        vclamp = neuron.h.SEClamp(0.5, sec=self.soma)
        vclamp.amp1 = level
        vclamp.dur1 = stop_time
        vclamp.dur2 = 0
        vclamp.dur3 = 0
        self.persistent.append(vclamp)

    def addSineCurrentInject(self, start_time, stop_time, freq,
                             amplitude, mid_level, dt=1.0):
        """Add a sinusoidal current injection.

        Returns
        -------

        (numpy array, numpy array) : time and current data

        """
        t_content = numpy.arange(start_time, stop_time, dt)
        i_content = [amplitude * math.sin(freq * (x - start_time) * (
            2 * math.pi)) + mid_level for x in t_content]
        self.injectCurrentWaveform(t_content, i_content)
        return (t_content, i_content)

    def get_time(self):
        """Get the time vector."""
        return numpy.array(self.get_recording('neuron.h._ref_t'))

    def get_soma_voltage(self):
        """Get a vector of the soma voltage."""
        return numpy.array(self.get_recording('self.soma(0.5)._ref_v'))

    def getNumberOfSegments(self):
        """Get the number of segments in the cell."""
        totalnseg = 0
        for section in self.all:
            totalnseg += section.nseg
        return totalnseg

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

    def add_dendrogram(self, variable=None, active=False):
        """Show a dendrogram of the cell."""
        cell_dendrogram = bglibpy.Dendrogram(
            self.psections, variable=variable, active=active)
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
                abs(self.delayed_weights.queue[0][0] -
                    neuron.h.t) < neuron.h.dt:
            (_, (sid, weight)) = self.delayed_weights.get()
            if sid in self.connections:
                self.connections[sid].post_netcon.weight[0] = weight
                # print "Changed weight of synapse id %d to %f at time %f" %
                # (sid, weight, neuron.h.t)

        if not self.delayed_weights.empty():
            neuron.h.cvode.event(self.delayed_weights.queue[
                                 0][0], self.weights_callback)

    def plot_callback(self):
        """Update all the windows."""
        for window in self.plot_windows:
            window.redraw()
        for cell_dendrogram in self.cell_dendrograms:
            cell_dendrogram.redraw()

        neuron.h.cvode.event(neuron.h.t + 1, self.plot_callback)

    def delete(self):
        """Delete the cell."""
        if self.cell:
            if self.cell.getCell():
                self.cell.getCell().clear()

            self.fih_plots = None
            self.fih_weights = None
            self.connections = None
            self.synapses = None

        for persistent_object in self.persistent:
            del(persistent_object)

    @property
    def hsynapses(self):
        """Contains a dictionary of all the hoc synapses
        in the cell with as key the gid"""
        return dict((gid, synapse.hsynapse) for (gid, synapse)
                    in self.synapses.iteritems())

    def __del__(self):
        self.delete()

    # Deprecated functions ###

    # pylint: disable=C0111, C0112

    @property
    @tools.deprecated("hsynapses")
    def syns(self):
        """Contains a list of the hoc synapses with as key the gid."""
        return self.hsynapses

    @tools.deprecated()
    def getThreshold(self):
        """Get the threshold current of the cell.

        warning: this is measured from hypamp"""
        return self.cell.threshold

    @tools.deprecated()
    def getHypAmp(self):
        """Get the current level necessary to bring the cell to -85 mV."""
        return self.cell.hypamp

    @tools.deprecated("add_recording")
    def addRecording(self, var_name):
        return self.add_recording(var_name)

    @tools.deprecated("add_recordings")
    def addRecordings(self, var_names):
        return self.add_recordings(var_names)

    @tools.deprecated("get_recording")
    def getRecording(self, var_name):
        return self.get_recording(var_name)

    @tools.deprecated()
    def addAllSectionsVoltageRecordings(self):
        """Deprecated."""
        self.add_allsections_voltagerecordings()

    @tools.deprecated()
    def getAllSectionsVoltageRecordings(self):
        """Deprecated."""
        return self.get_allsections_voltagerecordings()

    @tools.deprecated()
    def locateBAPSite(self, seclistName, distance):
        """Deprecated."""
        return self.locate_bapsite(seclistName, distance)

    def injectCurrentWaveform(self, t_content, i_content, location=None):
        """Inject a current in the cell"""
        start_time = t_content[0]
        stop_time = t_content[-1]
        time = neuron.h.Vector()
        currents = neuron.h.Vector()
        time = time.from_python(t_content)
        currents = currents.from_python(i_content)

        if location is None:
            location = self.soma
        pulse = neuron.h.new_IClamp(0.5, sec=location)
        self.persistent.append(pulse)
        self.persistent.append(time)
        self.persistent.append(currents)
        setattr(pulse, 'del', start_time)
        pulse.dur = stop_time - start_time
        # pylint: disable=W0212
        currents.play(pulse._ref_amp, time)

    @tools.deprecated("get_time")
    def getTime(self):
        return self.get_time()

    @tools.deprecated()
    def getSomaVoltage(self):
        """Deprecated by get_soma_voltage."""
        return self.get_soma_voltage()

    @tools.deprecated("add_plot_window")
    def addPlotWindow(self, *args, **kwargs):
        self.add_plot_window(*args, **kwargs)

    @tools.deprecated("add_dendrogram")
    def showDendrogram(self, *args, **kwargs):
        """"""
        self.add_dendrogram(*args, **kwargs)

    @tools.deprecated("add_ramp")
    def addRamp(self, *args, **kwargs):
        self.add_ramp(*args, **kwargs)

    # pylint: enable=C0111, C0112
