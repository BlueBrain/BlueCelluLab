import sys
sys.path = ["/usr/local/nrnnogui/lib/python2.7/site-packages"]  + sys.path
import neuron
import numpy
import re
import matplotlib as plt
import pylab
import multiprocessing
import math


neuron.h.nrn_load_dll('/home/vangeit/scripts/bglibpy/i686/.libs/libnrnmech.so')
neuron.h.load_file("nrngui.hoc")

neuron.h('obfunc new_IClamp() { return new IClamp($1) }')

class Cell:
    class persistent:
        objects = []

    def __init__(self, template_name, morphology_name):
        neuron.h.load_file(template_name)
        template_content = open(template_name, "r").read()
        match = re.search("begintemplate\s*(\S*)", template_content)
        cell_name = match.group(1)
        self.cell = eval("neuron.h."+ cell_name +"(0, morphology_name)")
        self.recordings = {}
        self.synapses = {}
        self.netstims = {}
        self.connections = {}

        neuron.h.finitialize()

        self.soma = [x for x in self.cell.getCell().somatic][0]
        self.somatic = [x for x in self.cell.getCell().somatic]
        self.basal = [x for x in self.cell.getCell().basal]
        self.apical = [x for x in self.cell.getCell().apical]

        self.addRecordings(['soma(0.5)._ref_v', 'neuron.h._ref_t'])
        self.cell_dendrograms = []
        #self.cell_dendrogram = None
        self.plotWindows = {}
        #self.activeDendrogram = False


    def addRecording(self, var_name):
        soma = [x for x in self.cell.getCell().somatic][0]
        soma = soma
        recording = neuron.h.Vector()
        recording.record(eval(var_name))
        self.recordings[var_name] = recording

    def addRecordings(self, var_names):
        for var_name in var_names:
            self.addRecording(var_name)

    def addAllSectionsVoltageRecordings(self):
        all_sections = self.cell.getCell().all
        for section in all_sections:
            var_name = 'neuron.h.' + section.name() + "(0.5)._ref_v"
            self.addRecording(var_name)

    def getAllSectionsVoltageRecordings(self):
        allSectionVoltages = {}
        all_sections = self.cell.getCell().all
        for section in all_sections:
            var_name = 'neuron.h.' + section.name() + "(0.5)._ref_v"
            allSectionVoltages[section.name()] = self.getRecording(var_name)
        return allSectionVoltages

    def getRecording(self, var_name):
        return self.recordings[var_name].to_python()

    def addSynapticStimulus(self, section, location):
        segname = section.name() + "(" + str(location) + ")"
        #print segname
        synapse = neuron.h.tmgInhSyn(location, sec=section)
        synapse.Use = 0.25 #0.02
        synapse.Dep = 706 #194
        synapse.Fac = 21 #507
        synapse.e = -80
        netstim = neuron.h.NetStim(sec=section)
        stimfreq = 70
        netstim.interval = 1000/stimfreq
        netstim.number = 1
        netstim.start = 150
        netstim.noise = 0
        connection = neuron.h.NetCon(netstim, synapse, 10, 0, 700, sec=section)
        self.synapses[segname] = synapse
        self.netstims[segname] = netstim
        self.connections[segname] = connection

    def removeSynapticStimulus(self, segname):
        self.synapses[segname] = None
        self.netstims[segname] = None
        self.connections[segname] = None

    def addAllSynapses(self):
        dendritic_sections = [x for x in self.cell.getCell().basal] + [x for x in self.cell.getCell().apical]
        for section in dendritic_sections:
            #var_name = 'neuron.h.' + section.name() + "(0.5)._ref_v"
            self.addSynapticStimulus(section, 0)
            self.addSynapticStimulus(section, 0.5)
            self.addSynapticStimulus(section, 1)

    def injectCurrentWaveform(self, t_content, i_content):
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

    def addRamp(self, start_time, stop_time, start_level, stop_level, dt=0.1):
        t_content = numpy.arange(start_time, stop_time, dt)
        i_content = [((stop_level-start_level)/(stop_time-start_time))*(x-start_time)+start_level for x in t_content]
        self.injectCurrentWaveform(t_content, i_content)

    def addVClamp(self, stop_time, level):
        vclamp = neuron.h.SEClamp(0.5, sec=self.soma)
        vclamp.amp1 = level
        vclamp.dur1 = stop_time
        vclamp.dur2 = 0
        vclamp.dur3 = 0
        self.persistent.objects.append(vclamp)
        #t_content = numpy.arange(start_time, stop_time, dt)
        #i_content = [((stop_level-start_level)/(stop_time-start_time))*(x-start_time)+start_level for x in t_content]
        #self.injectCurrentWaveform(t_content, i_content)

    def addSineCurrentInject(self, start_time, stop_time, freq, amplitude, mid_level, dt=1.0):
        t_content = numpy.arange(start_time, stop_time, dt)
        i_content = [amplitude*math.sin(freq*(x-start_time)*(2*math.pi))+mid_level for x in t_content]
        self.injectCurrentWaveform(t_content, i_content)
        return (t_content, i_content)

    def getTime(self):
        return numpy.array(self.getRecording('neuron.h._ref_t'))

    def getSomaVoltage(self):
        return numpy.array(self.getRecording('soma(0.5)._ref_v'))

    def addPlotWindow(self, var_name, xlim=None, ylim=None):
        xlim = [0, 1000] if xlim is None else xlim
        ylim = [-100, 100] if ylim is None else ylim
        if var_name not in self.recordings:
            self.addRecording(var_name)
        self.plotWindows[var_name] = PlotWindow(var_name, self, xlim, ylim)

    def showDendrogram(self, variable=None, active=False):
        cell_dendrogram = dendrogram([x for x in self.cell.getCell().all], variable=variable, active=active)
        cell_dendrogram.redraw()
        self.cell_dendrograms.append(cell_dendrogram)

    def update(self):
        for var_name in self.plotWindows:
            self.plotWindows[var_name].redraw()
        for cell_dendrogram in self.cell_dendrograms:
            cell_dendrogram.redraw()

    def delete(self):
        if self.cell:
            if self.cell.getCell():
                self.cell.getCell().clear()
        for var_name in self.plotWindows:
            self.plotWindows[var_name].process.join()

    def __del__(self):
        self.delete()

def calculate_SS_voltage(template_name, morphology_name, step_level):
    pool = multiprocessing.Pool(processes=1)
    SS_voltage = pool.apply(calculate_SS_voltage_subprocess, [template_name, morphology_name, step_level])
    pool.terminate()
    return SS_voltage

def calculate_SS_voltage_subprocess(template_name, morphology_name, step_level):
    cell = Cell(template_name, morphology_name)
    cell.addRamp(500, 5000, step_level, step_level, dt=1.0)
    simulation = Simulation()
    simulation.run(1000)
    time = cell.getTime()
    voltage = cell.getSomaVoltage()
    #pylab.plot(time, voltage)
    #pylab.show()
    SS_voltage = numpy.mean(voltage[numpy.where((time < 1000) & (time > 800))])
    cell.delete()
    #SS_voltage = numpy.mean(cell.getSomaVoltage()[int(400.0*constants.dt):int(500.0*constants.dt)])

    return SS_voltage

def search_hyp_current(template_name, morphology_name, hyp_voltage, start_current, stop_current):
    med_current = start_current + abs(start_current-stop_current)/2
    new_hyp_voltage = calculate_SS_voltage(template_name, morphology_name, med_current)
    print "Detected voltage: ", new_hyp_voltage
    if abs(new_hyp_voltage - hyp_voltage) < .5:
        return med_current
    elif new_hyp_voltage > hyp_voltage:
        return search_hyp_current(template_name, morphology_name, hyp_voltage, start_current, med_current)
    elif new_hyp_voltage < hyp_voltage:
        return search_hyp_current(template_name, morphology_name, hyp_voltage, med_current, stop_current)

def detect_hyp_current(template_name, morphology_name, hyp_voltage):
    return search_hyp_current(template_name, morphology_name, hyp_voltage, -1.0, 0.0)

def detect_spike_step(template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level):
    pool = multiprocessing.Pool(processes=1)
    spike_detected = pool.apply(detect_spike_step_subprocess, [template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level])
    pool.terminate()
    return spike_detected

def detect_spike_step_subprocess(template_name, morphology_name, hyp_level, inj_start, inj_stop, step_level):
    cell = Cell(template_name, morphology_name)
    cell.addRamp(0, 5000, hyp_level, hyp_level, dt=1.0)
    cell.addRamp(inj_start, inj_stop, step_level, step_level, dt=1.0)
    simulation = Simulation()
    simulation.run(int(inj_stop))

    time = cell.getTime()
    voltage = cell.getSomaVoltage()
    time_step = time[numpy.where((time > inj_start) & (time < inj_stop))]
    voltage_step = voltage[numpy.where((time_step > inj_start) & (time_step < inj_stop))]
    spike_detected = numpy.max(voltage_step) > -20

    cell.delete()

    return spike_detected

def search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, start_current, stop_current):
    med_current = start_current + abs(start_current-stop_current)/2
    spike_detected = detect_spike_step(template_name, morphology_name, hyp_level, inj_start, inj_stop, med_current)
    print "Spike threshold detection at: ", med_current, "mV", spike_detected

    if abs(stop_current - start_current) < .01:
        return stop_current
    elif spike_detected:
        return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, start_current, med_current)
    elif not spike_detected:
        return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, med_current, stop_current)

def detect_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop):
    return search_threshold_current(template_name, morphology_name, hyp_level, inj_start, inj_stop, 0.0, 1.0)


def calculateAllSynapticAttenuations(bglibcell):
    sim = Simulation()
    sim.addCell(bglibcell)

    bglibcell.addAllSectionsVoltageRecordings()

    #all_sections = [x for x in bglibcell.cell.getCell().all]
    basal_sections = [x for x in bglibcell.cell.getCell().basal]
    apical_sections = [x for x in bglibcell.cell.getCell().apical]
    dendritic_sections = basal_sections + apical_sections
    normdistances = []
    absdistances = []
    attenuations = []
    sectiontypes = []
    #pylab.ion()
    #pylab.figure(11)
    #xvalues = numpy.arange(0,1,0.01)
    #pylab.plot(xvalues,numpy.exp(numpy.divide(xvalues,0.2)))

    for section in dendritic_sections:
        for location in [0.0, 0.5, 1]:
            segname = section.name() + "(" + str(location) + ")"
            #section.push()
            print segname
            synapse = neuron.h.tmgExSyn(location, sec=section)
            synapse.Use = 0.02
            synapse.Dep = 194
            synapse.Fac = 507
            synapse.gmax = synapse.gmax/5
            netstim = neuron.h.NetStim(sec=section)
            stimfreq = 70
            netstim.interval = 1000/stimfreq
            netstim.number = 1
            netstim.start = 150
            netstim.noise = 0
            connection = neuron.h.NetCon(netstim, synapse, 10, 0, 700, sec=section)
            sim.run(300)
            time = bglibcell.getTime()
            allsectionrecordings = bglibcell.getAllSectionsVoltageRecordings()
            soma = [x for x in bglibcell.cell.getCell().somatic][0]
            somavoltage = numpy.array(allsectionrecordings[soma.name()])
            dendvoltage = numpy.array(allsectionrecordings[section.name()])
            del connection
            #neuron.h.pop_section()
            #pylab.figure()
            #pylab.plot(somavoltage, label='soma')
            #pylab.plot(dendvoltage, label='dend')
            #pylab.legend()
            #pylab.show()

            interval_indices = numpy.where((time > netstim.start) & (time < netstim.start+90))
            max_soma = max(somavoltage[numpy.where((time > netstim.start) & (time < netstim.start+90))])
            min_soma = numpy.min(somavoltage[interval_indices])
            amp_soma = max_soma - min_soma
            max_dend = numpy.max(dendvoltage[interval_indices])
            min_dend = numpy.min(dendvoltage[interval_indices])
            amp_dend = max_dend - min_dend
            attenuation = amp_dend / amp_soma
            attenuations.append(attenuation)
            neuron.h.distance(sec=soma)
            absdistance = neuron.h.distance(location, sec=section)
            absdistances.append(absdistance)
            if section in basal_sections:
                sectiontypes.append("basal")
                normdistance = absdistance/bglibcell.cell.getLongestBranch(bglibcell.cell.getCell().basal)
            elif section in apical_sections:
                sectiontypes.append("apical")
                normdistance = absdistance/bglibcell.cell.getLongestBranch(bglibcell.cell.getCell().apical)
            else:
                print "Section is not in apical nor basal"
                exit(1)
            normdistances.append(normdistance)


    return normdistances, absdistances, attenuations, sectiontypes

def calculateAllSSAttenuations(bglibcell):
    sim = Simulation()
    sim.addCell(bglibcell)

    bglibcell.addAllSectionsVoltageRecordings()

    #all_sections = [x for x in bglibcell.cell.getCell().all]
    basal_sections = [x for x in bglibcell.cell.getCell().basal]
    apical_sections = [x for x in bglibcell.cell.getCell().apical]
    dendritic_sections = basal_sections + apical_sections
    normdistances = []
    absdistances = []
    attenuations = []
    sectiontypes = []

    for section in dendritic_sections:
        for location in [0.0, 0.25, 0.5, 0.75, 1]:
            segname = section.name() + "(" + str(location) + ")"
            #section.push()
            print segname
            clamp = neuron.h.IClamp(0.5, sec=section)
            clamp.dur = 300
            clamp.delay = 150
            clamp.amp = 0.01

            '''
            synapse = neuron.h.tmgExSyn(location, sec=section)
            synapse.Use = 0.02
            synapse.Dep = 194
            synapse.Fac = 507
            synapse.gmax = synapse.gmax/5
            netstim = neuron.h.NetStim(sec=section)
            stimfreq = 70
            netstim.interval = 1000/stimfreq
            netstim.number = 1
            netstim.start = 150
            netstim.noise = 0
            connection = neuron.h.NetCon(netstim, synapse, 10, 0, 700, sec=section)
            '''
            sim.run(300)

            time = bglibcell.getTime()
            allsectionrecordings = bglibcell.getAllSectionsVoltageRecordings()
            soma = [x for x in bglibcell.cell.getCell().somatic][0]
            somavoltage = numpy.array(allsectionrecordings[soma.name()])
            dendvoltage = numpy.array(allsectionrecordings[section.name()])

            #del connection
            #neuron.h.pop_section()
            #pylab.figure()
            #pylab.plot(somavoltage, label='soma')
            #pylab.plot(dendvoltage, label='dend')
            #pylab.legend()
            #pylab.show()

            interval_indices = numpy.where((time > clamp.delay) & (time < clamp.delay+90))
            max_soma = max(somavoltage[numpy.where((time > clamp.delay) & (time < clamp.delay+90))])
            min_soma = numpy.min(somavoltage[interval_indices])
            amp_soma = max_soma - min_soma
            max_dend = numpy.max(dendvoltage[interval_indices])
            min_dend = numpy.min(dendvoltage[interval_indices])
            amp_dend = max_dend - min_dend
            attenuation = amp_dend / amp_soma
            attenuations.append(attenuation)
            neuron.h.distance(sec=soma)
            absdistance = neuron.h.distance(location, sec=section)
            absdistances.append(absdistance)
            if section in basal_sections:
                sectiontypes.append("basal")
                normdistance = absdistance/bglibcell.cell.getLongestBranch(bglibcell.cell.getCell().basal)
            elif section in apical_sections:
                sectiontypes.append("apical")
                normdistance = absdistance/bglibcell.cell.getLongestBranch(bglibcell.cell.getCell().apical)
            else:
                print "Section is not in apical nor basal"
                exit(1)
            normdistances.append(normdistance)


    return normdistances, absdistances, attenuations, sectiontypes

class PlotWindow:
    def __init__(self, var_name, cell, xlim, ylim):
        self.cell = cell
        self.var_name = var_name
        pylab.ion()
        self.figure = pylab.figure(figsize=(10, 10))
        pylab.ioff()

        self.ax = self.figure.gca()
        self.canvas = self.ax.figure.canvas

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel("ms")
        self.ax.set_ylabel("mV")

        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        recording = self.cell.getRecording(self.var_name)
        if recording:
            time = self.cell.getTime()
        else:
            time = self.cell.getTime()[1:]

        self.line = pylab.Line2D(time, recording, label=self.var_name)
        self.ax.add_line(self.line)
        self.ax.legend()

        self.figure.canvas.draw()

        self.drawCount = 10

    '''
    def drawLoop(self, q):
        prev_time = 0
        [time,voltage] = q.get()
        while q.qsize() is not 0:
            while time[-1] - prev_time < 10:
                try:
                    [time, voltage] = q.get(True,1)
                except Queue.Empty:
                    break
            prev_time = time[-1]
            self.redraw(time,voltage)
    '''

    def redraw(self):
        if not self.drawCount:
            time = self.cell.getTime()
            voltage = self.cell.getRecording(self.var_name)
            self.line.set_data(time, voltage)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
            self.drawCount = 10
        else:
            self.drawCount = self.drawCount - 1
        return True


class dendrogram:
    def __init__(self, sections, variable=None, active=False):
        pylab.ion()
        self.dend_figure = pylab.figure(figsize=(20, 12))
        pylab.ioff()


        self.sections = sections
        neuron.h.finitialize()

        self.hroot = neuron.h.SectionRef(sec=self.sections[0]).root
        self.proot = PSection(self.hroot, None)
        self.psections = [self.proot] + self.proot.getAllPDescendants()

        pylab.xlim([0, self.proot.treeWidth()])
        pylab.ylim([0, self.proot.treeHeight()])
        pylab.gca().set_xticks([])
        pylab.gca().set_yticks([])
        pylab.gcf().subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0.3)

        if variable is "v":
            varbounds = [-70, 50]
        else:
            varbounds = self.proot.getTreeVarBounds(variable)

        cax = pylab.imshow(numpy.outer(numpy.arange(0, 1, 0.01), numpy.ones(1)), aspect='auto', cmap=pylab.get_cmap("hot"), origin="lower")
        pylab.clim(varbounds[0], varbounds[1])

        cbar = self.dend_figure.colorbar(cax, ticks=[varbounds[0], varbounds[1]])
        cbar.ax.set_yticklabels(["%.2e"%(varbounds[0]), "%.2e"%(varbounds[1])])

        self.proot.drawTree(self.dend_figure, 0, 0, variable=variable, varbounds=varbounds)
        self.dend_figure.canvas.draw()

        for psection in self.psections:
            psection.redraw()

        self.canvas = self.dend_figure.gca().figure.canvas
        self.ax = self.dend_figure.gca()
        self.canvas.blit(self.ax.bbox)

        self.background = self.canvas.copy_from_bbox(self.dend_figure.gca().bbox)
        self.drawCount = 10

        self.active = active

    def redraw(self):
        if self.active:
            if not self.drawCount:
                for psection in self.psections:
                    psection.redraw()
                self.canvas.blit(self.ax.bbox)
                self.drawCount = 50
            else:
                self.drawCount = self.drawCount - 1
        return True


class PSection:
    def __init__(self, hsection, pparent):
        self.L = hsection.L
        self.diam = hsection.diam
        self.hsection = hsection
        self.name = neuron.h.secname(sec=hsection)
        self.pparent = pparent
        self.hchildren = [neuron.h.SectionRef(sec=self.hsection).child[index] for index in range(0, int(neuron.h.SectionRef(sec=self.hsection).nchild()))]
        self.pchildren = []
        for hchild in self.hchildren:
            self.pchildren.append(PSection(hchild, self))
        self.isLeaf = False
        if not self.hchildren:
            self.isLeaf = True

        self.psegments = []
        self.maxsegdiam = 0
        for hsegment in hsection:
            psegment = PSegment(hsection(hsegment.x), self)
            self.psegments.append(psegment)
            self.maxsegdiam = max(self.maxsegdiam, psegment.diam)

        self.xSpacing = 10
        self.ySpacing = 1

    def setupDraw(self, figure, x, y, variable=None, varbounds=None):
        x_accum = 0
        for psegment in self.psegments:
            psegment.setupDraw(figure, x + x_accum, y + (self.maxsegdiam-psegment.diam)/2, variable=variable, varbounds=varbounds)
            x_accum += psegment.L

    def redraw(self):
        for psegment in self.psegments:
            psegment.redraw()

    def getSectionVarBounds(self, variable):
        varmin = None
        varmax = None
        for psegment in self.psegments:
            value = psegment.getVariableValue(variable)
            if value:
                varmin = min(value, varmin) if varmin else value
                varmax = max(value, varmax) if varmax else value
        return [varmin, varmax]

    def getTreeVarBounds(self, variable):
        varbounds = self.getSectionVarBounds(variable)
        for child in self.pchildren:
            child_varbounds = child.getTreeVarBounds(variable)
            if child_varbounds[0] and child_varbounds[1]:
                varbounds[0] = min(varbounds[0], child_varbounds[0]) if varbounds[0] else child_varbounds[0]
                varbounds[1] = max(varbounds[1], child_varbounds[1]) if varbounds[1] else child_varbounds[1]

        return varbounds

    def drawTree(self, figure, x, y, variable=None, varbounds=None):
        self.setupDraw(figure, x, y, variable=variable, varbounds=varbounds)
        new_x = x + self.L + self.xSpacing
        new_y = y
        for child in self.pchildren:
            child.drawTree(figure, new_x, new_y, variable=variable, varbounds=varbounds)
            pylab.plot([x+self.L, new_x], [y+self.diam/2, new_y+child.diam/2], 'k')
            new_y = new_y + child.treeHeight()

    def treeHeight(self):
        if self.isLeaf:
            treeHeight = self.maxsegdiam + self.ySpacing
        else:
            treeHeight = 0
            for child in self.pchildren:
                treeHeight += child.treeHeight()

        return max(self.diam + self.ySpacing, treeHeight)

    def treeWidth(self):
        return self.L + self.xSpacing + (max([child.treeWidth() for child in self.pchildren]) if self.pchildren else 0)

    def getHChildren(self):
        return self.hchildren

    def getPParent(self):
        return self.parent

    def getHParent(self):
        return self.parent.hsection

    def getAllPDescendants(self):
        pdescendants = [x for x in self.pchildren]
        for child in self.pchildren:
            pdescendants += child.getAllPDescendants()
        return pdescendants

    def getAllPLeaves(self):
        pleaves = []
        if not self.pchildren:
            pleaves.append(self)
        else:
            for child in self.pchildren:
                pleaves += child.getAllPLeaves()
        return pleaves

class PSegment:
    def __init__(self, hsegment, parentsection):
        self.hsegment = hsegment
        self.parentsection = parentsection
        self.L = self.parentsection.L/self.parentsection.hsection.nseg
        self.diam = self.hsegment.diam

        self.figure = None
        self.figX = None
        self.figY = None
        self.color_map = plt.cm.get_cmap("hot")
        self.ax = None
        self.patch = None
        self.plotvariable = None
        self.mechanism_names = [mechanism.name() for mechanism in hsegment]
        self.varbounds = None

    def setupDraw(self, figure, x, y, variable=None, varbounds=None):
        self.figure = figure
        self.plotvariable = variable
        self.varbounds = varbounds
        self.ax = self.figure.gca()
        self.figX = x
        self.figY = y
        self.patch = plt.patches.Rectangle([self.figX, self.figY], self.L, self.diam, facecolor="white", edgecolor="black")
        self.ax.add_patch(self.patch)

    def redraw(self):
        if self.plotvariable:
            #print self.varbounds
            plotvariable_value = self.getVariableValue(self.plotvariable)
            #print plotvariable_value
            if not plotvariable_value is None:
                self.patch.set_facecolor(self.color_map((plotvariable_value-self.varbounds[0])/(self.varbounds[1]-self.varbounds[0])))
            else:
                self.patch.set_facecolor(self.color_map(1.0))
                self.patch.set_hatch("/")
            self.ax.draw_artist(self.patch)


    def getVariableValue(self, variable):
        if variable is "v" or neuron.h.execute1("{%s.%s(%f)}" % (neuron.h.secname(sec=self.parentsection.hsection), variable, self.hsegment.x), 0):
            return eval("self.hsegment."+variable)
        else:
            return None

class Simulation:
    def __init__(self, verbose_level=0):
        self.verbose_level = verbose_level
        self.cells = []
        self.steps_per_ms = 1

    def addCell(self, new_cell):
        self.cells.append(new_cell)

    def run(self, maxtime, cvode=True):
        neuron.h.tstop = 0.000001
        neuron.h.dt = 0.1

        if cvode:
            neuron.h('{cvode_active(1)}')
        else:
            neuron.h('{cvode_active(0)}')

        try:
            neuron.h.run()
        except Exception, e:
            print 'The Interneuron was eaten by the Python !\nReason: %s: %s' % (e.__class__.__name__, e)

        self.continuerun(maxtime)

    def continuerun(self, maxtime):
        while neuron.h.t < maxtime:
            for cell in self.cells:
                cell.update()
            if self.verbose_level >= 1:
                print str(neuron.h.t) + " ms"
            try:
                neuron.h.step()
            except Exception, e:
                print 'The Interneuron was eaten by the Python !\nReason: %s: %s' % (e.__class__.__name__, e)
                break

    def __del__(self):
        pass
