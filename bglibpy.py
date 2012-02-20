import neuron
import numpy
import re
import matplotlib as plt
import pylab

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
        self.addRecordings(['soma(0.5)._ref_v', 'neuron.h._ref_t'])
        self.cell_dendrogram = None
        self.plotWindows = {}
        self.activeDendrogram = False

        neuron.h.finitialize()

        neuron.h.topology()

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

    def addRamp(self, start_time, stop_time, start_level, stop_level, dt=0.1):
        t_content = numpy.arange(start_time, stop_time, dt)
        i_content = [((stop_level-start_level)/(stop_time-start_time))*(x-start_time)+start_level for x in t_content]

        soma = [x for x in self.cell.getCell().somatic][0]
        soma.push()

        time = neuron.h.Vector()
        currents = neuron.h.Vector()
        time = time.from_python(t_content)
        currents = currents.from_python(i_content)

        pulse = neuron.h.new_IClamp(0.5)
        self.persistent.objects.append(pulse)
        self.persistent.objects.append(time)
        self.persistent.objects.append(currents)
        setattr(pulse, 'del', start_time)
        pulse.dur = stop_time - start_time
        currents.play(pulse._ref_amp, time)

        neuron.h.pop_section()

    def getTime(self):
        return numpy.array(self.getRecording('neuron.h._ref_t'))

    def getSomaVoltage(self):
        return numpy.array(self.getRecording('soma(0.5)._ref_v'))

    def addPlotWindow(self, var_name):
        if var_name not in self.recordings:
            self.addRecording(var_name)
        self.plotWindows[var_name] = PlotWindow(var_name, self)
        #self.plotWindows[var_name].queue = multiprocessing.Queue()
        #self.plotWindows[var_name].process = multiprocessing.Process(target=self.plotWindows[var_name].drawLoop, args=(self.plotWindows[var_name].queue,))
        #self.plotWindows[var_name].process.start()

    def activateDendrogram(self):
        self.activeDendrogram = True

    def showDendrogram(self):
        self.cell_dendrogram = dendrogram([x for x in self.cell.getCell().all])
        self.cell_dendrogram.redraw()

    def update(self):
        for var_name in self.plotWindows:
            self.plotWindows[var_name].redraw()
        #queue.put([self.getTime(),self.getRecording(var_name)])
        if self.cell_dendrogram and self.activeDendrogram:
            self.cell_dendrogram.redraw()

    def delete(self):
        if self.cell:
            if self.cell.getCell():
                self.cell.getCell().clear()
        for var_name in self.plotWindows:
            self.plotWindows[var_name].process.join()

    def __del__(self):
        self.delete()


class PlotWindow:
    def __init__(self, var_name, cell):
        self.cell = cell
        self.var_name = var_name
        pylab.ion()
        self.figure = pylab.figure(figsize=(10, 10))
        pylab.ioff()

        self.figure.gca().set_ylim(-100, 100)
        self.figure.gca().set_xlim(0, 1000)

        self.ax = self.figure.gca()
        self.canvas = self.ax.figure.canvas
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.line = pylab.Line2D(self.cell.getTime(), self.cell.getRecording(self.var_name))
        self.ax.add_line(self.line)
        self.figure.canvas.draw()

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
        time = self.cell.getTime()
        voltage = self.cell.getRecording(self.var_name)
        self.line.set_data(time, voltage)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

        return True


class dendrogram:
    def __init__(self, sections):
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

        self.proot.drawTree(self.dend_figure, 0, 0)
        self.dend_figure.canvas.draw()

        self.canvas = self.dend_figure.gca().figure.canvas
        self.ax = self.dend_figure.gca()
        self.background = self.canvas.copy_from_bbox(self.dend_figure.gca().bbox)

    def redraw(self):
        for psection in self.psections:
            psection.redraw()
        self.canvas.blit(self.ax.bbox)
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

        self.xSpacing = 10
        self.ySpacing = 1
        self.figure = None
        self.figX = None
        self.figY = None
        self.color_map = plt.cm.get_cmap("hot")
        self.ax = None
        self.patch = None

    def setupDraw(self, figure, x, y):
        self.figure = figure
        self.ax = self.figure.gca()
        self.figX = x
        self.figY = y
        #color = self.color_map((self.hsection.v+100)/200)
        self.patch = plt.patches.Rectangle([self.figX, self.figY], self.L, self.diam, facecolor="white", edgecolor="black")

    def redraw(self):
        self.patch.set_facecolor(self.color_map((self.hsection.v+100)/200))
        self.ax.add_patch(self.patch)
        self.ax.draw_artist(self.patch)

    def drawTree(self, figure, x, y):
        self.setupDraw(figure, x, y)
        new_x = x + self.L + self.xSpacing
        new_y = y
        for child in self.pchildren:
            child.drawTree(figure, new_x, new_y)
            pylab.plot([x+self.L, new_x], [y+self.diam/2, new_y+child.diam/2], 'k')
            new_y = new_y + child.treeHeight()

    def treeHeight(self):
        if self.isLeaf:
            treeHeight = self.diam + self.ySpacing
        else:
            treeHeight = 0
            for child in self.pchildren:
                treeHeight += child.treeHeight()

        return treeHeight

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


class simulation:
    def __init__(self, verbose_level=0):
        self.verbose_level = verbose_level
        self.cells = []
        self.steps_per_ms = 1

    def addCell(self, new_cell):
        self.cells.append(new_cell)

    def run(self, maxtime):
        neuron.h.tstop = 0.000001
        neuron.h.dt = 0.1

        neuron.h('{cvode_active(1)}')

        try:
            neuron.h.run()
        except Exception, e:
            print 'The Interneuron was eaten by the Python !\nReason: %s: %s' % (e.__class__.__name__, e)

        self.continuerun(maxtime)

    def continuerun(self, maxtime):
        for i in range(0, maxtime/1):
            for cell in self.cells:
                cell.update()
            neuron.h.tstop = neuron.h.tstop + 1
            if self.verbose_level >= 1:
                print str(1*i) + " ms"
            try:
                neuron.h.continuerun(neuron.h.tstop)
            except Exception, e:
                print 'The Interneuron was eaten by the Python !\nReason: %s: %s' % (e.__class__.__name__, e)
                break

    def __del__(self):
        pass
