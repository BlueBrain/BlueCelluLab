"""
Synapse test file. Uses ball-and-stick models
Models simulated in N disticnt ways, all shoudl
"""

import sys
sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path

import bglibpy
import numpy
#from bglibpy import bluepy

#sys.path.append('/home/torben/sandbox/willem/DendriteApprox/')
#from greensFunctionCalculator import compute_system

'''Due to some path difficulties with BlueConfig, change directory'''
#import os
#os.chdir('test/ballstick_test')

cyl_surface = lambda diam, h: numpy.pi * diam * h

class Params:
    """Simulation parameters"""
    def __init__(self):
        Params.T_STOP = 200
        Params.V_INIT = -75
        Params.DT = 0.025

        Params.SYN_DECAY = 2.0
        Params.SYN_E = 0
        Params.SYN_ACTIVATION_T = 50
        Params.SYN_DELAY = 0.025
        Params.SYN_G = 0.001
        Params.SYN_LOC = 1.0

        cell = bglibpy.Cell("ballstick.hoc", "./")
        self.soma_L, self.soma_D, self.soma_A = cell.soma.L, cell.soma.diam, bglibpy.neuron.h.area(0.5, sec=cell.soma)
        #print 'SOMA L=%f, diam=%f,surf=%f' % (self.soma_L, self.soma_D, self.soma_A)

        self.dend0_L, self.dend0_D, self.dend0_A  = cell.basal[0].L, cell.basal[0].diam, bglibpy.neuron.h.area(0.5, sec=cell.basal[0])
        self.dend0_NSEG = cell.basal[0].nseg

        print 'DENDRITE L=%f, diam=%f,surf=%f' % (self.dend0_L, self.dend0_D, self.dend0_A)

        ''' I assume uniform passive properties shared by the soma and dendrites '''
        self.CM = cell.soma.cm
        self.RM = 1.0 / cell.soma(0.5).g_pas
        self.RA = cell.soma.Ra
        self.EL = cell.soma(0.5).e_pas

def run_pyneuron(soma_l, soma_d, params) :
    """Run ballstick with PyNeuron"""
    soma = bglibpy.neuron.h.Section()
    soma.L = soma_l
    soma.diam = soma_d
    soma.nseg = 1
    soma.cm = params.CM
    soma.Ra = params.RA

    soma.insert('pas')
    soma(0.5).e_pas = params.EL
    soma(0.5).g_pas = 1.0 / params.RM

    dend = bglibpy.neuron.h.Section()
    dend.L = params.dend0_L
    dend.diam = params.dend0_D
    dend.nseg = params.dend0_NSEG

    dend.cm = params.CM
    dend.Ra = params.RA
    dend.insert('pas')
    for seg in dend :
        seg.e_pas = params.EL
        seg.g_pas = 1.0/params.RM
    dend.connect(soma, 0.5, 0) # mid-soma to begin-den

    syn = bglibpy.neuron.h.ExpSyn(params.SYN_LOC, sec=dend)
    syn.tau = params.SYN_DECAY
    syn.e = params.SYN_E

    ns = bglibpy.neuron.h.NetStim()
    ns.interval = 100000
    ns.number = 1
    ns.start = params.SYN_ACTIVATION_T
    ns.noise = 0

    nc = bglibpy.neuron.h.NetCon(ns, syn, 0, params.SYN_DELAY, params.SYN_G)
    print nc

    v_vec = bglibpy.neuron.h.Vector()
    t_vec = bglibpy.neuron.h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(bglibpy.neuron.h._ref_t)

    bglibpy.neuron.h.finitialize(params.V_INIT)
    bglibpy.neuron.h.dt = params.DT
    print "PyNeuron: Soma L=%f, diam=%f, area=%f" % (soma.L, soma.diam, bglibpy.neuron.h.area(0.5, sec=soma))
    print "PyNeuron: Dend L=%f, diam=%f, area=%f" % (dend.L, dend.diam, bglibpy.neuron.h.area(0.5, sec=dend))
    bglibpy.neuron.run(params.T_STOP)

    voltage = numpy.array(v_vec)
    time = numpy.array(t_vec)

    del(syn)
    del(ns)
    del(nc)

    return time, voltage

def run_pyneuron_with_template(params):
    """Run ballstick with PyNeuron and template"""
    bglibpy.neuron.h.load_file("ballstick.hoc")
    cell = bglibpy.neuron.h.test_cell(0, "./")
    basal = [x for x in cell.getCell().basal]
    soma = [x for x in cell.getCell().somatic][0]
    syn = bglibpy.neuron.h.ExpSyn(params.SYN_LOC, sec=basal[0])
    syn.tau = params.SYN_DECAY
    syn.e = params.SYN_E
    ns = bglibpy.neuron.h.NetStim()
    ns.interval = 100000
    ns.number = 1
    ns.start = params.SYN_ACTIVATION_T
    ns.noise = 0
    nc = bglibpy.neuron.h.NetCon(ns, syn, 0, params.SYN_DELAY, params.SYN_G)

    v_vec = bglibpy.neuron.h.Vector()
    t_vec = bglibpy.neuron.h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(bglibpy.neuron.h._ref_t)

    bglibpy.neuron.h.finitialize(params.V_INIT)
    bglibpy.neuron.h.dt = params.DT
    bglibpy.neuron.run(params.T_STOP)

    voltage = numpy.array(v_vec)
    time = numpy.array(t_vec)

    del(syn)
    del(ns)
    del(nc)

    return time, voltage

def run_bglibpy(params):
    """Run ballstick with BGLibPy"""
    cell = bglibpy.Cell("ballstick.hoc", "./ballstick.asc")
    syn = bglibpy.neuron.h.ExpSyn(params.SYN_LOC, sec=cell.basal[0])
    syn.tau = params.SYN_DECAY
    syn.e = params.SYN_E
    ns = bglibpy.neuron.h.NetStim()
    ns.interval = 100000
    ns.number = 1
    ns.start = params.SYN_ACTIVATION_T
    ns.noise = 0
    nc = bglibpy.neuron.h.NetCon(ns, syn, 0, params.SYN_DELAY, params.SYN_G)

    sim = bglibpy.Simulation()
    sim.addCell(cell)
    print "BGLibPy: Soma L=%f, diam=%f, area=%f" % (cell.soma.L, cell.soma.diam, bglibpy.neuron.h.area(0.5, sec=cell.soma))
    print "BGLibPy: Dend L=%f, diam=%f, area=%f" % (cell.basal[0].L, cell.basal[0].diam, bglibpy.neuron.h.area(0.5, sec=cell.basal[0]))
    sim.run(params.T_STOP, v_init=params.V_INIT, cvode=False, dt=params.DT)
    bglibpy_t = cell.get_time()
    bglibpy_v = cell.get_soma_voltage()

    del(sim)
    del(syn)
    del(ns)
    del(nc)

    return bglibpy_t, bglibpy_v

def test_expsyn_pyneuron_vs_bglibpy(params, graph=False) :
    """Test ballstick with expsyn between pyneuron and bglibpy"""
    #global d_derived, l_derived
    '''
    The real stuff, part I
    Run the ball-and-stick model by 1. PyNEURON, 2. bglibpy, 3. analytic
    '''
    ''' Run the Golden Standard: PyNEURON '''
    d_derived = 10
    l_derived = params.soma_A / (2 * numpy.pi * d_derived / 2.0)
    print 'soma_A=%f, derived D=%f, L=%f -> A=%f' % (params.soma_A, d_derived, l_derived, cyl_surface(d_derived, l_derived))

    ''' Run in pure PyNeuron, with using the template '''
    pyneuron_t, pyneuron_v = run_pyneuron(l_derived, d_derived, params)

    ''' Run with template in PyNeuron '''
    pyneuron_template_t, pyneuron_template_v = run_pyneuron_with_template(params)

    ''' Run with bglibpy (Werner) '''
    bglibpy_t, bglibpy_v = run_bglibpy(params)

    if graph:
        import pylab
        pylab.plot(bglibpy_t, bglibpy_v, 'g', label='BGLibPy')
        pylab.plot(pyneuron_t, pyneuron_v, 'b', label='PyNeuron')
        pylab.plot(pyneuron_template_t, pyneuron_template_v, 'r', label='PyNeuron with template')
        pylab.legend(loc=0)
        pylab.show()


def main():
    """main"""
    params = Params()
    test_expsyn_pyneuron_vs_bglibpy(params, graph=True)

if __name__ == "__main__":
    main()
