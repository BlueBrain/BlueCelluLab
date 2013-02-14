"""
Synapse test file. Uses ball-and-stick models
Models simulated in N disticnt ways, all shoudl
"""

import numpy as np
#import numpy.testing
import matplotlib.pyplot as plt

#import sys
#sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path

import bglibpy
#from bglibpy import bluepy
#import bluepy

#sys.path.append('/home/torben/sandbox/willem/DendriteApprox/')
from greensFunctionCalculator import *

'''Due to some path difficulties with BlueConfig, change directory'''
#import os
#os.chdir('test/ballstick_test')

T_STOP = 200
V_INIT = -75
DT = 0.025

SYN_DECAY = 2.0
SYN_E = 0
SYN_ACTIVATION_T = 50
SYN_DELAY = 0.025
SYN_G = 0.001
SYN_LOC = 1.0

CM = -1
RM = -1
RA = -1
EL = -1
soma_L, soma_D, soma_A, dend0_L, dend0_D, dend0_A, dend0_NSEG = \
  -1,-1,-1,-1,-1,-1,-1
''' Actual configuration by calling compute_some_settings_from_ASC_file()
'''

surface = lambda r, h: 2*np.pi*r*h

def compute_some_settings_from_ASC_file() :
    global CM,RM,RA,EL, \
      soma_L, soma_D, soma_A, dend0_L, dend0_D, dend0_A,dend0_NSEG
    cell = bglibpy.Cell("ballstick.hoc", "./")
    soma_L,soma_D, soma_A = cell.soma.L, cell.soma.diam, bglibpy.neuron.h.area\
      (0.5, sec=cell.soma)
    print 'SOMA L=%f, diam=%f,surf=%f' % (soma_L,soma_D,soma_A)

    dend0_L,dend0_D,dend0_A  = cell.basal[0].L, cell.basal[0].diam, \
      bglibpy.neuron.h.area(0.5, sec=cell.basal[0])
    dend0_NSEG = cell.basal[0].nseg
    print 'DENDRITE L=%f, diam=%f,surf=%f' % (dend0_L,dend0_D,dend0_A)

    ''' I assume uniform passive properties shared by the soma and dendrites '''
    CM = cell.soma.cm
    RM = 1.0 / cell.soma(0.5).g_pas
    RA= cell.soma.Ra
    EL= cell.soma(0.5).e_pas

def run_hines_bs(soma_l,soma_d) :
    soma = bglibpy.neuron.h.Section()
    soma.L = soma_l
    soma.diam = soma_d
    soma.nseg = 1
    soma.cm = CM
    soma.Ra = RA
    soma.insert('pas')
    soma(0.5).e_pas = EL
    soma(0.5).g_pas = 1.0/RM

    dend = bglibpy.neuron.h.Section()
    dend.L = dend0_L
    dend.diam = dend0_D
    dend.nseg = dend0_NSEG
    print 'rune_hines: dend.nseg=%f' % (dend.nseg)
    dend.cm = CM
    dend.Ra = RA
    dend.insert('pas')
    for seg in dend :
        seg.e_pas = EL
        seg.g_pas = 1.0/RM
    dend.connect(soma,0.5,0) # mid-soma to begin-den

    syn = bglibpy.neuron.h.ExpSyn(SYN_LOC,sec=dend)
    syn.tau = SYN_DECAY
    syn.e = SYN_E

    ns = bglibpy.neuron.h.NetStim()
    ns.interval = 100000
    ns.number = 1
    ns.start = SYN_ACTIVATION_T
    ns.noise = 0

    nc= bglibpy.neuron.h.NetCon(ns,syn,0,SYN_DELAY,SYN_G)

    v_vec = bglibpy.neuron.h.Vector()
    t_vec = bglibpy.neuron.h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(bglibpy.neuron.h._ref_t)

    bglibpy.neuron.h.finitialize(V_INIT)
    bglibpy.neuron.h.dt = DT
    bglibpy.neuron.run(T_STOP)

    hines_v = np.array(v_vec)
    hines_t = np.array(t_vec)
    return hines_t,hines_v

def run_analytic(dt) :
    ''' write config file for Willem '''
    f_name = 'bs.cfg'
    outF = open(f_name,'w')
    outF.write('[neuron]\n')
    outF.write('CM: %f\n' % (CM))
    outF.write('RM: %f\n' % (RM))
    outF.write('RA: %f\n' % (RA))
    outF.write('EL: %f\n\n' % (EL))
    outF.write('[soma]\n')
    outF.write('D: ' + str(d_derived)+'\n')
    outF.write('L: ' + str(l_derived)+'\n\n')
    outF.write('[morph]\n')
    outF.write('lengths: [%f]\n' % (dend0_L))
    outF.write('diams: [%f]\n' % (dend0_D))
    outF.close()

    print 'going to do the analytical stuff'
    v_willem,t_willem = compute_system([[SYN_ACTIVATION_T]],T_STOP,dt,\
                                       conffile_name=f_name,numsyn=1,\
                                       syndend=[0],synloc=[SYN_LOC],\
                                       gbar=[SYN_G],decay=[SYN_DECAY],\
                                       E_rev=[SYN_E])
    return t_willem, v_willem

def run_bglib_bs() :
    cell = bglibpy.Cell("ballstick.hoc", "./")
    syn = bglibpy.neuron.h.ExpSyn(SYN_LOC,sec=cell.basal[0])
    syn.tau = SYN_DECAY
    syn.e = SYN_E
    ns = bglibpy.neuron.h.NetStim()
    ns.interval = 100000
    ns.number = 1
    ns.start = SYN_ACTIVATION_T
    ns.noise = 0
    nc= bglibpy.neuron.h.NetCon(ns,syn,0,SYN_DELAY,SYN_G)

    sim = bglibpy.Simulation()
    sim.add_cell(cell)
    sim.run(T_STOP,v_init=V_INIT,cvode=False,dt=DT)
    werner_t = cell.getTime()
    werner_v = cell.getSomaVoltage()
    del(sim)
    del(syn)
    del(ns)
    del(nc)
    return werner_t,werner_v

def test_bs_expsyn_pyneuron_vs_bglibpy(graph=False) :
    global d_derived, l_derived
    '''
    The real stuff, part I
    Run the ball-and-stick model by 1. PyNEURON, 2. bglibpy, 3. analytic
    '''
    compute_some_settings_from_ASC_file()
    ''' Run the Golden Standard: PyNEURON '''
    d_derived = 10
    l_derived = soma_A / (2*np.pi*d_derived/2.0)
    print 'soma_A=%f, derived D=%f, L=%f -> A=%f' % \
      (soma_A,d_derived,l_derived,surface(d_derived/2.0,l_derived))
    hines_t,hines_v = run_hines_bs(l_derived,d_derived)

    ''' Run with bglibpy (Werner) '''
    werner_t, werner_v = run_bglib_bs()
    #numpy.testing.assert_array_almost_equal(hines_v,werner_v,1,err_msg=\
    #                                        'Werner != Hines')

    if(graph) :
        plt.plot(werner_t,werner_v,'g',label='werner')
        plt.plot(hines_t,hines_v,'b',label='hines, derived')
        plt.legend(loc=0)
        plt.show()

def test_bs_expsyn_pyneuron_vs_analytic(graph=False) :
    global d_derived, l_derived
    '''
    The real stuff, part I
    Run the ball-and-stick model by 1. PyNEURON, 2. bglibpy, 3. analytic
    '''
    compute_some_settings_from_ASC_file()
    ''' Run the Golden Standard: PyNEURON '''
    d_derived = 10
    l_derived = soma_A / (2*np.pi*d_derived/2.0)
    print 'soma_A=%f, derived D=%f, L=%f -> A=%f' % \
      (soma_A,d_derived,l_derived,surface(d_derived/2.0,l_derived))
    hines_t,hines_v = run_hines_bs(l_derived,d_derived)

    ''' Compute the Pseudo/Semi-Analytical solution (Willem)'''
    t_willem, v_willem = run_analytic(dt=DT)

    #numpy.testing.assert_array_almost_equal(hines_v,v_willem,1,err_msg=\
    #                                        'Willem != Hines')

    if(graph) :
        plt.plot(t_willem,v_willem,'r',label='willem, DT=%f' % (DT))
        plt.plot(hines_t,hines_v,'b',label='hines, derived')
        plt.legend(loc=0)
        plt.show()

def test_bs_ProbAMPANMDAEMS_pyneuron_vs_bglibpy() :
    ''' pff. I'm done with nice encapsulation. just running a test!'''
    cell = bglibpy.Cell("ballstick.hoc", "./")
    syn = bglibpy.neuron.h.ProbAMPANMDA_EMS(SYN_LOC,sec=cell.basal[0])
    syn.tau_r_AMPA = 0.1
    syn.tau_r_NMDA = 0.1
    syn.tau_d_AMPA = SYN_DECAY
    syn.tau_d_NMDA = SYN_DECAY
    syn.NMDA_ratio = 0
    syn.Use = 1.0
    ns = bglibpy.neuron.h.NetStim()
    ns.interval = 100000
    ns.number = 1
    ns.start = SYN_ACTIVATION_T
    ns.noise = 0
    nc= bglibpy.neuron.h.NetCon(ns,syn,0,SYN_DELAY,SYN_G)

    sim = bglibpy.Simulation()
    sim.add_cell(cell)
    sim.run(T_STOP,v_init=V_INIT,cvode=False,dt=DT)
    werner_t = cell.getTime()
    werner_v = cell.getSomaVoltage()

    del(sim)
    del(ns);del(nc);del(syn)
    del(cell)

    #assert False == True

def test_bs_ProbAMPANMDAEMS_pyneuron_vs_bglib() :
    assert False == True

# ''' Eilif's BGLIB'''
# s = bluepy.Simulation("BlueConfig")
# # excitatory cell with an incoming inhibitory synapse
# #v_a1 = s.reports.soma.time_series(1)
# # inhibitory cell with an incoming excitatory synapse
# v_a2 = s.reports.soma.time_series(2)
# t = s.reports.soma.time_range- 401 + SYN_ACTIVATION_T
# plt.plot(t,v_a2,label='Eilif')
