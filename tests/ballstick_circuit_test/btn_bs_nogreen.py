"""
Synapse test file. Uses ball-and-stick models
Models simulated in N disticnt ways, all shoudl
"""

import psp
import numpy as np
import matplotlib.pyplot as plt

import bglibpy
import bluepy

import sys
# sys.path.append('/home/torben/sandbox/willem/DendriteApprox/')
# from greensFunctionCalculator import *

T_STOP = 100
V_INIT = -75
DT = 0.025

SYN_DECAY = 2.0
SYN_E = 0
SYN_ACTIVATION_T = 50
SYN_DELAY = 0.025
SYN_G = 0.001
SYN_LOC = 1.0


def surface(r, h):
    return 2 * np.pi * r * h


cell = bglibpy.Cell("ballstick.hoc", "./")
soma_L, soma_D, soma_A = cell.soma.L, cell.soma.diam, bglibpy.neuron.h.area(
    0.5, sec=cell.soma)
print 'SOMA L=%f, diam=%f,surf=%f' % (soma_L, soma_D, soma_A)

dend0_L, dend0_D, dend0_A = cell.basal[0].L, cell.basal[0].diam, \
    bglibpy.neuron.h.area(0.5, sec=cell.basal[0])
print 'DENDRITE L=%f, diam=%f,surf=%f' % (dend0_L, dend0_D, dend0_A)

''' I assume uniform passive properties shared by the soma and dendrites '''
CM = cell.soma.cm
RM = 1.0 / cell.soma(0.5).g_pas
RA = cell.soma.Ra
EL = cell.soma(0.5).e_pas

''' N simulations of ... the same '''

''' 1: PyNEURON (Hines) '''


def run_hines_bs(soma_l, soma_d):
    soma = bglibpy.neuron.h.Section()
    soma.L = soma_l
    soma.diam = soma_d
    soma.nseg = 1
    soma.cm = CM
    soma.Ra = RA
    soma.insert('pas')
    soma(0.5).e_pas = EL
    soma(0.5).g_pas = 1.0 / RM

    dend = bglibpy.neuron.h.Section()
    dend.L = dend0_L
    dend.diam = dend0_D
    dend.nseg = cell.basal[0].nseg
    print 'rune_hines: dend.nseg=%f' % (dend.nseg)
    dend.cm = CM
    dend.Ra = RA
    dend.insert('pas')
    for seg in dend:
        seg.e_pas = EL
        seg.g_pas = 1.0 / RM
    dend.connect(soma, 0.5, 0)  # mid-soma to begin-den

    syn = bglibpy.neuron.h.ExpSyn(SYN_LOC, sec=dend)
    syn.tau = SYN_DECAY
    syn.e = SYN_E

    ns = bglibpy.neuron.h.NetStim()
    ns.interval = 100000
    ns.number = 1
    ns.start = SYN_ACTIVATION_T
    ns.noise = 0

    nc = bglibpy.neuron.h.NetCon(ns, syn, 0, SYN_DELAY, SYN_G)

    v_vec = bglibpy.neuron.h.Vector()
    t_vec = bglibpy.neuron.h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(bglibpy.neuron.h._ref_t)

    bglibpy.neuron.h.finitialize(V_INIT)
    bglibpy.neuron.h.dt = DT
    bglibpy.neuron.run(T_STOP)

    hines_v = np.array(v_vec)
    hines_t = np.array(t_vec)
    return hines_t, hines_v

# hines_t,hines_v = run_hines_bs(soma_L,soma_D)
# plt.plot(hines_t,hines_v,label='hines, direct')


d_derived = 10
l_derived = soma_A / (2 * np.pi * d_derived / 2.0)
print 'soma_A=%f, derived D=%f, L=%f -> A=%f' % (soma_A, d_derived, l_derived, surface(d_derived / 2.0, l_derived))
hines_t, hines_v = run_hines_bs(l_derived, d_derived)
plt.plot(hines_t, hines_v, label='PyNEURON - ExpSyn')

''' 3: Pseudo/Semi-Analytical (Willem)'''

''' write config file for Willem '''
"""
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
v_willem,t_willem = compute_system([[SYN_ACTIVATION_T]],T_STOP,DT,\
                                   conffile_name=f_name,numsyn=1,syndend=[0],\
                                   synloc=[SYN_LOC],gbar=[SYN_G],\
                                   decay=[SYN_DECAY],E_rev=[SYN_E])
plt.plot(t_willem,v_willem,label='willem, DT=%f' % (DT))
"""

''' 2: bglibpy.Cell (Werner) '''
syn = bglibpy.neuron.h.ProbAMPANMDA_EMS(SYN_LOC, sec=cell.basal[0])
syn.tau_r_AMPA = 0.001
# syn.tau_r_NMDA = 0.5
syn.tau_d_AMPA = SYN_DECAY
# syn.tau_d_NMDA = SYN_DECAY
syn.NMDA_ratio = 0
syn.Use = 1.0
ns = bglibpy.neuron.h.NetStim()
ns.interval = 100000
ns.number = 1
ns.start = SYN_ACTIVATION_T
ns.noise = 0
nc = bglibpy.neuron.h.NetCon(ns, syn, 0, SYN_DELAY, SYN_G * 1000.0)

sim = bglibpy.Simulation()
sim.addCell(cell)
sim.run(T_STOP, v_init=V_INIT, cvode=False, dt=0.025)
werner_t = cell.get_time()
werner_v = cell.get_soma_voltage()
plt.plot(werner_t, werner_v, label='bglibpy dt=0.025 - ProbAMPANMDA_EMS')
sim.run(T_STOP, v_init=V_INIT, cvode=False, dt=0.001)
werner_t = cell.get_time()
werner_v = cell.get_soma_voltage()
plt.plot(werner_t, werner_v, label='bglibpy dt=0.001 - ProbAMPANMDA_EMS')

del(sim)
del(syn)
del(ns)
del(nc)

syn = bglibpy.neuron.h.ExpSyn(SYN_LOC, sec=cell.basal[0])
syn.tau = SYN_DECAY
syn.e = SYN_E
ns = bglibpy.neuron.h.NetStim()
ns.interval = 100000
ns.number = 1
ns.start = SYN_ACTIVATION_T
ns.noise = 0
nc = bglibpy.neuron.h.NetCon(ns, syn, 0, SYN_DELAY, SYN_G)

sim = bglibpy.Simulation()
sim.addCell(cell)
sim.run(T_STOP, v_init=V_INIT, cvode=False, dt=DT)
werner_t = cell.get_time()
werner_v = cell.get_soma_voltage()
plt.plot(werner_t, werner_v, label='bglibpy - ExpSyn')
del(sim)
del(syn)
del(ns)
del(nc)


''' Eilif's BGLIB'''
s = bluepy.Simulation("BlueConfig")
# excitatory cell with an incoming inhibitory synapse
# v_a1 = s.reports.soma.time_series(1)
# inhibitory cell with an incoming excitatory synapse
v_a2 = s.reports.soma.time_series(2)
t = s.reports.soma.time_range - 401 + SYN_ACTIVATION_T
plt.plot(t, v_a2, label='bglib, dt=0.025')

''' Ben's PSP amplitude code '''
sys.path += ['/home/ebmuller/src/bbp-user-ebmuller/experiments/synapse_psp_validation']
# shift epsp occurance back by 1.0ms (synaptic delay in circuit)
psp, g, dropped_count, v, t = psp.assess_psp_with_replay_synapses(
    "BlueConfig", 1, 2, None, -75.0, SYN_ACTIVATION_T - 1.0, 200.0, 1.0, 1)
plt.plot(t, v, label='ssim')


plt.legend(loc=0)
plt.show()
