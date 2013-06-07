"""Script to check if a certain simulation can be replicated by BGLibPy"""

import bglibpy
import bluepy
import pylab

bglibpy.tools.VERBOSE_LEVEL = 100
#blueconfig = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/L4_EXC/BlueConfig"

#blueconfig = "/bgscratch/bbp/projects/simulations/2013/11.02.13/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/weber1step_norestore/minicols310_I20_dI2.00_nosavestate/BlueConfig"

blueconfig = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/k_ca_scan/K5p0/Ca1p3/BlueConfig"
#blueconfig = '/bgscratch/bbp/release/23.07.12/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/BlueConfig'

# matches
#blueconfig = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/control/BlueConfig"

blueconfig = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/1x7/40Hz_minicols12/BlueConfig"


# a L5_TTPC1 cell in cental minicolumn of mc3_Column
# with 160 input synapses from c.projections.Thalamocortical_L4
# and 4398 incoming synapses
#gid = 113801

# A L5_SBC in a central minicolumn of mc3_Column
# with 67 input synapses from c.projections.Thalamocortical_L4
# and 812 incoming synapse
gid = 108849

ssim = bglibpy.SSim(blueconfig, record_dt=0.1)

sim = bluepy.Simulation(blueconfig)
circ = sim.circuit

# get previous results of the simulation
voltage_bglib = sim.reports.soma.time_series(gid)
time_bglib = sim.reports.soma.time_range[0:len(voltage_bglib)]

ssim.instantiate_gids([gid], synapse_detail=2, add_stimuli=True, add_replay=True)
ssim._add_synapses(add_minis=True, projection=circ.projections.Thalamocortical_L4.name)
# quick implementation of a SynapseReplay stimulus to Mosaic
#ssim._add_connections(add_replay=True, outdat_path="/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/1x7/stims/TC_yuste_Mosaic1x7_minicols12_40Hz.dat")
ssim._add_synapse_replay_stimuli()


ssim.run(t_stop=600)

time_bglibpy = ssim.get_time()
voltage_bglibpy = ssim.get_voltage_traces()[gid]

pylab.plot(time_bglib, voltage_bglib, label="BGLib_BGP")
pylab.plot(time_bglibpy, voltage_bglibpy, label="BGLibPy_Viz")
pylab.legend()
pylab.show()
