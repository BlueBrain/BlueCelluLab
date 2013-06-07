"""Script to check if a certain simulation can be replicated by BGLibPy"""

import bglibpy
import pylab

bglibpy.tools.VERBOSE_LEVEL = 100
blueconfig = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations/SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/L4_EXC/BlueConfig"
gids = range(101390, 101400)
gid = gids[0]

bglibpy.set_verbose(100)

ssim_bglibpy = bglibpy.SSim(blueconfig, record_dt=0.1)
ssim_bglib = bglibpy.SSim(blueconfig)

voltage_bglib = ssim_bglib.bc_simulation.reports.soma.time_series(gid)
time_bglib = ssim_bglib.bc_simulation.reports.soma.time_range[0:len(voltage_bglib)]

ssim_bglibpy.instantiate_gids(gids, synapse_detail=2, add_stimuli=True, add_replay=True)
ssim_bglibpy.run(t_stop=1000)

time_bglibpy = ssim_bglibpy.get_time()
voltage_bglibpy = ssim_bglibpy.get_voltage_traces()[gid]

pylab.plot(time_bglib, voltage_bglib, label="BGLib_BGP")
pylab.plot(time_bglibpy, voltage_bglibpy, label="BGLibPy_Viz")
pylab.legend()
pylab.show()
