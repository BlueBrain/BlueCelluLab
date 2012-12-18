"""Testing replay"""

import sys
sys.path = ["/home/vangeit/local/bglibpy/lib/python2.7/site-packages"]+ sys.path

import bglibpy
import pylab
import numpy

gid = 96517
tstop = 1000

ssim = bglibpy.SSim('/bgscratch/bbp/release/19.11.12/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/BlueConfig', record_dt=0.1)
ssim.instantiate_gids([gid], 3)
ssim.simulate(t_stop=tstop)

time = ssim.get_time()
voltage = ssim.get_voltage_traces()[gid]
pylab.plot(time, voltage, 'o-')
pylab.plot(time, ssim.bc_simulation.reports.soma.time_series(gid)[:len(time)], 'o-')
#pylab.plot(ssim.bc_simulation.reports.soma.time_range, ssim.bc_simulation.reports.soma.time_series(gid), 'o')

pre_spiketrains = ssim.cells[gid].pre_spiketrains
for sid in pre_spiketrains:
    pre_spiketrain = pre_spiketrains[sid]
    filtered_spiketrain = [x for x in pre_spiketrain if x < tstop]
    pylab.plot(filtered_spiketrain, numpy.ones(len(filtered_spiketrain)) * -60.0, "k|", markersize=10)

pylab.show()


