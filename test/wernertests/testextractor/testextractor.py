import sys
sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path

import tempfile

import bglibpy
from bglibpy import bluepy
import bluepy.extractor

import os

tempdir = tempfile.mkdtemp(dir="tmp")
print tempdir
outputdir = os.path.join(tempdir, "output")
os.makedirs(outputdir)
newcircuit_dir = os.path.join(tempdir, "Circuit")

circuit = bluepy.Circuit("/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig")

#gids = circuit.mvddb.select_gids(bluepy.targets.mvddb.Neuron.hyperColumn==2, bluepy.targets.mvddb.MType.name=="L5_TTPC1")[:2]
# [76477, 76478]
gids = [76477]

# add a cell which is presynaptic to 76477
gids+=[215690]

#gid_map = dict(zip(gids, range(1,len(gids)+1)))
extracted = bluepy.extractor.CircuitExtractor(circuit, gids)
extracted.extract_and_write(newcircuit_dir)

print gids

newcircuit_config = os.path.join(newcircuit_dir, "CircuitConfig")

newcircuit = bluepy.Circuit(newcircuit_config)

blueconfig_template = open("BlueConfig.template").read()

#todo: this thing has to set the prefix, metypepath etc
newblueconfig_content = blueconfig_template.format(circuit_path= newcircuit_dir, path= tempdir)

newblueconfig = os.path.join(tempdir, "BlueConfig")
newblueconfig_file = open(newblueconfig, "w")
newblueconfig_file.write(newblueconfig_content)
newblueconfig_file.close()

usertarget = os.path.join(tempdir, "user.target")
usertarget_file = open(usertarget, "w")
usertarget_file.close()

outdat = os.path.join(outputdir, "out.dat")
outdat_file = open(outdat, "w")
outdat_file.write("/scatter\n")
outdat_file.write("25.0 2\n")
outdat_file.write("50.0 2\n")
outdat_file.close()

ssim = bglibpy.SSim(newblueconfig)
ssim.instantiate_gids([1], 3)
ssim.run(100)

import pylab
pylab.plot(ssim.get_time(), ssim.get_voltage_traces()[1])
pylab.show()
