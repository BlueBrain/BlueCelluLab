import sys
sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path

import tempfile

from bglibpy import bluepy
import bluepy.extractor

d = tempfile.mkdtemp(dir=".")
print d

circuit = bluepy.Circuit("/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig")

gids = circuit.mvddb.select_gids(bluepy.targets.mvddb.Neuron.hyperColumn==2, bluepy.targets.mvddb.MType.name=="L5_TTPC1")[:2]
# [76477, 76478]

# add a cell which is presynaptic to 76477
gids+=[215690]

#gid_map = dict(zip(gids, range(1,len(gids)+1)))
extracted = bluepy.extractor.CircuitExtractor(circuit, gids)
extracted.extract_and_write(d)

print gids
