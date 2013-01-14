#!/usr/bin/env python

"""Test the BluePy extractor"""

import sys
sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path
from bglibpy import bluepy
import bluepy.extractor

def create_extracted_circuit(old_circuitname, output_path):
    "..."
    circuit = bluepy.Circuit(old_circuitname)

    #gids = circuit.mvddb.select_gids(bluepy.targets.mvddb.Neuron.hyperColumn==2, bluepy.targets.mvddb.MType.name=="L5_TTPC1")[:2]
    # [76477, 76478]
    gids = [76477]

    gids += [215690]

    extracted = bluepy.extractor.CircuitExtractor(circuit, gids)
    extracted.extract_and_write(output_path, keep_empty_targets=False)


def main():
    """Main"""

    print 'Create a test circuit with just two cells from %s' \
                        % "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"

    if len(sys.argv) == 2:
        output_path = sys.argv[1]
    else:
        print "Need to specify an output directory as first argument (will be created if it doesn't exist)"
        exit(1)

    old_circuitname = "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"

    create_extracted_circuit(old_circuitname, output_path)

if __name__ == "__main__":
    main()

