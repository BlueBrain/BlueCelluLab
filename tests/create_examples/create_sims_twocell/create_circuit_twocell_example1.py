#!/usr/bin/env python

"""Test the BluePy extractor"""

import os
import shutil

import bluepy.extractor

import bluepy


def create_extracted_circuit(old_circuitname, output_path):
    "..."
    circuit = bluepy.Circuit(old_circuitname)

    print '#########'
    print circuit
    print '#########'

    # gids = circuit.mvddb.select_gids(bluepy.targets.mvddb.Neuron.hyperColumn==2, bluepy.targets.mvddb.MType.name=="L5_TTPC1")[:2]
    # [76477, 76478]
    gids = [76477]

    gids += [215690]

    extracted = bluepy.extractor.CircuitExtractor(circuit, gids)
    extracted.extract_and_write(output_path, keep_empty_targets=False)

    new_circuitconfig = os.path.join(output_path, "CircuitConfig")
    print new_circuitconfig
    with open(new_circuitconfig, "r") as new_circuitconfig_file:
        new_circuitconfig_content = new_circuitconfig_file.read()

    correct_new_circuitconfig_content = ""

    for line in new_circuitconfig_content.split("\n")[:-1]:
        newline = line
        if "CircuitPath" in line:
            newline = "  CircuitPath ../circuit_twocell_example1"
        elif "nrnPath" in line:
            newline = "  nrnPath ../circuit_twocell_example1/ncsFunctionalAllRecipePathways"
        correct_new_circuitconfig_content += newline + "\n"

    with open(new_circuitconfig, "w") as new_circuitconfig_file:
        new_circuitconfig_file.write(correct_new_circuitconfig_content)


def main():
    """Main"""

    print 'Create a test circuit with just two cells from %s' \
        % "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"

    output_path = "../../examples/circuit_twocell_example1/"
    shutil.rmtree(output_path)
    old_circuitname = "/bgscratch/bbp/l5/release/2012.07.23/circuit/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"
    # /bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"

    create_extracted_circuit(old_circuitname, output_path)


if __name__ == "__main__":
    main()
