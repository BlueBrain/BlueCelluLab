"""Test the BluePy extractor"""

import sys
sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path
import tempfile
import bglibpy
from bglibpy import bluepy
import bluepy.extractor
import os

def create_extracted_circuit(old_circuitname, output_path):
    "..."
    circuit = bluepy.Circuit(old_circuitname)

    #gids = circuit.mvddb.select_gids(bluepy.targets.mvddb.Neuron.hyperColumn==2, bluepy.targets.mvddb.MType.name=="L5_TTPC1")[:2]
    # [76477, 76478]
    gids = [76477]

    gids += [215690]

    extracted = bluepy.extractor.CircuitExtractor(circuit, gids)
    extracted.extract_and_write(output_path, keep_empty_targets=False)

def create_extracted_simulation(output_path, blueconfig_template, runsh_template, tstop=None, dt=None, record_dt=None):
    """..."""

    outputdir = os.path.join(output_path, "output")
    os.makedirs(outputdir)

    #todo: this thing has to set the prefix, metypepath etc
    newblueconfig_content = blueconfig_template.format(circuit_path="./Circuit", path=".", tstop=tstop, dt=dt, record_dt=record_dt)

    newblueconfig = os.path.join(output_path, "BlueConfig")
    with open(newblueconfig, "w") as newblueconfig_file:
        newblueconfig_file.write(newblueconfig_content)

    newrunsh = os.path.join(output_path, "run.sh")
    with open(newrunsh, "w") as newrunsh_file:
        newrunsh_file.write(runsh_template)
    os.chmod(newrunsh, 0755)

    usertarget = os.path.join(output_path, "user.target")
    usertarget_file = open(usertarget, "w")
    usertarget_file.close()

    outdat = os.path.join(outputdir, "out.dat")
    outdat_file = open(outdat, "w")
    outdat_file.write("/scatter\n")
    outdat_file.write("15.0 2\n")
    outdat_file.close()


def main():
    """Main"""
    tstop = 100
    #dt = 1.0/64
    #record_dt = 1.0/8
    dt = 0.025
    record_dt = 0.1

    with open("BlueConfig.template") as blueconfig_templatefile:
        blueconfig_template = blueconfig_templatefile.read()

    with open("run.sh.template") as runsh_templatefile:
        runsh_template = runsh_templatefile.read()

    tempdir = tempfile.mkdtemp(dir="tmp")
    print tempdir

    old_circuitname = "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/O1/merged_circuit/CircuitConfig"

    create_extracted_circuit(old_circuitname, "./Circuit")
    create_extracted_simulation(".", blueconfig_template, runsh_template, tstop=tstop, dt=dt, record_dt=record_dt)

    ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=record_dt)
    ssim_bglibpy.instantiate_gids([1], 3)
    ssim_bglibpy.run(tstop, dt=dt)

    import subprocess
    subprocess.call("run.sh")
    ssim_bglib = bglibpy.SSim("BlueConfig")

    import pylab
    time_bglibpy = ssim_bglibpy.get_time()
    voltage_bglibpy = ssim_bglibpy.get_voltage_traces()[1]
    pylab.plot(time_bglibpy, voltage_bglibpy, 'b-', label="BGLibPy")
    pylab.plot(ssim_bglib.bc_simulation.reports.soma.time_range, ssim_bglib.bc_simulation.reports.soma.time_series(1), 'r-', label="BGLib")
    #import numpy
    #pylab.plot(numpy.diff(ssim_bglibpy.get_time()), 'o')
    pylab.legend()
    print "Simulation directory: %s" % tempdir
    pylab.show()

if __name__ == "__main__":
    main()

