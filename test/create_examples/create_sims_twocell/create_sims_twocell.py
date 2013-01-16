#!/usr/bin/env python

"""Test the BluePy extractor"""

import sys
sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path
import bglibpy
import os

def create_extracted_simulation(output_path, blueconfig_template, runsh_template, tstop=None, dt=None, record_dt=None, fill_outdat=False):
    """..."""

    outputdir = os.path.join(output_path, "output")
    # pylint: disable=W0704
    try:
        os.makedirs(outputdir)
    except OSError:
        pass

    #todo: this thing has to set the prefix, metypepath etc
    newblueconfig_content = blueconfig_template.format(circuit_path="../circuit_twocell_example1", path="./", tstop=tstop, dt=dt, record_dt=record_dt)

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

    old_cwd = os.getcwd()
    os.chdir(output_path)

    outdat = os.path.join(outputdir, "out.dat.original")
    outdat_file = open(outdat, "w")
    if fill_outdat:
        outdat_file.write("/scatter\n")
        outdat_file.write("15.0 2\n")
    outdat_file.close()

    import subprocess
    subprocess.call("run.sh")

    os.chdir(old_cwd)

def main():
    """Main"""
    tstop = 100
    #dt = 1.0/64
    #record_dt = 1.0/8
    dt = 0.025
    record_dt = 0.1

    print 'Create a test simulation with just two cells and no extra blocks in the BlueConfig'

    import create_circuit_twocell_example1
    create_circuit_twocell_example1.main()

    with open("run.sh.template") as runsh_templatefile:
        runsh_template = runsh_templatefile.read()

    output_path = "../../examples/sim_twocell_empty"
    with open("BlueConfig.empty.template") as blueconfig_templatefile:
        blueconfig_template = blueconfig_templatefile.read()
        create_extracted_simulation(output_path, blueconfig_template, runsh_template, tstop=tstop, dt=dt, record_dt=record_dt)

    output_path = "../../examples/sim_twocell_noisestim"
    with open("BlueConfig.noisestim.template") as blueconfig_templatefile:
        blueconfig_template = blueconfig_templatefile.read()
        create_extracted_simulation(output_path, blueconfig_template, runsh_template, tstop=tstop, dt=dt, record_dt=record_dt)

    output_path = "../../examples/sim_twocell_replay"
    with open("BlueConfig.replay.template") as blueconfig_templatefile:
        blueconfig_template = blueconfig_templatefile.read()
        create_extracted_simulation(output_path, blueconfig_template, runsh_template, tstop=tstop, dt=dt, record_dt=record_dt, fill_outdat=True)


    os.chdir("../../examples/sim_twocell_replay")

    ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=record_dt)
    ssim_bglibpy.instantiate_gids([1], 1, add_stimuli=True, add_replay=True)
    ssim_bglibpy.run(tstop, dt=dt)

    ssim_bglib = bglibpy.SSim("BlueConfig")

    import pylab
    #pylab.ion()
    pylab.figure()
    time_bglibpy = ssim_bglibpy.get_time()
    voltage_bglibpy = ssim_bglibpy.get_voltage_traces()[1]
    pylab.plot(time_bglibpy, voltage_bglibpy, 'b-', label="BGLibPy")
    pylab.plot(ssim_bglib.bc_simulation.reports.soma.time_range, ssim_bglib.bc_simulation.reports.soma.time_series(1), 'r-', label="BGLib")
    pylab.legend()
    pylab.show()
    #raw_input()

if __name__ == "__main__":
    main()

