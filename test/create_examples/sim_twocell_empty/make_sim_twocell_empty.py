"""Test the BluePy extractor"""

import sys
sys.path = ["/home/vangeit/local/bglibpy/lib64/python2.6/site-packages"]+ sys.path
import bglibpy
import os

def create_extracted_simulation(output_path, blueconfig_template, runsh_template, tstop=None, dt=None, record_dt=None):
    """..."""

    outputdir = os.path.join(output_path, "output")
    try:
        os.makedirs(outputdir)
    except OSError:
        pass

    #todo: this thing has to set the prefix, metypepath etc
    newblueconfig_content = blueconfig_template.format(circuit_path="../circuit_twocell_example1", path="../sim_twocell_empty", tstop=tstop, dt=dt, record_dt=record_dt)

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

    os.chdir("../../examples/sim_twocell_empty")

    outdat = os.path.join(outputdir, "out.dat.original")
    outdat_file = open(outdat, "w")
    outdat_file.write("/scatter\n")
    outdat_file.write("5015.0 2\n")
    outdat_file.close()

    import subprocess
    subprocess.call("run.sh")

def main():
    """Main"""
    tstop = 100
    #dt = 1.0/64
    #record_dt = 1.0/8
    dt = 0.025
    record_dt = 0.1

    print 'Create a test simulation with just two cells and no extra blocks in the BlueConfig'

    output_path = "../../examples/sim_twocell_empty"
    #if len(sys.argv) == 2:
    #    output_path = sys.argv[1]
    #else:
    #    print "Need to specify an output directory as first argument (will be created if it doesn't exist)"
    #    exit(1)

    with open("BlueConfig.template") as blueconfig_templatefile:
        blueconfig_template = blueconfig_templatefile.read()

    with open("run.sh.template") as runsh_templatefile:
        runsh_template = runsh_templatefile.read()

    create_extracted_simulation(output_path, blueconfig_template, runsh_template, tstop=tstop, dt=dt, record_dt=record_dt)

    ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=record_dt)
    ssim_bglibpy.instantiate_gids([1], 3)
    ssim_bglibpy.run(tstop, dt=dt)

    ssim_bglib = bglibpy.SSim("BlueConfig")

    import pylab
    pylab.ion()
    pylab.figure()
    time_bglibpy = ssim_bglibpy.get_time()
    voltage_bglibpy = ssim_bglibpy.get_voltage_traces()[1]
    pylab.plot(time_bglibpy, voltage_bglibpy, 'b-', label="BGLibPy")
    pylab.plot(ssim_bglib.bc_simulation.reports.soma.time_range, ssim_bglib.bc_simulation.reports.soma.time_series(1), 'r-', label="BGLib")
    pylab.legend()
    pylab.draw()
    raw_input()

if __name__ == "__main__":
    main()

