#!/usr/bin/env python

"""Test the BluePy extractor"""

import os


def create_extracted_simulation(
        output_path,
        blueconfig_template,
        runsh_template,
        tstop=None,
        dt=None,
        record_dt=None,
        fill_outdat=False):
    """..."""

    print "Creating new example in %s" % output_path
    outputdir = os.path.join(output_path, "output")
    # pylint: disable=W0704
    try:
        os.makedirs(outputdir)
    except OSError:
        pass

    newblueconfig_content = blueconfig_template.format(
        circuit_path="../circuit_twocell_example1",
        path="./",
        tstop=tstop,
        dt=dt,
        record_dt=record_dt)

    newblueconfig = os.path.join(output_path, "BlueConfig")
    with open(newblueconfig, "w") as newblueconfig_file:
        newblueconfig_file.write(newblueconfig_content)

    newrunsh = os.path.join(output_path, "run.sh")
    with open(newrunsh, "w") as newrunsh_file:
        newrunsh_file.write(runsh_template)

    os.chmod(newrunsh, 0o755)

    usertarget_content = "Target Cell PreCell\n{\na2\n}\n\n" \
        "Target Cell PostCell\n{\na1\n}\n\n" \
        "Target Cell MyPairs\n{\nPreCell PostCell\n}\n\n"

    usertarget = os.path.join(output_path, "user.target")
    usertarget_file = open(usertarget, "w")
    usertarget_file.write(usertarget_content)
    usertarget_file.close()

    old_cwd = os.getcwd()
    os.chdir(output_path)

    outdat = os.path.join(outputdir, "out.dat.original")
    outdat_file = open(outdat, "w")
    outdat_file.write("/scatter\n")
    if fill_outdat:
        outdat_file.write("15.0\t2\n")
        outdat_file.write("30.0\t2\n")
        outdat_file.write("45.0\t2\n")
        outdat_file.write("60.0\t2\n")
        outdat_file.write("75.0\t2\n")
        outdat_file.write("90.0\t2\n")
    # Add a large value, because BGLib cannot handle an empty out.dat
    outdat_file.write("5000000.0\t2\n")
    outdat_file.close()

    import subprocess
    print os.getcwd()
    print "Running Neurodamus ..."
    subprocess.call("./run.sh")
    print "Neurodamus finished"

    os.chdir("output")

    # soma2h5_v2_bglibpy.main(os.path.join(os.getcwd(), 'soma.bbp'))

    os.chdir("..")

    os.chdir(old_cwd)


def main():
    """Main"""
    tstop = 100
    # dt = 1.0/64
    # record_dt = 1.0/8
    dt = 0.025
    record_dt = 0.1

    print 'Create a test simulation with just two cells and no extra blocks in the BlueConfig'

    with open("templates/run.sh.template") as runsh_templatefile:
        runsh_template = runsh_templatefile.read()

    sims_info = [
        ('empty', False),
        ('noisestim', False),
        ('pulsestim', False),
        ('replay', True),
        ('minis_replay', True),
        ('all', True),
        ('all_mvr', True),
        ('neuronconfigure', True),
        ('synapseid', True),
        ('realconn', False)
    ]

    for sim_name, fill_outdat in sims_info:
        template_filename = 'BlueConfig.%s.template' % sim_name
        output_dirname = 'sim_twocell_%s' % sim_name
        template_path = os.path.join('templates', template_filename)
        output_path = os.path.join('../../examples/%s' % output_dirname)

        with open(template_path) as template_file:
            blueconfig_template = template_file.read()

        print('###########')
        print(sim_name)
        print('###########')

        create_extracted_simulation(
            output_path,
            blueconfig_template,
            runsh_template,
            tstop=tstop,
            dt=dt,
            record_dt=record_dt,
            fill_outdat=fill_outdat)

    """
    # Used for debugging

    os.chdir("../../examples/sim_twocell_pulsestim")
    ssim_bglibpy = bglibpy.SSim("BlueConfig", record_dt=record_dt)
    bglibpy.set_verbose(100)
    ssim_bglibpy.instantiate_gids(
        [1, 2], synapse_detail=2, add_stimuli=True, add_replay=True)
    ssim_bglibpy.run(tstop, dt=dt)

    ssim_bglib = bglibpy.SSim("BlueConfig")

    import pylab
    pylab.figure()
    time_bglibpy = ssim_bglibpy.get_time()
    voltage_bglibpy = ssim_bglibpy.get_voltage_traces()[1]
    pylab.plot(time_bglibpy, voltage_bglibpy, 'b-', label="BGLibPy")
    pylab.plot(
        ssim_bglib.bc_simulation.reports.soma.time_range,
        ssim_bglib.bc_simulation.reports.soma.time_series(1),
        'r-',
        label="BGLib")
    pylab.legend()
    pylab.show()
    # os.chdir("../../examples/sim_twocell_all")
    #ssim_bglib_all = bglibpy.SSim("BlueConfig", record_dt=record_dt)
    pylab.plot(
        ssim_bglib_all.bc_simulation.reports.soma.time_range,
        ssim_bglib_all.bc_simulation.reports.soma.time_series(1),
        'k-',
        label="BGLib all")
    """


if __name__ == "__main__":
    main()