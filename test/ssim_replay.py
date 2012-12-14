import matplotlib.pyplot as plt

import bglibpy
import bglibpy.ssim
import bluepy
import neuron

gids = [118583]#,118586 [107457]# 
t_stop=500

def test_compare_main_sim_vs_bglibpy_ssim() :
    ''' Replay a neuron with bglibpy.ssim.SSim and compare to the data from \
    the true cortical simulation performed on the BlueGene'''

    blue_config_filename = '/bgscratch/bbp/release/23.07.12/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/BlueConfig'
    bc_sim = bglibpy.ssim.SSim(blue_config_filename)
    bc_sim.instantiate_gids(gids,synapse_detail=3,full=True)

    bc_sim.simulate(t_stop=t_stop)
    t = bc_sim.get_time()
    vs = bc_sim.get_voltage_traces()
    for gid in gids :
        plt.plot(t,vs[gid],label='BTN')
    plt.legend(loc=0)

    bp_simulation = bluepy.Simulation(blue_config_filename)
    soma_report = bp_simulation.reports.soma
    voltage = soma_report.time_series(gids)[0]
    time = soma_report.time_range
    plt.plot(time,voltage,label='original')
    plt.show()
    #plt.savefig('auto_'+str(gids[0])+'_'+str(t_stop)+'.pdf')

    from neuron import gui
    raw_input('Test finished. Press ENTER')
    import sys
    sys.exit(0)

    assert False == True

# for testing purposes only
test_compare_main_sim_vs_bglibpy_ssim()
