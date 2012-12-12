import nose.tools as nt
import numpy as np
import bglibpy.ssim

def test_evaluate_connection_parameters():
    ssim = bglibpy.ssim.SSim("/bgscratch/bbp/release/19.11.12/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/BlueConfig")
    #/bgscratch/bbp/release/23.07.12/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/BlueConfig")
    s = ssim.bc_simulation

    # check a TTPC1 pair
    pre_gid, post_gid = list(ssim.bc_simulation.get_target("L5_TTPC1"))[:2]

    params = ssim._evaluate_connection_parameters(pre_gid, post_gid)
    assert params=={'SpontMinis': 0.067000000000000004, 'SynapseConfigure': ['%s.NMDA_ratio = 0.4', '%s.NMDA_ratio = 0.71'], 'Weight': 2.3500000000000001}
    #assert params=={'SpontMinis': 0.067000000000000004, 'SynapseConfigure': ['%s.NMDA_ratio = 0.4', '%s.NMDA_ratio = 0.71'], 'Weight': 1.0}

    pre_gid = list(ssim.bc_simulation.get_target("L5_MC"))[0]
    params = ssim._evaluate_connection_parameters(pre_gid, post_gid)
    assert params=={'SpontMinis': 0.012, 'SynapseConfigure': ['%s.e_GABAA = -80.0'], 'Weight': 2.0}

    pre_gid = list(ssim.bc_simulation.get_target("L5_LBC"))[0]
    params = ssim._evaluate_connection_parameters(pre_gid, post_gid)
    assert params=={'SpontMinis': 0.012, 'SynapseConfigure': ['%s.e_GABAA = -80.0'], 'Weight': 0.67000000000000004}

    pre_gid = list(ssim.bc_simulation.get_target("L1_HAC"))[0]
    params = ssim._evaluate_connection_parameters(pre_gid, post_gid)
    assert params=={'SpontMinis': 0.012, 'SynapseConfigure': ['%s.e_GABAA = -80.0', '%s.GABAB_ratio = 0.75'], 'Weight': 2.0}

def test_add_single_synapse_SynapseConfigure():
    ssim = bglibpy.ssim.SSim("/bgscratch/bbp/release/19.11.12/simulations/SomatosensoryCxS1-v4.lowerCellDensity.r151/Silberberg/knockout/control/BlueConfig")

    gid = list(ssim.bc_simulation.get_target("L5_MC"))[0]
    ssim.instantiate_gids([gid])
    pre_datas = ssim.bc_simulation.circuit.get_presynaptic_data(gid)
    # get second inh synapse (first fails)
    inh_synapses = np.nonzero(pre_datas[:,13]<100)
    sid = inh_synapses[0][1]
    syn_params = pre_datas[sid,:]
    pre_gid = syn_params[0]
    connection_modifiers = {'SynapseConfigure': ['%s.e_GABAA = -80.5 %s.e_GABAB = -101.0', '%s.tau_d_GABAA = 10.0 %s.tau_r_GABAA = 1.0', '%s.e_GABAA = -80.6'], 'Weight':2.0}
    ssim._add_single_synapse(gid,sid,syn_params,connection_modifiers)

    assert ssim.syns[gid][sid].e_GABAA==-80.6
    assert ssim.syns[gid][sid].e_GABAB==-101.0
    assert ssim.syns[gid][sid].tau_d_GABAA==10.0
    assert ssim.syns[gid][sid].tau_r_GABAA==1.0
