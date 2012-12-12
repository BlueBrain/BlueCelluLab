

from Cheetah.Template import Template
import tempfile
import os
import sys
import shutil
import bluepy
from matplotlib import pylab as plt
sys.path+=['/home/ebmuller/src/bbp-user-ebmuller/experiments/synapse_psp_validation']
import psp as psplib
import multiprocessing as mp
import pickle

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


class BGlibSim(object):
    def __init__(self, pre_gid, post_gid, t_stim=[400.0], dir=None):
        """ pre, post gids of neuron pair to simulate 
        whereby spikes from pre_gid (t_stim, list) are replayed on synapes on post_gid""" 

        if dir==None:
            self.dir = os.path.abspath(tempfile.mkdtemp(prefix=".bglib", dir="./"))
        else: self.dir = dir
        self.pre_gid = pre_gid
        self.post_gid = post_gid
        self.t_stim = t_stim

        # populate the sim dir 
        self.write_replay_spikes()
        self.write_blueconfig()
        self.write_usertarget()
        # copy run script
        shutil.copy(os.path.join("sim_files","run.sh"), self.dir)
        shutil.copy(os.path.join("sim_files","init.hoc"), self.dir)

    def write_replay_spikes(self):
        """ write spike times for pre_gid to replay.dat in the tempdir allocated for this sim """
        path = os.path.join(self.dir, "replay.dat")
        f = file(path,"w")
        f.writelines(["/scatter\n"]+["%s %d\n" % (t, self.pre_gid) for t in self.t_stim])
        f.close()
    
    def write_blueconfig(self):
        """ populate the blueconfig template and write it to the temp dir allocated for this sim"""
        in_path = os.path.join("sim_files", "BlueConfig.template")
        out_path = os.path.join(self.dir, "BlueConfig")
        f = file(out_path,"w")
        
        t = Template(file=in_path)
        t.path = self.dir
        print >>f, str(t)

    def write_usertarget(self):
        """ populate the user.target template and write it to the temp dir allocated for this sim"""
        in_path = os.path.join("sim_files", "user.target.template")
        out_path = os.path.join(self.dir, "user.target")
        f = file(out_path,"w")
        
        t = Template(file=in_path)
        #t.gid_list = ["a%d" % gid for gid in [self.post_gid]]
        t.gid_list = "a%d" % self.post_gid
        print >>f, str(t)


    def run(self):
        os.system("pushd %s; sh run.sh; popd" % self.dir)
    def get_soma_report(self):
        s = bluepy.Simulation(os.path.join(self.dir, "BlueConfig"))
        return s.reports.soma

    def get_vt(self):
        """ return tuple (v,t) which are the voltage trace and time points for the post_gid as numpy arrays """
        r = self.get_soma_report()
        return r.time_series(self.post_gid), r.time_range


    #def __del__(self):
    #    shutil.rmtree(self.dir)

        
class BGlibpySim(object):
    def __init__(self, pre_gid, post_gid, t_stim=[400.0], dir=""):
        self.pre_gid = pre_gid
        self.post_gid = post_gid
        self.t_stim = t_stim
        self.dir = dir

    def run(self):
        v,t, ssim, drop_count = psplib.mean_voltage_pair_pre_replay(os.path.join(self.dir,"BlueConfig"),self.pre_gid,self.post_gid,None,-75.0,self.t_stim,1000.0,1.0,1)
        self.v = v
        self.t = t
    def get_vt(self):
        return self.v, self.t

def rms(v1, v2):
    from numpy import sqrt, mean
    return sqrt(mean((v1-v2)**2))

def sample_mtype_pathway_rms_mp(args):
    try:
        val =  sample_mtype_pathway_rms(*args)
    except Exception, e:
        return "Unknown Failure '%s'" % repr(e)
    return val

def sample_mtype_pathway_rms(circuit_path, pre_mtype, post_mtype):

    from bluepy.targets.mvddb import Neuron
    import sqlalchemy as sqla
    c = bluepy.Circuit(circuit_path)
    syns = c.get_pathway_pairs(pre_mtype, post_mtype, sqla.or_(*(Neuron.miniColumn==mc_id for mc_id in xrange(3*310, 3*310+100))))

    if len(syns)==0:
        return None

    pre_gid, post_gid = syns[0][0:2]

    t_stim = [400.0, 450.0, 500, 550.0]
    #sim1 = BGlibSim(pre_gid, post_gid, t_stim, dir = "./.bglibeyQY6k")
    sim1 = BGlibSim(pre_gid, post_gid, t_stim)
    sim1.run()
    try:
        v1,t1 = sim1.get_vt()
    except KeyError:
        return "Failed"

    sim2 = BGlibpySim(pre_gid, post_gid, t_stim, sim1.dir)
    sim2.run()
    v2,t2 = sim2.get_vt()
    
    return rms(v1,v2[:-1])


def playing_around():

    pre_gid, post_gid = 108101, 107463

    t_stim = [400.0, 450.0, 500, 550.0]
    sim1 = BGlibSim(pre_gid, post_gid, t_stim, dir = "./.bglibeyQY6k")
    #sim1 = BGlibSim(pre_gid, post_gid, t_stim)
    #sim1.run()
    v1,t1 = sim1.get_vt()
    plt.plot(t,v, label="bglib")

    sim2 = BGlibpySim(pre_gid, post_gid, t_stim, sim1.dir)
    sim2.run()
    v2,t2 = sim2.get_vt()
    
    #assert(rms(v1,v2[:-1]<1e-3))
    plt.plot(t,v, label="bglibpy")
    plt.legend()
    plt.show()

    
if __name__=="__main__":
    
    circuit_path = "/bgscratch/bbp/circuits/23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/1x7_0/merged_circuit/CircuitConfig"

    c = bluepy.Circuit(circuit_path)
    mtypes = c.mtype_ids.keys()

    work_items = []
    for pre_mtype in mtypes:
        for post_mtype in mtypes:
            work_items.append((circuit_path, pre_mtype, post_mtype))

    
    rms_list = []
    cpu_count = mp.cpu_count()
    p = mp.Pool()

    """
    # serial
    for work_item in work_items:
        cp, pre_mtype, post_mtype = work_item
        rms_val = sample_mtype_pathway_rms_mp(work_item)
        rms_list.append(rms_val)
        print "RESULT: %s->%s rms=%s" % (pre_mtype, post_mtype, str(rms_val))
    """

    for work_chunk in chunks(work_items, cpu_count):
        p = mp.Pool(cpu_count)
        local_rms_list = p.map(sample_mtype_pathway_rms_mp, work_chunk)
        rms_list += local_rms_list

        for work_item, rms_val in zip(work_chunk, local_rms_list):
            cp, pre_mtype, post_mtype = work_item
            print "RESULT: %s->%s rms=%s" % (pre_mtype, post_mtype, str(rms_val))

        p.close()
        p.join()

    # save to disk
    f = file("test_psp.pickle","w")
    d = {"wi":work_items, "rms":rms_list}
    pickle.dump(d, f)
    f.close()




        
