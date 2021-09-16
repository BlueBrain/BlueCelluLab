
import h5py
import numpy

h5 = h5py.File("nrn.h5", "w")


# a1
#
a1 = numpy.array([[2.0, 1.0, 1.0, 0.0, 100.0,  # pre_gid, delay, postsec, post seg id, distance along seg
                   0.0, 0.0, 0.0,  # presec, preseg, preseg distance
                   # gsyn, u, d, f, dtc, synapse type (>=100 exc, <100 inh),
                   1.0, 0.5, 650.0, 10, 10.0, 0,
                   # mtype (index in circuit.mvd2 mtype list), BO dend (not
                   # used), BO axon (not used)
                   2.0, 1.0, 1.0,
                   1.0, 2.0]], dtype=">f4")  # ASE (not used), branch type on post


# a2
a2 = numpy.array([[1.0, 1.0, 1.0, 0.0, 150.0,  # pre_gid, delay, postsec, post seg id, distance along seg
                   0.0, 0.0, 0.0,  # presec, preseg, preseg distance
                   # gsyn, u, d, f, dtc, synapse type (>=100 exc, <100 inh),
                   1.0, 0.5, 650.0, 10, 2.0, 100,
                   # mtype (index in circuit.mvd2 mtype list), BO dend (not
                   # used), BO axon (not used)
                   1.0, 1.0, 1.0,
                   1.0, 2.0]], dtype=">f4")  # ASE (not used), branch type on post

h5.create_dataset('a1', data=a1)
h5.create_dataset('a2', data=a2)
h5.close()
