module load bglib/rhel6-mvapich2-psm-x86_64-shared-dev 
mpirun -n 2 -genvall $BGLIB_ROOT/bin/special $BGLIB_HOCLIB/init.hoc -mpi -NFRAME 256
