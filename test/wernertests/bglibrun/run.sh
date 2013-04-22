#module load bglib/rhel6-mvapich2-psm-x86_64-shared-dev 
module load $HOME/rhel6-mvapich2-psm-x86_64-shared-dev 
#mpirun -n 2 -genvall $BGLIB_ROOT/bin/special $BGLIB_HOCLIB/PrepConfig.hoc -c 'configFile="BlueConfig"' $BGLIB_HOCLIB/init.hoc -mpi -NFRAME 256
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs4/bbp.epfl.ch/sw/bbp-stack/reportinglib/dev-git/install/lib
mpirun -n 2 -genvall /home/vangeit/src/bbp/lib/x86_64/special  $BGLIB_HOCLIB/PrepConfig.hoc -c 'configFile="BlueConfig"' $BGLIB_HOCLIB/init.hoc -mpi -NFRAME 256
cd output
python /home/ebmuller/src/bbp-user-ebmuller/pybinreports/soma2h5.py
