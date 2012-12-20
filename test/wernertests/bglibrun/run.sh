#module load bglib/rhel6-mvapich2-psm-x86_64-shared-dev 
mpirun -n 2 -genvall $BGLIB_ROOT/bin/special $BGLIB_HOCLIB/PrepConfig.hoc -c 'configFile="BlueConfig"' $BGLIB_HOCLIB/init.hoc -mpi -NFRAME 256
#/home/vangeit/src/BlueBrain/lib/x86_64/special 
#mpirun -np 1 -genvall /home/vangeit/src/BlueBrain/lib/x86_64/special  $BGLIB_HOCLIB/PrepConfig.hoc -c configFile="BlueConfig.1" $BGLIB_HOCLIB/init.hoc -mpi -NFRAME 256
cd output
python /home/ebmuller/src/bbp-user-ebmuller/pybinreports/soma2h5.py
