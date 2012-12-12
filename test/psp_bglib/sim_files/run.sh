#export HOC_LIBRARY_PATH=$BGLIB_HOCLIB
mpirun -n 2 -genv HOC_LIBRARY_PATH $BGLIB_HOCLIB $BGLIB_ROOT/bin/special init.hoc -mpi -NFRAME 256 2>&1 | tee bglib_stdout_stderr.txt
python ~/src/bbp-user-ebmuller/pybinreports/soma2h5.py

