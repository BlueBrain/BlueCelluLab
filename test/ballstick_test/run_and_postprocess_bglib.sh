mpirun -n 2 -genvall $BGLIB_ROOT/bin/special $BGLIB_HOCLIB/init.hoc -mpi -NFRAME 256 2>&1 | tee bglib_stdout_stderr.txt
pushd bglib_output
python ~/src/bbp-user-ebmuller/pybinreports/soma2h5.py
popd
