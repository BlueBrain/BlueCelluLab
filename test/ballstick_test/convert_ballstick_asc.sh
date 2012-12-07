if [ ${HOSTNAME} = "bbplinsrv2" ]
then
        export HDF5_DISABLE_VERSION_CHECK=2
        /opt/neuronHDF5/convert.sh ballstick.asc .
else
        echo "This script has to be run on linsrv2"
fi
