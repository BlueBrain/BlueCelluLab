Installation
============

This installation guide will explain on how to install BGLibPy in your home 
directory on the viz cluster. In an ideal world the dependencies of BGLibPy are 
all pre-installed on the viz cluster, however, since in reality these 
installation are not kept up to date, we currently propose to install some of 
BGLibPy's dependency in your home directory 
(see section :ref:`installing-dependencies`)

Installing BGLibPy
------------------

(If necessary, **first install the dependencies** according to the 
:ref:`installing-dependencies` section.)

Clone the git repository::

    git clone ssh://bbpcode.epfl.ch/sim/BGLibPy
    cd BGLibPy

The source directory contains an example build script called 
"buid.sh.example.viz", make a copy of this script::

    cp build.sh.example.viz build.sh

The content of this file will be something like::

    mkdir build
    cd build
    cmake .. \
        -DMODLIBPATH=$HOME/src/bbp/lib/modlib \
        -DHOCLIBPATH=$HOME/src/bbp/lib/hoclib \
        -DNRNPATH=${NEURON_HOME} \
        -DNRNPYTHONPATH=${NEURON_PYTHON_HOME} \
        -DBLUEPYPATH=$HOME/local/bluepy/lib/python2.6/site-packages \
        -DCMAKE_INSTALL_PREFIX=$HOME/local/bglibpy
    make VERBOSE=1
    make install

After copying the script adapt the variables to your needs::

    MODLIBPATH = place where you checked out the BGLib modlib (lib/modlib subdirectory of the Neurodamus/bbp repository)
    HOCLIBPATH = place where you checked out the BGLib hoclib (lib/modlib subdirectory of the Neurodamus/bbp repository)
    NRNPATH = the installation directory of Neuron (in case you use the default install, this will be $NEURON_HOME)
    NRNPYTHONPATH = the path where the Neuron python module is installed (in case you use the default install, this will be $NRNPYTHONPATH)
    BLUEPYPATH = the path where the BluePy python module is installed 
    CMAKE_INSTALL_PREFIX = the directory where you want to install BGLibPy

Once this is done you execute the script::

    ./build.sh

Run the unit tests (this require /bgscratch to be 
mounted on the machine you are testing this on)::

   ./runtests.sh

Hopefully this installation went smoothly. If it didn't, please create a Jira 
ticket, and explain as detailed as possible the problems you encountered::
   
   https://bbpteam.epfl.ch/project/issues/browse/BGLPY

.. _installing-dependencies:

Installing Dependencies
-----------------------

The dependencies are::

    BluePy
    BGLib (Neurodamus)
    Neuron

Ideally you follow the installation instructions of the tools, 
in case these are missing:

*BluePy*
    Install the source from a git repository clone 
    (e.g. to $HOME/local/bluepy)::

        git clone ssh://bbpcode.epfl.ch/analysis/BluePy.git
        cd BluePy/bluepy
        python setup.py install --prefix=$HOME/local/


*BGLib*
    Just get the source from the git repository, no installation is required::

        git clone ssh://bbpcode.epfl.ch/sim/neurodamus/bbp.git

*Neuron*

    For Neuron you can probably use the version installed on the viz cluster
    Add this to your .bashrc (and re-login)::

        module load neuronYale/rhel6-mvapich2-psm-x86_64-shared-dev
