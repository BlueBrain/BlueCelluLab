.. _dependencies:

Dependencies
============

The main dependencies of BGLibPy are::

    Python 2.7+ or 3.5+ 
    Neuron
    Neurodamus
    BluePy

Ideally follow the installation instructions of these tools, or use 
pre-installed versions.

Python
------

Modern Linux systems will have Python 2.7 or 3 installed.

Make sure you're using a recent version of pip. It's best to run ::

    pip install pip --upgrade

before installing anything else using pip.

Possibly ways to acquire Python on BB5 are:

Pre-installed modules
~~~~~~~~~~~~~~~~~~~~~

First, you need to load an archive of your choice containing Python. Then you can load Python ::

    module load archive/2020-02
    module load python

Neuron
------

NEURON should be compiled with Python support. MPI support is not a 
requirement.

Versions that are supported:

- 7.4
- 7.5
- Latest git commit from https://github.com/nrnhines/nrn 
  (Before being release the BGLibPy package is tested against this release)

Possibly ways to acquire NEURON are:

Pre-installed modules
~~~~~~~~~~~~~~~~~~~~~

First, you need to load an archive of your choice containing NEURON. Then you can load NEURON ::

    module load archive/2020-02
    module load neuron

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

It's not too difficult to install NEURON from source in your home directory on
BB5.
If necessary change the SRC_DIR and INSTALL_DIR, and run the following code ::

    SRC_DIR=$HOME/src
    INSTALL_DIR=$HOME/local

    mkdir -p ${SRC_DIR}
    cd ${SRC_DIR}
    if [ ! -d nrn ]
    then
        echo "Downloading NEURON from github ..."
        git clone https://github.com/nrnhines/nrn.git
    else                                                                         
        echo "Neuron already downloaded"                                         
    fi                                                                           
    cd nrn
    echo "Preparing NEURON ..."
    ./build.sh
    echo "Configuring NEURON ..."                                                
    ./configure --prefix=${INSTALL_DIR} --without-x --with-nrnpython --disable-rx3d
    echo "Installing NEURON ..."
    make -j4 install
    
    export PATH="${INSTALL_DIR}/x86_64/bin":${PATH}
    export PYTHONPATH="${INSTALL_DIR}/lib64/python":${PYTHONPATH}

    echo "Testing NEURON import ...."
    python -c 'import neuron'
                                                                                 
    echo "NEURON successfully installed"
    echo "Set your PATH at login to: ${INSTALL_DIR}/x86_64/bin:\${PATH}"
    echo "Set your PYTHONPATH at login to: ${INSTALL_DIR}/lib64/python:\${PYTHONPATH}"

(The above code is based on a script called '.install_neuron.sh' in the BGLibPy
git repo)

Linux packages
~~~~~~~~~~~~~~

On RPM systems one can install NEURON and its python interface using the following command ::

    sudo dnf install python3-neuron

On Debian systems the corresponding command is ::

    sudo apt-get install python3-neuron

Neurodamus
----------

It's not necessary to fully install Neurodamus to use it with BGLibPy. 
The only required components are:

1. the HOC code (lib/hoclib subdir of neurodamus source).
2. the 'scientific' MOD files (ion channels, synapses, etc. 
   This doesn't include the 'technical' MOD files like hdf5 readers)

Installing from source
~~~~~~~~~~~~~~~~~~~~~~

First get the Neurodamus source using git::

    git clone ssh://bbpcode.epfl.ch/sim/neurodamus/bbp.git

The HOC code is located in the directory lib/hoclib of the newly created 'bbp'
subdir. Set the HOC_LIBRARY_PATH (add the resolved path to your login script 
if necessary) ::

    export HOC_LIBRARY_PATH=`pwd`/bbp/lib/hoclib

Place all the MOD files (ion channels, synapses, etc.) in
a single directory. 
Then, in the directory from where you want to run BGLibPy, run::

    nrnivmodl path_to_your_mod_dir

If you want to run a classical BBP somatosensory cortex simulation, you can
get the MOD files from lib/modlib directory from the repo you downloaded above.
You only have to remove some files to make the compilation easier::

    rm -rf lib/modlib/Bin*.mod                                             
    rm -rf lib/modlib/HDF*.mod 
    rm -rf lib/modlib/hdf*.mod
    rm -rf lib/modlib/MemUsage*.mod

(The above code is based on a script called '.install_neurodamus.sh' in the 
BGLibPy git repo)

Pre-installed modules
~~~~~~~~~~~~~~~~~~~~~

First, you need to load an archive of your choice containing Neurodamus. 
Then you can load Neurodamus compiled with the circuit specified such as::

    module load archive/2020-03
    module load neurodamus-thalamus/0.3

BluePy
~~~~~~

You won't have to manually install BluePy, it is automatically installed by
the pip-install of BGLibPy.

In case you get an error like::

    'Could not find a version that satisfies the requirement ...'

Check if there are wheels available for the dependencies of BluePy.
One common problem with this is that the Python binary you are using isn't 
compiled with::

    --enable-unicode=ucs4e

If you have problems with Brain / LibFlatIndex dependencies of BluePy, and
you don't need to read voltage reports from neurodamus simulations, one
option would be to drop the '[bbp]' when pip installing BGLibPy or BluePy. 
