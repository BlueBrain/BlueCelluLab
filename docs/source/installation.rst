Installation
============

This installation guide will explain on how to install BGLibPy.

Installing BGLibPy
------------------

You probably want to use a python virtual environment 
(https://bbpteam.epfl.ch/project/spaces/display/BBPWFA/virtualenv). 

Pip install bglibpy from the BBP Devpi server::

    pip install -i 'https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/' bglibpy[bbp]

Pip will install automatically the BluePy dependency. However you will still
have to load/install NEURON with Python support, and compile the MOD files of 
the circuit you want to use (for details see the :ref:`dependencies` section). 

Hopefully this installation went smoothly. If it didn't, please create a Jira 
ticket, and explain as detailed as possible the problems you encountered::
   
   https://bbpteam.epfl.ch/project/issues/browse/BGLPY


Installing from source 
----------------------

If you want to make changes to BGLibPy, you might want to install it using the 
source repository. The same remarks of the section above apply, 
the only difference is that you clone the git repo::

   git clone ssh://bbpcode.epfl.ch/sim/BGLibPy.git

and run pip from inside the newly created BGLibPy subdirectory 
(don't forget the dot at the end of the command)::

    pip install -i https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple --upgrade .[bbp]

If you run into permission issues when downloading the BGLibPy repo, make sure
your `Kerberos <https://bbpteam.epfl.ch/project/spaces/display/BLGTST/Kerberos+Authentication>`_ 
ticket is up-to-date (run 'kinit'). If the problem persist asked
to be added to the bbp-user-bglibpy permission group.

Supported systems
-----------------

The code of BGLibPy can be installed on any POSIX system that supports 
pip-installable python code.

However, the dependencies have stricter requirement:

- BluePy depends on several packages for which only binary packages are 
  available from the BBP dev server. The installation has only been tested on
  BBP Linux machines. It also depends on Brion which is e.g. not compatible
  with Mac OS X for the moment.
- A full installation of Neurodamus requires the hdf5 library, and the BBP
  reporting library. Installation of these might not be trivial, but for 
  BGLibPy it is not necessary to install Neurodamus fully. See the Neurodamus
  section on the :ref:`dependencies` page for more details.
