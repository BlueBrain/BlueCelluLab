BGLibPy
======

The pythonic interface to BGLib functionality 

Introduction
------------

BGLibPy is a scientist targeted streamlined interface to BGLib. 

Dependencies
------------

Third party dependencies are:

* python-numpy
* python-nose
* python-pylab (if one needs live plotting)
* PyNeuron

BBP dependencies are:

* BluePy
* BGLib (BlueBrain)

Building
--------

* There is an example build script provided called 'build.sh.example'
* Copy this script to 'build.sh'
* Edit this script so that it points to the write directories on your system

Testing
-------

Go to your build directory (after installing and building the code), 
and execute 'make test'. If you want to have more verbosity, you can type
'ctest -VV' instead of 'make test'.

Releasing
---------

TODO 

In this example we will release ``v0.3.0``.

#. Increment the version identifier. For example when releasing ``0.3.0`` the
   version identified in ``bluepy.__version__.py`` should be ``0.3.0-dev``.
   Simply remove the ``-dev`` part.
#. Update the changelog. If the changelog entry for the given version does not
   exist yet, create it. Otherwise complete it, and fix the date.
#. Commit the version bump and the changelog using for example
   ``v0.3.0 incl. changelog`` as message.
#. Tag the version using ``git tag v0.3.0``
#. Upload the tag using ``git push origin v0.3.0``
#. Generate and upload the documentation to a versioned folder. I.e. not
   ``../documentation/BluePy/dev/`` but ``../documentation/BluePy/v0.3.0/``.
#. Bump the version number to the next minor release. I.e. ``0.4.0-dev`` and
   commit and push that change. This will prevent future commits to have
   release version numbers.

In this example we will release ``v0.3.1``.

#. Create and checkout the maintenance branch ``maint-v0.3.x``. Use
   ``git checkout -b maint-v0.3.x v0.3.0`` if it does not exist already.
#. Commit the maintenance changes and push the branch.
#. Tag the maintenance release: ``git tag v0.3.1`` and upload the tag
#. Merge the maintenance branch into master and push the merge
#. Depending on the impact of the change, generate and upload the documentation
   as described above.

You may, or may not have to regenerate and upload the ubuntu and redhat
packages, see below for details.

Packages
--------


Changelog
---------


Authors and Contributors
------------------------

* Werner Van Geit
* Ben Torbien Nielsen
* Eilif Muller

Copyright
---------

Copyright (c) BBP/EPFL 2010-2012;
All rights reserved.
Do not distribute without further notice.
