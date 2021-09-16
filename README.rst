BGLibPy
=======

BGLibpy is a Pythonic API and infrastructure leveraging Neurodamus to facilitate the
detailed study of small numbers of neurons in simulations defined by Neurodamus and
a BlueConfig. Use cases for which BGLibpy is well suited include: scripting
and statistics across single or pairs of cells, light-weight detailed
reporting on a few state variables post-simulation, development of synaptic
plasticity rules, dynamics validations of e.g. synaptic properties, automation
of in-silico whole-cell patching experiments.

Documentation
-------------

Check the latest documentation for API usage, tutorial, FAQ, installation and dependencies.
https://bbpteam.epfl.ch/documentation/projects/BGLibPy/latest/index.html


Testing
-------

To run the tests in a python3.8 environment, use the following tox command::

    tox -e py38-test

To run the tests on the circuits (on bb5), you can speficy extra conditionals such as::

    tox -e py38-v5-v6-thal

`v5` and `v6` are enabling the tests on the cortex circuits.
`thal` is for the tests on the thalamus circuits.

To run all tests simply try::

    tox -e py38-test-v5-v6-thal

To test the coding style use::

    tox -e lint

More tox commands can be found at the tox.ini.


Changelog
---------

Refer to CHANGELOG.rst.

Authors and Contributors
------------------------

* Anil Tuncel
* Werner Van Geit
* Ben Torbien Nielsen
* Eilif Muller

Copyright
---------

Copyright (c) BBP/EPFL 2010-2021;
All rights reserved.
Do not distribute without further notice.
