Bluecellulab
=======

|tests| |license|


BlueCelluLab is designed to do simulation and experiment on a single cell or a group of cells. Use cases for which bluecellulab is well suited include: scripting and statistics across single or pairs of cells, light-weight detailed reporting on a few state variables post-simulation, development of synaptic plasticity rules, dynamics validations of e.g. synaptic properties, automation of in-silico whole-cell patching experiments, debugging both scientifically and computationally.


Testing
-------

To run the tests in a python environment, use the following command::

    tox


Changelog
---------

Refer to CHANGELOG.rst.

Copyright
---------

This work is licensed under `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`_ 

For MOD files for which the original source is available on ModelDB, any specific licenses on mentioned on ModelDB, or the generic License of ModelDB apply.

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
                :target: https://github.com/BlueBrain/BlueCelluLab/blob/main/LICENSE

.. |tests| image:: https://github.com/BlueBrain/BlueCelluLab/actions/workflows/test.yml/badge.svg?branch=main
   :target: https://github.com/BlueBrain/BlueCelluLab/actions/workflows/test.yml
   :alt: CI
