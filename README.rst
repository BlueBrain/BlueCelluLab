Bluecellulab
=======

+----------------+------------+
| Latest Release | |pypi|     |
+----------------+------------+
| Documentation  | |docs|     |
+----------------+------------+
| License        | |license|  |
+----------------+------------+
| Build Status 	 | |tests|    |
+----------------+------------+
| Coverage       | |coverage| |
+----------------+------------+
| Gitter         | |gitter|   |
+----------------+------------+


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

.. |pypi| image:: https://img.shields.io/pypi/v/bluecellulab.svg
               :target: https://pypi.org/project/bluecellulab/
               :alt: latest release

.. |docs| image:: https://readthedocs.org/projects/bluecellulab/badge/?version=latest
               :target: https://bluecellulab.readthedocs.io/
               :alt: latest documentation

.. |coverage| image:: https://codecov.io/github/BlueBrain/BlueCelluLab/coverage.svg?branch=main
                   :target: https://codecov.io/gh/BlueBrain/bluecellulab
                   :alt: coverage

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
                 :target: https://gitter.im/BlueBrain/BlueCelluLab
                 :alt: Join the chat at https://gitter.im/BlueBrain/BlueCelluLab
