BlueCelluLab
============

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

Citation
========

When you use this BlueCelluLab software for your research, we ask you to cite the following publication (this includes poster presentations):

.. code-block:: 

    @article {TBD Zenodo}

Support
=======

We are providing support on `Gitter <https://gitter.im/BlueBrain/BlueCelluLab>`_. We suggest you create tickets on the `Github issue tracker <https://github.com/BlueBrain/BlueCelluLab/issues>`_ in case you encounter problems while using the software or if you have some suggestions.

Main dependencies
=================

* `Python 3.8+ <https://www.python.org/downloads/release/python-380/>`_
* `Neuron 7.4+ <http://neuron.yale.edu/>`_ (compiled with Python support)

Installation
============

BlueCelluLab can be pip installed with the following command:

.. code-block:: python

    pip install bluecellulab

Quick Start
===========

The following example shows how to create a cell, add a stimulus and run a simulation:

.. code-block:: python

    from bluecellulab.cell import create_ball_stick
    from bluecellulab import Simulation

    cell = create_ball_stick()
    stimulus = cell.add_step(start_time=15.0, stop_time=20.0, level=1.5)
    sim = Simulation()
    sim.add_cell(cell)

    sim.run(200, cvode=False)
    time, voltage = cell.get_time(), cell.get_soma_voltage()
    # plotting time and voltage ...

.. image:: docs/images/voltage-readme.png
   :alt: Voltage plot

Tutorial
========

A more detailed explanation on how to use BlueCelluLab, as well as other examples can be found on the `examples page <examples/README.rst>`_.

API Documentation
=================

The API documentation can be found on `ReadTheDocs <https://bluecellulab.readthedocs.io>`_.

Funding & Acknowledgements
==========================

The development and maintenance of this code is supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government's ETH Board of the Swiss Federal Institutes of Technology.

Copyright
=========

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
