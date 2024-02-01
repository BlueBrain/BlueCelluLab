|banner|

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
| Citation       | |zenodo|   |
+----------------+------------+


The Blue Brain Cellular Laboratory is designed for simulations and experiments on individual cells or groups of cells.
Suitable use cases for BlueCelluLab include:

* Scripting and statistical analysis for single cells or cell pairs.

* Lightweight, detailed reporting on specific state variables after simulation.

* Developing synaptic plasticity rules.

* Validating dynamics of synaptic properties.

* Automating in-silico whole-cell patching experiments.

* Debugging, both scientifically and computationally.

Citation
========

When you use this BlueCelluLab software for your research, we ask you to cite the following reference(this includes poster presentations):

.. code-block::

    @software{bluecellulab_zenodo,
      author       = {Van Geit, Werner and Tuncel, Anil and Gevaert, Mike and Torben-Nielsen, Benjamin and Muller, Eilif},
      title        = {BlueCelluLab},
      month        = jul,
      year         = 2023,
      publisher    = {Zenodo},
      doi          = {10.5281/zenodo.8113483},
      url          = {https://doi.org/10.5281/zenodo.8113483}
    }

Support
=======

We are providing support on `Gitter <https://gitter.im/BlueBrain/BlueCelluLab>`_. We suggest you create tickets on the `Github issue tracker <https://github.com/BlueBrain/BlueCelluLab/issues>`_ in case you encounter problems while using the software or if you have some suggestions.

Main dependencies
=================

* `Python 3.8+ <https://www.python.org/downloads/release/python-380/>`_
* `Neuron 8.0.2+ <https://pypi.org/project/NEURON/>`_

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
    sim = Simulation()
    sim.add_cell(cell)
    stimulus = cell.add_step(start_time=5.0, stop_time=20.0, level=0.5)

    sim.run(25, cvode=False)
    time, voltage = cell.get_time(), cell.get_soma_voltage()
    # plotting time and voltage ...

.. image:: https://raw.githubusercontent.com/BlueBrain/BlueCelluLab/main/docs/images/voltage-readme.png
   :alt: Voltage plot

Tutorial
========

A more detailed explanation on how to use BlueCelluLab, as well as other examples can be found on the `examples page <https://github.com/BlueBrain/BlueCelluLab/blob/main/examples/README.rst>`_.

API Documentation
=================

The API documentation can be found on `ReadTheDocs <https://bluecellulab.readthedocs.io>`_.

Running the tests
=================

Testing is set up using `tox`:

.. code-block:: bash

    pip install tox

    tox -e py3  # runs the tests
    tox -e lint  # runs the format checks

Funding & Acknowledgements
==========================

The development and maintenance of this code is supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government's ETH Board of the Swiss Federal Institutes of Technology.

Copyright
=========

Copyright (c) 2023 Blue Brain Project/EPFL

This work is licensed under `Apache 2.0 <https://www.apache.org/licenses/LICENSE-2.0.html>`_

For MOD files for which the original source is available on ModelDB, any specific licenses on mentioned on ModelDB, or the generic License of ModelDB apply.

The licenses of the morphology files used in this repository are available on: https://zenodo.org/record/5909613


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

.. |zenodo| image:: https://zenodo.org/badge/640805129.svg
                 :target: https://zenodo.org/badge/latestdoi/640805129

..
    The following image is also defined in the index.rst file, as the relative path is
    different, depending from where it is sourced.
    The following location is used for the github README
    The index.rst location is used for the docs README; index.rst also defined an end-marker,
    to skip content after the marker 'substitutions'.

.. substitutions
.. |banner| image:: https://raw.githubusercontent.com/BlueBrain/BlueCelluLab/main/docs/source/logo/BlueCelluLabBanner.jpg
