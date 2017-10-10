Installation
============

This installation guide will explain on how to install BGLibPy in your home 
directory on the viz cluster. 
(see section :ref:`installing-dependencies`)

Installing BGLibPy
------------------

(**first install the dependencies** according to the 
:ref:`installing-dependencies` section.)

You probably want to use a python virtual environment

Pip install bglibpy from the BBP Devpi server::

    pip install -i 'https://bbpteam.epfl.ch/repository/devpi/bbprelman/dev/+simple/' bglibpy




Hopefully this installation went smoothly. If it didn't, please create a Jira 
ticket, and explain as detailed as possible the problems you encountered::
   
   https://bbpteam.epfl.ch/project/issues/browse/BGLPY

.. _installing-dependencies:

Installing Dependencies
-----------------------

The dependencies are::

    Python 2.7 (no 3.x for now)
    Neuron
    Neurodamus

Follow the installation instructions of the tools, 
