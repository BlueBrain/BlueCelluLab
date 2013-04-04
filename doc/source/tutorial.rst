********
Tutorial
********

Loading BGLibPy in Python
=========================

After installing BGLibPy, your PYTHONPATH environment variable should normally
contain the directory where the bglibpy module is installed. Loading BGLibPy 
in Python becomes then as easy as:

.. code:: python

        import bglibpy

Instantiating a single cell from a network simulation
=====================================================

First the network simulation needs to be loaded. These simulations are 
represented by an object of the class :py:class:`bglibpy.ssim.SSim`, 
and are constructed based on a BlueConfig of the simulation.

.. code:: python

        blueconfig = "/bgscratch/bbp/l5/projects/proj1/2013.02.11/simulations\
                /SomatosensoryCxS1-v4.SynUpdate.r151/Silberberg/knockout/\
                L4_EXC/BlueConfig"
        ssim = bglibpy.SSim(blueconfig)

Next, a cell can be instantiated from this network:

.. code:: python

        gid = 101390
        ssim.instantiate_gids([gid])

The :py:class:`bglibpy.cell.Cell` object of the instantiated gid can then be
accessed with:

.. code:: python

        cell = ssim.cells[gid]

The simulate that cell, the function :py:meth:`bglibpy.ssim.SSim.run` is 
called:

.. code:: python

        ssim.run(t_stop=1000)

To plot the result:

.. code:: python

        import pylab
        pylab.plot(cell.get_time(), cell.get_soma_voltage())
        pylab.show()


Changing the verbosity
======================
By default bglibpy will not print too much to stdout
If you want to get more insight of what is going on inside bglibpy, you can
set the verbose level

.. code:: python

        bglibpy.set_verbose(level=100)
