********
Tutorial
********

Loading BGLibPy in Python
=========================

After installing BGLibPy, your PYTHONPATH environment variable should normally
contain the directory where the bglibpy module is installed. Loading BGLibPy 
in Python becomes then as easy as:

.. code-block:: python

        import bglibpy

Instantiating a single cell from a network simulation
=====================================================

First the network simulation needs to be loaded. These simulations are 
represented by an object of the class :py:class:`~bglibpy.ssim.SSim`, 
and are constructed based on a BlueConfig of the simulation.

.. code-block:: python

        blueconfig = "PATH_TO_BGLibPy/bglibpy/tests/examples/sim_twocell_all/BlueConfig"
        ssim = bglibpy.SSim(blueconfig)

Next, a cell can be instantiated from this network:

.. code-block:: python

        gid = 1
        ssim.instantiate_gids([gid])

The :py:class:`~bglibpy.cell.Cell` object of the instantiated gid can then be
accessed with:

.. code-block:: python

        cell = ssim.cells[gid]

To simulate that cell, the function :py:meth:`~bglibpy.ssim.SSim.run` is
 called:

.. code-block:: python

        ssim.run(t_stop=1000)

To plot the result:

.. code-block:: python

        import pylab
        pylab.plot(cell.get_time(), cell.get_soma_voltage())
        pylab.show()

More details can be specified in the simulation. See the function
 :py:meth:`~bglibpy.ssim.SSim.instantiate_gids` for further information.

To enable the synapses and spont minis in the simulation:

.. code-block:: python

        ssim = bglibpy.SSim(blueconfig)
        ssim.instantiate_gids(
                [gid],
                add_synapses=True,
                add_minis=True
        )
        ssim.run(1000)
        cell = ssim.cells[gid]

To see how enabling minis and synapses affects the recorded voltage:

.. code-block:: python

        pylab.plot(cell.get_time(), cell.get_soma_voltage())
        pylab.show()

Changing the verbosity
======================
By default bglibpy will not print too much to stdout
If you want to get more insight of what is going on inside bglibpy, you can
set the verbose level

.. code-block:: python

        bglibpy.set_verbose(level=100)

Postsynaptic potential validation
=================================
BGLibPy can also be used in running pair simulations.
Further information can be found at the documentation of
 `psp-validation <https://bbp.epfl.ch/documentation/projects/psp-validation/latest/index.html>`_.

Jupyter notebook tutorial
=========================
An interactive scientific use-case demonstration
of BGLibPy on the neocortex circuit is available on the
`insilico-cookbook repository <https://github.com/BlueBrain/insilico-cookbook/tree/master/notebooks/Tutorials>`_.
