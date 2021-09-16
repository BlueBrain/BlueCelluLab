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


Instantiating and stimulating a single cell
=====================================================

First of all, let's create the cell using a hoc template and a morphology.

.. code-block:: python

        from bglibpy.cell import Cell

        template_path = "/your/path/to/cAD_noscltb.hoc"
        morph_dir = "/directory/containing/morphologies"
        morphology = "dend-jy180406_A_idB_axon-05311-05329-X10166-Y11730_-_Scale_x1.000_y0.950_z1.000.asc"
        extra_values = {"holding_current": None, "threshold_current": None}
        cell = Cell(
            template_filename=template_path,
            morphology_name=morphology,
            morph_dir=morph_dir,
            template_format="v6",
            extra_values=extra_values,
        )

Now we can inject a step current to the cell and start recording from its soma.

.. code-block:: python

        cell.add_step(start_time=15.0, stop_time=40.0, level=1.0)
        cell.add_recording("self.soma(0.5)._ref_v")

The next step is to create a simulation object and add our cell to the simulator.
Later the run method is called to initiate the simulation.
Here we specify the max runtime to be 50ms.
We can choose whether we want to use the cvode variable time step integrator or not.

.. code-block:: python

        sim = bglibpy.Simulation()
        sim.add_cell(cell)
        sim.run(maxtime=50, cvode=True)

The time and voltage can be retrieved as follows.

.. code-block:: python

        time = cell.get_time()
        voltage = cell.get_soma_voltage()

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
