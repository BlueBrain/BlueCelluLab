********
Tutorial
********

Loading bluecellulab in Python
==============================

After installing bluecellulab, your PYTHONPATH environment variable should normally
contain the directory where the bluecellulab module is installed. Loading bluecellulab 
in Python becomes then as easy as:

.. code-block:: python

        import bluecellulab


Instantiating and stimulating a single cell
=====================================================

First of all, let's create the cell using a hoc template and a morphology.

.. code-block:: python

        from bluecellulab.cell import Cell

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

        sim = bluecellulab.Simulation()
        sim.add_cell(cell)
        sim.run(maxtime=50, cvode=True)

The time and voltage can be retrieved as follows.

.. code-block:: python

        time = cell.get_time()
        voltage = cell.get_soma_voltage()

Instantiating a single cell from a network simulation
=====================================================

First the network simulation needs to be loaded. These simulations are 
represented by an object of the class :py:class:`~bluecellulab.ssim.SSim`, 
and are constructed based on a BlueConfig of the simulation.

.. code-block:: python

        blueconfig = "PATH_TO_bluecellulab/bluecellulab/tests/examples/sim_twocell_all/BlueConfig"
        ssim = bluecellulab.SSim(blueconfig)

Next, a cell can be instantiated from this network:

.. code-block:: python

        gid = 1
        ssim.instantiate_gids([gid])

The :py:class:`~bluecellulab.cell.Cell` object of the instantiated gid can then be
accessed with:

.. code-block:: python

        cell = ssim.cells[gid]

To simulate that cell, the function :py:meth:`~bluecellulab.ssim.SSim.run` is
 called:

.. code-block:: python

        ssim.run(t_stop=1000)

To plot the result:

.. code-block:: python

        import pylab
        pylab.plot(cell.get_time(), cell.get_soma_voltage())
        pylab.show()

More details can be specified in the simulation. See the function
 :py:meth:`~bluecellulab.ssim.SSim.instantiate_gids` for further information.

To enable the synapses and spont minis in the simulation:

.. code-block:: python

        ssim = bluecellulab.SSim(blueconfig)
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

Recording the spikes from a cell
=================================

bluecellulab has the functionality to record the spikes from a cell or a group of cells.

This feature can also be combined with a network simulation.

Let's start with creating a `Cell`, a `Simulation` and adding the cell to the simulation.

.. code-block:: python

        cell = bluecellulab.Cell(
        "%s/examples/cell_example1/test_cell.hoc" % script_dir,
        "%s/examples/cell_example1" % script_dir)
        sim = bluecellulab.Simulation()
        sim.add_cell(cell)

Now we can add a spike detector to the cell and enable recording from soma with a threshold of -30.

.. code-block:: python

        cell.start_recording_spikes(None, "soma", -30)

The first parameter of the `start_recording_spikes` method is the target.
That is the target point process that goes to the `NetCon` object of `NEURON`.
If it is specified, the `NetCon` connection object will be created between the specified location of the cell and the target.

After starting the recording process, here we add a step current to the cell and run the simulation.

.. code-block:: python

        cell.add_step(start_time=2.0, stop_time=22.0, level=1.0)
        sim.run(24, cvode=False)

The resulting spikes can be retrieved as follows.

.. code-block:: python

        spikes = cell.get_spikes(location="soma")

It returns the spikes in a list. A sample output is as follows:

.. code-block:: python

        >>> spikes
        [3.350000000100014, 11.52500000009988, 19.9750000000994]

Changing the verbosity
======================
By default bluecellulab will not print too much to stdout
If you want to get more insight of what is going on inside bluecellulab, you can
set the verbose level

.. code-block:: python

        bluecellulab.set_verbose(level=100)

Postsynaptic potential validation
=================================
bluecellulab can also be used in running pair simulations.
Further information can be found at the documentation of
`psp-validation <https://bbp.epfl.ch/documentation/projects/psp-validation/latest/index.html>`_.

Jupyter notebook tutorial
=========================
An interactive scientific use-case demonstration
of bluecellulab on the neocortex circuit is available on the
`insilico-cookbook repository <https://github.com/BlueBrain/insilico-cookbook/tree/master/notebooks/Tutorials>`_.
