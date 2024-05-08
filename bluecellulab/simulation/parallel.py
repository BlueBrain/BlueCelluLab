"""Controlled simulations in parallel."""

from __future__ import annotations
import multiprocessing
from multiprocessing.pool import Pool
from bluecellulab.simulation.neuron_globals import NeuronGlobals


class IsolatedProcess(Pool):
    """Multiprocessing Pool that restricts a worker to run max 1 process.

    Use this when running isolated NEURON simulations. Running 2 NEURON
    simulations on a single process is to be avoided. Required global
    NEURON simulation parameters will automatically be passed to each
    worker. `fork` mode of multiprocessing is used to pass any other
    global NEURON parameters to each worker.
    """

    def __init__(self, processes: int | None = 1):
        """Initialize the IsolatedProcess pool.

        Args:
            processes: The number of processes to use for running the simulations.
                       If set to None, then the number returned by os.cpu_count() is used.
        """
        ctx = multiprocessing.get_context("fork")
        neuron_global_params = NeuronGlobals.get_instance().export_params()
        super().__init__(
            processes=processes,
            initializer=self.init_worker,
            initargs=(neuron_global_params,),
            maxtasksperchild=1,
            context=ctx,
        )

    @staticmethod
    def init_worker(neuron_global_params):
        """Load global parameters for the NEURON environment in each worker
        process."""
        NeuronGlobals.get_instance().load_params(neuron_global_params)
