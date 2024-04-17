"""Package for performing single cell or network simulations and
experiments."""

try:
    import bluepy
    BLUEPY_AVAILABLE = True
except ImportError:
    BLUEPY_AVAILABLE = False

from bluecellulab.importer import import_hoc
from .verbosity import *
from .cell import Cell, create_ball_stick  # NOQA
from .circuit import EmodelProperties
from .connection import Connection  # NOQA
from .plotwindow import PlotWindow  # NOQA
from .dendrogram import Dendrogram  # NOQA
from .psection import PSection  # NOQA
from .psegment import PSegment  # NOQA
from .simulation import Simulation  # NOQA
from .rngsettings import RNGSettings  # NOQA
from .circuit_simulation import CircuitSimulation, CircuitSimulation  # NOQA
import neuron

from .simulation.neuron_globals import NeuronGlobals

logger.debug("Loading the hoc files.")
import_hoc(neuron)
_ = NeuronGlobals.get_instance()  # initiate the singleton
