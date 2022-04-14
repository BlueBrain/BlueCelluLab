"""Custom exceptions used within the package."""


class BGLibPyError(Exception):
    """Class to identify BGLibPy specific exceptions."""


class PopulationIDMissingError(BGLibPyError):
    """Raise when the population id of a projection is missing."""


class TargetDoesNotExist(BGLibPyError):
    """It is raised upon calling a target that does not exist."""


class UndefinedRNGException(BGLibPyError):
    """Raise when the RNG mode to be used does not exist."""


class OldNeurodamusVersionError(BGLibPyError):
    """Raise when the loaded neurodamus does not support new feature."""


class ConfigError(BGLibPyError):
    """Error due to invalid settings in BlueConfig"""


class NeuronEvalError(BGLibPyError):
    """Raise when an unsupported code string is sent to neuron."""
