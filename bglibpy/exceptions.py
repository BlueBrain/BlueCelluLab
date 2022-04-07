"""Custom exceptions used within the package."""


class BGLibPyError(Exception):
    """Class to identify BGLibPy specific exceptions."""
    pass


class PopulationIDMissingError(BGLibPyError):
    """Raise when the population id of a projection is missing."""
    pass


class TargetDoesNotExist(BGLibPyError):
    """It is raised upon calling a target that does not exist."""
    pass


class UndefinedRNGException(BGLibPyError):
    """Raise when the RNG mode to be used does not exist."""
    pass


class OldNeurodamusVersionError(BGLibPyError):
    """Raise when the loaded neurodamus does not support new feature."""
    pass


class ConfigError(BGLibPyError):
    """Error due to invalid settings in BlueConfig"""
    pass


class NeuronEvalError(BGLibPyError):
    """Raise when an unsupported code string is sent to neuron."""
    pass
