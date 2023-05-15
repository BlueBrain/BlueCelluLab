"""Custom exceptions used within the package."""


from contextlib import contextmanager


class BGLibPyError(Exception):
    """Class to identify BGLibPy specific exceptions."""


class TargetDoesNotExist(BGLibPyError):
    """It is raised upon calling a target that does not exist."""


class UndefinedRNGException(BGLibPyError):
    """Raise when the RNG mode to be used does not exist."""


class ConfigError(BGLibPyError):
    """Error due to invalid settings in BlueConfig"""


class NeuronEvalError(BGLibPyError):
    """Raise when an unsupported code string is sent to NEURON."""


class MissingSonataPropertyError(BGLibPyError):
    """Raise when a property is missing from SONATA."""


class ExtraDependencyMissingError(BGLibPyError):
    """Raise when an extra dependency is missing."""

    def __init__(self, dependency_name):
        self.dependency_name = dependency_name
        super().__init__(f"The extra dependency '{dependency_name}' is missing. "
                         f"Please install it to use this feature.")


@contextmanager
def error_context(context_info: str):
    """Use when the attribute/lookup error needs more context information.
    Useful for NEURON/HOC attribute/lookup errors.
    E.g. 'AttributeError: 'hoc.HocObject' object has no attribute' or
    LookupError: 'X' is not a defined hoc variable name messages are
      often not very helpful. Extra context information can be added
        to the error message.
    """
    try:
        yield
    except AttributeError as e:
        raise AttributeError(f"{context_info}: {e}")
    except LookupError as e:
        raise LookupError(f"{context_info}: {e}")
