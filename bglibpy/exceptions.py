"""Custom exceptions used within the package."""


class PopulationIDMissingError(Exception):
    """Raise when the population id of a projection is missing."""
    pass


class TargetDoesNotExist(Exception):
    """It is raised upon calling a target that does not exist."""
    pass


class UndefinedRNGException(Exception):
    """Raise when the RNG mode to be used does not exist."""
    pass


class OldNeurodamusVersionError(Exception):
    """Raise when the loaded neurodamus does not support new feature."""
    pass


class ConfigError(Exception):
    """Error due to invalid settings in BlueConfig"""
    pass
