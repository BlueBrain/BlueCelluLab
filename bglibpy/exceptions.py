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
