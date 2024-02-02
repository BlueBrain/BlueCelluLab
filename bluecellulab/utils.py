"""Utility functions."""


def run_once(func):
    """A decorator to ensure a function is only called once."""
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper
