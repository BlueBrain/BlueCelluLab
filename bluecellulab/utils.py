"""Utility functions."""
import contextlib
import io


def run_once(func):
    """A decorator to ensure a function is only called once."""
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


class CaptureOutput(list):
    def __enter__(self):
        self._stringio = io.StringIO()
        self._redirect_stdout = contextlib.redirect_stdout(self._stringio)
        self._redirect_stdout.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._redirect_stdout.__exit__(exc_type, exc_val, exc_tb)
        self.extend(self._stringio.getvalue().splitlines())
