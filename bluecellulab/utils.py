"""Utility functions used within BlueCellulab."""
from __future__ import annotations
import contextlib
import io
import json
from typing import Any

import numpy as np


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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Singleton(type):
    """Singleton metaclass implementation.

    Source: https://stackoverflow.com/a/6798042/1935611
    """
    _instances: dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        else:  # to run init on the same object
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]
