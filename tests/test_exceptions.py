"""Unit tests for the exceptions module."""

from bluecellulab.exceptions import error_context


def test_error_context():
    """Unit test for the error_context function."""
    attr_err_msg = "hoc.HocObject' object has no attribute 'minis_single_vesicle_"
    lookup_err_msg = "'X' is not a defined hoc variable name"
    context_info = "mechanism/s for minis_single_vesicle need to be compiled"
    try:
        with error_context(context_info):
            raise AttributeError(attr_err_msg)
    except AttributeError as error:
        assert str(error) == f"{context_info}: {attr_err_msg}"
    try:
        with error_context(context_info):
            raise LookupError(lookup_err_msg)
    except LookupError as error:
        assert str(error) == f"{context_info}: {lookup_err_msg}"
