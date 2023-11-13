import logging
import os
import pytest
from types import ModuleType
from unittest.mock import MagicMock, patch
from bluecellulab import importer
from bluecellulab.exceptions import BluecellulabError


@patch("os.path.isdir", return_value=False)  # when x86_64 isdir returns False
def test_import_mod_lib_no_env_no_folder(mocked_isdir):
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        assert importer.import_mod_lib(mock_neuron) == "No mechanisms are loaded."


def test_import_mod_lib_env_var_set_folder_exists():
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": "/fake/path"}):
        with patch(
            "os.path.isdir", return_value=True
        ):  # when x86_64 isdir returns True and env var is set
            with pytest.raises(
                BluecellulabError,
                match="BLUECELLULAB_MOD_LIBRARY_PATH is set and current directory contains the x86_64 folder. Please remove one of them.",
            ):
                importer.import_mod_lib(mock_neuron)


def test_import_mod_lib_env_var_set():
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": "/fake/path"}):
        with patch("os.path.isdir", return_value=False):
            assert importer.import_mod_lib(mock_neuron) == "/fake/path"


def test_import_mod_lib_so_file():
    mock_neuron = MagicMock()
    fake_so_path = "/fake/path/to/library.so"
    with patch.dict(os.environ, {"BLUECELLULAB_MOD_LIBRARY_PATH": fake_so_path}):
        with patch("os.path.isdir", return_value=False):
            importer.import_mod_lib(mock_neuron)
            mock_neuron.h.nrn_load_dll.assert_called_with(fake_so_path)


def test_import_mod_lib_no_env_with_folder():
    mock_neuron = MagicMock()
    with patch.dict(os.environ, {}, clear=True):
        with patch("os.path.isdir", return_value=True):
            assert importer.import_mod_lib(mock_neuron).endswith("x86_64")


@patch.object(
    importer.pkg_resources,
    "resource_filename",
    return_value="/fake/path/to/hoc_file.hoc",
)
def test_import_neurodamus(mocked_pkg_resources):
    mock_neuron = MagicMock()
    importer.import_neurodamus(mock_neuron)
    assert mock_neuron.h.load_file.called
    # Check that it was called with the expected arguments
    mock_neuron.h.load_file.assert_any_call("/fake/path/to/hoc_file.hoc")


def test_print_header(caplog):
    # Creating a dummy ModuleType object with an attribute '__file__'
    dummy_neuron = ModuleType("dummy_neuron")
    dummy_neuron.__file__ = "/path/to/neuron"

    mod_lib_path = "/path/to/mod_lib"
    with caplog.at_level(logging.INFO):
        importer.print_header(dummy_neuron, mod_lib_path)

    assert "Imported NEURON from: /path/to/neuron" in caplog.text
    assert "Mod lib: /path/to/mod_lib" in caplog.text
