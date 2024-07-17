# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'BlueCelluLab'
author = 'Blue Brain Project/EPFL'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

autosummary_mock_imports = [  # these modules are not publicly available
    'bluepy',
    'bluepy_configfile'
]

suppress_warnings = [
    'autosummary.import_cycle',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx-bluebrain-theme"
html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
]

# Configure doctest
doctest_test_doctest_blocks = 'default'

# For matplotlib plots in Sphinx
plot_include_source = True
plot_html_show_source_link = False

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_default_options = {
    "members": True,
}

# Autodoc-typehints

always_document_param_types = True
typehints_use_rtype = True
