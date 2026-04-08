"""Sphinx configuration for TOD Generation from Beam Convolution."""

import os
import sys

# Make the project root importable so autodoc can import the modules.
sys.path.insert(0, os.path.abspath(".."))

# ── Project information ────────────────────────────────────────────────────────
project = "TOD Generation from Beam Convolution"
copyright = "APC"
author = "APC"

# ── Sphinx extensions ──────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",  # Pull docstrings from source
    "sphinx.ext.napoleon",  # Parse Google-style docstrings
    "sphinx.ext.viewcode",  # Add [source] links
    "sphinx.ext.intersphinx",  # Cross-reference numpy / healpy docs
    "sphinx.ext.autosummary",  # Generate summary tables
]

# ── Napoleon (Google docstring) settings ──────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # only Google style in this project
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# ── Autodoc settings ──────────────────────────────────────────────────────────
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True

# ── Intersphinx mappings ───────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "healpy": ("https://healpy.readthedocs.io/en/latest", None),
}

# ── HTML output ───────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_static_path = []

# ── General ───────────────────────────────────────────────────────────────────
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
