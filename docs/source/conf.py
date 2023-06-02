# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'TRAK'
copyright = '2023, Kristian Georgiev'
author = 'Kristian Georgiev'

# The full version, including alpha/beta/rc tags
release = '0.2.1'
version = '0.2.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.doctest',
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    'sphinx.ext.autosectionlabel'
]


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)


templates_path = ['_templates']

html_theme = 'furo'
html_theme_options = {
    "source_repository": "https://github.com/MadryLab/trak",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
