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
sys.path.insert(0, os.path.abspath("../fsmpy"))


# -- Project information -----------------------------------------------------

project = 'fsmpy - fuzzy_set_measures'
copyright = '2022, Machine Learning and Vision Research Group'
author = 'Machine Learning and Vision Research Group'


# -- General configuration ---------------------------------------------------
viewcode_follow_imported_members = True
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
]

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    filepath = os.path.abspath(os.path.join("../src/", filename + "." + domain))
    if len(info["fullname"].split(".")) > 1:
        line_index = 0
        parts = info["fullname"].split(".")
        lines = open(filepath, "r").readlines()
        start_index = 0
        for part in parts:
            start_index = next(
                i + start_index
                for i, l in enumerate(lines[start_index:])
                if (start_index == 0 and part in l) or (start_index > 0 and part in l and "def" in l)
            )
        line_index = start_index
    else:
        try:
            lines = open(filepath, "r").readlines()
        except FileNotFoundError:
            filepath = os.path.abspath(os.path.join("../src/", filename, "__init__." + domain))
            lines = open(filepath, "r").readlines()
        line_index = next(
            i
            for i, l in enumerate(lines)
            if info["fullname"] in l
        )
    return "https://github.com/MachineLearningVisionRG/fsmpy/tree/main/src/{}.py#L{}".format(filename, line_index + 1)

numpydoc_validation_checks = {
    "all"
}

numpydoc_xref_aliases = {
    "BaseEstimator": "sklearn.base.BaseEstimator"
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

intersphinx_mapping = {
    'python': ('http://docs.python.org/2', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('http://matplotlib.sourceforge.net', None),
    'sklearn': ("https://scikit-learn.org/stable/", None)
}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
]
add_module_names = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
