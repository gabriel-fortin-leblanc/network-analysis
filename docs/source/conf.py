# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import re
import sys

# Add the project source code directory to the path
sys.path.append(os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Network Analysis"
copyright = "2023, Gabriel Fortin-Leblanc"
author = "Gabriel Fortin-Leblanc"

# Get the version from pyproject.toml
with open("../../pyproject.toml", encoding="utf-8") as f:
    string = f.read()
    release = re.search(r'version = "(.*?)"', string).group(1)
    version = release.rsplit(".", 1)[0]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "Network Analysis"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/gabriel-fortin-leblanc/network-analysis",
    # Header
    "use_repository_button": True,
    "use_issues_button": True,
    "logo": {
        "text": "Network Analysis",
    },
    "navbar_start": ["navbar-logo", "navbar-nav"],
    # Primary sidebar (left sidebar)
    "primary_sidebar_end": ["sidebar-ethical-ads"],
}
html_sidebars = {
    "**": [
        "search-field.html",
        "sidebar-nav-bs",
    ],
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
}

autodoc_default_options = {
    "inherited-members": True,
    "show-inheritance": True,
    "imported-members": True,
}
autodoc_class_signature = "separated"
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autosummary_imported_members = True
