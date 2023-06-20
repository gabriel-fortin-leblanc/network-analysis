# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Network Analysis"
copyright = "2023, Gabriel Fortin-Leblanc"
author = "Gabriel Fortin-Leblanc"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.apidoc", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_sidebars = {"**": ["search-field.html"]}

# html_logo = "path/to/myimage.png"

html_theme_options = {
    "repository_url": "https://github.com/gabriel-fortin-leblanc/network-analysis",
    # "path_to_docs": "{path-relative-to-site-root}",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}
