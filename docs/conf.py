# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HOME"
copyright = "2024, Nils Dittrich, Francis Barre, Zoe CordHomme"
author = "Nils Dittrich, Francis Barre, Zoe CordHomme"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc", 
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

# %% check path
# give docs the location: ../HOME
import os
import sys

root_dir = os.path.abspath("../HOME")
print(root_dir)
sys.path.insert(0, root_dir)
