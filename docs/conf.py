"""
Sphinx configuration file for the KodeAgent documentation.
This file sets up Sphinx to generate documentation from the source code
located in the 'src' directory, and includes support for Markdown files
using the MyST parser.
"""
import os
import sys

# --- Path setup ---
# Crucial: This tells Sphinx to look in 'src' to find the 'kodeagent' package.
sys.path.insert(0, os.path.abspath('../src'))

# --- Project information ---
project = 'KodeAgent'
copyright = '2025, Barun Saha'
author = 'Barun Saha'

# --- General configuration ---
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',    # Converts Google/NumPy style docstrings
    'sphinx.ext.viewcode',
    'myst_parser',            # Enables Markdown support (.md files)
    'sphinxcontrib.kroki',    # Use Kroki image rendering of Mermaid
]
autosummary_generate = True

# --- Autodoc configuration for sorting ---
autodoc_member_order = 'alphabetical'

# Tell Sphinx to look for custom templates
templates_path = ['_templates']

# Configure MyST to allow cross-referencing and nested structure
myst_enable_extensions = [
    'deflist',
    'html_image',
    'linkify',
    'replacements',
    'html_admonition'
]
# myst_fence_as_directive = ["mermaid"]
# Kroki configuration
kroki_server_url = "https://kroki.io"   # public server; use your own for offline/CI
kroki_output_format = "svg"             # or "png"
kroki_default_processing = "server"     # ensures server-side rendering
kroki_inline_svg = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'pydata_sphinx_theme'
master_doc = 'index'
html_show_sourcelink = True

html_context = {
    "display_github": True,
    "github_user": "barun-saha",
    "github_repo": "kodeagent",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_theme_options = {
    "github_url": "https://github.com/barun-saha/kodeagent",
}
