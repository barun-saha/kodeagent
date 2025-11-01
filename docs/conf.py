import os
import sys

# --- Path setup ---
# Crucial: This tells Sphinx to look in 'src' to find the 'kodeagent' package.
sys.path.insert(0, os.path.abspath('../src'))

# --- Project information ---------------------------------------------------
project = 'KodeAgent'  # <--- Set this to your desired name (e.g., 'KodeAgent')
copyright = '2025, Barun Saha'
author = 'Barun Saha'

# --- General configuration ---
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',    # Converts Google/NumPy style docstrings
    'sphinx_rtd_theme',
    'myst_parser',            # Enables Markdown support (.md files)
    'sphinx.ext.autosummary', # <--- Add this extension
]
autosummary_generate = True

# --- Autodoc configuration for sorting ---
autodoc_member_order = 'alphabetical'

# Tell Sphinx to look for custom templates
templates_path = ['_templates']

# Configure MyST to allow cross-referencing and nested structure
myst_enable_extensions = [
    "deflist",
    "html_image",
    "linkify",
    "replacements",
    "html_admonition"
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# ... other project information (project, author, release)
html_theme = 'pydata_sphinx_theme'

master_doc = 'index'
