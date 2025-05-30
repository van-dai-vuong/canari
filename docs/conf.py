import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

project = "canari"
copyright = "2025, Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
author = "Van-Dai Vuong, Luong-Ha Nguyen, James-A. Goulet"
release = "v.0.0.2"

extensions = [
    "sphinx.ext.autodoc",  # Auto-import docstrings
    "sphinx.ext.napoleon",  # Google/NumPy style
    "sphinx.ext.viewcode",  # Add source links
    "sphinx.ext.autosummary",  # Summary tables (optional)
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_book_theme",
]
nbsphinx_execute = "never"

autodoc_mock_imports = ["pytagi"]

templates_path = ["_templates"]
exclude_patterns = []

language = "EN"

autosummary_generate = False
html_theme = "sphinx_book_theme"
html_theme_options = {
    "collapse_navbar": False,
    "show_navbar_depth": 0,
    "max_navbar_depth": 5,
    "show_toc_level": 2,
    "includehidden": True,
    "titles_only": False,
}
html_logo = "_static/canari_logo.png"
