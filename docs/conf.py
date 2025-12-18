#!/usr/bin/env python3
# -*- coding: utf-8 -*-

extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.todo", "nbsphinx"]

import os
import zipfile

os.environ["WEBGPU_EXPORTING"] = "1"
master_doc = "index"
source_suffix = [".rst", ".md"]

language = "en"
nbsphinx_execute = "auto"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".virtual_documents/*",
    # Ignore a previously generated HTML tree under docs/html so that
    # its notebooks and doctrees are not treated as source documents.
    "html/**",
]

html_theme = "pydata_sphinx_theme"
# html_theme = "piccolo_theme"

html_static_path = ["_static"]

html_logo = "_static/logo.svg"

html_theme_options = {
    "logo": {
        "text": "Webgpu Docs",
        "image_dark": "_static/logo_dark.png",
        "image_light": "_static/logo.svg",
    },
    # Show deeper levels in the right-hand "On this page" TOC so that
    # class entries under sections like "Scenes and renderers" are
    # always visible, not only when their section is active.
    "show_toc_level": 2,
}

todo_include_todos = True

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
# autodoc_typehints = "description"

# Introduce line breaks if the line length is greater than 80 characters.
python_maximum_signature_line_length = 80


def setup(app):
    app.add_css_file("custom.css")

    # After the HTML build finishes, create a ZIP archive of the
    # top-level ``tutorials`` folder (if it exists) and place it in
    # the HTML static output directory so it can be downloaded from
    # the documentation.

    def build_tutorials_zip(app, exception):
        if exception is not None:
            return

        # Tutorials are expected in ``../tutorials`` relative to the
        # documentation source directory.
        tutorials_src = os.path.abspath(os.path.join(app.srcdir, "..", "tutorials"))
        if not os.path.isdir(tutorials_src):
            return

        static_dir = os.path.join(app.outdir, "_static")
        os.makedirs(static_dir, exist_ok=True)
        zip_path = os.path.join(static_dir, "tutorials.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tutorials_src):
                for name in files:
                    if not name.endswith(".ipynb"):
                        continue
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, tutorials_src)
                    zf.write(full_path, rel_path)

    app.connect("build-finished", build_tutorials_zip)
