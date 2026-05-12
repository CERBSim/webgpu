Deployment and Execution Modes
==============================

This page describes how the framework connects to WebGPU in different
environments and how to export scenes for use without Python.


Execution modes
---------------

The :mod:`webgpu.jupyter` module supports two main execution modes in
notebooks:

- **Python via websocket**

  - Your Python kernel runs on a local machine or remote server.
  - The browser connects back via a websocket and forwards WebGPU calls
    to the browser instance.
  - This is the default when running notebooks on a normal Jupyter
    server.

- **Pyodide**

  - Python runs inside the browser using Pyodide.
  - The same high-level API (``Draw``, ``Scene``, renderers) is used, but
    the bridge to WebGPU goes through the Pyodide integration instead of
    a websocket.

In most cases you can use the same notebook code in both modes; only
infrastructure and packaging differ.


Sphinx documentation builds
----------------------------

This project's ``docs/conf.py`` sets ``WEBGPU_EXPORTING=1`` so that
notebooks processed by ``nbsphinx`` automatically produce interactive
3D scenes in the generated HTML pages.

Prerequisites for the build host:

- ``pip install playwright && playwright install chrome``
- Vulkan support (real GPU or ``mesa-vulkan-drivers`` for software
  rendering via lavapipe)

The JavaScript engine embedded in the output supports camera interaction
(rotate, pan, zoom) and, for scenes that include clipping or colormap
uniforms, interactive GUI controls powered by lil-gui.
