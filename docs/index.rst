WebGPU from Python
==================

**Interactive 3D visualization and GPU compute in Jupyter notebooks.**

.. raw:: html
   :file: _static/showcase.html

The ``webgpu`` package gives Python direct access to the browser's
`WebGPU API <https://www.w3.org/TR/webgpu/>`_ — a modern, cross-platform
GPU interface.  It is designed for **scientific applications** (finite
elements, vector fields, point clouds) but exposes the full WebGPU
pipeline so you can build any GPU-accelerated visualization or
computation.


Key features
------------

- **Full WebGPU access** — write WGSL shaders directly, create custom
  render and compute pipelines from Python.
- **Scientific renderers** — built-in support for instanced shapes,
  triangle meshes, vector fields, colormaps, clipping planes, and labels.
- **Jupyter integration** — ``Draw(renderer)`` produces an interactive 3D
  canvas in any notebook cell with rotate/pan/zoom.
- **Export to HTML** — scenes serialize to standalone HTML files with a
  built-in JavaScript WebGPU engine.  No Python runtime needed in the browser.
- **Compute shaders** — run WGSL compute kernels on GPU buffers from Python.
- **Selection & picking** — click objects in the canvas and react in Python.


.. toctree::
   :titlesonly:
   :maxdepth: 1
   :caption: Contents

   installation
   guide
   interactions
   deployment
   performance
   testing
   api
