Getting started
===============

Installation
------------

The ``webgpu`` package is available on PyPI and can also be installed
from source. A recent Python (3.8+) and a browser with WebGPU support
(Chrome, Edge, or another browser with WebGPU enabled) are required.

**Install from PyPI**::

   pip install webgpu

**Install from a local checkout**::

   git clone https://github.com/CERBSim/webgpu
   cd webgpu
   pip install -e .


Your first scene in a notebook
------------------------------

The easiest way to start experimenting is inside a Jupyter notebook.

1. Start Jupyter (Notebook or Lab) and create a new Python notebook.
2. Run the following minimal example cell:

   .. code-block:: python

      import numpy as np

      from webgpu.shapes import generate_cylinder, ShapeRenderer
      from webgpu.scene import Scene
      from webgpu.jupyter import Draw

      # 1) Create some simple geometry on the CPU
      shape = generate_cylinder(n=32, radius=0.5, height=1.0)

      # 2) Wrap it in a renderer that knows how to talk to WebGPU
      renderer = ShapeRenderer(shape_data=shape, label="demo")

      # 3) Put renderers into a Scene and draw it into a WebGPU canvas
      scene = Scene([renderer])
      Draw(scene, width=480, height=480)

If your browser supports WebGPU, you should see a shaded cylinder rendered
in the notebook output. From here, you can start modifying the geometry,
camera and colors to explore the API.


Main concepts
-------------

The library is built around a small set of concepts:

**Device and canvas (hidden by default)**
   A WebGPU *device* and HTML *canvas* are created for you when you call
   ``Draw(...)`` in a notebook. You normally do not manage adapters,
   devices or swapchains directly.

**Renderer**
   A ``Renderer`` turns some domain-specific data (triangles, shapes,
   vectors, labels, etc.) into GPU buffers and a WebGPU render pipeline.

**Scene**
   A ``Scene`` holds one or more renderers, plus common state like camera,
   lights and clipping configuration. The scene is what you pass to
   ``Draw(...)``.

**Jupyter / Pyodide integration**
   The ``webgpu.jupyter`` module takes care of connecting your Python
   process to the browser's WebGPU implementation. It supports both local
   Python (via a websocket link) and pure-browser Pyodide deployments (via
   an exported HTML using Pyodide).

The tutorial notebooks expand on these ideas with more complex examples,
including instancing, selection, compute shaders and more.
