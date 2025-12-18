Core concepts
=============

This page describes the core pieces of the ``webgpu`` framework and how
they fit together.


Data flow and building blocks
-----------------------------

A typical visualization follows these steps:

1. Prepare CPU-side data (meshes, vectors, labels, scalar fields).
2. Wrap the data in one or more *renderers*.
3. Put the renderers into a :class:`webgpu.scene.Scene`.
4. Call :func:`webgpu.jupyter.Draw` in a notebook cell.

``Draw`` takes care of creating a WebGPU device, a Canvas connected to an
HTML canvas element, and a small GUI panel for per-object options.

The main building blocks are:

- **Renderers** – create GPU buffers and pipelines from domain data.
- **Scene** – groups renderers plus common state such as camera and
  lights.
- **Camera** – defines how the scene is projected to the screen and
  handles user input.
- **Light** – defines lighting direction and intensity.
- **Canvas** – wraps the HTML canvas, manages color, depth and selection
  textures, and owns the WebGPU context.

You usually work at the level of renderers and scenes; device and canvas
management is handled behind the scenes.


Notebook workflow
-----------------

In a notebook, a minimal yet realistic workflow looks like::

   import numpy as np

   from webgpu.shapes import generate_cylinder, ShapeRenderer
   from webgpu.scene import Scene
   from webgpu.jupyter import Draw

   shape = generate_cylinder(n=64, radius=0.5, height=1.0)
   renderer = ShapeRenderer(shape_data=shape, label="example")

   scene = Scene([renderer])
   Draw(scene, width=512, height=512)

After this, you can interact with the view (rotate/zoom) and adjust
renderer parameters via the automatically created GUI.

The remaining user-guide pages build on this workflow and explain how to
customize each part.
