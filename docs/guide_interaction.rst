Interaction, scenes and camera
==============================

This page focuses on how user interaction, scenes, cameras and selection
work together.


Scenes and camera
-----------------

A :class:`webgpu.scene.Scene` groups one or more renderers and shared
state:

- :class:`webgpu.camera.Camera` – view and projection, plus interaction
  logic (orbiting, panning, zoom).
- :class:`webgpu.light.Light` – light direction and intensity.
- :class:`webgpu.canvas.Canvas` – wraps the HTML canvas and manages the
  color, depth and selection textures.

When you construct a ``Scene``, the camera is automatically initialized
from the combined bounding boxes of all renderers. Calling
``Draw(scene, ...)`` binds the scene to a canvas, creates the GUI and
starts rendering.

For many use cases, the default camera and light settings are sufficient
and you only modify per-renderer options.


Camera controls
---------------

By default, the camera reacts to mouse and touch input:

- Rotate/orbit around the scene.
- Zoom in and out.
- Pan the view.

The exact bindings can be inspected and customized via the
:mod:`webgpu.input_handler` module, but for most users the defaults give
sensible behaviour without extra configuration.


Selection and picking
---------------------

The framework supports picking objects and reacting to clicks.

Each renderer instance gets a unique numeric id which is written into a
hidden selection buffer during a special render pass. On a click, the
framework reads back a small region of this buffer and constructs a
:class:`webgpu.renderer.SelectEvent` with:

- ``x, y`` – pixel coordinates in the canvas.
- ``obj_id`` – the selected object id (or 0 if nothing was hit).
- ``z`` – depth value at the hit point.
- ``user_data`` – optional extra data encoded by the renderer.

To react to selections on a renderer::

   def on_pick(ev):
       print("picked id", ev.obj_id, "extra", ev.float32)

   renderer.on_select(on_pick)

Inside the callback you can read per-instance data from ``ev.float32`` or
``ev.uint32`` and update the scene (for example highlight an instance or
show details in a separate widget).

For one-off queries of the world-space position under the cursor, use
:meth:`webgpu.scene.Scene.get_position`, which returns a 3D point or
``None`` if nothing was hit.
