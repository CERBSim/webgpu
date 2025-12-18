Performance and best practices
==============================

This page summarizes patterns that keep visualizations responsive and
scalable.


Renderer lifecycle
------------------

Each renderer tracks when its GPU state was last updated using an
internal timestamp. On a new render pass, only renderers whose data changed
need to rebuild pipelines or buffers.

To work with this mechanism:

- Reuse renderer instances and update their data instead of recreating
  them on every change.
- Call ``renderer.set_needs_update()`` after modifying large arrays or
  other state that should be copied to GPU buffers.


Data layout
-----------

Careful data layout avoids unnecessary copies and conversions:

- Use contiguous ``numpy`` arrays with explicit ``dtype`` (typically
  ``float32`` or ``uint32``) before creating GPU buffers.
- Pack related attributes together where it matches the shader layout to
  minimise the number of separate buffers.


Drawing many objects
--------------------

For large numbers of similar objects, prefer instanced rendering over
many independent renderers:

- Use :class:`webgpu.shapes.ShapeRenderer` (or similar instanced
  renderers) with per-instance attribute arrays.
- Batch updates to instance data where possible instead of modifying
  thousands of small Python objects.


Interaction and responsiveness
------------------------------

- Avoid performing heavy CPU work inside selection callbacks or GUI
  event handlers; move such work into separate functions or background
  jobs where practical.
- Keep shader code simple and data-parallel; avoid unnecessary branching
  in inner loops.

These guidelines, together with the architectural overview and examples
in the other user-guide pages, should give you a solid baseline for
building efficient visualizations with ``webgpu``.
