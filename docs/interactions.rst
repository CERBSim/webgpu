GUI Interactions
================

The ``GuiParam`` class provides a declarative way to create interactive
GUI controls (dropdowns, sliders, checkboxes) that write to GPU uniform
buffers.  A single ``GuiParam`` instance can be shared across multiple
renderers — it appears once in the GUI and writes to all bound targets.

Quick start
-----------

.. code-block:: python

   from webgpu import GuiParam

   # A slider that writes directly to a uniform field
   shrink = GuiParam("slider", "Shrink", default=1.0, min=0.0, max=1.0)
   shrink.bind(mesh_uniform, "shrink")

   # A dropdown whose value is written to a buffer field
   component = GuiParam("dropdown", "Component",
                        options={"Norm": -1, "x": 0, "y": 1}, default=-1)
   component.bind(settings, "component")

   # Side effects: when the dropdown changes, also update colormap range
   component.affects(colormap, "min", values={-1: 0.1, 0: -1.0, 1: 0.5})
   component.affects(colormap, "max", values={-1: 5.0, 0: 1.0, 1: 3.0})

Renderers expose their params via ``_gui_params``.  The scene collects
them automatically at render time — no manual wiring needed.


Concepts
--------

**bind(ref, field_name)**
   Write the selected value directly to a uniform buffer field.
   The byte offset and data type are derived automatically from the
   ctypes structure definition.

**affects(ref, field_name, values=...)**
   On change, look up the selected value in a dict and write the
   result to a different field.  Use this for side effects like
   updating a colormap range when a component changes.

**Sharing**
   Multiple renderers can call ``bind()`` on the same ``GuiParam``.
   The GUI shows one control that writes to all bound targets.
   This is the primary mechanism for keeping multiple views in sync.

**Lazy resolution**
   ``bind`` and ``affects`` accept references that are resolved at
   export time.  You can pass:

   - A ``UniformBase`` (ctypes Structure) directly
   - An object with a ``.uniform`` attribute (e.g. ``FunctionSettings``)
   - An object with a ``.uniforms`` attribute (e.g. ``Colormap``)
   - A callable that returns a uniform


Control types
-------------

Dropdown
~~~~~~~~

.. code-block:: python

   mode = GuiParam("dropdown", "Mode",
                   options={"Real": 0, "Imag": 1, "Abs": 2},
                   default=0)
   mode.bind(complex_settings, "mode")

The ``options`` dict maps display labels to values.  The selected value
is what gets written to the buffer.

Slider
~~~~~~

.. code-block:: python

   phase = GuiParam("slider", "Phase",
                    default=0.0, min=0.0, max=6.283, step=0.01)
   phase.bind(complex_settings, "phase")

Checkbox
~~~~~~~~

.. code-block:: python

   enabled = GuiParam("checkbox", "Clipping", default=True)
   enabled.affects(clipping_uniforms, "mode", values={True: 1, False: 0})


Shared parameters across renderers
-----------------------------------

When multiple renderers visualize the same data, they can share a
``GuiParam`` so that one control updates all of them:

.. code-block:: python

   from webgpu import GuiParam
   from ngsolve_webgpu import FunctionData, CFRenderer, ClippingCF

   function_data = FunctionData(mesh_data, cf, order=5)

   colormap = Colormap()
   cfr = CFRenderer(function_data, colormap=colormap, clipping=clipping)
   clip = ClippingCF(function_data, clipping=clipping, colormap=colormap)

   # Both renderers auto-bind to function_data.component_param
   # One dropdown appears, writes to both, rescales colormap.
   Draw([cfr, clip, Colorbar(colormap)])

In this example, ``FunctionData`` creates a shared ``component_param``
internally.  Both ``CFRenderer`` and ``ClippingCF`` bind their settings
uniform to it during construction.  The colormap min/max side effects
are also set up automatically when the colormap has ``autoscale=True``.


Custom parameters
-----------------

You can create your own ``GuiParam`` and attach it to a renderer:

.. code-block:: python

   my_param = GuiParam("slider", "Threshold", default=0.5, min=0.0, max=1.0)
   my_param.bind(my_renderer.uniforms, "threshold")

   # Register so the scene picks it up
   my_renderer._gui_params.append(my_param)

   Draw([my_renderer])


How it works internally
-----------------------

1. Renderers store ``GuiParam`` instances in ``self._gui_params``.
2. When the scene is initialized, it collects all params (deduplicated
   by identity — same object = one control).
3. Each ``GuiParam`` exports itself as an ``Interaction`` entry containing
   the control definition and write targets.
4. The JavaScript engine creates a lil-gui control and executes the
   buffer writes on change.


API Reference
-------------

.. automodule:: webgpu.gui_param
   :members:
   :undoc-members:
