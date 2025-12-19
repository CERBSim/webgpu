Renderers and shapes
====================

This page describes how renderers turn your data into GPU draw calls and
how to work with shapes and instance data.


Built-in renderers
------------------

The library ships several renderers for common visualization tasks:

- :class:`webgpu.shapes.ShapeRenderer`

  - Renders many copies of a base shape (cylinder, cone, circle).
  - Supports per-instance positions, directions, scalar values and
    explicit colors.
  - Integrates with :class:`webgpu.colormap.Colormap` to map scalar
    values to colors.

- Triangle-based renderers in :mod:`webgpu.triangles`

  - Work with arbitrary meshes defined by vertex positions, normals and
    triangle indices.
  - Useful when you already have a mesh from another tool or solver.

- Additional helpers such as :mod:`webgpu.labels`,
  :mod:`webgpu.vectors`, :mod:`webgpu.colormap` and
  :mod:`webgpu.clipping` provide labels, vector glyphs, colormaps and
  clipping planes.

The tutorials show concrete combinations of these building blocks for
instanced geometry, vector fields and selection highlights.


Instance data
-------------

Many renderers, in particular :class:`webgpu.shapes.ShapeRenderer`, can
render thousands of instances efficiently. Typical instance attributes
include:

- positions (3D translation per instance)
- directions (orientation or direction vectors)
- values (scalars mapped to colors via a colormap)
- colors (explicit per-instance RGBA values)

These attributes can be passed either as ``numpy`` arrays or as
pre-created GPU buffers. For example::

   renderer.positions = np.random.randn(N, 3)
   renderer.values = my_scalar_field

Changing instance data usually requires a redraw. Call
``renderer.set_needs_update()`` after updating large arrays to ensure the
next frame rebuilds the relevant GPU buffers.


Writing a custom renderer
-------------------------

For data types that do not fit the existing renderers, you can implement
your own by subclassing :class:`webgpu.renderer.Renderer` (most common)
or, for advanced use, :class:`webgpu.renderer.BaseRenderer`.

High‑level lifecycle
~~~~~~~~~~~~~~~~~~~~

Renderers live inside a :class:`webgpu.scene.Scene`:

* you create one or more renderers and pass them to :class:`Scene`,
* :class:`Scene` creates a :class:`webgpu.renderer.RenderOptions`
  object that contains the camera, light, canvas and command encoder,
* on every render, each renderer is asked to ``update`` (if needed) and
  then ``render`` using those options.

The base classes handle most of this for you:

* :class:`BaseRenderer` tracks a timestamp and only calls
  ``update(options)`` when something changed (for example after
  ``set_needs_update()``),
* :class:`Renderer`` builds a ``RenderPipeline`` and, by default,
  performs a simple non‑indexed ``draw`` call plus an optional separate
  selection pass.

In a custom renderer you mainly describe **what to draw** (WGSL shader,
buffers and bindings) and **when to (re)build GPU resources**.

Choosing a base class
~~~~~~~~~~~~~~~~~~~~~

* Subclass :class:`Renderer` when you want a standard graphics pipeline
  (color + depth, optional selection) and are happy with the default
  ``render`` and ``select`` implementations. This is appropriate for
  most visualization use cases.
* Subclass :class:`BaseRenderer` directly only when you need full
  control over pipeline creation or draw calls (for example multiple
  passes, unusual topologies or compute‑driven rendering). Then you are
  responsible for implementing ``create_render_pipeline`` and
  ``render`` yourself.
* Use :class:`webgpu.renderer.MultipleRenderer` to group several
  renderer instances into one logical object that shares selection
  behaviour and ``on_select`` callbacks.

Core responsibilities of a renderer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regardless of the base class, a renderer is responsible for three
things:

* defining shader code,
* defining GPU bindings (buffers, textures, samplers),
* updating those GPU resources when the Python‑side data changes.

The key methods and attributes are:

* ``get_shader_code(self) -> str``

  Return WGSL shader source. The recommended pattern is to store WGSL
  in ``your_module/shaders/`` and load it via
  :func:`webgpu.utils.read_shader_file`. The string is run through
  :func:`webgpu.utils.preprocess_shader_code`, which supports
  ``#import`` (for shared code like camera and lighting) and simple
  ``#ifdef``/``@TOKEN@`` replacement.

  Two defines are injected automatically:

  * ``RENDER_OBJECT_ID`` – a unique integer for this renderer. In
    shaders you typically use ``@RENDER_OBJECT_ID@`` when writing to the
    selection buffer.
  * ``SELECT_PIPELINE`` – defined only for the selection pipeline,
    which makes it easy to branch between normal and selection output
    in the same WGSL file if desired.

* ``get_bindings(self) -> list[webgpu.utils.BaseBinding]``

  Return a list of bindings (uniform buffers, storage buffers,
  textures, samplers) that the renderer needs. Use helper classes such
  as :class:`webgpu.utils.BufferBinding`,
  :class:`webgpu.utils.UniformBinding` and
  :class:`webgpu.utils.TextureBinding`. These bindings are combined with
  scene‑wide bindings from the camera and lights to build the final bind
  group.

* ``update(self, options: webgpu.renderer.RenderOptions) -> None``

  Prepare or refresh GPU‑side state when the scene timestamp changes.
  This often means:

  * creating or updating vertex/index/instance buffers from NumPy
    arrays using :func:`webgpu.utils.buffer_from_array`,
  * updating uniform buffers that depend on current camera or light
    configuration,
  * filling ``self.vertex_buffers`` and ``self.vertex_buffer_layouts``
    (when using classic vertex attributes),
  * setting ``self.n_vertices``, ``self.n_instances``,
    ``self.topology`` and, if needed, alternative WGSL entry points via
    ``self.vertex_entry_point``, ``self.fragment_entry_point`` and
    ``self.select_entry_point``.

  The base class decorator ensures ``update`` is only called when
  necessary. When you change large arrays from Python, call
  ``renderer.set_needs_update()`` so the next frame rebuilds buffers.

How drawing and selection work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you subclass :class:`Renderer`, you normally do not override
``render`` and ``select``:

* :meth:`Renderer.create_render_pipeline` compiles your shader code,
  creates a bind group from ``options.get_bindings() + self.get_bindings()``
  and sets up a graphics pipeline for both color and selection passes.
* :meth:`Renderer.render` opens a render pass from
  :class:`RenderOptions`, binds the pipeline, bind group and any vertex
  buffers in ``self.vertex_buffers``, then calls
  ``draw(self.n_vertices, self.n_instances)``.
* :meth:`Renderer.select` does the same using ``self._select_pipeline``
  and writes into the offscreen selection texture. The
  :class:`Scene` then reads back a single pixel, decodes
  ``obj_id`` and forwards a :class:`webgpu.renderer.SelectEvent` to any
  renderer that registered ``on_select`` callbacks.

For more advanced layouts you can override these methods. For example,
the shape renderer in :mod:`webgpu.shapes` performs indexed drawing
(``drawIndexed``) and sets up multiple vertex buffers for positions,
directions, per‑instance colors/values and additional per‑mesh data.

Patterns for custom renderers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built‑in renderers show common patterns you can copy:

* :class:`webgpu.shapes.ShapeRenderer` uses classic vertex and instance
  attributes. In ``update`` it:

  * converts NumPy arrays into GPU vertex buffers,
  * fills ``self.vertex_buffers`` and ``self.vertex_buffer_layouts`` to
    describe positions, directions, colors and per‑shape data,
  * chooses fragment entry points depending on whether you pass scalar
    values (colormap look‑up) or explicit RGBA colors.

* :class:`webgpu.triangles.TriangulationRenderer` uses storage buffers
  only: it uploads vertex positions and normals via
  :class:`webgpu.utils.BufferBinding` and accesses them from WGSL using
  ``@builtin(vertex_index)`` and ``@builtin(instance_index)``. This is a
  good template when you prefer to keep all geometry in storage buffers
  instead of vertex attributes.

Adapting these patterns
~~~~~~~~~~~~~~~~~~~~~~~

To create your own renderer:

1. Decide whether you want vertex attributes (``vertex_buffers``) or
   storage buffers (:class:`BufferBinding`) or a combination.
2. Subclass :class:`Renderer`, set up your CPU‑side attributes and
   determine ``n_vertices`` and ``n_instances``.
3. In ``update``, create or update GPU buffers, vertex layouts and any
   uniforms. Call ``set_needs_update`` whenever Python‑side data
   changes.
4. Implement ``get_shader_code`` (and, if needed, ``get_bindings``) so
   the WGSL code and bindings match the buffers you created.
5. Optionally customise selection by providing a
   ``select_entry_point`` that writes ``@RENDER_OBJECT_ID@`` and any
   per‑instance information you want to receive in
   :class:`SelectEvent`.

With these pieces in place, your renderer can be dropped into any
existing :class:`Scene` alongside the built‑in renderers.
