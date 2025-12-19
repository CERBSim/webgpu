Compute and deployment
======================

This page explains how to run compute shaders from Python and how the
framework can be used in different execution environments.


Running compute shaders
-----------------------

For standalone compute workloads that do not go through a renderer, use
:func:`webgpu.utils.run_compute_shader` from :mod:`webgpu.utils`.

The helper function:

- Compiles WGSL code (with ``preprocess_shader_code`` and
  ``#define/#import`` support).
- Creates a compute pipeline and bind group from a list of bindings.
- Dispatches a user-specified number of workgroups.

Sketch of a typical pattern::

   from math import ceil
   import numpy as np

   from webgpu.utils import run_compute_shader, buffer_from_array

   data = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
   buf = buffer_from_array(data, usage=...)  # include STORAGE usage

   code = """
   @group(0) @binding(0) var<storage, read_write> data: array<f32>;

   @compute @workgroup_size(64)
   fn main(@builtin(global_invocation_id) id: vec3<u32>) {
       let i = id.x;
       if (i < arrayLength(&data)) {
           data[i] = data[i] * 2.0;
       }
   }
   """

   n_workgroups = ceil(len(data) / 64)
   run_compute_shader(code, [buf], n_workgroups=n_workgroups)

For more advanced scenarios you can reuse an existing command encoder and
queue multiple compute or render passes before submitting.


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
