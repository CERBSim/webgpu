# WebGPU API for Python for local and Pyodide development

This library provides a Python wrapper around the WebGPU API that runs in the browser via Pyodide or connects to a local Python instance. It is intended for interactive visualization and compute in environments such as Jupyter notebooks and plain HTML pages.

The package bundles a small rendering framework (scenes, cameras, labels, colormaps) and utilities to work with WGSL shaders from Python. It is used in higher level frameworks like ngsolve_webgpu for GPU-accelerated finite element visualizations in the browser.

---

## Features

- **Browser-based GPU access** via WebGPU from local Python and Pyodide.
- **Python-first API** for creating renderers, scenes and simple GUI controls.
- **Jupyter integration** through a `webgpu.jupyter` helper module.
- **WGSL tooling**:
  - simple `#import` mechanism for shader code
  - conditional compilation using `#ifdef/#else/#endif`
- **Ready-to-use building blocks**:
  - `Renderer`, `Scene`, `Camera`, `Labels`, `Colormap`, `Clipping`, `Font`
  - utilities for GPU buffers and bindings

---

## Requirements

- Python 3.8 or newer
- A browser with WebGPU support (for example a recent Chrome or Edge)
- For notebook use: JupyterLab or Jupyter Notebook

To check WebGPU support in your browser, you can use common WebGPU capability pages (for example WebGPU report pages linked from the project website).

---

## Installation

### From PyPI

```bash
pip install webgpu
```

### From source

```bash
git clone https://github.com/CERBSim/webgpu
cd webgpu
pip install -e .
```

## Documentation

The full project documentation, including API reference, tutorials and user guide, is published at:

- https://cerbsim.github.io/webgpu/

## Examples

The repository contains several examples:

- examples/first_shader.ipynb – minimal example drawing a triangle with custom WGSL shader code.
- examples/websocket.ipynb – example using the websocket link.
- docs/introduction.ipynb – overview and introductory material.

Additional, more advanced examples in docs/*.ipynb (compute, instanced shapes, selection).

## Browser support

This library relies on the browser’s WebGPU implementation. In practice this means:

- A recent Chromium-based browser (Chrome, Edge) or Firefox is recommended.
- Some platforms may require enabling experimental WebGPU flags.
- Mobile and non-Chromium browsers may have partial or no support.

If WebGPU is not available, initialization will fail. See project documentation and browser-specific WebGPU status pages for details and troubleshooting.

## License

This project is licensed under the terms of the LGPL-2.1-or-later. See the LICENSE file for details.

