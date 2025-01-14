# WebGPU API for Python using Pyodide

This repository contains a Python wrapper for the WebGPU API to be used in browser environments (for instance plain HTML or Jupyter Notebooks).

An example jupyter notebook file is [here](example.ipynb), the executed html is available [here](https://cerbsim.github.io/webgpu/).

## Setup

To get started, just install the python package and run the jupyter notebook.

```bash
git clone https://github.com/cerbsim/webgpu
cd webgpu
python3 -m pip install .
python3 -m jupyter notebook example.ipynb
```

Note that your browser needs WebGPU support, you can verify that by opening the [webgpureport](https://webgpureport.org) website. See [here](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) for more information about the current status and needed settings.
