# NGSolve with Pyodide and WebGPU

This repository contains an example setup for using NGSolve with Pyodide and WebGPU. Follow the instructions below to set up and run the project, either as a standalone HTML server or within a Jupyter Notebook.

You can test the hosted example here (takes a few seconds to init Pyodide): https://mhochsteger.github.io/webgpu_pyodide_experiments/

Note that you need a browser with webgpu support.

## Setup

To get started, download and extract the required NGSolve Pyodide package:

```bash
wget https://ngsolve.org/files/ngsolve_pyodide_0.26.2.tar.bz2
tar xvf ngsolve_pyodide_0.26.2.tar.bz2
python3 -m pip install -e .
```

## Running the server

Run `python dev.py` and open `http://localhost:8000/index.html`. Everytime you change some code, the website is automatically hot-reloaded (pyodide stays alive) and the `webgpu` package is reloaded and executed.
For the hot-reload feature you need to have the python packages `websockets` and `watchdog` installed

## Jupyter notebook

```
jupyter notebook webgpu.ipynb
```
