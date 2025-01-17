import base64
import pickle

from .utils import reload_package, _is_pyodide

import base64


def create_package_zip(module_name="webgpu"):
    """
    Creates a zip file containing all files in the specified Python package.
    """
    import importlib.util
    import os
    import tempfile
    import zipfile

    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ValueError(f"Package {module_name} not found.")

    package_dir = os.path.dirname(spec.origin)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, f"{module_name}.zip")
        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(
                        file_path, start=os.path.dirname(package_dir)
                    )
                    zipf.write(file_path, arcname)

        return open(output_filename, "rb").read()


_package_b64 = base64.b64encode(create_package_zip()).decode("utf-8")

_init_js_code = (
    r"""
const SNAPSHOT_URL = 'https://cdn.jsdelivr.net/gh/mhochsteger/ngsolve_pyodide@webgpu1/snapshot.bin.gz';

function decodeB64(base64String) {
    const binaryString = atob(base64String);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

async function fetchSnapshot() {
  const blob = await (await fetch(SNAPSHOT_URL)).blob();
  const decompressor = new DecompressionStream('gzip');
  const stream = blob.stream().pipeThrough(decompressor);
  const response = new Response(stream);
  return await response.arrayBuffer();
};

async function main() {
  if(window.webgpu_ready === undefined) {
      const pyodide_module = await import("https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.mjs");
      window.pyodide = await pyodide_module.loadPyodide( {
        // _loadSnapshot: await fetchSnapshot(),
        lockFileURL: 'https://cdn.jsdelivr.net/gh/mhochsteger/ngsolve_pyodide@webgpu2/pyodide-lock.json',
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/",
        }
      );
      pyodide.setDebug(true);
      await pyodide.loadPackage(['netgen', 'ngsolve', 'numpy', 'packaging']);
  }
  else {
      await webgpu_ready;
  }
  const webgpu_b64 = `"""
    + _package_b64
    + r"""`;
  const webpgu_zip = decodeB64(webgpu_b64);
  await pyodide.unpackArchive(webpgu_zip, 'zip');
  await pyodide.runPythonAsync('import webgpu.utils');
  await pyodide.runPythonAsync('await webgpu.utils.init_device()');
}
window.webgpu_ready = main();

"""
)


def _encode_data(data):
    binary_chunk = pickle.dumps(data)
    return base64.b64encode(binary_chunk).decode("utf-8")


def _decode_data(data):
    binary_chunk = base64.b64decode(data.encode("utf-8"))
    return pickle.loads(binary_chunk)


def _encode_function(func):
    import inspect

    return [func.__name__, inspect.getsource(func)]


def _decode_function(encoded_func):
    import __main__

    func_name, func_str = encoded_func
    symbols = __main__.__dict__
    exec(func_str, symbols, symbols)
    return symbols[func_name]


_render_canvases = {}


def _init(canvas_id="canvas"):
    import js
    from webgpu.gpu import init_webgpu

    print("init with canvas id", canvas_id)
    canvas = js.document.getElementById(canvas_id)
    print("canvas", canvas)

    gpu = init_webgpu(canvas)
    _render_canvases[canvas_id] = gpu
    gpu.update_uniforms()
    return gpu


def get_render_canvas(canvas_id):
    return _render_canvases[canvas_id]


def _draw_client(canvas_id, scene, assets, globs):
    import js
    import pyodide.ffi

    from webgpu.jupyter import _decode_data, _decode_function

    from pathlib import Path

    assets = _decode_data(assets)

    for module_data in assets.get("modules", {}).values():
        # extract zipfile from binary chunk
        import io
        import zipfile

        zipf = zipfile.ZipFile(io.BytesIO(module_data))
        zipf.extractall()

    for file_name, file_data in assets.get("files", {}).items():
        with open(file_name, "wb") as f:
            f.write(file_data)

    for module_name in assets.get("modules", {}):
        reload_package(module_name)

    if "redraw" in assets and assets["redraw"]:
        gpu = get_render_canvas(canvas_id)
    else:
        gpu = _init(canvas_id)

    if "init_function" in assets:
        func = _decode_function(assets["init_function"])
    else:
        func = globs[assets["init_function_name"]]
    func(gpu, _decode_data(scene))


_draw_js_code_template = r"""
async function draw() {{
    var canvas = document.createElement('canvas');
    var canvas_id = "{canvas_id}";
    canvas.id = canvas_id;
    canvas.width = {width};
    canvas.height = {height};
    canvas.style = "background-color: #d0d0d0";
    console.log("create canvas with id", canvas.id, canvas);
    console.log("got id", canvas_id);
    element.appendChild(canvas);
    await window.webgpu_ready;
    await window.pyodide.runPythonAsync('import webgpu.jupyter; webgpu.jupyter._draw_client("{canvas_id}", "{data}", globals())');
}}
draw();
    """

if not _is_pyodide:
    from IPython.core.magic import register_cell_magic
    from IPython.core.magics.display import Javascript, display

    display(Javascript(_init_js_code))

    _call_counter = 0

    def _get_canvas_id():
        global _call_counter
        _call_counter += 1
        return f"canvas_{_call_counter}"

    def _run_js_code(data, width, height):
        display(
            Javascript(
                _draw_js_code_template.format(
                    canvas_id=_get_canvas_id(),
                    data=_encode_data(data),
                    width=width,
                    height=height,
                )
            )
        )

    def Draw(cf, mesh, width=600, height=600):
        data = {"cf": cf, "mesh": mesh}
        _run_js_code(data, width=width, height=height)

    def DrawCustom(
        client_function,
        kwargs={},
        modules: list[str] = [],
        files: list[str] = [],
        width=600,
        height=600,
    ):
        data = {}
        data["kwargs"] = kwargs
        if isinstance(client_function, str):
            data["init_function_name"] = client_function
        else:
            data["init_function"] = _encode_function(client_function)
        data["modules"] = {module: create_package_zip(module) for module in modules}
        data["files"] = {f: open(f, "rb").read() for f in files}
        _run_js_code(data, width=width, height=height)

    def run_code_in_pyodide(code: str):
        display(
            Javascript(
                f"window.webgpu_ready.then(() => {{ window.pyodide.runPythonAsync(`{code}`) }});"
            )
        )

    @register_cell_magic
    def pyodide(line, cell):
        run_code_in_pyodide(str(cell))

    del pyodide

    class Pyodide:
        def __setattr__(self, key, value):
            data = _encode_data(value)
            display(
                Javascript(
                    f"window.webgpu_ready.then(() => {{ window.pyodide.runPythonAsync(`import webgpu.jupyter; {key} = webgpu.jupyter._decode_data('{data}')`) }});"
                )
            )

    pyodide = Pyodide()
