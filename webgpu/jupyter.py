import base64
import pickle
from .utils import reload_package

try:
    import js

    _is_pyodide = True
except:
    _is_pyodide = False

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
  pyodide.runPython("import glob; print(glob.glob('**', recursive=True))");
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

    return inspect.getsource(func)


def _decode_function(func_str):
    d = {}
    exec(func_str, d, d)
    func_name = sorted(d.keys())[-1]
    return d[func_name]


def _draw_client(data):
    import js
    import pyodide.ffi

    from webgpu.jupyter import _decode_data, _decode_function, gpu

    data = _decode_data(data)

    from pathlib import Path

    for module_data in data.get("modules", {}).values():
        # extract zipfile from binary chunk
        import zipfile
        import io

        zipf = zipfile.ZipFile(io.BytesIO(module_data))
        zipf.extractall()

    for module_name in data.get("modules", {}):
        reload_package(module_name)

    if "_init_function" in data:
        func = _decode_function(data["_init_function"])
        func(data)
    else:
        # from ngsolve_webgpu import *
        import ngsolve as ngs

        mesh = data["mesh"]
        cf = data["cf"]
        order = data.get("order", 1)

        mesh_data = MeshData(mesh, cf, order)
        mesh_object = MeshRenderObject(gpu, mesh_data)

        def render_function(t):
            gpu.update_uniforms()

            encoder = gpu.device.createCommandEncoder()
            mesh_object.render(encoder)
            gpu.device.queue.submit([encoder.finish()])

        render_function = pyodide.ffi.create_proxy(render_function)
        gpu.input_handler.render_function = render_function
        js.requestAnimationFrame(render_function)


gpu = None


async def _init(canvas_id="canvas"):
    global gpu
    from webgpu.gpu import init_webgpu

    print("init with canvas id", canvas_id)
    canvas = js.document.getElementById(canvas_id)
    print("canvas", canvas)

    gpu = await init_webgpu(canvas)
    gpu.update_uniforms()


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
    await window.pyodide.runPythonAsync('import webgpu.jupyter; await webgpu.jupyter._init("{canvas_id}")');
    const data_string = "{data}";
    window.pyodide.runPython("import webgpu.jupyter; webgpu.jupyter._draw_client")(data_string);
}}
draw();
    """

if not _is_pyodide:
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
        data, client_function, modules: list[str] = [], width=600, height=600
    ):
        data["_init_function"] = _encode_function(client_function)
        data["modules"] = {module: create_package_zip(module) for module in modules}
        _run_js_code(data, width=width, height=height)
