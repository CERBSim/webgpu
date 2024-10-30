import base64
import marshal
import pickle

try:
    import js

    _is_pyodide = True
except:
    _is_pyodide = False

import base64


def create_package_zip():
    """
    Creates a zip file containing all files in the specified Python package.

    Parameters:
    - package_name (str): Name of the Python package.
    - output_filename (str): Name of the output zip file.
    """
    import importlib.util
    import os
    import tempfile
    import zipfile

    spec = importlib.util.find_spec("webgpu")
    if spec is None or spec.origin is None:
        raise ValueError(f"Package webgpu not found.")

    package_dir = os.path.dirname(spec.origin)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_filename = os.path.join(temp_dir, "webgpu.zip")
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
function decodeB64(base64String) {
    const binaryString = atob(base64String);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

async function main() {
  if(window.webgpu_ready === undefined) {
      const pyodide_module = await import("../../files/pyodide/pyodide.mjs");
      window.pyodide = await pyodide_module.loadPyodide();
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
    return base64.b64encode(marshal.dumps(func.__code__)).decode("utf-8")


def _decode_function(func_str):
    import types

    code = marshal.loads(base64.b64decode(func_str.encode("utf-8")))
    return types.FunctionType(code, globals(), "_decoded_function")


def _draw_client(data):
    import js
    import pyodide.ffi

    import webgpu.mesh
    from webgpu.jupyter import _decode_data, _decode_function, gpu

    data = _decode_data(data)
    if "_init_function" in data:
        print("have init function")
        func = _decode_function(data["_init_function"])
        func(data)
    else:
        import ngsolve as ngs

        mesh = data["mesh"]
        cf = data["cf"]
        order = data.get("order", 1)

        mesh_data = webgpu.mesh.MeshData(mesh, cf, order)
        mesh_object = webgpu.mesh.MeshRenderObject(gpu, mesh_data)

        def render_function(t):
            gpu.uniforms.update_buffer()

            encoder = gpu.create_command_encoder()
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

    for i in [0, 5, 10]:
        gpu.uniforms.mat[i] = 1.8

    gpu.uniforms.mat[15] = 1.0

    gpu.uniforms.mat[12] = -0.5 * 1.8
    gpu.uniforms.mat[13] = -0.5 * 1.8


_draw_js_code_template = r"""
async function draw() {{
    var canvas = document.createElement('canvas');
    var canvas_id = "{canvas_id}";
    canvas.id = canvas_id;
    canvas.width = 400;
    canvas.height = 400;
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

    def _run_js_code(data):
        display(
            Javascript(
                _draw_js_code_template.format(
                    canvas_id=_get_canvas_id(), data=_encode_data(data)
                )
            )
        )

    def Draw(cf, mesh):
        data = {"cf": cf, "mesh": mesh}
        _run_js_code(data)

    def DrawCustom(data, client_function):
        data["_init_function"] = _encode_function(client_function)
        _run_js_code(data)
