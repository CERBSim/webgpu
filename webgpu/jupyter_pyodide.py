import base64
import pickle

from .draw import Draw as DrawPyodide
from .lilgui import LilGUI
from .render_object import RenderObject, _render_objects
from .scene import Scene
from .utils import _is_pyodide, reload_package


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

function initLilGUI() {
    // In generated html files, requirejs is imported before lil-gui is loaded.
    // Thus, we must load lil-gui using require, use import otherwise.
    const lil_url = "https://cdn.jsdelivr.net/npm/lil-gui@0.20";
    if(window.define === undefined){
        import(lil_url);
    } else {
        require([lil_url], (module) => {
            window.lil = module;
        });
    }
}

async function main() {
  if(window.webgpu_ready === undefined) {
      initLilGUI();
      const pyodide_module = await import("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.mjs");
      window.pyodide = await pyodide_module.loadPyodide( {
        // _loadSnapshot: await fetchSnapshot(),
        // lockFileURL: 'https://cdn.jsdelivr.net/gh/mhochsteger/ngsolve_pyodide@webgpu2/pyodide-lock.json',
        // indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/",
        }
      );
      pyodide.setDebug(true);
      window.pyodide_ready = pyodide.loadPackage(['micropip', 'numpy', 'packaging']);
      await window.pyodide_ready;
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


def _init(canvas_id="canvas"):
    import js

    from webgpu.canvas import init_webgpu

    return init_webgpu(js.document.getElementById(canvas_id))


def get_render_canvas(canvas_id):
    return _render_canvases[canvas_id]


def _draw_client(canvas_id, scene, assets, globs):
    from pathlib import Path

    import js
    import pyodide.ffi

    from webgpu.jupyter import _decode_data, _decode_function

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

    canvas = _init(canvas_id)
    scene = _decode_data(scene)

    if "init_function" in assets:
        func = _decode_function(assets["init_function"])
        func(canvas, **scene)
    elif "init_function_name" in assets:
        func = globs[assets["init_function_name"]]
        func(canvas, **scene)
    else:
        scene.init(canvas)
        DrawPyodide(scene, canvas)


_draw_js_code_template = r"""
async function draw() {{
    var canvas = document.createElement('canvas');
    var canvas_id = "{canvas_id}";
    canvas.id = canvas_id;
    canvas.width = {width};
    canvas.height = {height};
    canvas.style = "background-color: #d0d0d0";
    element.appendChild(canvas);
    await window.webgpu_ready;
    await window.pyodide.runPythonAsync('import webgpu.jupyter; webgpu.jupyter._draw_client("{canvas_id}", "{data}", "{assets}", globals())');
}}
draw();
    """

if not _is_pyodide:
    from IPython.core.magic import register_cell_magic
    from IPython.display import HTML, Javascript, display

    display(Javascript(_init_js_code))

    _call_counter = 0

    def _get_canvas_id():
        global _call_counter
        _call_counter += 1
        return f"canvas_{_call_counter}"

    def _run_js_code(data, assets, width, height):
        display(
            Javascript(
                _draw_js_code_template.format(
                    canvas_id=_get_canvas_id(),
                    data=_encode_data(data),
                    assets=_encode_data(assets),
                    width=width,
                    height=height,
                )
            )
        )

    html_code = r"""
<div id="{canvas_id}_row" style="display: flex; justify-content: space-between;">
    <canvas id="{canvas_id}" style="flex: 3; margin-right: 10px; padding: 10px; height: {height}px; width: {width}px; background-color: #d0d0d0;"></canvas>
    <div id="{canvas_id}_gui" style="flex: 1; margin-left: 10px; padding: 10px;"></div>
</div>
"""
    js_code = r"""
async function draw() {{
    await window.webgpu_ready;
    var gui_element = document.getElementById('{canvas_id}' + '_gui');
    console.log('gui_element =', gui_element);
    if(window.lil_guis === undefined) {{
      window.lil_guis = new Object();
    }}
    window.lil_guis['{canvas_id}'] = new lil.GUI({{container: gui_element}});
    // var canvas2 = document.createElement('canvas');
    // console.log("canvas2 =", canvas2);
    var canvas = document.getElementById("{canvas_id}");
    console.log('canvas size', canvas.clientWidth, canvas.clientHeight);
    console.log(canvas);
    canvas.width = Math.floor(canvas.clientWidth/32)*32;
    canvas.height = Math.floor(canvas.clientHeight/32)*32;
    canvas.style = "background-color: #d0d0d0; max-width: {width}px; max-height: {height}px;";
    await window.pyodide.runPythonAsync('import webgpu.jupyter; webgpu.jupyter._draw_client("{canvas_id}", "{scene}", "{assets}", globals())');
}}
draw();
    """

    def Draw(
        scene: Scene | list[RenderObject] | RenderObject,
        width=608,
        height=608,
        modules=[],
    ):
        if isinstance(scene, RenderObject):
            scene = [scene]
        if isinstance(scene, list):
            scene = Scene(scene)
        canvas_id = _get_canvas_id()
        scene.gui = LilGUI(canvas_id, scene._id)
        assets = {"modules": {module: create_package_zip(module) for module in modules}}
        display(
            HTML(html_code.format(canvas_id=canvas_id, width=width, height=height)),
            Javascript(
                js_code.format(
                    canvas_id=canvas_id,
                    scene=_encode_data(scene),
                    assets=_encode_data(assets),
                    width=width,
                    height=height,
                )
            ),
        )

        return scene

    def DrawCustom(
        client_function,
        kwargs={},
        modules: list[str] = [],
        files: list[str] = [],
        width=608,
        height=608,
    ):
        assets = {
            "modules": {module: create_package_zip(module) for module in modules},
            "files": {f: open(f, "rb").read() for f in files},
        }
        if isinstance(client_function, str):
            assets["init_function_name"] = client_function
        else:
            assets["init_function"] = _encode_function(client_function)
        canvas_id = _get_canvas_id()
        display(
            HTML(html_code.format(canvas_id=canvas_id, height=height, width=width)),
            Javascript(
                js_code.format(
                    canvas_id=canvas_id,
                    scene=_encode_data(kwargs),
                    assets=_encode_data(assets),
                    width=width,
                    height=height,
                )
            ),
        )

    def run_code_in_pyodide(code: str):
        display(
            Javascript(
                f"window.webgpu_ready.then(() => {{ window.pyodide.runPythonAsync(`{code}`) }});"
            )
        )

    @register_cell_magic
    def pyodide(line, cell):
        run_code_in_pyodide(str(cell))

    @register_cell_magic
    def pyodide_and_kernel(line, cell):
        run_code_in_pyodide(str(cell))
        ip = get_ipython()
        exec(cell, ip.user_global_ns)

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


def pyodide_install_packages(packages):
    if not _is_pyodide:
        display(
            Javascript(
                f"window.webgpu_ready = window.webgpu_ready.then(() => {{ return window.pyodide.loadPackage({packages}); }})"
            )
        )


def update_render_object(id, **kwargs):
    if _is_pyodide:
        obj = _render_objects[id]
        obj.update(**kwargs)
    else:
        kwargs = _encode_data(kwargs)
        run_code_in_pyodide(
            f"import webgpu.jupyter; webgpu.jupyter.update_render_object('{id}', **webgpu.jupyter._decode_data('{kwargs}'));"
        )


def redraw_canvas(canvas_id: str):
    run_code_in_pyodide(
        f"import webgpu.draw, js; js.requestAnimationFrame(webgpu.draw._canvas_id_to_gpu['{canvas_id}'].input_handler.render_function)"
    )
