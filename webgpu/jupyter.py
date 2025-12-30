import base64
import itertools
import os
import pickle
import time

from . import platform, utils
from .canvas import Canvas
from .lilgui import LilGUI
from .link import js_code as _link_js_code
from .renderer import *
from .scene import Scene
from .triangles import *
from .utils import init_device_sync
from .webgpu_api import *


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
                    arcname = os.path.relpath(file_path, start=os.path.dirname(package_dir))
                    zipf.write(file_path, arcname)

        return open(output_filename, "rb").read()


_id_counter = itertools.count()


def _init_html(scene, width, height, flex=None):
    from IPython.display import HTML, display

    if isinstance(scene, Renderer):
        scene = [scene]
    if isinstance(scene, list):
        scene = Scene(scene)

    id_ = f"__webgpu_{next(_id_counter)}_"

    style = f"background-color: #d0d0d0; width: {width}px; height: {height}px;"
    if flex is not None:
        style += f" flex: {flex};"

    display(
        HTML(
            f"""
            <div id='{id_}root'
            style="display: flex; justify-content: space-between;"
            >
                <canvas 
                    id='{id_}canvas'
                    style='{style}'
                >
                </canvas>
                <div id='{id_}lilgui'
                    style='flex: 1;'

                ></div>
            </div>
            """
        )
    )

    return scene, id_


def _draw_scene(scene: Scene, width, height, id_):
    html_canvas = platform.js.document.getElementById(f"{id_}canvas")

    while html_canvas is None:
        html_canvas = platform.js.document.getElementById(f"{id_}canvas")
    html_canvas.width = width
    html_canvas.height = height
    gui_element = platform.js.document.getElementById(f"{id_}lilgui")

    # Lazily initialize the WebGPU device the first time we draw.
    canvas = Canvas(init_device_sync(), html_canvas)
    scene.gui = LilGUI(gui_element, scene)
    scene.init(canvas)
    scene.render()


def _DrawPyodide(b64_data: str):
    data = base64.b64decode(b64_data.encode("utf-8"))
    id_, scene, width, height = pickle.loads(data)

    _draw_scene(scene, width, height, id_)
    return scene


def _DrawHTML(
    scene: Scene | list[Renderer] | Renderer,
    width=640,
    height=640,
):
    """Draw a scene using display(Javascript()) with all information in the HTML
    This way, data is kept in the converted html when running nbconvert
    The scene object is unpickled and drawn within a pyodide instance in the browser when the html is opened
    """
    from IPython.display import Javascript, display

    scene, id_ = _init_html(scene, width, height)

    data = pickle.dumps((id_, scene, width, height))
    b64_data = base64.b64encode(data).decode("utf-8")

    display(Javascript(f"window.draw_scene('{b64_data}');"))
    return scene


def Draw(
    scene: Scene | list[Renderer] | Renderer,
    width: int | None = None,
    height: int | None = None,
):
    flex = 3 if width is None else None

    width = width if width is not None else 640
    height = height if height is not None else 640

    scene, id_ = _init_html(scene, width, height, flex)

    # In classic Jupyter we already have a websocket connection at import
    # time, so this callback runs immediately. In VS Code, outputs are only
    # processed once the cell has finished executing; using execute_when_init
    # ensures that drawing happens once the websocket connection is ready
    # instead of blocking the import.
    platform.execute_when_init(lambda js: _draw_scene(scene, width, height, id_))
    return scene


_js_init_pyodide = """
async function init_pyodide(webgpu_b64) {
  const pyodide_module = await import(
    "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.mjs"
  );
  window.pyodide = await pyodide_module.loadPyodide();
  pyodide.setDebug(true);
  await pyodide.loadPackage([
    "micropip",
    "numpy",
    "packaging",
  ]);
  const webpgu_zip = decodeB64(webgpu_b64);
  await pyodide.unpackArchive(webpgu_zip, "zip");
  await pyodide.runPythonAsync("import webgpu.utils");
  await pyodide.runPythonAsync("await webgpu.utils.init_device()");
}

window.draw_scene = async (data) => {
  await window.pyodide_ready;
  window.pyodide.runPython(`import webgpu.jupyter; webgpu.jupyter._DrawPyodide("${data}")`)
}
"""

if not platform.is_pyodide:
    from IPython.display import Javascript, display

    is_exporting = "WEBGPU_EXPORTING" in os.environ

    if is_exporting:
        Draw = _DrawHTML
        webgpu_module = create_package_zip("webgpu")
        webgpu_module_b64 = base64.b64encode(webgpu_module).decode("utf-8")
        js_code = _link_js_code
        js_code += _js_init_pyodide
        js_code += f"\nwindow.pyodide_ready = init_pyodide('{webgpu_module_b64}');"
        display(Javascript(js_code))
    else:
        # Not exporting and not running in pyodide -> Start a websocket server
        # and wait for the client to connect.
        #
        # In VS Code notebooks, outputs are typically only processed once the
        # cell has completed execution. If we were to block here waiting for
        # the websocket connection, the JavaScript that establishes the
        # connection would never run, leading to a deadlock. We therefore
        # avoid blocking on the connection in that environment and instead
        # defer drawing until the link is ready via execute_when_init.

        def _webgpu_js(server):
            js = _link_js_code + """
const __is_vscode = (typeof location !== 'undefined' && location.protocol === 'vscode-webview:');
const __webgpu_host = __is_vscode ? '127.0.0.1' : ((typeof location !== 'undefined' && location.hostname) || '127.0.0.1');
WebsocketLink('ws://' + __webgpu_host + ':{port}');
""".format(port=server.port)
            display(Javascript(js))

        is_vscode = "VSCODE_PID" in os.environ
        platform.init(
            before_wait_for_connection=_webgpu_js,
            block_on_connection=not is_vscode,
        )
