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

_PYODIDE_VERSION = "0.27.6"


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


def _init_html(scene, width, height):
    from IPython.display import HTML, display

    if isinstance(scene, Renderer):
        scene = [scene]
    if isinstance(scene, list):
        scene = Scene(scene)

    id_ = f"__webgpu_{next(_id_counter)}_"

    display(
        HTML(
            f"""
            <div id='{id_}root'
            style="display: flex; justify-content: space-between;"
            >
                <canvas 
                    id='{id_}canvas'
                    style='background-color: #d0d0d0; flex: 3; width: {width}px; height: {height}px;'
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

    canvas = Canvas(utils.get_device(), html_canvas)
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
    print("draw html")
    from IPython.display import Javascript, display

    scene, id_ = _init_html(scene, width, height)

    data = pickle.dumps((id_, scene, width, height))
    b64_data = base64.b64encode(data).decode("utf-8")

    display(Javascript(f"window.draw_scene('{b64_data}');"))
    return scene


def Draw(
    scene: Scene | list[Renderer] | Renderer,
    width=640,
    height=640,
):
    scene, id_ = _init_html(scene, width, height)
    _draw_scene(scene, width, height, id_)
    return scene


_js_init_pyodide = """
async function init_pyodide(webgpu_b64) {
  const pyodide_module = await import(
    "https://cdn.jsdelivr.net/pyodide/v{PYODIDE_VERSION}/full/pyodide.mjs"
  );
  window.pyodide = await pyodide_module.loadPyodide(
      {lockFileURL: "https://cdn.jsdelivr.net/gh/mhochsteger/ngsolve_pyodide@{PYODIDE_VERSION}/pyodide-lock.json"}
  );
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
  console.log("draw scene, wati for pyoidde");
  await window.pyodide_ready;
  console.log("draw scene, have pyoidde");
  window.pyodide.runPythonAsync(`import webgpu.jupyter; webgpu.jupyter._DrawPyodide("${data}")`)
  console.log("draw scene, done");
}
""".replace(
    "{PYODIDE_VERSION}", _PYODIDE_VERSION
)

is_exporting = False

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
        # Not exporting and not running in pyodide -> Start a websocket server and wait for the client to connect
        platform.init(
            before_wait_for_connection=lambda server: display(
                Javascript(_link_js_code + f"WebsocketLink('ws://localhost:{server.port}');")
            )
        )
        device = init_device_sync()

_code_counter = 0


def add_init_js_code(js_code: str):
    global _code_counter
    if not is_exporting:
        return

    from IPython.display import Javascript, display

    display(
        Javascript(
            """
window._webgpu_ready_{{COUNTER}} = window.pyodide_ready;
window.pyodide_ready = async function() {
    await window._webgpu_ready_{{COUNTER}};
    {{CODE}} 
}();
""".replace(
                "{{COUNTER}}", str(_code_counter)
            ).replace(
                "{{CODE}}", js_code
            )
        )
    )


def add_zipped_module_on_export(module_name: str):
    print("add module on export", module_name)
    global _module_counter
    if not is_exporting:
        return

    module_code = create_package_zip(module_name)
    module_b64 = base64.b64encode(module_code).decode("utf-8")
    code = f'await pyodide.unpackArchive(decodeB64("{module_b64}"), "zip");'
    add_init_js_code(code)


def add_imports_on_export(modules: list[str]):
    """Add imports to the pyodide_ready function on export"""
    if not is_exporting:
        return

    add_init_js_code(
        f'await pyodide.runPythonAsync("import micropip; await micropip.install({modules})")'
    )


def install_wheels_on_export(urls: list[str]):
    """Add imports to the pyodide_ready function on export"""
    if not is_exporting:
        return

    code = ""
    for url in urls:
        file_name = url.split("/")[-1]
        code += f"""
        {{
          const response = await fetch("{url}");
          const blob = await response.blob();
          await window.pyodide._api.install(await blob.arrayBuffer(), "{file_name}", 'site');
        }}
"""
    add_init_js_code(code)
