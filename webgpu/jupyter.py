import itertools
import time
from pathlib import Path

from IPython.display import HTML, Javascript, display

from . import proxy, utils
from .canvas import Canvas
from .lilgui import LilGUI
from .render_object import *
from .scene import Scene
from .triangles import *
from .webgpu_api import *

time.sleep(0.1)

proxy.remote = proxy.JsRemote()

js_code = (
    Path("../webgpu/proxy.js")
    .read_text()
    .replace("WEBSOCKET_PORT", str(proxy.remote._websocket_port))
)

display(Javascript(js_code))

while not proxy.remote._connected_clients:
    time.sleep(1 / 60)

_id = itertools.count()


def init_device():
    if not js.navigator.gpu:
        js.alert("WebGPU is not supported")
        sys.exit(1)

    reqAdapter = js.navigator.gpu.requestAdapter
    options = RequestAdapterOptions(
        powerPreference=PowerPreference.high_performance
    ).toJS()
    adapter = reqAdapter(options)
    if not adapter:
        js.alert("WebGPU is not supported")
        sys.exit(1)
    one_gig = 1024**3
    utils._device = Device(
        adapter.requestDevice(
            [],
            Limits(
                maxBufferSize=one_gig - 16,
                maxStorageBufferBindingSize=one_gig - 16,
            ),
            None,
            "WebGPU device",
        )
    )
    return utils._device


device = init_device()
js.console.log("have device", device.handle)


def Draw(
    scene: Scene | list[RenderObject] | RenderObject,
    width=640,
    height=640,
):
    if isinstance(scene, RenderObject):
        scene = [scene]
    if isinstance(scene, list):
        scene = Scene(scene)

    id_ = next(_id)
    root_id = f"__webgpu_root_{id_}"
    canvas_id = f"__webgpu_canvas_{id_}"
    lilgui_id = f"__webgpu_lilgui_{id_}"

    display(
        HTML(
            f"""
            <div id='{root_id}'
            style="display: flex; justify-content: space-between;"
            >
                <canvas 
                    id='{canvas_id}'
                    style='background-color: #d0d0d0; flex: 3; width: {width}px; height: {height}px;'
                >
                </canvas>
                <div id='{lilgui_id}'
                    style='flex: 1;'

                ></div>
            </div>
            """
        )
    )
    display(
        Javascript(
            f"""window.lil_guis['{lilgui_id}'] = new lil.GUI({{
                container: document.getElementById('{lilgui_id}')
            }});"""
        )
    )
    html_canvas = js.document.getElementById(canvas_id)
    # proxy.remote.on_canvas_resize(html_canvas)

    canvas = Canvas(device, html_canvas)
    scene.gui = LilGUI(lilgui_id, scene)
    scene.init(canvas)
    scene.render()

    return scene
