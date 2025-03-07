import time
from pathlib import Path

from IPython.display import Javascript, display

from . import proxy, utils
from .canvas import Canvas
from .render_object import *
from .scene import Scene
from .triangles import *
from .webgpu_api import *

time.sleep(0.1)
js_code = Path("../webgpu/proxy.js").read_text()

display(Javascript(js_code))

while not proxy.remote._connected_clients:
    time.sleep(0.1)


def init_device():
    if not js.navigator.gpu:
        js.alert("WebGPU is not supported")
        sys.exit(1)

    reqAdapter = js.navigator.gpu.requestAdapter
    options = RequestAdapterOptions(powerPreference=PowerPreference.high_performance).toJS()
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

# ori_loop = asyncio.get_running_loop()
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)
# device = loop.run_until_complete(init_device())
# asyncio.set_event_loop(ori_loop)
# print('device:', device)
# # device = await init_device()


def Draw(
    scene: Scene | list[RenderObject] | RenderObject,
    width=640,
    height=640,
):
    if isinstance(scene, RenderObject):
        scene = [scene]
    if isinstance(scene, list):
        scene = Scene(scene)

    # scene.gui = LilGUI(canvas_id, scene._id)

    display(Javascript("window._webgpu_element = element;"))
    html_canvas = js.document.createElement("canvas")
    html_canvas.id = 1
    html_canvas.width = width
    html_canvas.height = height
    html_canvas.style = "background-color: #d0d0d0"
    # div_root = js.document.getElementById("root")
    div_root = js._webgpu_element
    div_root.appendChild(html_canvas)
    proxy.remote.on_canvas_resize(html_canvas)

    canvas = Canvas(device, html_canvas)
    scene.init(canvas)
    scene.render()

    return scene
