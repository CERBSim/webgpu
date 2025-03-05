import time
import asyncio

from . import proxy
from .webgpu_api import *
from .utils import init_device
from .canvas import Canvas

from .triangles import *
# from .draw import Draw
from .render_object import *

while not proxy.remote._connected_clients:
    time.sleep(0.1)

loop = asyncio.new_event_loop()
device = loop.run_until_complete(init_device())

html_canvas = js.document.createElement('canvas');
html_canvas.id = 1
html_canvas.width = 500
html_canvas.height = 500
html_canvas.style = "background-color: #d0d0d0";
div_root = js.document.getElementById("root")
div_root.appendChild(html_canvas);

canvas = Canvas(device, html_canvas)

triangle = TriangulationRenderer([0,0,0, 1,0,0, 0,1,0])

# Draw(triangle, canvas, lilgui = False)

def render(*args, **kwargs):
    encoder = device.createCommandEncoder()
    triangle.render(encoder)
    device.queue.submit([encoder.finish()])

options = RenderOptions(canvas, render)

triangle.options = options
triangle.options.update_buffers()
triangle.update()

pmin, pmax = triangle.get_bounding_box()

camera = options.camera
camera.transform._center = 0.5 * (pmin + pmax)
camera.transform._scale = 2 / np.linalg.norm(pmax - pmin)

if not (pmin[2] == 0 and pmax[2] == 0):
    camera.transform.rotate(30, -20)
camera._update_uniforms()

def f(*args, **kwargs):
    pass

render_proxy = create_proxy(render)
js.requestAnimationFrame(render_proxy)

while True:
    time.sleep(1)
    js.requestAnimationFrame(render_proxy)
