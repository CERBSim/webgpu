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

def render():
    encoder = device.createCommandEncoder()
    triangle.render(encoder)
    device.queue.submit([encoder.finish()])

options = RenderOptions(canvas, render)

triangle.options = options
triangle.update()

js.requestAnimationFrame(toJS(render))
