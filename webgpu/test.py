import asyncio
import time

from . import proxy
from .canvas import Canvas

# from .draw import Draw
from .render_object import *
from .triangles import *
from .utils import init_device
from .webgpu_api import *

while not proxy.remote._connected_clients:
    time.sleep(0.1)

loop = asyncio.new_event_loop()
device = loop.run_until_complete(init_device())

html_canvas = js.document.createElement("canvas")
html_canvas.id = 1
html_canvas.width = 500
html_canvas.height = 500
html_canvas.style = "background-color: #d0d0d0"
div_root = js.document.getElementById("root")
div_root.appendChild(html_canvas)

canvas = Canvas(device, html_canvas)

triangle = TriangulationRenderer([0, 0, 0, 1, 0, 0, 0, 1, 0])

# Draw(triangle, canvas, lilgui = False)

def render(t):
    global options
    encoder = device.createCommandEncoder()
    triangle.render(encoder)
    device.queue.submit([encoder.finish()])
    js.patchedRequestAnimationFrame(
        device.handle._id, canvas.context._id, canvas.target_texture._id
    )


options = RenderOptions(canvas, render)

triangle.options = options
triangle.options.update_buffers()
triangle.update()

pmin, pmax = triangle.get_bounding_box()

print("pmin", pmin)
print("pmax", pmax)
camera = options.camera
camera.transform._center = 0.5 * (pmin + pmax)
camera.transform._scale = 2 / np.linalg.norm(pmax - pmin)

print(camera.transform.mat )
print("center", camera.transform._center)
print("scale", camera.transform._scale)


if not (pmin[2] == 0 and pmax[2] == 0):
    camera.transform.rotate(30, -20)
camera._update_uniforms()

render_proxy = create_proxy(
    render,
)

# js.requestAnimationFrame(render_proxy)
render(0)

while True:
    time.sleep(1)
    camera.transform.rotate(0, 10)
    triangle.options.update_buffers()
    render(0)
#     js.requestAnimationFrame(render_proxy)
