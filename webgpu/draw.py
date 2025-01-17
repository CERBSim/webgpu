from .render_object import RenderObject
from .utils import Scene, max_bounding_box

_canvas_id_to_gpu = {}


def Draw(scene: Scene | RenderObject | list[RenderObject]):
    import js
    import numpy as np
    import pyodide.ffi

    if isinstance(scene, RenderObject):
        scene = [scene]
    if isinstance(scene, list):
        scene = Scene(scene)

    gpu = scene.gpu
    _canvas_id_to_gpu[gpu.canvas.id] = gpu

    objects = scene.render_objects
    gpu = objects[0].gpu
    pmin, pmax = max_bounding_box([o.get_bounding_box() for o in objects])
    gpu.input_handler.transform._center = 0.5 * (pmin + pmax)
    gpu.input_handler.transform._scale = 2 / np.linalg.norm(pmax - pmin)
    if not (pmin[2] == 0 and pmax[2] == 0):
        gpu.input_handler.transform.rotate(30, -20)
    gpu.input_handler._update_uniforms()

    def render_function(t):
        gpu.update_uniforms()
        encoder = gpu.device.createCommandEncoder()
        for obj in objects:
            obj.render(encoder)
        gpu.device.queue.submit([encoder.finish()])

    render_function = pyodide.ffi.create_proxy(render_function)
    gpu.input_handler.render_function = render_function
    js.requestAnimationFrame(render_function)
    scene.render_function = render_function
    return scene
