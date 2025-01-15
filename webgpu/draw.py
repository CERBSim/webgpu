
from .render_object import RenderObject

def max_bounding_box(boxes):
    import numpy as np
    pmin = np.array(boxes[0][0])
    pmax = np.array(boxes[0][1])
    for b in boxes[1:]:
        pmin = np.min(pmin, b[0])
        pmax = np.max(pmax, b[1])
    return (pmin, pmax)

def Draw(objects: list[RenderObject] | RenderObject):
    import numpy as np
    import js
    import pyodide.ffi
    if isinstance(objects, RenderObject):
        objects = [objects]
    if len(objects) == 0:
        return
    gpu = objects[0].gpu
    pmin, pmax = max_bounding_box([o.get_bounding_box() for o in objects])
    gpu.input_handler.transform._center = 0.5 * (pmin + pmax)
    gpu.input_handler.transform._scale = 2/np.linalg.norm(pmax - pmin)
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

