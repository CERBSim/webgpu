from .render_object import RenderObject
from .utils import max_bounding_box
from .scene import Scene


def Draw(scene: Scene):
    import js
    import numpy as np

    objects = scene.render_objects
    pmin, pmax = max_bounding_box([o.get_bounding_box() for o in objects])

    camera = scene.options.camera
    camera.transform._center = 0.5 * (pmin + pmax)
    camera.transform._scale = 2 / np.linalg.norm(pmax - pmin)

    if not (pmin[2] == 0 and pmax[2] == 0):
        camera.transform.rotate(30, -20)
    camera._update_uniforms()
    js.requestAnimationFrame(scene._js_render)

    return scene
