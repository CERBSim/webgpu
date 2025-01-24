from .render_object import BaseRenderObject
from .utils import max_bounding_box
from .scene import Scene
from .canvas import Canvas


def Draw(
    scene: Scene | BaseRenderObject | list[BaseRenderObject],
    canvas: Canvas | None = None,
) -> Scene:
    import js
    import numpy as np

    if isinstance(scene, BaseRenderObject):
        scene = Scene([scene])
    elif isinstance(scene, list):
        scene = Scene(scene)

    if canvas is not None:
        scene.init(canvas)

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
