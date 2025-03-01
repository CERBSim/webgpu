from .render_object import BaseRenderObject
from .utils import max_bounding_box
from .scene import Scene
from .canvas import Canvas
from .lilgui import LilGUI


def Draw(
    scene: Scene | BaseRenderObject | list[BaseRenderObject],
    canvas: Canvas, lilgui=True
) -> Scene:
    import js
    import numpy as np

    if isinstance(scene, BaseRenderObject):
        scene = Scene([scene])
    elif isinstance(scene, list):
        scene = Scene(scene)
    scene.init(canvas)
    if lilgui:
        scene.gui = LilGUI(canvas.canvas.id, scene._id)

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
