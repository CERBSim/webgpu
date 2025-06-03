from .canvas import Canvas
from .lilgui import LilGUI
from .renderer import BaseRenderer
from .scene import Scene
from .utils import max_bounding_box


def Draw(
    scene: Scene | BaseRenderer | list[BaseRenderer],
    canvas: Canvas,
    lilgui=True,
) -> Scene:
    import numpy as np

    if isinstance(scene, BaseRenderer):
        scene = Scene([scene])
    elif isinstance(scene, list):
        scene = Scene(scene)
    scene.init(canvas)
    if lilgui:
        scene.gui = LilGUI(canvas.canvas.id, scene._id)

    scene.render()

    return scene
