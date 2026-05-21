from .canvas import Canvas
from .renderer import BaseRenderer
from .scene import Scene
from .utils import max_bounding_box


def Draw(
    scene: Scene | BaseRenderer | list[BaseRenderer],
    canvas: Canvas,
) -> Scene:
    import numpy as np

    if isinstance(scene, BaseRenderer):
        scene = Scene([scene])
    elif isinstance(scene, list):
        scene = Scene(scene)
    scene.init(canvas)
    scene.render()

    return scene
