from .canvas import Canvas
from .render_object import BaseRenderObject, RenderOptions
from .utils import _is_pyodide
from .webgpu_api import *


class Scene:
    canvas: Canvas = None
    render_objects: list[BaseRenderObject]
    options: RenderOptions

    def __init__(
        self,
        render_objects: list[BaseRenderObject],
        id: str | None = None,
        canvas: Canvas | None = None,
    ):
        if id is None:
            import uuid

            id = str(uuid.uuid4())

        self._id = id
        self.render_objects = render_objects

        if _is_pyodide:
            _scenes_by_id[id] = self
            if canvas is not None:
                self.init(canvas)

    def __repr__(self):
        return ""

    @property
    def id(self) -> str:
        return self._id

    @property
    def device(self) -> Device:
        return self.canvas.device

    def init(self, canvas):
        import pyodide.ffi

        self.canvas = canvas
        self.options = RenderOptions(self.canvas, self.render)

        for obj in self.render_objects:
            obj.options = self.options
            obj.update()

        self._js_render = pyodide.ffi.create_proxy(self.render)
        self.options.camera.register_callbacks(canvas.input_handler, self._render)
        self.options.update_buffers()
        _scenes_by_id[self.id] = self

    def redraw(self):
        import time

        if _is_pyodide:
            import js

            js.requestAnimationFrame(self._js_render)
        else:
            # TODO: check if we are in a jupyter kernel
            from .jupyter import run_code_in_pyodide

            ts = time.time()
            for obj in self.render_objects:
                obj.redraw(timestamp=ts)

            run_code_in_pyodide(
                f"import webgpu.scene; webgpu.scene.redraw_scene('{self.id}')"
            )

    def _render(self):
        import js

        js.requestAnimationFrame(self._js_render)

    def render(self, t=0):
        encoder = self.device.createCommandEncoder()
        for obj in self.render_objects:
            obj.render(encoder)
        current = self.canvas.context.getCurrentTexture()
        target = self.canvas.target_texture
        encoder.copyTextureToTexture(
            TexelCopyTextureInfo(target),
            TexelCopyTextureInfo(current),
            [current.width, current.height, 1],
        )
        self.device.queue.submit([encoder.finish()])


if _is_pyodide:
    _scenes_by_id: dict[str, Scene] = {}

    def get_scene(id: str) -> Scene:
        return _scenes_by_id[id]

    def redraw_scene(id: str):
        get_scene(id).redraw()
