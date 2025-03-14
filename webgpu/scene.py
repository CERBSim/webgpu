from .canvas import Canvas
from . import proxy
from .render_object import BaseRenderObject, RenderOptions
from .utils import _is_pyodide, max_bounding_box
from .webgpu_api import *
import math


class Scene:
    canvas: Canvas = None
    render_objects: list[BaseRenderObject]
    options: RenderOptions
    gui: object = None

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
        self.canvas = canvas
        self.options = RenderOptions(self.canvas, self.render)

        for obj in self.render_objects:
            obj.options = self.options
            obj.update()

        pmin, pmax = max_bounding_box([o.get_bounding_box() for o in self.render_objects])
        camera = self.options.camera
        camera.transform._center = 0.5 * (pmin + pmax)
        def norm(v):
            return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        camera.transform._scale = 2 / norm(pmax - pmin)
        if not (pmin[2] == 0 and pmax[2] == 0):
            camera.transform.rotate(270, 0)
            camera.transform.rotate(0,-20)
            camera.transform.rotate(20, 0)


        self._js_render = proxy.create_proxy(self._render_direct)
        camera.register_callbacks(canvas.input_handler, self.render)
        self.options.update_buffers()
        if _is_pyodide:
            _scenes_by_id[self.id] = self

    def redraw(self):
        import time

        ts = time.time()
        for obj in self.render_objects:
            obj.redraw(timestamp=ts)

        self.render()
        return

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
        proxy.js.requestAnimationFrame(self._js_render)

    def _render_direct(self, t=0):
        encoder = self.device.createCommandEncoder()

        for obj in self.render_objects:
            obj.render(encoder)

        encoder.copyTextureToTexture(
            TexelCopyTextureInfo(self.canvas.target_texture),
            TexelCopyTextureInfo(self.canvas.context.getCurrentTexture()),
            [self.canvas.canvas.width, self.canvas.canvas.height, 1],
        )
        self.device.queue.submit([encoder.finish()])

    def render(self, t=0):
        if _is_pyodide:
            self._render()
            return
        # print("render")
        # print("canvas", self.canvas.canvas)
        # from . import proxy
        # proxy.js.console.log("canvas", self.canvas.canvas)
        # print("canvas size ", self.canvas.canvas.width, self.canvas.canvas.height)
        # print(
        #     "texture size",
        #     self.canvas.target_texture.width,
        #     self.canvas.target_texture.height,
        # )
        encoder = self.device.createCommandEncoder()

        for obj in self.render_objects:
            obj.render(encoder)

        self.device.queue.submit([encoder.finish()])

        if not _is_pyodide:
            proxy.js.patchedRequestAnimationFrame(
                self.canvas.device.handle._id,
                self.canvas.context._id,
                self.canvas.target_texture._id,
            )


if _is_pyodide:
    _scenes_by_id: dict[str, Scene] = {}

    def get_scene(id: str) -> Scene:
        return _scenes_by_id[id]

    def redraw_scene(id: str):
        get_scene(id).redraw()
