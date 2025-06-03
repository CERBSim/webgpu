import time
from threading import Timer

import numpy as np

from . import platform
from .canvas import Canvas
from .renderer import BaseRenderer, RenderOptions, SelectEvent
from .utils import is_pyodide, max_bounding_box, read_buffer, read_texture
from .webgpu_api import *
from .input_handler import InputHandler

_TARGET_FPS = 60


def debounce(render_function):
    if platform.is_pyodide:
        return render_function

    # Render only once every 1/_TARGET_FPS seconds
    def debounced(*args, **kwargs):
        if debounced.timer is not None:
            # we already have a render scheduled, so do nothing
            return

        def f():
            # clear the timer, so we can schedule a new one with the next function call
            t = time.time()
            render_function(*args, **kwargs)
            debounced.timer = None
            debounced.t_last = t

        t_wait = max(1 / _TARGET_FPS - (time.time() - debounced.t_last), 0)
        debounced.timer = Timer(t_wait, f)
        debounced.timer.start()

    debounced.timer = None
    debounced.t_last = time.time()
    return debounced


class Scene:
    canvas: Canvas = None
    render_objects: list[BaseRenderer]
    options: RenderOptions
    gui: object = None

    def __init__(
        self,
        render_objects: list[BaseRenderer],
        id: str | None = None,
        canvas: Canvas | None = None,
    ):
        if id is None:
            import uuid

            id = str(uuid.uuid4())
        import threading

        self.options = RenderOptions()
        self._render_mutex = threading.Lock()

        self._id = id
        self.render_objects = render_objects

        if is_pyodide:
            _scenes_by_id[id] = self
            if canvas is not None:
                self.init(canvas)

        self.t_last = 0
        objects = render_objects
        pmin, pmax = max_bounding_box([o.get_bounding_box() for o in objects])

        camera = self.options.camera
        camera.transform._center = 0.5 * (pmin + pmax)
        camera.transform._scale = 2 / np.linalg.norm(pmax - pmin)

        if not (pmin[2] == 0 and pmax[2] == 0):
            camera.transform.rotate(270, 0)
            camera.transform.rotate(0, -20)
            camera.transform.rotate(20, 0)
        # if not (pmin[2] == 0 and pmax[2] == 0):
        #     camera.transform.rotate(30, -20)
        camera._update_uniforms()
        self.input_handler = InputHandler()

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
        self.input_handler.set_canvas(canvas.canvas)
        self.options.set_canvas(canvas)

        self.options.timestamp = time.time()
        for obj in self.render_objects:
            obj._update_and_create_render_pipeline(self.options)

        camera = self.options.camera
        self._js_render = platform.create_proxy(self._render_direct)
        camera.register_callbacks(self.input_handler, self.render)
        self.options.update_buffers()
        if is_pyodide:
            _scenes_by_id[self.id] = self

        self._select_buffer = self.device.createBuffer(
            size=4 * 4,
            usage=BufferUsage.COPY_DST | BufferUsage.MAP_READ,
            label="select",
        )
        self._select_buffer_valid = False

        canvas.on_resize(self.render)

    @debounce
    def select(self, x: int, y: int):
        objects = self.render_objects

        have_select_callback = False
        for obj in objects:
            if obj.active and obj._select_pipeline and obj._on_select:
                have_select_callback = True
                break

        if not have_select_callback:
            return

        with self._render_mutex:
            select_texture = self.canvas.select_texture
            bytes_per_row = (select_texture.width * 16 + 255) // 256 * 256

            options = self.options
            options.command_encoder = self.device.createCommandEncoder()

            if not self._select_buffer_valid:
                for obj in objects:
                    if obj.active:
                        obj._update_and_create_render_pipeline(options)

                for obj in objects:
                    if obj.active:
                        obj.select(options, x, y)

                self._select_buffer_valid = True

            buffer = self._select_buffer
            options.command_encoder.copyTextureToBuffer(
                TexelCopyTextureInfo(select_texture, origin=Origin3d(x, y, 0)),
                TexelCopyBufferInfo(buffer, 0, bytes_per_row),
                [1, 1, 1],
            )

            self.device.queue.submit([options.command_encoder.finish()])
            options.command_encoder = None

            ev = SelectEvent(x, y, read_buffer(buffer))
            if ev.obj_id > 0:
                for obj in objects:
                    if obj._id == ev.obj_id:
                        obj._handle_on_select(ev)
                        break

            return ev

    def _render_objects(self, to_canvas=True):
        self._select_buffer_valid = False
        options = self.options
        for obj in self.render_objects:
            if obj.active:
                obj._update_and_create_render_pipeline(options)

        options.command_encoder = self.device.createCommandEncoder()
        for obj in self.render_objects:
            if obj.active:
                obj.render(options)

        if to_canvas:
            options.command_encoder.copyTextureToTexture(
                TexelCopyTextureInfo(self.canvas.target_texture),
                TexelCopyTextureInfo(self.canvas.context.getCurrentTexture()),
                [self.canvas.width, self.canvas.height, 1],
            )
        self.device.queue.submit([options.command_encoder.finish()])
        options.command_encoder = None

    def _redraw_blocking(self):
        with self._render_mutex:
            import time

            self.options.timestamp = time.time()
            for obj in self.render_objects:
                obj._update_and_create_render_pipeline(self.options)

            self.render()

    @debounce
    def _redraw_debounced(self):
        self._redraw_blocking()

    def redraw(self, blocking=False):
        if blocking:
            self._redraw_blocking()
        else:
            self._redraw_debounced()

    def _render(self):
        platform.js.requestAnimationFrame(self._js_render)

    def _render_direct(self, t=0):
        self._render_objects(to_canvas=True)

    @debounce
    def render(self, t=0):
        # self.canvas.resize()
        if self.canvas is None:
            return
        if is_pyodide:
            self._render()
            return
        with self._render_mutex:
            self._render_objects(to_canvas=False)

            if not is_pyodide:
                platform.js.patchedRequestAnimationFrame(
                    self.canvas.device.handle,
                    self.canvas.context,
                    self.canvas.target_texture,
                )

    def cleanup(self):
        with self._render_mutex:
            if self.canvas is not None:
                self.options.camera.unregister_callbacks(self.input_handler)
                self.options.camera._render_function = None
                self.input_handler.unregister_callbacks()
                platform.destroy_proxy(self._js_render)
                del self._js_render
                self.canvas._on_resize_callbacks.remove(self.render)
                self.canvas = None

                if is_pyodide:
                    del _scenes_by_id[self.id]


if is_pyodide:
    _scenes_by_id: dict[str, Scene] = {}

    def get_scene(id: str) -> Scene:
        return _scenes_by_id[id]

    def redraw_scene(id: str):
        get_scene(id).redraw()
