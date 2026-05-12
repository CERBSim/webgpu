import time

from . import platform
from .canvas import Canvas, debounce
from .input_handler import InputHandler
from .renderer import BaseRenderer, RenderOptions, SelectEvent
from .utils import max_bounding_box, read_buffer, Lock, print_communications
from .platform import is_pyodide, is_pyodide_main_thread
from .webgpu_api import *
from .camera import Camera
from .light import Light


class Scene:
    """Container that ties render objects, camera, canvas, and input into a live WebGPU scene."""
    canvas: Canvas = None
    render_objects: list[BaseRenderer]
    options: RenderOptions
    gui: object = None

    def __init__(
        self,
        render_objects: list[BaseRenderer],
        id: str | None = None,
        canvas: Canvas | None = None,
        camera: Camera | None = None,
        light: Light | None = None,
    ):
        """Create a scene from render objects and optional canvas/camera/light overrides."""
        if id is None:
            import uuid

            id = str(uuid.uuid4())

        objects = render_objects
        pmin, pmax = max_bounding_box([o.get_bounding_box() for o in objects])
        self.bounding_box = (pmin, pmax)
        if camera is None:
            camera = Camera()
            camera.reset(pmin, pmax)
        light = light or Light()
        self.options = RenderOptions(camera, light)
        self._render_mutex = None

        self._id = id
        self.render_objects = render_objects
        self._on_click_background = []

        self.t_last = 0

        self.input_handler = InputHandler()

    def __getstate__(self):
        """Return picklable state so scenes can be serialized between processes/notebooks."""
        state = {
            "render_objects": self.render_objects,
            "id": self._id,
            "render_options": self.options,
        }
        return state

    def on_click_background(self, callback):
        self._on_click_background.append(callback)

    def __setstate__(self, state):
        """Restore a pickled scene and reinitialize input handling (canvas is set later)."""
        self.render_objects = state["render_objects"]
        self._id = state["id"]
        self.options = state["render_options"]
        self.canvas = None
        self.input_handler = InputHandler()
        self._render_mutex = None

        if is_pyodide:
            _scenes_by_id[self._id] = self

    def __repr__(self):
        return ""

    def export(self, path=None):
        """Export scene to binary blob for JS engine."""
        from .export.capture import capture_scene
        from .export.serialize import serialize_scene
        blob = serialize_scene(capture_scene(self))
        if path:
            from pathlib import Path
            Path(path).write_bytes(blob)
        return blob

    @property
    def id(self) -> str:
        return self._id

    @property
    def device(self) -> Device:
        return self.canvas.device

    def init(self, canvas):
        """Attach the scene to a canvas and initialize GPU resources and event handlers."""
        self.canvas = canvas
        self.input_handler.set_canvas(canvas.canvas)
        self.options.set_canvas(canvas)

        self._render_mutex = Lock(True) if is_pyodide else canvas._update_mutex

        with self._render_mutex:
            self.options.timestamp = time.time()
            self.options.update_buffers()
            for obj in self.render_objects:
                if not obj.active:
                    continue
                try:
                    obj._update_and_create_render_pipeline(self.options)
                except Exception as e:
                    print(f'Warning: failed to init renderer {type(obj).__name__}: {e}')
                    obj.active = False

            camera = self.options.camera
            self._js_render = platform.create_proxy(self._render_direct)
            camera.register_callbacks(self.input_handler, self.get_position)
            camera.register_observer(self._on_camera_changed)

            self._select_buffer = self.device.createBuffer(
                size=4 * 4,
                usage=BufferUsage.COPY_DST | BufferUsage.MAP_READ,
                label="select",
            )
            self._select_buffer_valid = False

            canvas.on_resize(self._on_resize)
            canvas.on_visibility(self._on_visibility_changed)

            canvas.on_update_html_canvas(self.__on_update_html_canvas)

    def __on_update_html_canvas(self, html_canvas):
        """Update event wiring when the underlying HTML canvas element changes."""
        camera = self.options.camera
        # Always unregister first to avoid duplicates
        camera.unregister_callbacks(self.input_handler)
        camera.unregister_observer(self._on_camera_changed)

        if html_canvas is not None:
            self.input_handler.set_canvas(html_canvas)
            camera.register_callbacks(self.input_handler, self.get_position)
            camera.register_observer(self._on_camera_changed)
        else:
            self.input_handler.set_canvas(None)

    def get_position(self, x: int, y: int):
        """Return the 3D position under canvas pixel (x, y) using the selection buffer."""
        objects = self.render_objects

        with self._render_mutex:
            select_texture = self.canvas.select_texture
            bytes_per_row = (select_texture.width * 16 + 255) // 256 * 256

            options = self.options
            options.update_buffers()
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
                p = ev.calculate_position(self.options)
                return p
            return None

    @debounce
    def select(self, x: int, y: int):
        """Perform an object selection at (x, y) and dispatch callbacks on matching renderers."""
        if self._render_mutex is None:
            return
        objects = self.render_objects

        have_select_callback = len(self._on_click_background) != 0
        for obj in objects:
            if obj.active and obj.on_select_set:
                have_select_callback = True
                break

        if not have_select_callback:
            return

        with self._render_mutex:
            select_texture = self.canvas.select_texture
            bytes_per_row = (select_texture.width * 16 + 255) // 256 * 256

            options = self.options
            options.update_buffers()
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
                for parent in objects:
                    for obj in parent.all_renderer():
                        if obj._id == ev.obj_id:
                            obj._handle_on_select(ev)
                            break
            else:
                for cb in self._on_click_background:
                    cb(ev)
            return ev

    # @print_communications
    def _render_objects(self, to_canvas=True, update_pipelines=True):
        """Update pipelines and render all active objects, optionally copying to the canvas."""
        if self.canvas is None:
            return
        options = self.options

        if update_pipelines:
            self._select_buffer_valid = False
            options.timestamp = time.time()
            for obj in self.render_objects:
                if obj.active:
                    obj._update_and_create_render_pipeline(options)
                    if obj.needs_update:
                        print("warning: object still needs update after update was done:", obj)

        options.command_encoder = self.device.createCommandEncoder()
        for obj in self.render_objects:
            if obj.active:
                obj.render_opaque(options)
        for obj in self.render_objects:
            if obj.active:
                obj.render_transparent(options)

        if to_canvas:
                target_texture = self.canvas.target_texture
                if target_texture is not None:
                    # Skip if the underlying JS texture has been destroyed already.
                    handle = getattr(target_texture, "handle", None)
                    if handle is not None and getattr(handle, "__webgpu_destroyed__", False):
                        current_texture = None
                    else:
                        current_texture = self.canvas.context.getCurrentTexture()

                    if current_texture is not None:
                        copy_width = min(target_texture.width, current_texture.width)
                        copy_height = min(target_texture.height, current_texture.height)

                        if copy_width > 0 and copy_height > 0:
                            options.command_encoder.copyTextureToTexture(
                                TexelCopyTextureInfo(target_texture),
                                TexelCopyTextureInfo(current_texture),
                                [copy_width, copy_height, 1],
                            )
        self.device.queue.submit([options.command_encoder.finish()])
        options.command_encoder = None

    def _render_highlight(self):
        """Fast re-render for highlight-only uniform changes.

        Skips pipeline rebuild and select buffer invalidation.
        Caller must already hold _render_mutex.
        """
        if self.canvas is None or self.canvas.height == 0:
            return
        self._render_objects(to_canvas=False, update_pipelines=False)
        platform.js.patchedRequestAnimationFrame(
            self.canvas.device.handle,
            self.canvas.context,
            self.canvas.target_texture,
        )

    def redraw(self, blocking=False, fps=10):
        """Request a redraw, either blocking immediately or debounced on the event loop."""
        self.options.timestamp = time.time()
        if blocking:
            self.render._original(self)
        else:
            self.render()

    def _render(self):
        """Schedule a frame render via requestAnimationFrame on the JS side."""
        platform.js.requestAnimationFrame(self._js_render)

    def _render_direct(self, t=0):
        """Internal render entry point used from JS, rendering directly to the canvas texture."""
        self._render_objects(to_canvas=True)

    @debounce
    def render(self, t=0, rerender_if_update_needed=True):
        """Main render loop: enqueue a frame and optionally keep rendering while objects update."""
        if self.canvas is None or self.canvas.height == 0:
            return

        if is_pyodide_main_thread:
            self._render()
            return
            
        if self._render_mutex is None:
            return

        with self._render_mutex:
            if self.canvas is None or self.canvas.height == 0:
                return
            self._render_objects(to_canvas=False)

            platform.js.patchedRequestAnimationFrame(
                self.canvas.device.handle,
                self.canvas.context,
                self.canvas.target_texture,
            )

    def cleanup(self):
        """Detach the scene from its canvas, unregister callbacks, and release JS proxies."""
        if self._render_mutex is None:
            return
        with self._render_mutex:
            if self.canvas is not None:
                self.options.camera.unregister_callbacks(self.input_handler)
                self.options.camera.unregister_observer(self._on_camera_changed)
                self.input_handler.unregister_callbacks()
                if hasattr(self, '_js_render'):
                    platform.destroy_proxy(self._js_render)
                    del self._js_render
                if self._on_resize in self.canvas._on_resize_callbacks:
                    self.canvas._on_resize_callbacks.remove(self._on_resize)
                if self._on_visibility_changed in self.canvas._on_visibility_callbacks:
                    self.canvas._on_visibility_callbacks.remove(self._on_visibility_changed)
                if self.__on_update_html_canvas in self.canvas._on_update_html_canvas:
                    self.canvas._on_update_html_canvas.remove(self.__on_update_html_canvas)
                self.canvas = None

    def _on_camera_changed(self):
        """Called by the camera when its transform changes. Update uniforms and re-render."""
        if self.canvas is None:
            return
        html_canvas = self.canvas.canvas
        if html_canvas is None:
            return
        # Check if canvas has layout (display:none → offsetWidth/Height = 0)
        if not html_canvas.offsetWidth or not html_canvas.offsetHeight:
            self.options.camera.unregister_callbacks(self.input_handler)
            self.options.camera.unregister_observer(self._on_camera_changed)
            return
        self.options.update_buffers()
        self.render()

    def _on_resize(self):
        """Called on canvas resize. Update camera uniforms (aspect ratio) and re-render."""
        self.options.update_buffers()
        self.render()

    def _on_visibility_changed(self, visible):
        """Called by canvas IntersectionObserver when visibility changes."""
        camera = self.options.camera
        if visible:
            camera.register_callbacks(self.input_handler, self.get_position)
            camera.register_observer(self._on_camera_changed)
            self.options.update_buffers()
            self.render()
        else:
            camera.unregister_callbacks(self.input_handler)
            camera.unregister_observer(self._on_camera_changed)
