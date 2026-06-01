import time
import os
import pathlib
from base64 import b64decode

from . import platform
from .canvas import Canvas, debounce
from .input_handler import InputHandler
from .renderer import BaseRenderer, RenderOptions, SelectEvent
from .utils import max_bounding_box, read_buffer, read_texture, Lock, print_communications
from .platform import is_pyodide, is_pyodide_main_thread
from .webgpu_api import *
from .camera import Camera
from .light import Light


class Scene:
    """Container that ties render objects, camera, canvas, and input into a live WebGPU scene."""
    canvas: Canvas = None
    render_objects: list[BaseRenderer]
    options: RenderOptions


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
        self._js_engine = None

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
        self._js_engine = None

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

    def save_screenshot(self, filename: str):
        """Save a screenshot of the current rendered frame to *filename*.

        Supports both the live JS engine path and the legacy Python render
        path.  The file format is inferred from the extension (e.g. ".png").
        """
        import numpy as np

        if self.canvas is None or self.canvas.width == 0 or self.canvas.height == 0:
            raise RuntimeError("Cannot save screenshot: no canvas or canvas has zero size")

        if self._js_engine is not None:
            # JS engine renders directly to the canvas texture.
            # Use captureFrameBuffer() which renders a frame, does GPU readback
            # on the JS side, and returns the raw pixel ArrayBuffer.
            data_bytes = self._js_engine.captureFrameBuffer()
            width = self.canvas.width
            height = self.canvas.height
            fmt = str(self.canvas.format)
            data = np.frombuffer(data_bytes, dtype=np.uint8).reshape((height, width, 4))
            if fmt == "bgra8unorm":
                data = data[:, :, [2, 1, 0, 3]]
        else:
            # Legacy Python render path: render to target_texture, then readback.
            with self._render_mutex:
                self._render_objects(to_canvas=False)
            data = read_texture(self.canvas.target_texture)

        path = pathlib.Path(filename)
        fmt_ext = path.suffix[1:]  # e.g. "png"

        canvas_el = platform.js.document.createElement("canvas")
        canvas_el.width = self.canvas.width
        canvas_el.height = self.canvas.height
        ctx = canvas_el.getContext("2d")
        u8 = platform.js.Uint8ClampedArray._new(data.tobytes())
        image_data = platform.js.ImageData._new(u8, self.canvas.width, self.canvas.height)
        ctx.putImageData(image_data, 0, 0)
        canvas_el.remove()
        path.write_bytes(b64decode(canvas_el.toDataURL(fmt_ext).split(",")[1]))

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

        # After core init, try to install the JS engine as the live renderer.
        # Drops to legacy Python rendering if RenderEngine isn't reachable.
        # Skipped during export and tests (the export path emits a fresh blob-
        # driven engine; tests rely on the legacy Python render path for now).
        self._js_engine = None
        if not os.environ.get("WEBGPU_EXPORTING") and not os.environ.get("WEBGPU_TESTING"):
            self._install_live_engine()

    def reconnect(self, canvas):
        """Re-attach a previously initialized scene to a (new) canvas.

        Unlike :meth:`init`, this skips the expensive pipeline rebuild for
        all render objects.  Use after :meth:`cleanup` when the scene data
        (buffers, pipelines) is still valid but the canvas changed.
        """
        self.canvas = canvas
        self.input_handler.set_canvas(canvas.canvas)
        self.options.set_canvas(canvas)

        self._render_mutex = Lock(True) if is_pyodide else canvas._update_mutex

        with self._render_mutex:
            self.options.update_buffers()

            camera = self.options.camera
            if not hasattr(self, '_js_render') or self._js_render is None:
                self._js_render = platform.create_proxy(self._render_direct)
            camera.register_callbacks(self.input_handler, self.get_position)
            camera.register_observer(self._on_camera_changed)

            self._select_buffer_valid = False

            canvas.on_resize(self._on_resize)
            canvas.on_visibility(self._on_visibility_changed)
            canvas.on_update_html_canvas(self.__on_update_html_canvas)

        self._js_engine = None
        if not os.environ.get("WEBGPU_EXPORTING") and not os.environ.get("WEBGPU_TESTING"):
            self._install_live_engine()

    def _install_live_engine(self):
        """Build a live descriptor and hand it to RenderEngine.createLive.

        Idempotent: safe to call again after pipelines change (rebuilds via
        ``engine.update`` if an engine is already installed).
        """
        if not hasattr(platform, 'js') or platform.js is None:
            return
        RenderEngine = getattr(platform.js, 'RenderEngine', None)
        if RenderEngine is None:
            return

        from .export.capture import capture_scene_live, build_live_resource_maps
        from dataclasses import asdict

        with self._render_mutex:
            export, registry = capture_scene_live(self)

        buffers, textures, samplers, frame_buffers = build_live_resource_maps(registry)

        descriptor = {
            "device": self.canvas.device.handle,
            "context": self.canvas.context,
            "canvasFormat": self.canvas.format,
            "buffers": buffers,
            "textures": textures,
            "samplers": samplers,
            "frame_buffers": frame_buffers,
            "render_passes":  [asdict(p) for p in export.render_passes],
            "compute_passes": [asdict(p) for p in export.compute_passes],
            "interactions":   [asdict(i) for i in export.interactions],
            "camera": export.camera,
            "light":  export.light,
        }

        js_descriptor = platform.toJS(descriptor)

        # Mark renderers that have JS-side compute so they skip Python dispatch
        for obj in self.render_objects:
            if hasattr(obj, '_js_compute') and export.compute_passes:
                if hasattr(obj, 'get_export_compute_passes'):
                    obj._js_compute = True

        if self._js_engine is None:
            promise = RenderEngine.createLive(self.canvas.canvas, js_descriptor)
            # In Pyodide we get a JsPromise; in websocket mode the call is sync.
            try:
                self._js_engine = promise.syncify() if hasattr(promise, 'syncify') else promise
            except Exception:
                # Fallback: store the promise; render() will await/skip until ready.
                self._js_engine = promise
        else:
            self._js_engine.update(platform.toJS({
                "render_passes":  descriptor["render_passes"],
                "compute_passes": descriptor["compute_passes"],
                "interactions":   descriptor["interactions"],
                "buffers":        buffers,
                "textures":       textures,
                "samplers":       samplers,
                "frame_buffers":  frame_buffers,
            }))

    def __on_update_html_canvas(self, html_canvas):
        """Update event wiring when the underlying HTML canvas element changes."""
        camera = self.options.camera
        # Always unregister first to avoid duplicates
        camera.unregister_callbacks(self.input_handler)
        camera.unregister_observer(self._on_camera_changed)

        # Dispose the old JS engine — its context/canvas references are stale.
        if self._js_engine is not None:
            try:
                self._js_engine.dispose()
            except Exception:
                pass
            self._js_engine = None
            # Reset _js_compute flag on renderers so they fall back to Python-
            # side compute dispatch until a new engine is installed.
            for obj in self.render_objects:
                if hasattr(obj, '_js_compute'):
                    obj._js_compute = False

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
        if self.canvas is None:
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
        if not os.environ.get("WEBGPU_TESTING"):
            platform.js.patchedRequestAnimationFrame(
                self.canvas.device.handle,
                self.canvas.context,
                self.canvas.target_texture,
            )

    def redraw(self, blocking=False, fps=10):
        """Request a redraw, either blocking immediately or debounced on the event loop."""
        for obj in self.render_objects:
            obj.set_needs_update()
        self.options.timestamp = time.time()
        if blocking:
            self.render._original()
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

        # Live JS engine path: it owns the rAF loop; we only ensure
        # buffer contents are up-to-date and renderers that explicitly need
        # rebuild are re-captured into the engine.
        if self._js_engine is not None:
            with self._render_mutex:
                self.options.update_buffers()
                any_dirty = any(
                    obj.needs_update for obj in self.render_objects if obj.active
                )
                if any_dirty:
                    self.options.timestamp = time.time()
                    for obj in self.render_objects:
                        if obj.active:
                            obj._update_and_create_render_pipeline(self.options)
                    self._install_live_engine()  # idempotent → engine.update()
            try:
                if any_dirty:
                    self._js_engine.notifyDirty(None)
                self._js_engine.render()
            except Exception as e:
                print(f'warning: js_engine.render() failed: {e}')
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

            # patchedRequestAnimationFrame copies target_texture → canvas for
            # display. In headless test mode this is unnecessary and its async
            # queue.submit interferes with bridge mapAsync calls.
            if not os.environ.get("WEBGPU_TESTING"):
                platform.js.patchedRequestAnimationFrame(
                    self.canvas.device.handle,
                    self.canvas.context,
                    self.canvas.target_texture,
                )

    def cleanup(self):
        """Detach the scene from its canvas, unregister callbacks, and release JS proxies."""
        if self._js_engine is not None:
            try:
                self._js_engine.dispose()
            except Exception:
                pass
            self._js_engine = None

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
        with self._render_mutex:
            self._select_buffer_valid = False
            self.options.update_buffers()
        self.render()

    def _on_resize(self):
        """Called on canvas resize. Update camera uniforms (aspect ratio) and re-render."""
        self._select_buffer_valid = False
        self.options.update_buffers()
        if self._js_engine is not None:
            try:
                self._js_engine.handleResize()
            except Exception as e:
                print(f'warning: js_engine.handleResize() failed: {e}')
        elif not os.environ.get("WEBGPU_EXPORTING") and not os.environ.get("WEBGPU_TESTING"):
            # Engine was disposed (e.g. canvas element changed). Reinstall now
            # that the canvas has valid dimensions.
            if self.canvas is not None and self.canvas.width > 0 and self.canvas.height > 0:
                self._install_live_engine()
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
