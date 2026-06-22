import time
import os
import json
import pathlib
import threading
from base64 import b64decode

from . import platform
from .canvas import Canvas, debounce
from .input_handler import InputHandler
from .renderer import BaseRenderer, RenderOptions, SelectEvent
from .utils import max_bounding_box, read_buffer, read_texture, Lock, print_communications, take_dirty_buffers
from .platform import is_pyodide, is_pyodide_main_thread
from .webgpu_api import *
from .camera import Camera
from .light import Light


_default_use_js_engine = True


def set_default_use_js_engine(value: bool):
    """Set the default backend for scenes: True = JS engine, False = legacy
    Python render/camera. Override per scene via ``Scene(use_js_engine=...)``."""
    global _default_use_js_engine
    _default_use_js_engine = bool(value)


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
        use_js_engine: bool | None = None,
        show_gui_controls: bool = True,
    ):
        """Create a scene from render objects and optional canvas/camera/light overrides.

        ``use_js_engine`` selects the backend (see :func:`set_default_use_js_engine`).
        ``show_gui_controls`` toggles the JS engine's built-in lil-gui panel.
        """
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
        self._on_event_proxy = None
        self._on_camera_changed_proxy = None
        self._select_lock = threading.Lock()
        self._pending_select = None
        self._select_running = False
        self._use_js_engine = (
            _default_use_js_engine if use_js_engine is None else bool(use_js_engine)
        )
        self._show_gui_controls = bool(show_gui_controls)
        # Latest live buffer registry (set on each capture). render() uses it to
        # map host-written buffers to engine ids for targeted compute re-triggers.
        self._registry = None

    def __getstate__(self):
        """Return picklable state so scenes can be serialized between processes/notebooks."""
        state = {
            "render_objects": self.render_objects,
            "id": self._id,
            "render_options": self.options,
            "use_js_engine": self._use_js_engine,
            "show_gui_controls": self._show_gui_controls,
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
        self._on_event_proxy = None
        self._on_camera_changed_proxy = None
        self._select_lock = threading.Lock()
        self._pending_select = None
        self._select_running = False
        self._use_js_engine = state.get("use_js_engine", _default_use_js_engine)
        self._show_gui_controls = state.get("show_gui_controls", True)
        self._registry = None

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

        self._wire_input(camera)

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
            camera.register_observer(self._on_camera_changed)

            self._select_buffer_valid = False

            canvas.on_resize(self._on_resize)
            canvas.on_visibility(self._on_visibility_changed)
            canvas.on_update_html_canvas(self.__on_update_html_canvas)

        self._js_engine = None
        if not os.environ.get("WEBGPU_EXPORTING") and not os.environ.get("WEBGPU_TESTING"):
            self._install_live_engine()

        self._wire_input(camera)

    def _wire_input(self, camera):
        """Route input to the JS engine (camera owned in the browser, only
        double-click-to-center kept in Python) or to the legacy Python path."""
        if self._js_engine is not None:
            camera.unregister_callbacks(self.input_handler)
            self.input_handler.set_engine_mode(True)
            camera.register_dblclick_center(self.input_handler, self.get_position)
            self.options.skip_camera_buffer_write = True
        else:
            self.input_handler.set_engine_mode(False)
            camera.register_callbacks(self.input_handler, self.get_position)
            self.options.skip_camera_buffer_write = False

    def _ensure_engine_js(self):
        """Load the JS ``RenderEngine`` into the browser if not already present
        (some hosts, e.g. the ngapp app, don't inject it). Returns True if
        available afterwards."""
        if not hasattr(platform, 'js') or platform.js is None:
            return False
        if getattr(platform.js, 'RenderEngine', None) is not None:
            return True
        try:
            from .engine import engine_js

            doc = platform.js.document
            script = doc.createElement("script")
            script.textContent = engine_js
            doc.head.appendChild(script)
        except Exception as e:
            print(f"warning: could not inject engine_js: {e}")
            return False
        return getattr(platform.js, 'RenderEngine', None) is not None

    def _install_live_engine(self):
        """Build a live descriptor and hand it to RenderEngine.createLive.

        Idempotent: safe to call again after pipelines change (rebuilds via
        ``engine.update`` if an engine is already installed).
        """
        if not self._use_js_engine:
            return
        if not hasattr(platform, 'js') or platform.js is None:
            return
        if not self._ensure_engine_js():
            return
        RenderEngine = platform.js.RenderEngine

        canvas = self.canvas
        if canvas is None:
            return

        from .export.capture import capture_scene_live, build_live_resource_maps
        from dataclasses import asdict

        with self._render_mutex:
            export, registry = capture_scene_live(self)

        # Keep the registry so render() can resolve host-written buffers (clip
        # plane, grid size, …) to engine ids for targeted compute re-triggers.
        self._registry = registry

        buffers, textures, samplers, frame_buffers = build_live_resource_maps(registry)

        # Empty interactions → JS engine builds no lil-gui panel.
        interactions = (
            [asdict(i) for i in export.interactions] if self._show_gui_controls else []
        )

        # Input events are backpressured: the JS input handler keeps at most one
        # high-frequency move event in flight and waits for an explicit ack
        # before sending the next, so a fast drag can't pile a backlog into the
        # async callback queue that then replays slowly. The proxy itself is
        # fire-and-forget (its return value is dropped at enqueue time on the
        # websocket bridge), so we ack *after* the handler actually runs.
        if getattr(self, "_on_event_proxy", None) is None:
            self._on_event_proxy = platform.create_proxy(
                self._handle_input_event_and_ack, ignore_return_value=True
            )
        # Camera-changed stays fire-and-forget so the browser never blocks on it.
        if getattr(self, "_on_camera_changed_proxy", None) is None:
            self._on_camera_changed_proxy = platform.create_proxy(
                self._apply_camera_from_js, ignore_return_value=True
            )

        descriptor = {
            "device": canvas.device.handle,
            "context": canvas.context,
            "canvasFormat": canvas.format,
            "buffers": buffers,
            "textures": textures,
            "samplers": samplers,
            "frame_buffers": frame_buffers,
            "render_passes":  [asdict(p) for p in export.render_passes],
            "compute_passes": [asdict(p) for p in export.compute_passes],
            "interactions":   interactions,
            "camera": export.camera,
            "light":  export.light,
            "theme": export.theme,
            "on_event": self._on_event_proxy,
            "on_camera_changed": self._on_camera_changed_proxy,
            "clear_color": self._canvas_clear_color(),
        }

        # Signature of everything engine.update() actually rebuilds: the pass
        # descriptors, interactions, and the *set* of resource ids (not their
        # contents). A transient sim redraws every timestep with new field DATA
        # but an unchanged structure — and the data is already written to the
        # GPU in place (device.queue.writeBuffer, called from each renderer's
        # update() in _update_and_create_render_pipeline) before we get here.
        # So re-shipping the descriptor and rebuilding all JS pipelines is pure
        # overhead, and the blocking update() round-trip is what eventually
        # trips the 120s link timeout. Skip it when nothing structural changed;
        # render() still issues notifyDirty()+render() to show the new data.
        # A buffer that grows gets a fresh id (the registry keys by object
        # identity), which changes the id set here and forces a real update.
        cam = descriptor["camera"]
        structural_sig = json.dumps(
            {
                "render_passes":    descriptor["render_passes"],
                "compute_passes":   descriptor["compute_passes"],
                "interactions":     descriptor["interactions"],
                "camera_buffer_id": cam.get("buffer_id") if isinstance(cam, dict) else None,
                "buffer_ids":       sorted(buffers.keys()),
                "texture_ids":      sorted(textures.keys()),
                "sampler_ids":      sorted(samplers.keys()),
                "frame_buffer_ids": sorted(frame_buffers.keys()),
            },
            sort_keys=True,
            default=str,
        )
        if self._js_engine is not None and structural_sig == getattr(
            self, "_installed_descriptor_sig", None
        ):
            return

        js_descriptor = platform.toJS(descriptor)

        # Mark renderers that have JS-side compute so they skip Python dispatch
        for obj in self.render_objects:
            if hasattr(obj, '_js_compute') and export.compute_passes:
                if hasattr(obj, 'get_export_compute_passes'):
                    obj._js_compute = True

        if self._js_engine is None:
            promise = RenderEngine.createLive(canvas.canvas, js_descriptor)
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
                # The buffer registry reassigns string ids each capture, so the
                # camera's id can change when the render-object set changes. Send
                # the fresh camera so the engine re-resolves _cameraBufferId
                # instead of writing camera data to a stale (wrong-sized) buffer.
                "camera":         descriptor["camera"],
            }))
        self._installed_descriptor_sig = structural_sig

        # The engine has just rebuilt (and pruned) this.buffers to the current
        # set, so any resource whose Python wrapper was GC'd is no longer
        # referenced by a frame — safe to free now. Runs under the render mutex
        # held by our caller (render()/init).
        flush_pending_destroys()

    def _handle_input_event_and_ack(self, event):
        """Process a forwarded input event, then ack the JS input handler so it
        releases backpressure and sends the next (coalesced) move. Runs on the
        link's callback thread; the ack is sent only after the handler (incl.
        its render) has run, which is what actually bounds the event rate."""
        try:
            self.input_handler.handle_engine_event(event)
        finally:
            eng = self._js_engine
            if eng is not None:
                try:
                    # Fire-and-forget so the callback thread isn't blocked on it.
                    if hasattr(eng, "_call_method_ignore_return"):
                        eng._call_method_ignore_return("ackInput", [])
                    else:
                        eng.ackInput()
                except Exception:
                    pass

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
            # Engine gone — fall back to Python input until _on_resize reinstalls it.
            self.input_handler.set_engine_mode(False)

        if html_canvas is not None:
            self.input_handler.set_canvas(html_canvas)
            camera.register_callbacks(self.input_handler, self.get_position)
            camera.register_observer(self._on_camera_changed)
        else:
            self.input_handler.set_canvas(None)

    def get_position(self, x: int, y: int):
        """Return the 3D position under canvas pixel (x, y) using the selection buffer."""
        if self.canvas is None or self.canvas.height == 0:
            return None
        objects = self.render_objects

        with self._render_mutex:
            canvas = self.canvas
            select_texture = canvas.select_texture if canvas is not None else None
            if select_texture is None:
                return None
            bytes_per_row = (select_texture.width * 16 + 255) // 256 * 256
            x = min(max(int(x), 0), int(select_texture.width) - 1)
            y = min(max(int(y), 0), int(select_texture.height) - 1)

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

    def select(self, x: int, y: int):
        """Queue an object selection at (x, y).

        A single worker processes only the most recent request, so a backlog
        (e.g. hover moves piled up while Python was busy) collapses to the last
        position instead of replaying every queued move. Never runs two selects
        concurrently.
        """
        if self._render_mutex is None:
            return
        if self.canvas is None or self.canvas.height == 0:
            return
        # Pyodide: JS is only reachable from this thread — run synchronously.
        if is_pyodide:
            self._do_select(int(x), int(y))
            return
        with self._select_lock:
            self._pending_select = (int(x), int(y))
            if self._select_running:
                return
            self._select_running = True
        threading.Thread(target=self._select_worker, daemon=True).start()

    def _select_worker(self):
        while True:
            with self._select_lock:
                pending = self._pending_select
                self._pending_select = None
                if pending is None:
                    self._select_running = False
                    return
            try:
                self._do_select(*pending)
            except Exception as e:
                print(f"warning: select failed: {e}")

    def _do_select(self, x: int, y: int):
        """Perform an object selection at (x, y) and dispatch callbacks on matching renderers."""
        if self._render_mutex is None:
            return
        if self.canvas is None or self.canvas.height == 0:
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
            canvas = self.canvas
            select_texture = canvas.select_texture if canvas is not None else None
            if select_texture is None:
                return None
            bytes_per_row = (select_texture.width * 16 + 255) // 256 * 256
            x = min(max(int(x), 0), int(select_texture.width) - 1)
            y = min(max(int(y), 0), int(select_texture.height) - 1)

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
        if self.canvas is None or self.canvas.height == 0:
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
        render_pass = options.begin_render_pass()
        options.render_pass = render_pass
        try:
            for obj in self.render_objects:
                if obj.active:
                    obj.render_opaque(options)
            for obj in self.render_objects:
                if obj.active:
                    obj.render_transparent(options)
        finally:
            render_pass.end()
            options.render_pass = None

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

        # Render-safe flush point for the direct/legacy path: this frame was
        # drawn from the current render objects, so any resource whose Python
        # wrapper was GC'd is no longer referenced and can be freed. Only after
        # a full render (update_pipelines); the highlight fast-path passes
        # update_pipelines=False and must not free engine-referenced buffers.
        if update_pipelines:
            flush_pending_destroys()

    def _canvas_clear_color(self):
        """Return the canvas clear color as [r, g, b, a] (or None)."""
        cc = getattr(self.canvas, "clear_color", None) if self.canvas else None
        if cc is None:
            return None
        return [float(cc.r), float(cc.g), float(cc.b), float(cc.a)]

    def _render_highlight(self):
        """Fast re-render for highlight-only uniform changes.

        Skips pipeline rebuild and select buffer invalidation.
        Caller must already hold _render_mutex.
        """
        if self.canvas is None or self.canvas.height == 0:
            return
        # Live engine: uniforms already written; just redraw (a Python render
        # here would fight the engine for the canvas).
        engine = self._js_engine
        if engine is not None:
            try:
                engine.render()
            except Exception as e:
                print(f"warning: js_engine.render() failed: {e}")
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

        if self._js_engine is not None:
            with self._render_mutex:
                self.options.update_buffers()
                self._select_buffer_valid = False
                any_dirty = any(
                    obj.needs_update for obj in self.render_objects if obj.active
                )
                active_ids = frozenset(
                    id(obj) for obj in self.render_objects if obj.active
                )
                if any_dirty or active_ids != getattr(self, "_installed_active_set", None):
                    if any_dirty:
                        self.options.timestamp = time.time()
                    for obj in self.render_objects:
                        if obj.active:
                            obj._update_and_create_render_pipeline(self.options)
                    self._install_live_engine()  # idempotent → engine.update()
                    self._installed_active_set = active_ids
            engine = self._js_engine
            if engine is None:
                return
            try:
                cc = self._canvas_clear_color()
                if cc is not None and cc != getattr(self, "_pushed_clear_color", None):
                    engine.setClearColor(platform.toJS(cc))
                    self._pushed_clear_color = cc
                dirty_bufs = take_dirty_buffers()
                if any_dirty:
                    engine.notifyDirty(None)
                elif dirty_bufs and self._registry is not None:
                    ids = []
                    for buf in dirty_bufs:
                        try:
                            ids.append(self._registry.get_id(buf))
                        except KeyError:
                            pass  # buffer not in the engine (e.g. CPU-only) — skip
                    if ids:
                        engine.notifyDirty(ids)
                engine.render()
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
                for attr in ("_on_event_proxy", "_on_camera_changed_proxy"):
                    proxy = getattr(self, attr, None)
                    if proxy is not None:
                        try:
                            platform.destroy_proxy(proxy)
                        except Exception:
                            pass
                        setattr(self, attr, None)
                if self._on_resize in self.canvas._on_resize_callbacks:
                    self.canvas._on_resize_callbacks.remove(self._on_resize)
                if self._on_visibility_changed in self.canvas._on_visibility_callbacks:
                    self.canvas._on_visibility_callbacks.remove(self._on_visibility_changed)
                if self.__on_update_html_canvas in self.canvas._on_update_html_canvas:
                    self.canvas._on_update_html_canvas.remove(self.__on_update_html_canvas)
                self.canvas = None

    def _on_camera_changed(self):
        """Python-initiated camera change (reset/view/bookmark): push to the JS
        engine if live, else update the uniform and re-render (legacy)."""
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

        if self._js_engine is not None:
            with self._render_mutex:
                self._select_buffer_valid = False
            t = self.options.camera.transform
            center = t._center.tolist() if hasattr(t._center, "tolist") else list(t._center)
            try:
                self._js_engine.setCameraTransform(
                    platform.toJS(t._mat.flatten().tolist()),
                    platform.toJS(list(center)),
                )
            except Exception as e:
                print(f"warning: js_engine.setCameraTransform() failed: {e}")
            return

        with self._render_mutex:
            self._select_buffer_valid = False
            self.options.update_buffers()
        self.render()

    def _apply_camera_from_js(self, payload):
        """Mirror a JS-engine camera move back into the Python camera (for
        bookmarks/screenshots/picking). Does not re-render or push back to JS."""
        import numpy as np

        try:
            try:
                import pyodide.ffi

                if isinstance(payload, pyodide.ffi.JsProxy):
                    payload = payload.to_py()
            except ImportError:
                pass
            mat = list(payload["matrix"])
            center = list(payload["center"])
            t = self.options.camera.transform
            t._mat = np.array(mat, dtype=float).reshape(4, 4)
            t._center = np.array(center, dtype=float)
            if self._render_mutex is not None:
                with self._render_mutex:
                    self._select_buffer_valid = False
        except Exception as e:
            print(f"warning: apply camera from js failed: {e}")

    def _on_resize(self):
        """Called on canvas resize. Update camera uniforms (aspect ratio) and re-render."""
        with self._render_mutex:
            self._select_buffer_valid = False
            self.options.update_buffers()
        if self._js_engine is not None:
            try:
                self._js_engine.handleResize()
            except Exception as e:
                print(f'warning: js_engine.handleResize() failed: {e}')
        elif not os.environ.get("WEBGPU_EXPORTING") and not os.environ.get("WEBGPU_TESTING"):
            # Engine was disposed (e.g. canvas changed). Reinstall and re-route input.
            if self.canvas is not None and self.canvas.width > 0 and self.canvas.height > 0:
                self._install_live_engine()
                self._wire_input(self.options.camera)
        self.render()

    def _on_visibility_changed(self, visible):
        """Called by canvas IntersectionObserver when visibility changes."""
        camera = self.options.camera
        if visible:
            camera.register_observer(self._on_camera_changed)
            self._wire_input(camera)
            self.options.update_buffers()
            self.render()
        else:
            camera.unregister_callbacks(self.input_handler)
            camera.unregister_observer(self._on_camera_changed)
