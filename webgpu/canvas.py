from base64 import b64decode
from typing import Callable
import threading
import time
import functools
import pathlib

from . import platform
from .utils import get_device, read_texture, Lock
from .webgpu_api import *
from functools import wraps

_TARGET_FPS = 60


# @dataclass
# class _DebounceData:
#     t_last_frame: float = 0
#     t_last_call: float = 0
#     timer: threading.Timer | None = None
#     lock: Lock = None
#     running: bool = False
#     pending: bool = False

def debounce(arg=None, *, rate_hz=60):

    if platform.is_pyodide:
      if callable(arg):
        return arg
      def _rate_limited(fn, *args, **kwargs):
        return fn

      return _rate_limited

    def _rate_limited(fn, rate_hz):
        interval = 1.0 / rate_hz
        lock = threading.RLock()
        last_call = 0.0
        timer = None
        pending = None

        def schedule(delay):
            nonlocal timer
            timer = threading.Timer(delay, run_pending)
            timer.daemon = True
            timer.start()

        def run_pending():
            nonlocal last_call, timer, pending

            with lock:
                if pending is None:
                    timer = None
                    return

                args, kwargs = pending
                pending = None
                # print("call frequency = ", 1.0 / (time.monotonic() - last_call))
                last_call = time.monotonic()

            fn(*args, **kwargs)

            with lock:
                timer = None
                if pending is not None:
                    delay = max(0.0, interval - (time.monotonic() - last_call))
                    schedule(delay)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal last_call, pending

            with lock:
                now = time.monotonic()
                elapsed = now - last_call
                if elapsed >= interval and timer is None:
                    # print("call frequency = ", 1.0 / elapsed if elapsed > 0 else float('inf'))
                    last_call = now
                    run_now = True
                else:
                    pending = (args, kwargs)
                    run_now = False

                    if timer is None:
                        schedule(max(0.0, interval - elapsed))

            if run_now:
                fn(*args, **kwargs)

        return wrapper

    if callable(arg):
        return _rate_limited(arg, rate_hz)

    if arg is not None:
        rate_hz = arg

    def decorate(fn):
        return _rate_limited(fn, rate_hz)
    return decorate



# def debounce(arg=None):
#     def decorator(func):
#         # Render only once every 1/_TARGET_FPS seconds
#         @functools.wraps(func)
#         def debounced(obj, *args, **kwargs):
#             if not hasattr(obj, "_debounce_data"):
#                 obj._debounce_data = {}

#             fname = func.__name__
#             if obj._debounce_data.get(fname, None) is None:
#                 obj._debounce_data[fname] = _DebounceData(0, 0, None, Lock())

#             data = obj._debounce_data[fname]
#             frame_time = 1.0 / target_fps

#             def run():
#                 while True:
#                     # Call func OUTSIDE the lock to avoid deadlocks
#                     func(obj, *args, **kwargs)

#                     with data.lock:
#                         if not data.pending:
#                             data.running = False
#                             return
#                         data.pending = False
#                         elapsed = time.time() - data.t_last_frame
#                         t_wait = frame_time - elapsed
#                         if t_wait > 0:
#                             # Schedule deferred re-run to respect frame rate
#                             if platform.is_pyodide:
#                                 import asyncio
#                                 async def _rerun():
#                                     await asyncio.sleep(t_wait)
#                                     with data.lock:
#                                         data.t_last_frame = time.time()
#                                     run()
#                                 asyncio.create_task(_rerun())
#                             else:
#                                 def _deferred():
#                                     with data.lock:
#                                         data.t_last_frame = time.time()
#                                     run()
#                                 data.timer = threading.Timer(t_wait, _deferred)
#                                 data.timer.start()
#                             return
#                         data.t_last_frame = time.time()

#             def f():
#                 with data.lock:
#                     if t_call != data.t_last_call and t_call - data.t_last_frame < frame_time:
#                         return
#                     if data.running:
#                         data.pending = True
#                         return
#                     data.running = True
#                     data.t_last_frame = time.time()

#                 run()

#             with data.lock:
#                 t_call = time.time()
#                 data.t_last_call = t_call
#                 t_wait = frame_time - (t_call - data.t_last_frame)

#                 if t_wait <= 0:
#                     if data.running:
#                         data.pending = True
#                         return
#                     data.running = True
#                     data.t_last_frame = time.time()
#                     if data.timer is not None:
#                         data.timer.cancel()
#                         data.timer = None
#                 else:
#                     if data.timer is not None:
#                         data.timer.cancel()
#                     if platform.is_pyodide:
#                         import asyncio
#                         async def _runner():
#                             await asyncio.sleep(t_wait)
#                             f()
#                         asyncio.create_task(_runner())
#                     else:
#                         data.timer = threading.Timer(t_wait, f)
#                         data.timer.start()
#                     return

#             run()

#         debounced._original = func
#         return debounced



def init_webgpu(html_canvas):
    """Initialize WebGPU, create device and canvas"""
    device = get_device()
    return Canvas(device, html_canvas)


class Canvas:
    """Canvas management class, handles "global" state, like webgpu device, canvas, frame and depth buffer"""

    device: Device
    depth_format: TextureFormat
    depth_texture: Texture = None
    multisample_texture: Texture = None
    multisample: MultisampleState = None
    target_texture: Texture = None
    select_depth_texture: Texture = None
    select_texture: Texture = None

    width: int = 0
    height: int = 0

    _on_resize_callbacks: list[Callable]
    _on_update_html_canvas: list[Callable]

    def __init__(self, device, canvas, multisample_count=4):
        self._update_mutex = Lock()
        self.target_texture = None

        self._on_resize_callbacks = []
        self._on_update_html_canvas = []
        self._on_visibility_callbacks = []

        self._resize_observer = None
        self._intersection_observer = None

        self.device = device
        self.context = None
        self.format = platform.js.navigator.gpu.getPreferredCanvasFormat()
        self.color_target = ColorTargetState(
            format=self.format,
            blend=BlendState(
                color=BlendComponent(
                    srcFactor=BlendFactor.one,
                    dstFactor=BlendFactor.one_minus_src_alpha,
                    operation=BlendOperation.add,
                ),
                alpha=BlendComponent(
                    srcFactor=BlendFactor.one,
                    dstFactor=BlendFactor.one_minus_src_alpha,
                    operation=BlendOperation.add,
                ),
            ),
        )
        self.depth_format = TextureFormat.depth24plus

        self.select_format = TextureFormat.rgba32uint
        self.select_target = ColorTargetState(
            format=self.select_format,
        )
        self.multisample = MultisampleState(count=multisample_count)

        self.update_html_canvas(canvas)

    def __del__(self):
        disconnect = getattr(self._resize_observer, "disconnect", None)
        if callable(disconnect):
            disconnect()
        disconnect = getattr(self._intersection_observer, "disconnect", None)
        if callable(disconnect):
            disconnect()

    def update_html_canvas(self, html_canvas):
        """Reconfigure the canvas with the current HTML canvas element. This is necessary when the HTML canvas element changes, disappears (e.g. when switching a tab) and appears again."""

        self.width = self.height = 0  # disable rendering until resize is called

        with self._update_mutex:
            if self.context is not None:
                self.context.unconfigure()

            self.canvas = html_canvas
            self.destroy_textures()

            if html_canvas is None:
                self.context = None
                for func in self._on_update_html_canvas:
                    func(html_canvas)
                return

            self.context = html_canvas.getContext("webgpu")
            self.context.configure(
                {
                    "device": self.device.handle,
                    "format": self.format,
                    "alphaMode": "premultiplied",
                    "usage": TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_DST | TextureUsage.COPY_SRC,
                }
            )

            def on_resize(*args):
                self.resize()

            def on_intersection(observer_entry, args):
                if observer_entry[0].isIntersecting:
                    for func in self._on_resize_callbacks:
                        func()
                for func in self._on_visibility_callbacks:
                    func(observer_entry[0].isIntersecting)

            if self._resize_observer is not None:
                self._resize_observer.disconnect()
            if self._intersection_observer is not None:
                self._intersection_observer.disconnect()
            self._resize_observer = platform.js.ResizeObserver._new(
                platform.create_proxy(on_resize, True)
            )
            self._intersection_observer = platform.js.IntersectionObserver._new(
                platform.create_proxy(on_intersection, True),
                {
                    "root": None,
                    "rootMargin": "0px",
                    "threshold": 0.01,  # Trigger when at least 10% of the canvas is visible
                },
            )

            self._resize_observer.observe(self.canvas)
            self._intersection_observer.observe(self.canvas)

            for func in self._on_update_html_canvas:
                func(html_canvas)

            self.width = self.height = 0  # force resize
            self.resize()

    def on_resize(self, func: Callable):
        self._on_resize_callbacks.append(func)

    def on_visibility(self, func: Callable):
        """Register callback for visibility changes. Called with True/False."""
        self._on_visibility_callbacks.append(func)

    def on_update_html_canvas(self, func: Callable):
        self._on_update_html_canvas.append(func)

    def save_screenshot(self, filename: str):
        with self._update_mutex:
            path = pathlib.Path(filename)
            format = path.suffix[1:]
            data = read_texture(self.target_texture)
            canvas = platform.js.document.createElement("canvas")
            canvas.width = self.width
            canvas.height = self.height
            ctx = canvas.getContext("2d")
            u8 = platform.js.Uint8ClampedArray._new(data.tobytes())
            image_data = platform.js.ImageData._new(u8, self.width, self.height)
            ctx.putImageData(image_data, 0, 0)
            canvas.remove()
            path.write_bytes(b64decode(canvas.toDataURL(format).split(",")[1]))

    def destroy_textures(self):
        with self._update_mutex:
            for tex in [
                self.target_texture,
                self.multisample_texture,
                self.depth_texture,
                self.select_texture,
                self.select_depth_texture,
            ]:
                if tex is not None:
                    tex.destroy()

    @debounce(5)
    def resize(self):
        with self._update_mutex:
            canvas = self.canvas
            if canvas is None:
                return
            rect = canvas.getBoundingClientRect()
            dpr = getattr(platform.js.window, 'devicePixelRatio', 1) or 1
            width = round(rect.width * dpr)
            height = round(rect.height * dpr)

            if width == self.width and height == self.height:
                for func in self._on_resize_callbacks:
                    func()
                return False

            if width == 0 or height == 0:
                self.height = 0
                self.width = 0
                return False

            canvas.width = width
            canvas.height = height

            device = self.device

            self.destroy_textures()

            self.target_texture = device.createTexture(
                size=[width, height, 1],
                sampleCount=1,
                format=self.format,
                usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
                label="target",
            )
            if self.multisample.count > 1:
                self.multisample_texture = device.createTexture(
                    size=[width, height, 1],
                    sampleCount=self.multisample.count,
                    format=self.format,
                    usage=TextureUsage.RENDER_ATTACHMENT,
                    label="multisample",
                )

            self.depth_texture = device.createTexture(
                size=[width, height, 1],
                format=self.depth_format,
                usage=TextureUsage.RENDER_ATTACHMENT,
                label="depth_texture",
                sampleCount=self.multisample.count,
            )

            self.target_texture_view = self.target_texture.createView()

            self.select_texture = device.createTexture(
                size=[width, height, 1],
                format=self.select_format,
                usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
                label="select",
            )
            self.select_depth_texture = device.createTexture(
                size=[width, height, 1],
                format=self.depth_format,
                usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
                label="select_depth",
            )
            self.select_texture_view = self.select_texture.createView()

            self.width = width
            self.height = height

        for func in self._on_resize_callbacks:
            func()

    def color_attachments(self, loadOp: LoadOp):
        have_multisample = self.multisample.count > 1
        return [
            RenderPassColorAttachment(
                view=(
                    self.multisample_texture.createView()
                    if have_multisample
                    else self.target_texture_view
                ),
                resolveTarget=self.target_texture_view if have_multisample else None,
                clearValue=Color(1, 1, 1, 1),
                loadOp=loadOp,
            ),
        ]

    def select_attachments(self, loadOp: LoadOp):
        return [
            RenderPassColorAttachment(
                view=self.select_texture_view,
                clearValue=Color(0, 0, 0, 0),
                loadOp=loadOp,
            ),
        ]

    def select_depth_stencil_attachment(self, loadOp: LoadOp):
        return RenderPassDepthStencilAttachment(
            self.select_depth_texture.createView(),
            depthClearValue=1.0,
            depthLoadOp=loadOp,
        )

    def depth_stencil_attachment(self, loadOp: LoadOp):
        return RenderPassDepthStencilAttachment(
            self.depth_texture.createView(),
            depthClearValue=1.0,
            depthLoadOp=loadOp,
        )
