from base64 import b64decode
from typing import Callable
import threading
import time
import functools
import pathlib

from . import platform
from .utils import get_device, read_texture
from .webgpu_api import *

_TARGET_FPS = 60


def debounce(arg=None):
    def decorator(func):
        if platform.is_pyodide:
            return arg

        # Render only once every 1/_TARGET_FPS seconds
        @functools.wraps(func)
        def debounced(*args, **kwargs):
            if debounced.timer is not None:
                # we already have a render scheduled, so do nothing
                return

            def f():
                # clear the timer, so we can schedule a new one with the next function call
                t = time.time()
                func(*args, **kwargs)
                debounced.timer = None
                debounced.t_last = t

            t_wait = max(1 / target_fps - (time.time() - debounced.t_last), 0)
            debounced.timer = threading.Timer(t_wait, f)
            debounced.timer.start()

        debounced.timer = None
        debounced.t_last = time.time()
        return debounced

    if callable(arg):
        target_fps = _TARGET_FPS
        return decorator(arg)
    else:
        target_fps = arg
        return decorator


def init_webgpu(html_canvas):
    """Initialize WebGPU, create device and canvas"""
    device = get_device()
    return Canvas(device, html_canvas)


class Canvas:
    """Canvas management class, handles "global" state, like webgpu device, canvas, frame and depth buffer"""

    device: Device
    depth_format: TextureFormat
    depth_texture: Texture
    multisample_texture: Texture
    multisample: MultisampleState

    width: int = 0
    height: int = 0

    _on_resize_callbacks: list[Callable]
    _on_update_html_canvas: list[Callable]

    def __init__(self, device, canvas, multisample_count=4):
        self._update_mutex = threading.Lock()

        self._on_resize_callbacks = []
        self._on_update_html_canvas = []

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
        if self._resize_observer is not None:
            self._resize_observer.disconnect()
        if self._intersection_observer is not None:
            self._intersection_observer.disconnect()

    def update_html_canvas(self, html_canvas):
        """Reconfigure the canvas with the current HTML canvas element. This is necessary when the HTML canvas element changes, disappears (e.g. when switching a tab) and appears again."""

        self.width = self.height = 0  # disable rendering until resize is called

        with self._update_mutex:
            if self.context is not None:
                self.context.unconfigure()

            self.canvas = html_canvas
            self.context = html_canvas.getContext("webgpu")
            self.context.configure(
                {
                    "device": self.device.handle,
                    "format": self.format,
                    "alphaMode": "premultiplied",
                    "sampleCount": self.multisample.count,
                    "usage": TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_DST,
                }
            )

            def on_resize(*args):
                self.resize()

            def on_intersection(observer_entry, args):
                if observer_entry[0].isIntersecting:
                    for func in self._on_resize_callbacks:
                        func()

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

    @debounce(5)
    def resize(self):
        with self._update_mutex:
            canvas = self.canvas
            rect = canvas.getBoundingClientRect()
            width = int(rect.width)
            height = int(rect.height)

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
