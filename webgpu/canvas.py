from typing import Callable

from . import platform
from .utils import get_device
from .webgpu_api import *


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

    _on_resize_callbacks: list[Callable] = []

    def __init__(self, device, canvas, multisample_count=4):

        self._on_resize_callbacks = []

        self.device = device
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
        self.canvas = canvas

        self.context = canvas.getContext("webgpu")
        self.context.configure(
            toJS(
                {
                    "device": device.handle,
                    "format": self.format,
                    "alphaMode": "premultiplied",
                    "sampleCount": multisample_count,
                    "usage": TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_DST,
                }
            )
        )

        self.multisample = MultisampleState(count=multisample_count)

        self.resize()

        # platform.js.webgpuOnResize(canvas, create_proxy(self.resize, True))

    def on_resize(self, func: Callable):
        self._on_resize_callbacks.append(func)

    def resize(self, *args):
        canvas = self.canvas
        rect = canvas.getBoundingClientRect()
        width = int(rect.width)
        height = int(rect.height)

        if width == self.width and height == self.height:
            return False

        if width == 0 or height == 0:
            return False

        canvas.width = width
        canvas.height = height

        self.width = width
        self.height = height

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
