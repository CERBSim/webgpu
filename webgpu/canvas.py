from typing import Callable

from .input_handler import InputHandler
from .utils import get_device
from .webgpu_api import *
from . import platform


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
        self.depth_format = TextureFormat.depth24plus
        self.input_handler = InputHandler(canvas)

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

        for func in self._on_resize_callbacks:
            func()

    def color_attachments(self, loadOp: LoadOp):
        return [
            RenderPassColorAttachment(
                view=self.multisample_texture.createView(),
                resolveTarget=self.target_texture_view,
                # view=self.context.getCurrentTexture().createView(),
                clearValue=Color(1, 1, 1, 1),
                loadOp=loadOp,
            )
        ]

    def depth_stencil_attachment(self, loadOp: LoadOp):
        return RenderPassDepthStencilAttachment(
            self.depth_texture.createView(),
            depthClearValue=1.0,
            depthLoadOp=loadOp,
        )

    def __del__(self):
        # unregister is needed to remove circular references
        self.input_handler.unregister_callbacks()
