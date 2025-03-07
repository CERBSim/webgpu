from .input_handler import InputHandler
from .utils import get_device, to_js
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

    def __init__(self, device, canvas, multisample_count=4):

        self.render_function = None
        self.device = device
        self.format = js.navigator.gpu.getPreferredCanvasFormat()
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
        js.console.log("context", self.context)
        self.context.configure(
            {
                "device": device.handle,
                "format": self.format,
                "alphaMode": "premultiplied",
                "sampleCount": multisample_count,
                "usage": TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_DST,
            }
        )

        self.target_texture = device.createTexture(
            size=[canvas.width, canvas.height, 1],
            sampleCount=1,
            format=self.format,
            usage=TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC,
            label="target",
        )
        self.multisample_texture = device.createTexture(
            size=[canvas.width, canvas.height, 1],
            sampleCount=multisample_count,
            format=self.format,
            usage=TextureUsage.RENDER_ATTACHMENT,
            label="multisample",
        )
        self.multisample = MultisampleState(count=multisample_count)
        self.depth_format = TextureFormat.depth24plus

        self.depth_texture = device.createTexture(
            size=[canvas.width, canvas.height, 1],
            format=self.depth_format,
            usage=js.GPUTextureUsage.RENDER_ATTACHMENT,
            label="depth_texture",
            sampleCount=multisample_count,
        )
        self.input_handler = InputHandler(canvas)

        self.target_texture_view = self.target_texture.createView()

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
