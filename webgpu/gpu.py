import sys

import js

from . import utils
from .colormap import Colormap
from .input_handler import InputHandler
from .uniforms import (
    ClippingUniforms,
    FontUniforms,
    FunctionUniforms,
    MeshUniforms,
    ViewUniforms,
)
from .utils import to_js
from .webgpu_api import *


async def init_webgpu(canvas):
    """Initialize WebGPU, create device and canvas"""
    adapter = await requestAdapter(powerPreference=PowerPreference.low_power)

    required_features = []
    if adapter.features.has("timestamp-query"):
        print("have timestamp query")
        required_features.append("timestamp-query")
    else:
        print("no timestamp query")

    one_meg = 1024**2
    one_gig = 1024**3
    device = await adapter.requestDevice(
        requiredLimits=Limits(
            maxBufferSize=one_gig - 16,
            maxStorageBufferBindingSize=one_gig - 16,
        )
    )
    js.console.log("device limits\n", device.limits)
    js.console.log("adapter info\n", adapter.info)

    print(
        "max storage buffer binding size",
        device.limits.maxStorageBufferBindingSize / one_meg,
    )
    print("max buffer size", device.limits.maxBufferSize / one_meg)

    return WebGPU(utils.Device(device.handle), canvas)


class WebGPU:
    """WebGPU management class, handles "global" state, like device, canvas, frame/depth buffer, colormap and uniforms"""

    def __init__(self, device, canvas):
        self._is_first_render_pass = True
        self.render_function = None
        self.native_device = device
        self.device = device
        self.format = js.navigator.gpu.getPreferredCanvasFormat()
        self.canvas = canvas

        print("canvas", canvas.width, canvas.height, canvas)

        self.u_clipping = ClippingUniforms(self.device)
        self.u_view = ViewUniforms(self.device)
        self.u_font = FontUniforms(self.device)
        self.u_function = FunctionUniforms(self.device)
        self.u_mesh = MeshUniforms(self.device)
        self.u_mesh.shrink = 0.5

        self.context = canvas.getContext("webgpu")
        self.context.configure(
            to_js(
                {
                    "device": device.handle,
                    "format": self.format,
                    "alphaMode": "premultiplied",
                }
            )
        )
        self.colormap = Colormap(device)
        self.depth_format = "depth24plus"
        self.depth_stencil = {
            "format": self.depth_format,
            "depthWriteEnabled": True,
            "depthCompare": "less",
        }

        self.depth_texture = device.createTexture(
            size=[canvas.width, canvas.height, 1],
            format=self.depth_format,
            usage=js.GPUTextureUsage.RENDER_ATTACHMENT,
        )
        self.input_handler = InputHandler(canvas, self.u_view)

    def update_uniforms(self):
        self.u_view.update_buffer()
        self.u_clipping.update_buffer()
        self.u_font.update_buffer()
        self.u_function.update_buffer()
        self.u_mesh.update_buffer()

    def get_bindings(self):
        return [
            *self.u_view.get_bindings(),
            *self.u_clipping.get_bindings(),
            *self.u_font.get_bindings(),
            *self.u_function.get_bindings(),
            *self.u_mesh.get_bindings(),
            *self.colormap.get_bindings(),
        ]

    def begin_render_pass(self, command_encoder, args={}):
        load_op = "clear" if self._is_first_render_pass else "load"
        options = {
            "colorAttachments": [
                {
                    "view": self.context.getCurrentTexture().createView(),
                    "clearValue": {"r": 1, "g": 1, "b": 1, "a": 1},
                    "loadOp": load_op,
                    "storeOp": "store",
                }
            ],
            "depthStencilAttachment": {
                "view": self.depth_texture.createView(
                    to_js({"format": self.depth_format, "aspect": "all"})
                ),
                "depthLoadOp": load_op,
                "depthStoreOp": "store",
                "depthClearValue": 1.0,
            },
        } | args

        render_pass_encoder = command_encoder.beginRenderPass(to_js(options))
        render_pass_encoder.setViewport(
            0, 0, self.canvas.width, self.canvas.height, 0.0, 1.0
        )
        self._is_first_render_pass = False
        return render_pass_encoder

    def create_command_encoder(self):
        self._is_first_render_pass = True
        return self.native_device.createCommandEncoder()

    def __del__(self):
        self.depth_texture.destroy()
        del self.u_view
        del self.u_clipping
        del self.u_font
        del self.u_function
        del self.u_mesh
        del self.colormap

        # unregister is needed to remove circular references
        self.input_handler.unregister_callbacks()
        del self.input_handler
