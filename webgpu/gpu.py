from .colormap import Colormap
from .input_handler import InputHandler
from .uniforms import (
    ClippingUniforms,
    FontUniforms,
    FunctionUniforms,
    MeshUniforms,
    ViewUniforms,
)
from .utils import to_js, get_shader_code
from .webgpu_api import *


async def init_webgpu(canvas):
    """Initialize WebGPU, create device and canvas"""
    import js
    adapter = await requestAdapter(powerPreference=PowerPreference.low_power)

    required_features = []
    if "timestamp-query" in adapter.features:
        print("have timestamp query")
        required_features.append("timestamp-query")
    else:
        print("no timestamp query")

    one_meg = 1024**2
    one_gig = 1024**3
    device = await adapter.requestDevice(
        label="WebGPU device",
        requiredLimits=Limits(
            maxBufferSize=one_gig - 16,
            maxStorageBufferBindingSize=one_gig - 16,
        ),
    )
    limits = device.limits
    js.console.log("device limits\n", limits)
    js.console.log("adapter info\n", adapter.info)

    print(
        f"max storage buffer binding size {limits.maxStorageBufferBindingSize / one_meg:.2f} MB"
    )
    print(f"max buffer size {limits.maxBufferSize / one_meg:.2f} MB")

    return WebGPU(device, canvas)


class WebGPU:
    """WebGPU management class, handles "global" state, like device, canvas, frame/depth buffer, colormap and uniforms"""

    def __init__(self, device, canvas):
        import js
        self.render_function = None
        self.device = device
        self.format = js.navigator.gpu.getPreferredCanvasFormat()
        self.canvas = canvas

        print("canvas", canvas.width, canvas.height, canvas)
        self.shader_module = device.createShaderModule(get_shader_code())

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
        self.depth_format = TextureFormat.depth24plus
        self.depth_stencil = {
            "format": self.depth_format,
            "depthWriteEnabled": True,
            "depthCompare": "less",
        }

        self.depth_texture = device.createTexture(
            size=[canvas.width, canvas.height, 1],
            format=self.depth_format,
            usage=js.GPUTextureUsage.RENDER_ATTACHMENT,
            label="depth_texture",
        )
        self.input_handler = InputHandler(canvas, self.u_view)

    def color_attachments(self, loadOp: LoadOp):
        return [
            RenderPassColorAttachment(
                self.context.getCurrentTexture().createView(),
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

    def begin_render_pass(self, command_encoder: CommandEncoder, **kwargs):
        load_op = command_encoder.getLoadOp()

        render_pass_encoder = command_encoder.beginRenderPass(
            self.color_attachments(load_op),
            self.depth_stencil_attachment(load_op),
            **kwargs,
        )

        render_pass_encoder.setViewport(
            0, 0, self.canvas.width, self.canvas.height, 0.0, 1.0
        )

        return render_pass_encoder

    def __del__(self):
        print("destroy WebGPU")
        del self.u_view
        del self.u_clipping
        del self.u_font
        del self.u_function
        del self.u_mesh
        del self.colormap

        # unregister is needed to remove circular references
        self.input_handler.unregister_callbacks()
        # del self.input_handler
        # del self.depth_texture
        # del self.device
