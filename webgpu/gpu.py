from .light import Light
from .colormap import Colormap
from .camera import Camera
from .input_handler import InputHandler
from .uniforms import (
    ClippingUniforms,
    MeshUniforms,
)
from .utils import BaseBinding, to_js
from .webgpu_api import *


async def init_webgpu(canvas):
    """Initialize WebGPU, create device and canvas"""
    import js

    adapter = await requestAdapter(powerPreference=PowerPreference.high_performance)

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

    device: Device
    depth_format: TextureFormat
    depth_texture: Texture
    multisample_texture: Texture
    multisample: MultisampleState

    def __init__(self, device, canvas, multisample_count=4):
        print("init gpu")
        import js

        self.render_function = None
        self.device = device
        self.format = js.navigator.gpu.getPreferredCanvasFormat()
        self.canvas = canvas

        print("canvas", canvas.width, canvas.height, canvas)

        self.u_clipping = ClippingUniforms(self.device)
        self.u_mesh = MeshUniforms(self.device)
        self.u_mesh.shrink = 0.5

        self.context = canvas.getContext("webgpu")
        self.context.configure(
            to_js(
                {
                    "device": device.handle,
                    "format": self.format,
                    "alphaMode": "premultiplied",
                    "sampleCount": multisample_count,
                }
            )
        )

        self.multisample_texture = device.createTexture(
            size=[canvas.width, canvas.height, 1],
            sampleCount=multisample_count,
            format=self.format,
            usage=TextureUsage.RENDER_ATTACHMENT,
        )
        self.multisample = MultisampleState(count=multisample_count)

        self.colormap = Colormap(device)
        self.light = Light(device)
        self.camera = Camera(device)
        self.depth_format = TextureFormat.depth24plus

        self.depth_texture = device.createTexture(
            size=[canvas.width, canvas.height, 1],
            format=self.depth_format,
            usage=js.GPUTextureUsage.RENDER_ATTACHMENT,
            label="depth_texture",
            sampleCount=multisample_count,
        )
        self.input_handler = InputHandler(canvas, self.camera.uniforms)

    def color_attachments(self, loadOp: LoadOp):
        return [
            RenderPassColorAttachment(
                view=self.multisample_texture.createView(),
                resolveTarget=self.context.getCurrentTexture().createView(),
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

    def update_uniforms(self):
        self.camera.uniforms.update_buffer()
        self.u_clipping.update_buffer()
        self.colormap.uniforms.update_buffer()
        self.u_mesh.update_buffer()

    def get_bindings(self):
        return [
            *self.u_clipping.get_bindings(),
            *self.u_mesh.get_bindings(),
            *self.camera.get_bindings(),
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
        del self.u_clipping
        del self.u_mesh
        del self.colormap
        del self.camera

        # unregister is needed to remove circular references
        self.input_handler.unregister_callbacks()
        # del self.input_handler
        # del self.depth_texture
        # del self.device


class RenderObject:
    """Base class for render objects"""

    gpu: WebGPU
    label: str = ""

    def __init__(self, gpu, label=None):
        self.gpu = gpu

        if label is None:
            self.label = self.__class__.__name__
        else:
            self.label = label

    def render(self, encoder: CommandEncoder) -> None:
        raise NotImplementedError

    def get_bindings(self) -> list[BaseBinding]:
        raise NotImplementedError

    def get_shader_code(self) -> str:
        raise NotImplementedError

    @property
    def device(self) -> Device:
        return self.gpu.device
