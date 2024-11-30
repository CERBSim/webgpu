import base64
import zlib

from . import webgpu_api as wgpu
from .shader import get_shader_code
from .webgpu_api import *
from .webgpu_api import toJS as to_js


def encode_bytes(data: bytes) -> str:
    if data == b"":
        return ""
    return base64.b64encode(zlib.compress(data)).decode("utf-8")


def decode_bytes(data: str) -> bytes:
    if data == "":
        return b""
    return zlib.decompress(base64.b64decode(data.encode()))


class BaseBinding:
    """Base class for any object that has a binding number (uniform, storage buffer, texture etc.)"""

    def __init__(
        self,
        nr,
        visibility=ShaderStage.ALL,
        resource=None,
        layout=None,
        binding=None,
    ):
        self.nr = nr
        self.visibility = visibility
        self._layout_data = layout or {}
        self._binding_data = binding or {}
        self._resource = resource or {}

    @property
    def layout(self):
        return {
            "binding": self.nr,
            "visibility": self.visibility,
        } | self._layout_data

    @property
    def binding(self):
        return {
            "binding": self.nr,
            "resource": self._resource,
        }


class UniformBinding(BaseBinding):
    def __init__(self, nr, buffer, visibility=ShaderStage.ALL):
        super().__init__(
            nr=nr,
            visibility=visibility,
            layout={"buffer": {"type": "uniform"}},
            resource={"buffer": buffer},
        )


class StorageTextureBinding(BaseBinding):
    def __init__(
        self,
        nr,
        texture,
        visibility=ShaderStage.COMPUTE,
        dim=2,
        access="write-only",
    ):
        super().__init__(
            nr=nr,
            visibility=visibility,
            layout={
                "storageTexture": {
                    "access": access,
                    "format": texture.format,
                    "viewDimension": f"{dim}d",
                }
            },
            resource=texture.createView(),
        )


class TextureBinding(BaseBinding):
    def __init__(
        self,
        nr,
        texture,
        visibility=ShaderStage.FRAGMENT,
        sample_type="float",
        dim=1,
        multisamples=False,
    ):
        super().__init__(
            nr=nr,
            visibility=visibility,
            layout={
                "texture": {
                    "sampleType": sample_type,
                    "viewDimension": f"{dim}d",
                    "multisamples": multisamples,
                }
            },
            resource=texture.createView(),
        )


class SamplerBinding(BaseBinding):
    def __init__(self, nr, sampler, visibility=ShaderStage.FRAGMENT):
        super().__init__(
            nr=nr,
            visibility=visibility,
            layout={"sampler": {"type": "filtering"}},
            resource=sampler,
        )


class BufferBinding(BaseBinding):
    def __init__(self, nr, buffer, read_only=True, visibility=ShaderStage.ALL):
        type_ = "read-only-storage" if read_only else "storage"
        super().__init__(
            nr=nr,
            visibility=visibility,
            layout={"buffer": {"type": type_}},
            resource={"buffer": buffer},
        )


def create_bind_group(device, bindings: list, label=""):
    """creates bind group layout and bind group from a list of BaseBinding objects"""
    layouts = []
    resources = []
    for binding in bindings:
        layouts.append(BindGroupLayoutEntry(**binding.layout))
        resources.append(BindGroupEntry(**binding.binding))

    layout = device.createBindGroupLayout(entries=layouts, label=label)
    group = device.createBindGroup(
        label=label,
        layout=layout,
        entries=resources,
    )
    return layout, group


class Device(wgpu.Device):
    """Helper class to wrap device functions"""

    @property
    def shader_module(self):
        code = get_shader_code()
        return self.createShaderModule(code)

    def create_pipeline_layout(self, binding_layout, label=""):
        return self.handle.createPipelineLayout(
            to_js({"label": label, "bindGroupLayouts": [binding_layout]})
        )

    def create_render_pipeline(self, binding_layout, options: dict):
        options["layout"] = self.create_pipeline_layout(
            binding_layout, label=options.get("label", "")
        )
        return self.handle.createRenderPipeline(to_js(options))

    def create_compute_pipeline(self, binding_layout, options: dict):
        options["layout"] = self.create_pipeline_layout(
            binding_layout, label=options.get("label", "")
        )
        return self.handle.createComputePipeline(to_js(options))

    def create_buffer(self, size_or_data: int | bytes, usage=None):
        import js

        usage = (
            js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST
            if usage is None
            else usage
        )
        if isinstance(size_or_data, int):
            size = size_or_data
            data = None
        else:
            size = len(size_or_data)
            data = size_or_data
        buffer = self.handle.createBuffer(size=size, usage=usage)
        if data is not None:
            self.handle.queue.writeBuffer(buffer, 0, js.Uint8Array.new(data))
        return buffer


class TimeQuery:
    def __init__(self, device):
        import js

        self.device = device
        self.query_set = self.device.createQuerySet(
            to_js({"type": "timestamp", "count": 2})
        )
        self.buffer = self.device.createBuffer(
            size=16,
            usage=BufferUsage.COPY_DST | BufferUsage.MAP_READ,
        )
