import base64
import zlib
from pathlib import Path

from .shader import get_shader_code


def encode_bytes(data: bytes) -> str:
    if data == b"":
        return ""
    return base64.b64encode(zlib.compress(data)).decode()


def decode_bytes(data: str) -> bytes:
    if data == "":
        return b""
    return zlib.decompress(base64.b64decode(data.encode()))


class ShaderStage:
    VERTEX = 0x1
    FRAGMENT = 0x2
    COMPUTE = 0x4
    ALL = VERTEX | FRAGMENT | COMPUTE


def to_js(value):
    import js
    from pyodide.ffi import to_js as _to_js

    return _to_js(value, dict_converter=js.Object.fromEntries)


# any object that has a binding number (uniform, storage buffer, texture etc.)
class BaseBinding:
    def __init__(
        self, nr, visibility=ShaderStage.ALL, resource=None, layout=None, binding=None
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


class Device:
    """Helper class to wrap device functions"""

    def __init__(self, device):
        self.device = device

    def create_bind_group(self, bindings: list, label=""):
        """creates bind group layout and bind group from a list of BaseBinding objects"""
        layouts = []
        resources = []
        for binding in bindings:
            layouts.append(binding.layout)
            resources.append(binding.binding)

        layout = self.device.createBindGroupLayout(to_js({"entries": layouts}))
        group = self.device.createBindGroup(
            to_js(
                {
                    "label": label,
                    "layout": layout,
                    "entries": resources,
                }
            )
        )
        return layout, group

    def create_pipeline_layout(self, binding_layout, label=""):
        return self.device.createPipelineLayout(
            to_js({"label": label, "bindGroupLayouts": [binding_layout]})
        )

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
        buffer = self.device.createBuffer(to_js({"size": size, "usage": usage}))
        if data is not None:
            self.device.queue.writeBuffer(buffer, 0, js.Uint8Array.new(data))
        return buffer

    def data_to_buffers(self, data: dict):
        buffers = {}
        for name, value in data.items():
            buffers[name] = self.create_buffer(value)
        return buffers

    def compile_files(self, *files):
        code = get_shader_code(files)
        return self.device.createShaderModule(to_js({"code": code}))
