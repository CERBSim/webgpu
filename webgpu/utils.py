import base64
import zlib
from pathlib import Path

from . import webgpu_api as wgpu
from .webgpu_api import *
from .webgpu_api import toJS as to_js

def read_shader_file(file_name, module_file) -> str:
    shader_dir = Path(module_file).parent / "shaders"
    return (shader_dir / file_name).read_text()

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


class TimeQuery:
    def __init__(self, device):
        self.device = device
        self.query_set = self.device.createQuerySet(
            to_js({"type": "timestamp", "count": 2})
        )
        self.buffer = self.device.createBuffer(
            size=16,
            usage=BufferUsage.COPY_DST | BufferUsage.MAP_READ,
        )


def reload_package(package_name):
    """Reload python package and all submodules (searches in modules for references to other submodules)"""
    import importlib
    import os
    import types

    package = importlib.import_module(package_name)
    assert hasattr(package, "__package__")
    file_name = package.__file__
    package_dir = os.path.dirname(file_name) + os.sep
    reloaded_modules = {file_name: package}

    def reload_recursive(module):
        module = importlib.reload(module)

        for var in vars(module).values():
            if isinstance(var, types.ModuleType):
                file_name = getattr(var, "__file__", None)
                if file_name is not None and file_name.startswith(package_dir):
                    if file_name not in reloaded_modules:
                        reloaded_modules[file_name] = reload_recursive(var)

        return module

    reload_recursive(package)
    return reloaded_modules


