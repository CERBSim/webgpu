import base64
import zlib
from pathlib import Path

from .webgpu_api import *
from .webgpu_api import toJS as to_js
from . import platform

_device: Device = None


def init_device_sync():
    global _device
    if _device is not None:
        return _device

    if not platform.js.navigator.gpu:
        platform.js.alert("WebGPU is not supported")
        sys.exit(1)

    reqAdapter = platform.js.navigator.gpu.requestAdapter
    options = RequestAdapterOptions(
        powerPreference=PowerPreference.low_power,
    ).toJS()
    adapter = reqAdapter(options)
    if not adapter:
        platform.js.alert("WebGPU is not supported")
        sys.exit(1)
    one_gig = 1024**3
    _device = Device(
        adapter.requestDevice(
            [],
            Limits(
                maxBufferSize=one_gig - 16,
                maxStorageBufferBindingSize=one_gig - 16,
            ),
            None,
            "WebGPU device",
        )
    )
    return _device


async def init_device() -> Device:
    global _device

    if _device is not None:
        return _device

    adapter = requestAdapter(powerPreference=PowerPreference.low_power)
    try:
        adapter = await adapter
    except:
        pass

    required_features = []
    if "timestamp-query" in adapter.features:
        print("have timestamp query")
        required_features.append("timestamp-query")
    else:
        print("no timestamp query")

    one_meg = 1024**2
    one_gig = 1024**3
    _device = adapter.requestDevice(
        label="WebGPU device",
        requiredLimits=Limits(
            maxBufferSize=one_gig - 16,
            maxStorageBufferBindingSize=one_gig - 16,
        ),
    )
    try:
        _device = await _device
    except:
        pass

    limits = _device.limits
    js.console.log("device limits\n", limits)
    js.console.log("adapter info\n", adapter.info)

    print(
        f"max storage buffer binding size {limits.maxStorageBufferBindingSize / one_meg:.2f} MB"
    )
    print(f"max buffer size {limits.maxBufferSize / one_meg:.2f} MB")

    return _device


def get_device() -> Device:
    if _device is None:
        raise RuntimeError("Device not initialized")
    return _device


class Pyodide:
    def __init__(self):
        pass

    def __setattr__(self, key, value):
        pass


def find_shader_file(file_name, module_file) -> Path:
    for path in [module_file, __file__]:
        file_path = Path(path).parent / "shaders" / file_name
        if file_path.exists():
            return file_path

    raise FileNotFoundError(f"Shader file {file_name} not found")


def read_shader_file(file_name, module_file) -> str:
    code = find_shader_file(file_name, module_file).read_text()
    if not "#import" in code:
        return code
    lines = code.split("\n")
    code = ""
    for line in lines:
        if line.startswith("#import"):
            imported_file = line.split()[1] + ".wgsl"
            code += f"// start file {imported_file}\n"
            code += read_shader_file(imported_file, module_file) + "\n"
            code += f"// end file {imported_file}\n"
        else:
            code += line + "\n"
    return code


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
        if not read_only:
            visibility = ShaderStage.COMPUTE
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


def run_compute_shader(
    encoder, code, bindings, n_workgroups, label="compute", entry_point="main"
):
    from webgpu.utils import create_bind_group, get_device

    device = get_device()

    shader_module = device.createShaderModule(code)

    layout, bind_group = create_bind_group(device, bindings, label)
    pipeline = device.createComputePipeline(
        device.createPipelineLayout([layout], label),
        ComputeState(
            shader_module,
            entry_point,
        ),
        label,
    )

    pass_encoder = encoder.beginComputePass()
    pass_encoder.setPipeline(pipeline)
    pass_encoder.setBindGroup(0, bind_group)
    pass_encoder.dispatchWorkgroups(*n_workgroups)
    pass_encoder.end()


def buffer_from_array(array, usage=BufferUsage.STORAGE | BufferUsage.COPY_DST):
    device = get_device()
    buffer = device.createBuffer(array.size * array.itemsize, usage=usage)
    device.queue.writeBuffer(buffer, 0, array.tobytes())
    return buffer


def uniform_from_array(array):
    return buffer_from_array(array, usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST)


class ReadBuffer:
    def __init__(self, buffer, encoder):
        self.buffer = buffer
        self.read_buffer = get_device().createBuffer(
            buffer.size, BufferUsage.MAP_READ | BufferUsage.COPY_DST
        )
        encoder.copyBufferToBuffer(self.buffer, 0, self.read_buffer, 0, buffer.size)

    def get_array(self, dtype):
        import numpy as np

        self.read_buffer.handle.mapAsync(MapMode.READ, 0, self.read_buffer.size)
        data = self.read_buffer.handle.getMappedRange(0, self.read_buffer.size)
        res = np.frombuffer(data, dtype=dtype)
        self.read_buffer.unmap()
        return res


def max_bounding_box(boxes):
    import numpy as np

    boxes = [b for b in boxes if b is not None]
    pmin = np.array(boxes[0][0])
    pmax = np.array(boxes[0][1])
    for b in boxes[1:]:
        pmin = np.minimum(pmin, np.array(b[0]))
        pmax = np.maximum(pmax, np.array(b[1]))
    return (pmin, pmax)


def format_number(n):
    if n == 0:
        return "0"
    abs_n = abs(n)
    # Use scientific notation for numbers smaller than 0.001 or larger than 9999
    if abs_n < 1e-2 or abs_n >= 1e3:
        return f"{n:.2e}"
    else:
        return f"{n:.3g}"
