"""Python equivalents to all uniforms defined in shader code

The UniformBase class are derived from ctypes.Structure to ensure correct memory layout.

CAUTION: 
- The Binding numbers must match the numbers defined in the shader code. 
- Uniforms structs must match exactly the memory layout defined in the shader code.
- The size of each struct must be a multiple of 16 bytes.
"""

import ctypes as ct

from .utils import UniformBinding
from .webgpu_api import BufferUsage, Device


class Binding:
    """Binding numbers for uniforms in shader code in uniforms.wgsl"""

    CAMERA = 0
    CLIPPING = 1
    FONT = 2
    FONT_TEXTURE = 3
    FUNCTION = 5
    COLORMAP_TEXTURE = 6
    COLORMAP_SAMPLER = 7

    EDGES = 8
    TRIGS = 9
    TRIG_FUNCTION_VALUES = 10
    SEG_FUNCTION_VALUES = 11
    VERTICES = 12
    TRIGS_INDEX = 13
    GBUFFERLAM = 14

    MESH = 20
    EDGE = 21
    SEG = 22
    TRIG = 23
    QUAD = 24
    TET = 25
    PYRAMID = 26
    PRISM = 27
    HEX = 28

    LINE_INTEGRAL_CONVOLUTION = 40
    LINE_INTEGRAL_CONVOLUTION_INPUT_TEXTURE = 41
    LINE_INTEGRAL_CONVOLUTION_OUTPUT_TEXTURE = 42


class UniformBase(ct.Structure):

    def __init__(self, device: Device, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self._buffer = device.createBuffer(
            size=len(bytes(self)),
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
        )

        size = len(bytes(self))
        if size % 16:
            raise ValueError(
                f"Size of type {type(self)} must be multiple of 16, current size: {size}"
            )

    def update_buffer(self):
        self.device.queue.writeBuffer(self._buffer, 0, bytes(self))

    def get_bindings(self) -> list[UniformBinding]:
        return [UniformBinding(self._binding, self._buffer)]

    def __del__(self):
        self._buffer.destroy()


class CameraUniforms(UniformBase):
    """Uniforms class, derived from ctypes.Structure to ensure correct memory layout"""

    _binding = Binding.CAMERA

    _fields_ = [
        ("model_view", ct.c_float * 16),
        ("model_view_projection", ct.c_float * 16),
        ("normal_mat", ct.c_float * 16),
        ("aspect", ct.c_float),
        ("padding", ct.c_uint32 * 3),
    ]


class ClippingUniforms(UniformBase):
    _binding = Binding.CLIPPING
    _fields_ = [
        ("plane", ct.c_float * 4),
        ("sphere", ct.c_float * 4),
        ("mode", ct.c_uint32),
        ("padding", ct.c_uint32 * 3),
    ]

    def __init__(self, device, mode=0, **kwargs):
        super().__init__(device, mode=mode, **kwargs)


class FunctionUniforms(UniformBase):
    _binding = Binding.FUNCTION
    _fields_ = [("min", ct.c_float), ("max", ct.c_float), ("padding", ct.c_float * 2)]


class FontUniforms(UniformBase):
    _binding = Binding.FONT
    _fields_ = [
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("padding", ct.c_uint32 * 2),
    ]


class MeshUniforms(UniformBase):
    _binding = Binding.MESH
    _fields_ = [
        ("subdivision", ct.c_uint32),
        ("shrink", ct.c_float),
        ("padding", ct.c_float * 2),
    ]

    def __init__(self, device, subdivision=1, shrink=1.0, **kwargs):
        super().__init__(device, subdivision=subdivision, shrink=shrink, **kwargs)


class LineIntegralConvolutionUniforms(UniformBase):
    _binding = Binding.LINE_INTEGRAL_CONVOLUTION
    _fields_ = [
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("kernel_length", ct.c_uint32),
        ("oriented", ct.c_uint32),
        ("thickness", ct.c_uint32),
        ("padding", ct.c_float * 3),
    ]

    def __init__(self, device, kernel_length=25, oriented=0, thickness=5, **kwargs):
        super().__init__(
            device,
            kernel_length=kernel_length,
            oriented=oriented,
            thickness=thickness,
            **kwargs,
        )
