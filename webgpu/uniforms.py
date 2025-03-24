"""Python equivalents to all uniforms defined in shader code

The UniformBase class are derived from ctypes.Structure to ensure correct memory layout.

CAUTION: 
- The Binding numbers must match the numbers defined in the shader code. 
- Uniforms structs must match exactly the memory layout defined in the shader code.
- The size of each struct must be a multiple of 16 bytes.
"""

import ctypes as ct

from .utils import BaseBinding, UniformBinding
from .webgpu_api import BufferUsage, Device


class Binding:
    """Binding numbers for uniforms in shader code in uniforms.wgsl"""

    CAMERA = 0
    CLIPPING = 1
    FONT = 2
    FONT_TEXTURE = 3
    COLORMAP = 5
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

    TEXT = 30

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
            label=type(self).__name__,
        )

        size = len(bytes(self))
        if size % 16:
            raise ValueError(
                f"Size of type {type(self)} must be multiple of 16, current size: {size}"
            )

    def update_buffer(self):
        self.device.queue.writeBuffer(self._buffer, 0, bytes(self))

    def get_bindings(self) -> list[BaseBinding]:
        return [UniformBinding(self._binding, self._buffer)]

    def __del__(self):
        self._buffer.destroy()


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
