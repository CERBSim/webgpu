import ctypes as ct

import js

from .utils import UniformBinding, to_js


# These values must match the numbers defined in the shader
class Binding:
    UNIFORMS = 0
    COLORMAP_TEXTURE = 1
    COLORMAP_SAMPLER = 2

    EDGES = 4
    TRIGS = 5
    TRIG_FUNCTION_VALUES = 6
    VERTICES = 8
    INDEX = 9
    GBUFFERLAM = 10
    FONT_TEXTURE = 11


class ClippingPlaneUniform(ct.Structure):
    _fields_ = [("normal", ct.c_float * 3), ("dist", ct.c_float)]


class ComplexUniform(ct.Structure):
    _fields_ = [("re", ct.c_float), ("im", ct.c_float)]


class ColormapUniform(ct.Structure):
    _fields_ = [("min", ct.c_float), ("max", ct.c_float)]


class Uniforms(ct.Structure):
    """Uniforms class, derived from ctypes.Structure to ensure correct memory layout"""

    _fields_ = [
        ("mat", ct.c_float * 16),
        ("clipping_plane", ClippingPlaneUniform),
        ("colormap", ColormapUniform),
        ("scaling", ComplexUniform),
        ("aspect", ct.c_float),
        ("eval_mode", ct.c_uint32),
        ("do_clipping", ct.c_uint32),
        ("font_width", ct.c_uint32),
        ("font_height", ct.c_uint32),
        ("padding0", ct.c_uint32),
        ("padding1", ct.c_uint32),
        ("padding2", ct.c_uint32),
    ]

    def __init__(self, device):
        self.device = device
        self.do_clipping = 1
        self.clipping_plane.normal[0] = 1
        self.clipping_plane.normal[1] = 0
        self.clipping_plane.normal[2] = 0
        self.clipping_plane.dist = 1
        self.colormap.min = 0.0
        self.colormap.max = 1.0
        self.scaling.im = 0.0
        self.scaling.re = 0.0
        self.aspect = 0.0
        self.eval_mode = 0

        for i in range(16):
            self.mat[i] = 0.0

        uniforms_size = len(bytes(self))
        if uniforms_size % 16:
            raise ValueError(
                f"Uniforms size must be multiple of 16, current size: {uniforms_size}"
            )

        self.buffer = device.createBuffer(
            to_js(
                {
                    "size": len(bytes(self)),
                    "usage": js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST,
                }
            )
        )

    def get_bindings(self):
        return [UniformBinding(Binding.UNIFORMS, self.buffer)]

    def update_buffer(self):
        """Copy the current data to the GPU buffer"""
        data = js.Uint8Array.new(bytes(self))
        self.device.queue.writeBuffer(self.buffer, 0, data)

    def __del__(self):
        self.buffer.destroy()
