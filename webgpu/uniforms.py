import ctypes as ct

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
    TRIGS_INDEX = 9
    GBUFFERLAM = 10
    FONT_TEXTURE = 11

    MESH_UNIFORMS = 20
    TET = 21
    PYRAMID = 22
    PRISM = 23
    HEX = 24


class UniformBase(ct.Structure):

    def __init__(self, device):
        import js

        self.device = device
        self._buffer = device.create_buffer(
            len(bytes(self)),
            js.GPUBufferUsage.UNIFORM | js.GPUBufferUsage.COPY_DST,
        )

        size = len(bytes(self))
        if size % 16:
            raise ValueError(
                f"Size of type {type(self)} must be multiple of 16, current size: {size}"
            )

    def update_buffer(self):
        self.device.write_buffer(self._buffer, bytes(self))

    def get_bindings(self):
        return [UniformBinding(self._binding, self._buffer)]

    def __del__(self):
        self._buffer.destroy()


class MeshUniforms(UniformBase):
    _binding = Binding.MESH_UNIFORMS
    _fields_ = [("shrink", ct.c_float), ("padding", ct.c_float * 3)]


class Uniforms(UniformBase):
    """Uniforms class, derived from ctypes.Structure to ensure correct memory layout"""

    _binding = Binding.UNIFORMS

    class ClippingPlaneUniform(ct.Structure):
        _fields_ = [("normal", ct.c_float * 3), ("dist", ct.c_float)]

    class ComplexUniform(ct.Structure):
        _fields_ = [("re", ct.c_float), ("im", ct.c_float)]

    class ColormapUniform(ct.Structure):
        _fields_ = [("min", ct.c_float), ("max", ct.c_float)]

    _fields_ = [
        ("model_view", ct.c_float * 16),
        ("model_view_projection", ct.c_float * 16),
        ("normal_mat", ct.c_float * 16),
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
        super().__init__(device)

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
