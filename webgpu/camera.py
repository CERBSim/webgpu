from .uniforms import BaseBinding, UniformBase, ct, Binding
from .utils import read_shader_file


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


class Camera:
    def __init__(self, device):
        self.device = device
        self.uniforms = CameraUniforms(device)

    def get_bindings(self) -> list[BaseBinding]:
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("camera.wgsl", __file__)

    def __del__(self):
        del self.uniforms
