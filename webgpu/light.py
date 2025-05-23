from .utils import read_shader_file
from .uniforms import UniformBase, Binding, ct


class LightUniforms(UniformBase):
    """Uniforms class for light settings, derived from ctypes.Structure to ensure correct memory layout"""

    _binding = Binding.LIGHT

    _fields_ = [
        ("direction", ct.c_float * 3),
        ("ambient", ct.c_float),
        ("diffuse", ct.c_float),
        ("specular", ct.c_float),
        ("shininess", ct.c_float),
        ("padding", ct.c_uint32),
    ]


class Light:
    def __init__(self):
        self.uniforms = LightUniforms()
        self.uniforms.direction = (0.5, 0.5, 1.5)
        self.uniforms.ambient = 0.3
        self.uniforms.diffuse = 0.7
        self.uniforms.specular = 0.3
        self.uniforms.shininess = 10.0

    def get_bindings(self):
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("light.wgsl")

    def _update_uniforms(self):
        self.uniforms.update_buffer()
