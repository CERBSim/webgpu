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
        self.direction = (0.5, 0.5, 1.5)
        self.ambient = 0.3
        self.diffuse = 0.7
        self.specular = 0.3
        self.shininess = 10.0
        self.uniforms = None

    def __getstate__(self):
        state = {
            "direction": self.direction,
            "ambient": self.ambient,
            "diffuse": self.diffuse,
            "specular": self.specular,
            "shininess": self.shininess,
        }
        return state

    def __setstate__(self, state):
        self.direction = state["direction"]
        self.ambient = state["ambient"]
        self.diffuse = state["diffuse"]
        self.specular = state["specular"]
        self.shininess = state["shininess"]
        self.uniforms = None

    def update(self, options):
        if self.uniforms is None:
            self.uniforms = LightUniforms()
        self.uniforms.direction = self.direction
        self.uniforms.ambient = self.ambient
        self.uniforms.diffuse = self.diffuse
        self.uniforms.specular = self.specular
        self.uniforms.shininess = self.shininess
        self._update_uniforms()

    def get_bindings(self):
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("light.wgsl")

    def _update_uniforms(self):
        self.uniforms.update_buffer()
