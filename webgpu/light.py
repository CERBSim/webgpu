from .utils import read_shader_file
from .uniforms import UniformBase, Binding, ct


class LightUniforms(UniformBase):
    """Uniforms class for light settings, derived from ctypes.Structure to ensure correct memory layout"""

    _binding = Binding.LIGHT

    _fields_ = [
        ("key", ct.c_float * 4),
        ("fill", ct.c_float * 4),
        ("ambient", ct.c_float),
        ("diffuse", ct.c_float),
        ("specular", ct.c_float),
        ("shininess", ct.c_float),
        ("wrap", ct.c_float),
        ("min_diffuse", ct.c_float),
        ("rim_strength", ct.c_float),
        ("rim_power", ct.c_float),
    ]


class Light:
    def __init__(self):
        self.key = [-0.6, 0.2, .8, 1.0]
        self.fill = [0.4, 0.6, 1.0, 0.25]
        self.key = [1, 3, 3, 1.0]
        self.fill = [-1, -2, 3, 0.5]
        self.ambient = 0.15 
        self.diffuse = 0.75 
        self.specular = 0.03
        self.shininess = 16

        self.wrap = 0.2
        self.min_diffuse = 0.2

        self.rim_strength = 0.2
        self.rim_power = 2

        self.uniforms = None

    def __getstate__(self):
        state = {
            "key": self.key,
            "fill": self.fill,
            "ambient": self.ambient,
            "specular": self.specular,
            "shininess": self.shininess,
            "wrap": self.wrap,
            "min_diffuse": self.min_diffuse,
            "rim_strength": self.rim_strength,
            "rim_power": self.rim_power,
        }
        return state

    def __setstate__(self, state):
        self.keys = state["key"]
        self.fill = state["fill"]
        self.ambient = state["ambient"]
        self.specular = state["specular"]
        self.shininess = state["shininess"]
        self.wrap = state.get("wrap", 0.3)
        self.min_diffuse = state.get("min_diffuse", 0.2)
        self.rim_strength = state.get("rim_strength", 0.1)
        self.rim_power = state.get("rim_power", 2)
        self.uniforms = None

    def update(self, options):
        if self.uniforms is None:
            self.uniforms = LightUniforms()
        self.uniforms.key = tuple(self.key)
        self.uniforms.fill = tuple(self.fill)
        self.uniforms.ambient = self.ambient
        self.uniforms.diffuse = self.diffuse
        self.uniforms.specular = self.specular
        self.uniforms.shininess = self.shininess
        self.uniforms.wrap = self.wrap
        self.uniforms.min_diffuse = self.min_diffuse
        self.uniforms.rim_strength = self.rim_strength
        self.uniforms.rim_power = self.rim_power
        self._update_uniforms()

    def get_bindings(self):
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("light.wgsl")

    def _update_uniforms(self):
        self.uniforms.update_buffer()
