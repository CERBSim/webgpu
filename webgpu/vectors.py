import numpy as np

from webgpu.uniforms import UniformBase, ct
from webgpu import read_shader_file, RenderObject, BufferBinding, Colormap
from webgpu.webgpu_api import PrimitiveTopology
from webgpu.utils import buffer_from_array


class Binding:
    POINTS = 81
    VECTORS = 82
    OPTIONS = 83


class VectorUniform(UniformBase):
    _binding = Binding.OPTIONS
    _fields_ = [
        ("size", ct.c_float),
        ("_padding1", ct.c_float),
        ("_padding2", ct.c_float),
        ("_padding3", ct.c_float),
    ]


class BaseVectorRenderObject(RenderObject):
    topology = PrimitiveTopology.triangle_strip
    n_vertices = 10

    def __init__(self, label="VectorField"):
        super().__init__(label=label)
        self.colormap = Colormap()

    def update(self):
        self.colormap.options = self.options
        self.colormap.update()

    def get_bindings(self):
        return [
            *self.options.camera.get_bindings(),
            *self.options.light.get_bindings(),
            BufferBinding(Binding.POINTS, self._buffers["points"]),
            BufferBinding(Binding.VECTORS, self._buffers["vectors"]),
            *self.vec_uniforms.get_bindings(),
            *self.colormap.get_bindings(),
        ]

    def create_vector_data(self):
        raise NotImplementedError

    def get_shader_code(self):
        shader_code = read_shader_file("vector.wgsl", __file__)
        shader_code += self.options.camera.get_shader_code()
        shader_code += self.options.light.get_shader_code()
        shader_code += self.colormap.get_shader_code()
        return shader_code

    def render(self, encoder):
        super().render(encoder)


class VectorRenderer(BaseVectorRenderObject):
    def __init__(self, points, vectors, size=None):
        super().__init__(label="VectorField")
        self.points = np.asarray(points, dtype=np.float32).reshape(-1)
        self.vectors = np.asarray(vectors, dtype=np.float32).reshape(-1)
        self.bounding_box = self.points.reshape(-1, 3).min(axis=0), self.points.reshape(
            -1, 3
        ).max(axis=0)
        self.size = size or 1 / 10 * np.linalg.norm(
            self.bounding_box[1] - self.bounding_box[0]
        )

    def update(self):
        super().update()
        self._buffers = {
            "points": buffer_from_array(self.points),
            "vectors": buffer_from_array(self.vectors),
        }
        self.vec_uniforms = VectorUniform(self.device)
        self.vec_uniforms.size = self.size
        self.vec_uniforms.update_buffer()
        min_vec, max_vec = (
            np.linalg.norm(self.vectors.reshape(-1, 3), axis=1).min(),
            np.linalg.norm(self.vectors.reshape(-1, 3), axis=1).max(),
        )
        if self.colormap.autoupdate:
            self.colormap.set_min_max(min_vec, max_vec, set_autoupdate=False)
        self.colormap.update()
        self.n_instances = len(self.points) // 3
        self.create_render_pipeline()

    def get_bounding_box(self):
        return self.bounding_box
