import numpy as np

from .render_object import RenderObject
from .utils import BufferBinding, buffer_from_array, read_shader_file


class Binding:
    VERTICES = 90
    NORMALS = 91
    INDICES = 92


class TriangulationRenderer(RenderObject):
    n_vertices: int = 3

    def __init__(
        self, points, normals=None, color=(0.0, 1.0, 0.0, 1.0), label="Triangulation"
    ):
        super().__init__(label=label)
        self.color = color
        self.points = np.asarray(points, dtype=np.float32).reshape(-1)
        assert len(self.points) % 9 == 0, "Invalid number of points"
        if normals is None:
            ps = self.points.reshape(-1, 3, 3)
            normals = np.cross((ps[:, 1] - ps[:, 0]), (ps[:, 2] - ps[:, 0]))
            normals = normals / np.linalg.norm(normals, axis=1)[:, None]
            self.normals = np.concatenate([normals, normals, normals], axis=1).flatten()
        else:
            self.normals = np.asarray(normals, dtype=np.float32).reshape(-1)
        ps = self.points.reshape(-1, 3)
        self._bounding_box = ps.min(axis=0), ps.max(axis=0)
        self.n_instances = len(self.points) // 9

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp

        self.point_buffer = buffer_from_array(self.points)
        self.normal_buffer = buffer_from_array(self.normals)
        self.create_render_pipeline()

    def get_bounding_box(self):
        return self._bounding_box

    def get_color_shader(self):
        return """
fn getColor(vertId: u32, trigId: u32) -> vec4f {{
  return vec4f{color};
}}""".format(
            color=self.color
        )

    def get_shader_code(self) -> str:
        return (
            read_shader_file("triangulation.wgsl", __file__)
            + self.options.camera.get_shader_code()
            + self.options.light.get_shader_code()
            + self.get_color_shader()
        )

    def get_bindings(self):
        return [
            *self.options.camera.get_bindings(),
            *self.options.light.get_bindings(),
            BufferBinding(Binding.VERTICES, self.point_buffer),
            BufferBinding(Binding.NORMALS, self.normal_buffer),
        ]
