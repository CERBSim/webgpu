"""
Simple shapes (cylinder, cone, circle) generation and render objects
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .renderer import RenderOptions, Renderer
from .colormap import Colormap
from .utils import (
    BufferBinding,
    buffer_from_array,
    read_shader_file,
)
from .webgpu_api import (
    BufferUsage,
    CommandEncoder,
    IndexFormat,
    ShaderStage,
    VertexAttribute,
    VertexBufferLayout,
    VertexFormat,
)


@dataclass
class ShapeData:
    vertices: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray

    _buffers: dict = field(default_factory=dict)

    def create_buffers(self):
        vertex_data = np.concatenate((self.vertices, self.normals), axis=1)
        self._buffers = {
            "vertex_data": buffer_from_array(
                np.array(vertex_data, dtype=np.float32),
                usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
                label="vertex_data",
            ),
            "triangles": buffer_from_array(
                np.array(self.triangles, dtype=np.uint32),
                label="triangles",
                usage=BufferUsage.INDEX | BufferUsage.COPY_DST,
            ),
        }
        return self._buffers


def generate_circle(n, radius: float = 1.0):
    angles = np.linspace(0, 2 * math.pi, n + 1)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius
    z = np.zeros_like(x)

    vertices = np.column_stack((x, y, z))
    normals = np.zeros((n, 3))
    normals[:, 2] = 1

    triangles = np.zeros((n, 3), dtype=np.uint32)

    for i in range(n - 2):
        next_i = (i + 1) % n
        triangles[i] = [i, next_i, n - 1]

    return ShapeData(
        vertices,
        normals,
        triangles,
    )


def generate_cylinder(
    n: int,
    radius: float = 1.0,
    height: float = 1.0,
    include_top=False,
    include_bottom=False,
    radius_top=None,
):
    if radius_top is None:
        radius_top = radius

    circle_bot = generate_circle(n, radius)
    circle_top = generate_circle(n, radius_top)
    circle_top.vertices[:, 2] = height

    vertices = np.concatenate((circle_bot.vertices, circle_top.vertices), axis=0)

    normals = height * circle_bot.vertices
    normals[:, 2] = radius_top - radius
    # normals = np.linalg.norm(normals, axis=1)
    normals = np.concatenate((normals, normals), axis=0)

    triangles = []
    for i in range(n + 1):
        next_i = (i + 1) % n
        triangles.append([i, next_i, i + n + 1])
        triangles.append([next_i, next_i + n + 1, i + n + 1])

    triangles = np.array(triangles, dtype=np.uint32)

    if include_bottom:
        n0 = vertices.shape[0]
        vertices = np.concatenate((vertices, v0), axis=0)
        normals = np.concatenate((vertices, -1 * circle.normals), axis=0)
        triangles = np.concatenate(triangles, n0 + circle.triangles, axis=0)

    if include_top:
        n0 = vertices.shape[0]
        vertices = np.concatenate((vertices, v1), axis=0)
        normals = np.concatenate((vertices, circle.normals), axis=0)
        triangles = np.concatenate(triangles, n0 + circle.triangles, axis=0)

    return ShapeData(
        vertices,
        normals,
        triangles,
    )


def generate_cone(n, radius=1, height=1, include_bottom=False):
    return generate_cylinder(n, radius, height, include_top=False, include_bottom=include_bottom)


_VALUES_BINDING_NUMBER = 101
_POSITIONS_BINDING_NUMBER = 102


class ShapeRenderer(Renderer):
    def __init__(
        self, shape_data: ShapeData, positions: np.ndarray, values: np.ndarray, label=None
    ):
        super().__init__(label=label)
        self.colormap = Colormap()
        self.positions = np.array(positions, dtype=np.float32)
        self.values = np.array(values, dtype=np.float32)
        self.shape_data = shape_data
        self.n_vertices = shape_data.triangles.size
        self.n_instances = self.positions.size // 6
        self.vertex_entry_point = "cylinder_vertex_main"
        self.fragment_entry_point = "shape_fragment_main"

    def update(self, options: RenderOptions):
        self.colormap.update(options)
        buffers = self.shape_data.create_buffers()
        self.vertex_buffer = buffers["vertex_data"]
        self.triangle_buffer = buffers["triangles"]
        self.positions_buffer = buffer_from_array(self.positions, label="positions")
        self.values_buffer = buffer_from_array(self.values, label="values")
        self.vertex_buffer_layouts = [
            VertexBufferLayout(
                arrayStride=2 * 3 * 4,
                attributes=[
                    VertexAttribute(
                        format=VertexFormat.float32x3,
                        offset=0,
                        shaderLocation=0,
                    ),
                    VertexAttribute(
                        format=VertexFormat.float32x3,
                        offset=3 * 4,
                        shaderLocation=1,
                    ),
                ],
            )
        ]

    def get_shader_code(self) -> str:
        return read_shader_file("shapes.wgsl")

    def get_bindings(self, options):
        return [
            *options.get_bindings(),
            *self.colormap.get_bindings(options),
            BufferBinding(
                _VALUES_BINDING_NUMBER,
                self.values_buffer,
                visibility=ShaderStage.VERTEX,
            ),
            BufferBinding(
                _POSITIONS_BINDING_NUMBER,
                self.positions_buffer,
                visibility=ShaderStage.VERTEX,
            ),
        ]

    def render(self, options: RenderOptions) -> None:
        render_pass = options.begin_render_pass()
        render_pass.setPipeline(self.pipeline)
        render_pass.setBindGroup(0, self.group)
        render_pass.setVertexBuffer(0, self.vertex_buffer)
        render_pass.setIndexBuffer(self.triangle_buffer, IndexFormat.uint32)
        render_pass.drawIndexed(
            self.n_vertices,
            self.n_instances,
        )
        render_pass.end()
