"""
Simple shapes (cylinder, cone, circle) generation and render objects
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .colormap import Colormap
from .renderer import Renderer, RenderOptions
from .utils import (
    buffer_from_array,
    read_shader_file,
)
from .webgpu_api import (
    BufferUsage,
    IndexFormat,
    VertexAttribute,
    VertexBufferLayout,
    VertexFormat,
    VertexStepMode,
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

    def get_bounding_box(self):
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def move(self, v):
        self.vertices[:, 0] += v[0]
        self.vertices[:, 1] += v[1]
        self.vertices[:, 2] += v[2]
        return self

    def normalize_z(self):
        self.move([0, 0, -self.get_bounding_box()[0][2]])

    def __add__(self, other):
        if not isinstance(other, ShapeData):
            raise TypeError("Can only add ShapeData objects")

        return ShapeData(
            np.concatenate((self.vertices, other.vertices)),
            np.concatenate((self.normals, other.normals)),
            np.concatenate((self.triangles, other.triangles + self.vertices.shape[0])),
        )


def generate_circle(n, radius: float = 1.0) -> ShapeData:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
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
    top_face=False,
    bottom_face=False,
    radius_top=None,
):
    if radius_top is None:
        radius_top = radius

    circle_bot = generate_circle(n, radius)
    circle_top = generate_circle(n, radius_top).move([0, 0, height])

    vertices = np.concatenate((circle_bot.vertices, circle_top.vertices), axis=0)

    normals = height * circle_bot.vertices
    normals[:, 2] = radius - radius_top
    normals = np.concatenate((normals, normals), axis=0)

    triangles = []
    for i in range(n):
        next_i = (i + 1) % n
        triangles.append([i, next_i, next_i + n])
        triangles.append([i, next_i + n, i + n])

    triangles = np.array(triangles, dtype=np.uint32)

    if bottom_face:
        n0 = vertices.shape[0]
        vertices = np.concatenate((vertices, circle_bot.vertices))
        normals = np.concatenate((normals, -1 * circle_bot.normals))
        triangles = np.concatenate((triangles, n0 + circle_bot.triangles))

    if top_face:
        n0 = vertices.shape[0]
        vertices = np.concatenate((vertices, circle_top.vertices))
        normals = np.concatenate((normals, circle_top.normals))
        triangles = np.concatenate((triangles, n0 + circle_top.triangles))

    return ShapeData(
        vertices,
        normals,
        triangles,
    )


def generate_cone(n, radius=1, height=1, bottom_face=False):
    return generate_cylinder(
        n, radius, height, top_face=False, bottom_face=bottom_face, radius_top=0
    )


class ShapeRenderer(Renderer):
    def __init__(
        self,
        shape_data: ShapeData,
        positions: np.ndarray,
        directions: np.ndarray,
        values: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        label=None,
        colormap=None,
    ):
        if colors is None and values is None:
            raise ValueError("Either colors or values must be provided")

        super().__init__(label=label)

        self.colormap = colormap or Colormap()
        self.positions = np.array(positions, dtype=np.float32).reshape(-1)
        self.values = values and np.array(values, dtype=np.float32).reshape(-1)
        self.directions = np.array(directions, dtype=np.float32).reshape(-1)

        if colors:
            colors = np.array(colors, dtype=np.float32).reshape(-1)
            colors = np.array(np.round(255 * colors), dtype=np.uint8).flatten()
        self.colors = colors

        self.shape_data = shape_data

    def get_bindings(self):
        return self.colormap.get_bindings()

    def update(self, options: RenderOptions):
        self.n_vertices = self.shape_data.triangles.size
        self.n_instances = self.positions.size // 3
        self.colormap.update(options)
        buffers = self.shape_data.create_buffers()
        self.triangle_buffer = buffers["triangles"]
        positions_buffer = buffer_from_array(
            self.positions, label="positions", usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
        )
        directions_buffer = buffer_from_array(
            self.directions, label="directions", usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
        )

        self.vertex_entry_point = "shape_vertex_main"
        if self.colors is not None:
            itemsize = self.colors.itemsize * 4
            color_format = VertexFormat.unorm8x4
            n_colors = self.colors.size // 4
            self.fragment_entry_point = "shape_fragment_main_color"
            colors_buffer = buffer_from_array(
                self.colors, label="colors", usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
            )
        else:
            itemsize = self.values.itemsize
            color_format = VertexFormat.float32
            n_colors = self.values.size
            if self.colormap.autoscale:
                self.colormap.set_min_max(
                    self.values.min(), self.values.max(), set_autoscale=False
                )
            self.fragment_entry_point = "shape_fragment_main_value"
            colors_buffer = buffer_from_array(
                self.values, label="values", usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
            )

        if n_colors == self.n_instances:
            color_stride = itemsize
            color_top_offset = 0
        elif n_colors == 2 * self.n_instances:
            color_stride = 2 * itemsize
            color_top_offset = itemsize
        elif n_colors == 1:
            color_stride = 0
            color_top_offset = 0
        elif n_colors == 2:
            color_stride = 0
            color_top_offset = itemsize

        bmin, bmax = self.shape_data.get_bounding_box()
        z_range = [bmin[2], bmax[2]]
        total_height_buffer = buffer_from_array(
            np.array(z_range, dtype=np.float32),
            label="z_range",
            usage=BufferUsage.VERTEX | BufferUsage.COPY_DST,
        )
        self.vertex_buffers = [
            buffers["vertex_data"],
            positions_buffer,
            directions_buffer,
            colors_buffer,
            total_height_buffer,
        ]

        direction_stride = 0 if self.directions.size == 3 else self.directions.itemsize * 3

        self.vertex_buffer_layouts = [
            VertexBufferLayout(
                arrayStride=2 * 3 * 4,
                stepMode=VertexStepMode.vertex,
                attributes=[
                    # vertex position
                    VertexAttribute(
                        format=VertexFormat.float32x3,
                        offset=0,
                        shaderLocation=0,
                    ),
                    # vertex normal
                    VertexAttribute(
                        format=VertexFormat.float32x3,
                        offset=3 * 4,
                        shaderLocation=1,
                    ),
                ],
            ),
            VertexBufferLayout(
                arrayStride=3 * 4,
                stepMode=VertexStepMode.instance,
                attributes=[
                    # instance position
                    VertexAttribute(
                        format=VertexFormat.float32x3,
                        offset=0,
                        shaderLocation=2,
                    ),
                ],
            ),
            VertexBufferLayout(
                arrayStride=direction_stride,
                stepMode=VertexStepMode.instance,
                attributes=[
                    # instance direction
                    VertexAttribute(
                        format=VertexFormat.float32x3,
                        offset=0,
                        shaderLocation=3,
                    ),
                ],
            ),
            VertexBufferLayout(
                arrayStride=color_stride,
                stepMode=VertexStepMode.instance,
                attributes=[
                    # color/value bottom
                    VertexAttribute(
                        format=color_format,
                        offset=0,
                        shaderLocation=4,
                    ),
                    # color/value top
                    VertexAttribute(
                        format=color_format,
                        offset=color_top_offset,
                        shaderLocation=5,
                    ),
                ],
            ),
            VertexBufferLayout(
                arrayStride=0,
                attributes=[
                    # total_height
                    VertexAttribute(
                        format=VertexFormat.float32x2,
                        offset=0,
                        shaderLocation=6,
                    ),
                ],
            ),
        ]

    def get_shader_code(self) -> str:
        return read_shader_file("shapes.wgsl")

    def render(self, options: RenderOptions) -> None:
        render_pass = options.begin_render_pass()
        render_pass.setPipeline(self.pipeline)
        render_pass.setBindGroup(0, self.group)
        for i, vertex_buffer in enumerate(self.vertex_buffers):
            render_pass.setVertexBuffer(i, vertex_buffer)
        render_pass.setIndexBuffer(self.triangle_buffer, IndexFormat.uint32)
        render_pass.drawIndexed(
            self.n_vertices,
            self.n_instances,
        )
        render_pass.end()

    def get_bounding_box(self):
        bmin, bmax = self.shape_data.get_bounding_box()
        r = np.linalg.norm(bmax - bmin) / 2
        r *= self.directions.max()
        for i in range(3):
            bmin[i] = self.positions[i::3].min()
            bmax[i] = self.positions[i::3].max()

        bmin = bmin - r
        bmax = bmax + r

        return bmin, bmax
