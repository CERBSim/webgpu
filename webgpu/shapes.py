"""
Simple shapes (cylinder, cone, circle) generation and render objects
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .colormap import Colormap
from .renderer import Renderer, RenderOptions
from .uniforms import UniformBase, ct
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


class ShapeUniforms(UniformBase):
    _binding = 10
    _fields_ = [("scale", ct.c_float), ("scale_mode", ct.c_uint32), ("padding", ct.c_float * 2)]


class ShapeRenderer(Renderer):
    SCALE_UNIFORM = ct.c_uint32(0)
    SCALE_Z = ct.c_uint32(1)
    vertex_entry_point = "shape_vertex_main"
    select_entry_point = "shape_fragment_main_select"

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

        super().__init__(label=label)

        if positions is None:
            positions = []

        if directions is None:
            directions = []

        self.colormap = colormap or Colormap()
        self._positions = np.array(positions, dtype=np.float32).reshape(-1)
        self._values = (
            np.array(values, dtype=np.float32).reshape(-1) if values is not None else None
        )
        self._directions = np.array(directions, dtype=np.float32).reshape(-1)

        if colors:
            colors = np.array(colors, dtype=np.float32).reshape(-1)
            colors = np.array(np.round(255 * colors), dtype=np.uint8).flatten()
        self._colors = colors
        self._scale = 1.0
        self._scale_mode = ShapeRenderer.SCALE_UNIFORM
        self._scale_range = (0.01, 2, 0.01)
        self._uniforms = None
        self.shape_data = shape_data

    def get_bindings(self):
        return self.colormap.get_bindings() + self._uniforms.get_bindings()

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        self._positions = np.array(value, dtype=np.float32).reshape(-1)
        self.set_needs_update()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value
        if self._uniforms is not None:
            self._uniforms.scale = value
            self._uniforms.update_buffer()

    @property
    def scale_mode(self):
        return self._scale_mode

    @scale_mode.setter
    def scale_mode(self, value):
        self._scale_mode = value
        if self._uniforms is not None:
            self._uniforms.scale_mode = value
            self._uniforms.update_buffer()

    @property
    def directions(self):
        return self._directions

    @directions.setter
    def directions(self, value):
        self._directions = np.array(value, dtype=np.float32).reshape(-1)
        self.set_needs_update()

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        self._values = np.array(value, dtype=np.float32).reshape(-1)
        self.set_needs_update()

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = np.array(value, dtype=np.uint8).reshape(-1)
        self.set_needs_update()

    def update(self, options: RenderOptions):
        if self.colors is None and self.values is None:
            raise ValueError("Either colors or values must be provided")
        self.n_vertices = self.shape_data.triangles.size
        self.n_instances = self.positions.size // 3
        self.colormap.update(options)
        self._uniforms = ShapeUniforms()
        self._uniforms.scale = self.scale
        self._uniforms.scale_mode = self.scale_mode
        self._uniforms.update_buffer()
        buffers = self.shape_data.create_buffers()
        self.triangle_buffer = buffers["triangles"]
        positions_buffer = buffer_from_array(
            self.positions, label="positions", usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
        )
        directions_buffer = buffer_from_array(
            self.directions, label="directions", usage=BufferUsage.VERTEX | BufferUsage.COPY_DST
        )

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
                self.colormap.set_min_max(self.values.min(), self.values.max(), set_autoscale=False)
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

    def add_options_to_gui(self, gui):
        if gui is None:
            return

        def set_scale(value):
            self.scale = value

        gui.slider(
            value=self.scale,
            func=set_scale,
            min=self._scale_range[0],
            max=self._scale_range[1],
            step=self._scale_range[2],
            label="Scale Shapes",
        )

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

    def select(self, options: RenderOptions, x, y) -> None:
        render_pass = options.begin_select_pass(x, y)
        render_pass.setPipeline(self._select_pipeline)
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
        if self.positions.size == 0:
            return None
        bmin, bmax = self.shape_data.get_bounding_box()
        r = np.linalg.norm(bmax - bmin) / 2
        r *= self.directions.max()
        for i in range(3):
            bmin[i] = self.positions[i::3].min()
            bmax[i] = self.positions[i::3].max()

        bmin = bmin - r
        bmax = bmax + r

        return bmin, bmax
