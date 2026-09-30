"""Append-only triangle soup for live previews (e.g. a surface mesh while it is generated)."""

import threading

import numpy as np

from .clipping import Clipping
from .renderer import Renderer, RenderOptions
from .uniforms import UniformBase, ct
from .utils import BufferBinding, create_buffer, read_shader_file
from .webgpu_api import BufferUsage

# u32 words per triangle: 9 coordinates, group id | hidden edge bits << 29, packed rgba8 colour
_STRIDE = 11
_GROUP_BITS = 29
_GROUP_MASK = (1 << _GROUP_BITS) - 1


class Binding:
    TRIANGLES = 95
    COLORS = 96
    PARAMS = 97


class DynamicTrianglesUniforms(UniformBase):
    _binding = Binding.PARAMS
    _fields_ = [
        ("wireframe", ct.c_uint32),
        ("line_width", ct.c_float),
        ("depth_bias", ct.c_float),
        ("wireframe_opacity", ct.c_float),
        ("wireframe_color", ct.c_float * 4),
    ]


def _pack_rgba8(colors):
    c = np.clip(np.round(np.asarray(colors, dtype=np.float32) * 255), 0, 255).astype(np.uint32)
    return c[:, 0] | (c[:, 1] << 8) | (c[:, 2] << 16) | (c[:, 3] << 24)


class DynamicTriangles(Renderer):
    """Triangles that are appended in chunks and drawn with flat shading.

    ``append`` only touches a CPU mirror; the next ``scene.render()`` uploads the
    triangles added since the previous upload with a single ``writeBuffer``.
    All mutating methods are safe to call from any thread.

    ``depth_bias`` pulls the triangles toward the camera by that amount of NDC
    depth (times w), so they draw on top of coincident surfaces such as the CAD
    faces they are meshed on; geometry edges (2e-4) stay in front. ``line_width``,
    ``wireframe_color`` (rgba) and ``wireframe_opacity`` style the wireframe.
    Changing any of these only rewrites a uniform on the next render.
    """

    n_vertices: int = 3
    # slope part of the bias for grazing angles; the constant part is depth_bias
    depthBias: int = 0
    depthBiasSlopeScale: float = -1.0
    select_entry_point: str = ""

    def __init__(
        self,
        capacity=4096,
        clipping=None,
        wireframe=True,
        pickable=False,
        line_width=1.0,
        depth_bias=5e-5,
        wireframe_color=(0.0, 0.0, 0.0, 1.0),
        wireframe_opacity=1.0,
        label="DynamicTriangles",
    ):
        super().__init__(label=label)
        self.gpu_objects.clipping = clipping or Clipping()
        if pickable:
            self.select_entry_point = "fragment_select"
        self._lock = threading.Lock()
        self._capacity = max(int(capacity), 1)
        self._mirror = np.zeros((self._capacity, _STRIDE), dtype=np.uint32)
        self._n = 0
        self._dirty_from = 0
        self._pmin = np.full(3, np.inf, dtype=np.float32)
        self._pmax = np.full(3, -np.inf, dtype=np.float32)
        self._any_alpha = False
        self._group_colors = None
        self._colors_dirty = True
        self._uniforms = DynamicTrianglesUniforms(
            wireframe=int(wireframe),
            line_width=line_width,
            depth_bias=depth_bias,
            wireframe_opacity=wireframe_opacity,
        )
        self.wireframe_color = wireframe_color
        self._trig_buffer = None
        self._color_buffer = None
        self._indirect_buffer = None
        self._uploaded_n = -1

    @property
    def num_triangles(self) -> int:
        return self._n

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def wireframe(self) -> bool:
        return bool(self._uniforms.wireframe)

    @wireframe.setter
    def wireframe(self, value):
        self._uniforms.wireframe = int(bool(value))

    @property
    def line_width(self) -> float:
        return self._uniforms.line_width

    @line_width.setter
    def line_width(self, value):
        self._uniforms.line_width = float(value)

    @property
    def depth_bias(self) -> float:
        return self._uniforms.depth_bias

    @depth_bias.setter
    def depth_bias(self, value):
        self._uniforms.depth_bias = float(value)

    @property
    def wireframe_color(self) -> tuple:
        return tuple(self._uniforms.wireframe_color)

    @wireframe_color.setter
    def wireframe_color(self, rgba):
        rgba = [float(c) for c in rgba]
        if len(rgba) == 3:
            rgba.append(1.0)
        self._uniforms.wireframe_color[:] = rgba

    @property
    def wireframe_opacity(self) -> float:
        return self._uniforms.wireframe_opacity

    @wireframe_opacity.setter
    def wireframe_opacity(self, value):
        self._uniforms.wireframe_opacity = float(value)

    def append(self, coords, groups, colors=None, edge_mask=None):
        """Append triangles.

        coords (n, 3, 3), groups (n,) int in [0, 2**29), colors (n, 4) rgba 0..1
        overriding the group colour, edge_mask (n,) uint8 with bits 0, 1, 2 for the
        wireframe edges v0v1, v1v2, v2v0 to draw (default: all), e.g. 0b011 hides
        the diagonal v2v0 of a quad split into (v0, v1, v2), (v0, v2, v3).
        """
        coords = np.ascontiguousarray(coords, dtype=np.float32).reshape(-1, 9)
        k = len(coords)
        if k == 0:
            return
        groups = np.asarray(groups, dtype=np.int32).reshape(-1)
        assert len(groups) == k, "groups must have one entry per triangle"
        assert groups.min() >= 0 and groups.max() <= _GROUP_MASK, "group ids must be in [0, 2**29)"
        packed = groups.view(np.uint32)
        if edge_mask is not None:
            edge_mask = np.asarray(edge_mask, dtype=np.uint32).reshape(-1)
            assert len(edge_mask) == k, "edge_mask must have one entry per triangle"
            packed = packed | ((~edge_mask & 7) << _GROUP_BITS)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float32).reshape(k, 4)
        pts = coords.reshape(-1, 3)
        with self._lock:
            n = self._n
            if n + k > self._capacity:
                cap = self._capacity
                while cap < n + k:
                    cap *= 2
                mirror = np.zeros((cap, _STRIDE), dtype=np.uint32)
                mirror[:n] = self._mirror[:n]
                self._mirror, self._capacity = mirror, cap
            rows = self._mirror[n : n + k]
            rows[:, :9] = coords.view(np.uint32)
            rows[:, 9] = packed
            if colors is None:
                rows[:, 10] = 0
            else:
                rows[:, 10] = _pack_rgba8(colors)
                if (colors[:, 3] < 1.0).any():
                    self._any_alpha = True
            np.minimum(self._pmin, pts.min(axis=0), out=self._pmin)
            np.maximum(self._pmax, pts.max(axis=0), out=self._pmax)
            self._dirty_from = min(self._dirty_from, n)
            self._n = n + k

    def set_group_colors(self, rgba):
        """Colour per group id, (ngroups, 4) rgba 0..1; ids without colour use a default palette."""
        rgba = np.array(rgba, dtype=np.float32).reshape(-1, 4)
        with self._lock:
            self._group_colors = rgba
            self._colors_dirty = True

    def drop_groups(self, ids):
        """Remove all triangles of the given group ids."""
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        with self._lock:
            n = self._n
            if n == 0 or len(ids) == 0:
                return
            keep = ~np.isin((self._mirror[:n, 9] & _GROUP_MASK).astype(np.int32), ids)
            m = int(keep.sum())
            if m == n:
                return
            self._mirror[:m] = self._mirror[:n][keep]
            self._n = m
            self._dirty_from = 0
            self._update_bounding_box()

    def clear(self):
        with self._lock:
            self._n = 0
            self._dirty_from = 0
            self._any_alpha = False
            self._pmin[:] = np.inf
            self._pmax[:] = -np.inf

    def _update_bounding_box(self):
        if self._n == 0:
            self._pmin[:] = np.inf
            self._pmax[:] = -np.inf
            return
        pts = self._mirror[: self._n, :9].view(np.float32).reshape(-1, 3)
        self._pmin[:] = pts.min(axis=0)
        self._pmax[:] = pts.max(axis=0)

    def get_bounding_box(self):
        with self._lock:
            if self._n == 0:
                return None
            return self._pmin.copy(), self._pmax.copy()

    def _is_transparent(self):
        gc = self._group_colors
        return self._any_alpha or (gc is not None and bool((gc[:, 3] < 1.0).any()))

    def _upload_pending(self):
        """Write everything changed since the last upload; reallocations are deferred to update()."""
        if self._trig_buffer is None:
            return
        with self._lock:
            n = self._n
            if (
                self._capacity * _STRIDE * 4 > self._trig_buffer.size
                or (self._colors_dirty and self._color_rows() * 16 > self._color_buffer.size)
                or self._is_transparent() != self.transparent
            ):
                self.set_needs_update()
                return
            data = self._mirror[self._dirty_from : n].tobytes() if self._dirty_from < n else None
            offset = self._dirty_from * _STRIDE * 4
            colors = self._color_data() if self._colors_dirty else None
            self._dirty_from = n
            self._colors_dirty = False
        queue = self.device.queue
        if data:
            queue.writeBuffer(self._trig_buffer, offset, data)
        if colors is not None:
            queue.writeBuffer(self._color_buffer, 0, colors)
        if n != self._uploaded_n:
            queue.writeBuffer(self._indirect_buffer, 0, np.array([3 * n, 1, 0, 0], dtype=np.uint32).tobytes())
            self._uploaded_n = n
        self._uniforms.update_buffer()

    def _color_rows(self):
        gc = self._group_colors
        k = 1 if gc is None else len(gc)
        return max(64, 1 << (k - 1).bit_length())

    def _color_data(self):
        data = np.full((self._color_rows(), 4), -1.0, dtype=np.float32)
        if self._group_colors is not None:
            data[: len(self._group_colors)] = self._group_colors
        return data.tobytes()

    def update(self, options: RenderOptions):
        usage = BufferUsage.STORAGE | BufferUsage.COPY_DST | BufferUsage.COPY_SRC
        with self._lock:
            self.transparent = self._is_transparent()
            size = self._capacity * _STRIDE * 4
            if self._trig_buffer is None or self._trig_buffer.size < size:
                self._trig_buffer = create_buffer(size, usage, "dynamic_triangles", self._trig_buffer)
                self._dirty_from = 0
            size = self._color_rows() * 16
            if self._color_buffer is None or self._color_buffer.size < size:
                self._color_buffer = create_buffer(size, usage, "dynamic_triangles_colors", self._color_buffer)
                self._colors_dirty = True
        if self._indirect_buffer is None:
            self._indirect_buffer = create_buffer(
                16, BufferUsage.INDIRECT | usage, "dynamic_triangles_indirect"
            )
            self._uploaded_n = -1
        self._upload_pending()

    def get_bindings(self):
        return [
            BufferBinding(Binding.TRIANGLES, self._trig_buffer),
            BufferBinding(Binding.COLORS, self._color_buffer),
            *self._uniforms.get_bindings(),
            *self.gpu_objects.clipping.get_bindings(),
        ]

    def get_shader_code(self):
        return read_shader_file("dynamic_triangles.wgsl")

    def _draw(self, render_pass):
        render_pass.setBindGroup(0, self.group)
        render_pass.drawIndirect(self._indirect_buffer, 0)

    def render(self, options: RenderOptions) -> None:
        with options.render_pass_scope() as render_pass:
            render_pass.setPipeline(self.pipeline)
            self._draw(render_pass)

    def select(self, options: RenderOptions, x: int, y: int) -> None:
        if not self._select_pipeline or not getattr(self, "_select_active", True):
            return
        render_pass = options.begin_select_pass(x, y)
        render_pass.setPipeline(self._select_pipeline)
        self._draw(render_pass)
        render_pass.end()

    def get_export_descriptor(self, options, buffer_registry):
        desc = super().get_export_descriptor(options, buffer_registry)
        desc.draw_indirect = buffer_registry.register_buffer(self._indirect_buffer, "indirect")
        return desc
