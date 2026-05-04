"""Coordinate axes (3D arrows) and navigation cube renderers.

Both render in fixed screen corners, rotated by the camera's rotation matrix.
Uses the Labels renderer for text (overlay mode) and minimal custom renderers
for the mesh and edge geometry.
"""

import numpy as np

from .labels import Labels
from .renderer import MultipleRenderer, Renderer, RenderOptions
from .utils import (
    BufferBinding,
    UniformBinding,
    buffer_from_array,
    uniform_from_array,
    read_shader_file,
)
from .webgpu_api import PrimitiveTopology

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rotation_to_direction(direction):
    """3x3 rotation mapping +Z to *direction*."""
    z = np.asarray(direction, dtype=np.float64)
    z = z / np.linalg.norm(z)
    up = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _cylinder_tris(segs, r, z0, z1, rot):
    verts, norms = [], []
    angles = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    for i in range(segs):
        j = (i + 1) % segs
        c0, s0 = np.cos(angles[i]), np.sin(angles[i])
        c1, s1 = np.cos(angles[j]), np.sin(angles[j])
        p00 = rot @ [r * c0, r * s0, z0]
        p10 = rot @ [r * c1, r * s1, z0]
        p01 = rot @ [r * c0, r * s0, z1]
        p11 = rot @ [r * c1, r * s1, z1]
        n0 = rot @ [c0, s0, 0.0]
        n1 = rot @ [c1, s1, 0.0]
        verts += [p00, p10, p01, p10, p11, p01]
        norms += [n0, n1, n0, n1, n1, n0]
    return verts, norms


def _cone_tris(segs, r, z0, z1, rot):
    verts, norms = [], []
    angles = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    tip = rot @ [0, 0, z1]
    slope = r / (z1 - z0)
    base_center = rot @ [0, 0, z0]
    base_n = rot @ [0, 0, -1.0]
    for i in range(segs):
        j = (i + 1) % segs
        c0, s0 = np.cos(angles[i]), np.sin(angles[i])
        c1, s1 = np.cos(angles[j]), np.sin(angles[j])
        p0 = rot @ [r * c0, r * s0, z0]
        p1 = rot @ [r * c1, r * s1, z0]
        raw0 = np.array([c0, s0, slope])
        raw1 = np.array([c1, s1, slope])
        n0 = rot @ raw0 / np.linalg.norm(raw0)
        n1 = rot @ raw1 / np.linalg.norm(raw1)
        nt = (n0 + n1)
        nt /= np.linalg.norm(nt)
        verts += [p0, p1, tip]
        norms += [n0, n1, nt]
        verts += [base_center, p1, p0]
        norms += [base_n, base_n, base_n]
    return verts, norms


def _generate_arrows():
    """Return (positions, normals, colors) flat float32 arrays for 3 arrows."""
    axes = [
        ([1, 0, 0], [0.85, 0.20, 0.20, 1.0]),
        ([0, 1, 0], [0.20, 0.75, 0.20, 1.0]),
        ([0, 0, 1], [0.25, 0.35, 0.90, 1.0]),
    ]
    segs = 12
    shaft_r, shaft_len = 0.018, 0.65
    head_r, head_len = 0.055, 0.28
    all_v, all_n, all_c = [], [], []
    for d, col in axes:
        rot = _rotation_to_direction(d)
        sv, sn = _cylinder_tris(segs, shaft_r, 0, shaft_len, rot)
        cv, cn = _cone_tris(segs, head_r, shaft_len, shaft_len + head_len, rot)
        n = len(sv) + len(cv)
        all_v += sv + cv
        all_n += sn + cn
        all_c += [col] * n
    return (
        np.array(all_v, dtype=np.float32).flatten(),
        np.array(all_n, dtype=np.float32).flatten(),
        np.array(all_c, dtype=np.float32).flatten(),
    )


def _generate_cube_faces(h=0.45):
    """Return (positions, normals, colors) for 6 coloured cube faces."""
    face_defs = [
        ([[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]], [1, 0, 0], [0.82, 0.55, 0.55, 1.0]),
        ([[-h, -h, h], [-h, h, h], [-h, h, -h], [-h, -h, -h]], [-1, 0, 0], [0.58, 0.40, 0.40, 1.0]),
        ([[-h, h, -h], [h, h, -h], [h, h, h], [-h, h, h]], [0, 1, 0], [0.55, 0.82, 0.55, 1.0]),
        ([[h, -h, -h], [-h, -h, -h], [-h, -h, h], [h, -h, h]], [0, -1, 0], [0.40, 0.58, 0.40, 1.0]),
        ([[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], [0, 0, 1], [0.55, 0.55, 0.82, 1.0]),
        ([[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]], [0, 0, -1], [0.40, 0.40, 0.58, 1.0]),
    ]
    all_v, all_n, all_c = [], [], []
    for corners, normal, color in face_defs:
        c = [np.array(p, dtype=np.float64) for p in corners]
        all_v += [c[0], c[1], c[2], c[0], c[2], c[3]]
        all_n += [normal] * 6
        all_c += [color] * 6
    return (
        np.array(all_v, dtype=np.float32).flatten(),
        np.array(all_n, dtype=np.float32).flatten(),
        np.array(all_c, dtype=np.float32).flatten(),
    )


def _generate_cube_edges(h=0.45):
    """Return flat float32 array of edge pairs (p1 xyz, p2 xyz) x 12 edges."""
    v = [
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    data = []
    for a, b in edges:
        data += v[a] + v[b]
    return np.array(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class GizmoMeshRenderer(Renderer):
    """Renders a triangle mesh in a fixed corner, rotated by camera."""

    select_entry_point: str = ""

    def __init__(self, positions, normals, colors, corner, scale, label="GizmoMesh"):
        super().__init__(label=label)
        self._raw_pos = positions
        self._raw_nrm = normals
        self._raw_col = colors
        self._corner = corner
        self._scale = scale
        self._pos_buf = None
        self._nrm_buf = None
        self._col_buf = None
        self._uni_buf = None

    def update(self, options: RenderOptions):
        self.n_vertices = len(self._raw_pos) // 3
        self._pos_buf = buffer_from_array(self._raw_pos, label="gizmo_pos", reuse=self._pos_buf)
        self._nrm_buf = buffer_from_array(self._raw_nrm, label="gizmo_nrm", reuse=self._nrm_buf)
        self._col_buf = buffer_from_array(self._raw_col, label="gizmo_col", reuse=self._col_buf)
        uni = np.array([self._corner[0], self._corner[1], self._scale, 0], dtype=np.float32)
        self._uni_buf = uniform_from_array(uni, label="gizmo_uni", reuse=self._uni_buf)

    def get_shader_code(self):
        return read_shader_file("gizmo_mesh.wgsl")

    def get_bindings(self):
        return [
            BufferBinding(90, self._pos_buf),
            BufferBinding(91, self._nrm_buf),
            BufferBinding(92, self._col_buf),
            UniformBinding(93, self._uni_buf),
        ]

    def get_bounding_box(self):
        return None


class GizmoEdgesRenderer(Renderer):
    """Renders thick lines in a fixed corner, rotated by camera."""

    n_vertices: int = 4
    topology: PrimitiveTopology = PrimitiveTopology.triangle_strip
    select_entry_point: str = ""

    def __init__(self, edge_data, corner, scale, thickness=0.003, label="GizmoEdges"):
        super().__init__(label=label)
        self._raw_edges = edge_data
        self._corner = corner
        self._scale = scale
        self._thickness = thickness
        self._edge_buf = None
        self._uni_buf = None

    def update(self, options: RenderOptions):
        self.n_instances = len(self._raw_edges) // 6
        self._edge_buf = buffer_from_array(self._raw_edges, label="gizmo_edges", reuse=self._edge_buf)
        uni = np.array([self._corner[0], self._corner[1], self._scale, self._thickness], dtype=np.float32)
        self._uni_buf = uniform_from_array(uni, label="gizmo_edge_uni", reuse=self._uni_buf)

    def get_shader_code(self):
        return read_shader_file("gizmo_edges.wgsl")

    def get_bindings(self):
        return [
            BufferBinding(90, self._edge_buf),
            UniformBinding(93, self._uni_buf),
        ]

    def get_bounding_box(self):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ARROW_COLORS = [
    [0.85, 0.20, 0.20, 1.0],
    [0.20, 0.75, 0.20, 1.0],
    [0.25, 0.35, 0.90, 1.0],
]

_AXES_CORNER = (-0.78, -0.78)
_AXES_SCALE = 0.15
_CUBE_CORNER = (0.78, -0.78)
_CUBE_SCALE = 0.13


class CoordinateAxes(MultipleRenderer):
    """3D arrows with X/Y/Z labels in the bottom-left corner."""

    def __init__(self):
        pos, nrm, col = _generate_arrows()
        tip_offset = 1.12
        self.arrows = GizmoMeshRenderer(pos, nrm, col, _AXES_CORNER, _AXES_SCALE, "Arrows")
        self.labels = Labels(
            labels=["X", "Y", "Z"],
            positions=[[tip_offset, 0, 0], [0, tip_offset, 0], [0, 0, tip_offset]],
            colors=_ARROW_COLORS,
            overlay={"corner": _AXES_CORNER, "scale": _AXES_SCALE},
            h_align="center",
            v_align="center",
            font_size=16,
        )
        super().__init__([self.arrows, self.labels])

    def get_bounding_box(self):
        return None


class NavigationCube(MultipleRenderer):
    """Coloured orientation cube with edge wireframe and face labels."""

    FACE_VIEWS = ["yz", "yz_flip", "xz", "xz_flip", "xy", "xy_flip"]

    def __init__(self):
        h = 0.45
        fp, fn, fc = _generate_cube_faces(h)
        edges = _generate_cube_edges(h)
        self.faces = GizmoMeshRenderer(fp, fn, fc, _CUBE_CORNER, _CUBE_SCALE, "CubeFaces")
        self.faces.select_entry_point = "fragment_select"
        self.edges = GizmoEdgesRenderer(edges, _CUBE_CORNER, _CUBE_SCALE, thickness=0.003, label="CubeEdges")

        label_offset = h + 0.08
        self.labels = Labels(
            labels=["X", "x", "Y", "y", "Z", "z"],
            positions=[
                [label_offset, 0, 0], [-label_offset, 0, 0],
                [0, label_offset, 0], [0, -label_offset, 0],
                [0, 0, label_offset], [0, 0, -label_offset],
            ],
            normals=[
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
            ],
            colors=[
                [0.75, 0.15, 0.15, 1.0], [0.45, 0.25, 0.25, 1.0],
                [0.15, 0.65, 0.15, 1.0], [0.25, 0.45, 0.25, 1.0],
                [0.15, 0.25, 0.80, 1.0], [0.25, 0.25, 0.45, 1.0],
            ],
            overlay={"corner": _CUBE_CORNER, "scale": _CUBE_SCALE},
            h_align="center",
            v_align="center",
            font_size=14,
        )
        super().__init__([self.faces, self.edges, self.labels])

    def get_bounding_box(self):
        return None
