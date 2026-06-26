import numpy as np

from .font import Font
from .renderer import Renderer, RenderOptions, check_timestamp
from .uniforms import Binding
from .utils import BufferBinding, UniformBinding, buffer_from_array, uniform_from_array, read_shader_file
from .webgpu_api import *


class Labels(Renderer):
    vertex_entry_point: str = "vertexText"
    fragment_entry_point: str = "fragmentFont"
    select_entry_point: str = ""
    n_vertices: int = 6
    transparent: bool = True

    """Render a list of strings on screen
    @param labels: list of strings to render
    @param positions: list of positions to render the labels at
    @param apply_camera: whether to apply the camera transformation to the labels
    @param h_align: horizontal alignment of the labels. Can be one of: left, l, center, c, right, r
    @param v_align: vertical alignment of the labels. Can be one of: bottom, b, center, c, top, t
    @param font_size: font size
    @param overlay: dict with 'corner' (x,y) and 'scale' for fixed-screen overlay mode, or None
    @param normals: per-label normals for visibility culling in overlay mode, or None
    @param colors: per-label RGBA colors (list of [r,g,b,a] floats 0-1), or None (renders black)

    If any of apply_camera, h_align, or v_align is a list, it must have the same length as labels.
    """

    def __init__(
        self,
        labels: list[str],
        positions: list[tuple],
        apply_camera: bool | list[bool] = False,
        h_align: str | list[str] = "left",
        v_align: str | list[str] = "bottom",
        font_size=20,
        overlay: dict | None = None,
        normals: list | None = None,
        colors: list | None = None,
    ):
        super().__init__()
        self.labels = labels
        self.positions = positions
        self.font_size = font_size
        self.apply_camera = apply_camera
        self.h_align = h_align
        self.v_align = v_align
        self.overlay = overlay
        self.normals = normals
        self.colors = colors
        self.buffer = None
        self._overlay_buf = None
        self.font = None
        self._text_color = None  # (r,g,b) for the default (non-per-label) path

        if colors is not None:
            self.fragment_entry_point = "fragmentFontColor"

    @property
    def text_color(self):
        return self._text_color

    @text_color.setter
    def text_color(self, value):
        self._text_color = None if value is None else tuple(value[:3])
        if self.font is not None and self._text_color is not None:
            self.font.set_color(self._text_color)

    def get_theme_label_target(self, registry):
        """Expose the font color uniform for theme-dependent defaulting.

        When the app left the label color at the default (neither per-label
        ``colors`` nor ``text_color`` were set), the engine should pick a color
        based on the canvas background so the text stays legible: black on a
        light background, off-white on a dark one. Returns the buffer id and
        byte offset of the color field, or ``None`` when a color was set
        explicitly (in which case the app's choice is respected).
        """
        if self.colors is not None or self._text_color is not None:
            return None
        if self.font is None or self.font.uniforms is None:
            return None
        buf = self.font.uniforms._buffer
        if buf is None:
            return None
        key = id(buf)
        if key not in registry._buffers:
            return None
        from .font import FontUniforms

        return {
            "buffer_id": registry._buffers[key][0],
            "offset": FontUniforms.color.offset,
        }

    def update(self, options: RenderOptions):
        n_chars = sum(len(label) for label in self.labels)
        n_labels = len(self.labels)
        self.n_vertices = 6
        self.n_instances = n_chars

        char_t = np.dtype(
            [
                ("itext", np.uint32),
                ("ichar", np.uint16),
                ("char", np.uint16),
            ]
        )
        char_data = np.zeros(n_chars, dtype=char_t)

        # 8 u32s per text: pos(3f) + packed(1u) + normal(3f) + color_packed(1u)
        text_t = np.dtype(
            [
                ("pos", np.float32, 3),
                ("packed", np.uint32),
                ("normal", np.float32, 3),
                ("color_packed", np.uint32),
            ]
        )
        text_data = np.zeros(n_labels, dtype=text_t)

        align_map = {
            "c": 1, "center": 1,
            "r": 2, "right": 2, "t": 2, "top": 2,
            "b": 0, "bottom": 0, "l": 0, "left": 0,
        }

        if self.font is None:
            self.font = Font(options.canvas, self.font_size)
        else:
            self.font.set_font_size(self.font_size)
        if self._text_color is not None:
            self.font.set_color(self._text_color)

        char_map = self.font.atlas.char_map

        ichar = 0
        for i, (label, pos) in enumerate(zip(self.labels, self.positions)):
            h_align = self.h_align if isinstance(self.h_align, str) else self.h_align[i]
            v_align = self.v_align if isinstance(self.v_align, str) else self.v_align[i]
            align = align_map[h_align] + 4 * align_map[v_align]

            if self.overlay is not None:
                apply_camera = 2
            elif isinstance(self.apply_camera, bool):
                apply_camera = int(self.apply_camera)
            else:
                apply_camera = int(self.apply_camera[i])

            if len(pos) == 2:
                pos = (*pos, 0)

            text_data[i]["pos"] = pos
            text_data[i]["packed"] = (
                (len(label) & 0xFFFF)
                | ((apply_camera & 0xFF) << 16)
                | ((align & 0xFF) << 24)
            )

            if self.normals is not None:
                text_data[i]["normal"] = self.normals[i]

            if self.colors is not None:
                c = self.colors[i]
                r = int(c[0] * 255)
                g = int(c[1] * 255)
                b = int(c[2] * 255)
                a = int(c[3] * 255) if len(c) > 3 else 255
                text_data[i]["color_packed"] = r | (g << 8) | (b << 16) | (a << 24)

            i0 = ichar
            for c in label:
                char_data[ichar]["itext"] = i
                char_data[ichar]["ichar"] = ichar - i0
                char_data[ichar]["char"] = char_map.get(ord(c), char_map.get(ord("?"), 0))
                ichar += 1

        data = (
            np.array([n_labels], dtype=np.uint32).tobytes()
            + text_data.tobytes()
            + char_data.tobytes()
        )

        self.buffer = buffer_from_array(data, BufferUsage.STORAGE | BufferUsage.COPY_DST, "labels", self.buffer)

        # Overlay uniform
        if self.overlay is not None:
            corner = self.overlay["corner"]
            scale = self.overlay["scale"]
            overlay_data = np.array([corner[0], corner[1], scale, 0.0], dtype=np.float32)
        else:
            overlay_data = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._overlay_buf = uniform_from_array(overlay_data, label="overlay_uni", reuse=self._overlay_buf)

    def get_shader_code(self):
        return read_shader_file("text.wgsl")

    def get_bindings(self):
        return [
            *self.font.get_bindings(),
            BufferBinding(Binding.TEXT, self.buffer),
            UniformBinding(31, self._overlay_buf),
        ]
