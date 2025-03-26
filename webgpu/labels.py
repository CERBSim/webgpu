import numpy as np

from .font import Font
from .render_object import RenderObject
from .uniforms import Binding
from .utils import BufferBinding, read_shader_file
from .webgpu_api import *


class Labels(RenderObject):
    vertex_entry_point: str = "vertexText"
    fragment_entry_point: str = "fragmentFont"
    n_vertices: int = 6

    """Render a list of strings on screen
    @param labels: list of strings to render
    @param positions: list of positions to render the labels at
    @param apply_camera: whether to apply the camera transformation to the labels
    @param h_align: horizontal alignment of the labels. Can be one of: left, l, center, c, right, r
    @param v_align: horizontal alignment of the labels. Can be one of: bottom, b, center, c, top, t
    @param font_size: font size

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
    ):
        self.labels = labels
        self.positions = positions
        self.font_size = font_size
        self.apply_camera = apply_camera
        self.h_align = h_align
        self.v_align = v_align

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp
        n_chars = sum(len(label) for label in self.labels)
        n_labels = len(self.labels)
        self.n_vertices = 6
        self.n_instances = n_chars
        char_t = np.dtype(
            [
                ("itext", np.uint32),
                ("ichar", np.uint16),
                ("char", np.uint8),
                ("padding", np.uint8),
            ]
        )
        char_data = np.zeros(n_chars, dtype=char_t)
        text_t = np.dtype(
            [
                ("pos", np.float32, 3),
                ("length", np.uint16),
                ("apply_camera", np.uint8),
                ("alignment", np.uint8),
            ]
        )
        text_data = np.zeros(n_labels, dtype=text_t)

        align_map = {
            "c": 1,
            "center": 1,
            "r": 2,
            "right": 2,
            "t": 2,
            "top": 2,
            "b": 0,
            "bottom": 0,
            "l": 0,
            "left": 0,
        }

        ichar = 0
        for i, label, pos in zip(range(len(self.labels)), self.labels, self.positions):
            h_align = self.h_align if isinstance(self.h_align, str) else self.h_align[i]
            v_align = self.v_align if isinstance(self.v_align, str) else self.v_align[i]
            align = align_map[h_align] + 4 * align_map[v_align]
            apply_camera = (
                self.apply_camera
                if isinstance(self.apply_camera, bool)
                else self.apply_camera[i]
            )

            if len(pos) == 2:
                pos = (*pos, 0)

            text_data[i]["pos"] = pos
            text_data[i]["length"] = len(label)
            text_data[i]["apply_camera"] = apply_camera
            text_data[i]["alignment"] = align

            i0 = ichar
            for c in label:
                char_data[ichar]["itext"] = i
                char_data[ichar]["ichar"] = ichar - i0
                char_data[ichar]["char"] = ord(c)
                ichar += 1

        self.font = Font(self.canvas, self.font_size)

        data = (
            np.array([len(self.labels)], dtype=np.uint32).tobytes()
            + text_data.tobytes()
            + char_data.tobytes()
        )

        self.buffer = self.device.createBuffer(
            len(data),
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
        )
        self.device.queue.writeBuffer(self.buffer, 0, data)
        self.create_render_pipeline()

    def get_shader_code(self):
        shader_code = read_shader_file("text.wgsl", __file__)
        shader_code += self.font.get_shader_code()
        shader_code += self.options.camera.get_shader_code()
        return shader_code

    def get_bindings(self):
        return [
            *self.font.get_bindings(),
            *self.options.camera.get_bindings(),
            BufferBinding(Binding.TEXT, self.buffer),
        ]
