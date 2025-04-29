from .render_object import BaseRenderObject
from .utils import read_shader_file

from .uniforms import UniformBase, ct

import time


class Binding:
    CLIPPING = 1


class ClippingUniforms(UniformBase):
    _binding = Binding.CLIPPING
    _fields_ = [
        ("plane", ct.c_float * 4),
        ("sphere", ct.c_float * 4),
        ("mode", ct.c_uint32),
        ("padding", ct.c_uint32 * 3),
    ]

    def __init__(self, device, mode=0, **kwargs):
        super().__init__(device, mode=mode, **kwargs)


class Clipping(BaseRenderObject):
    class Mode:
        DISABLED = 0
        PLANE = 1
        SPHERE = 2

    def __init__(
        self,
        mode=Mode.DISABLED,
        center=[0.0, 0.0, 0.0],
        normal=[0.0, -1.0, 0.0],
        radius=1.0,
    ):
        self.mode = mode
        self.center = center
        self.normal = normal
        self.radius = radius
        self.callbacks = []

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp
        if not hasattr(self, "uniforms"):
            self.uniforms = ClippingUniforms(self.device)
        import numpy as np

        c, n = (
            np.array(self.center, dtype=np.float32),
            np.array(self.normal, dtype=np.float32),
        )
        if np.linalg.norm(n) == 0:
            n = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            n = n / np.linalg.norm(n)
        # convert to normal and distance from origin
        d = -np.dot(c, n)
        self.uniforms.mode = self.mode
        for i in range(4):
            self.uniforms.plane[i] = [*n, d][i]
            self.uniforms.sphere[i] = [*c, self.radius][i]
        self.update_buffer()

    def update_buffer(self):
        self.uniforms.update_buffer()

    def get_bindings(self):
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("clipping.wgsl", __file__)

    def get_bounding_box(self) -> tuple[list[float], list[float]] | None:
        return None

    def __del__(self):
        if hasattr(self, "uniforms"):
            self.uniforms._buffer.destroy()

    def add_options_to_gui(self, gui):
        folder = gui.folder("Clipping", closed=True)
        folder.checkbox(
            "enabled", self.mode != self.Mode.DISABLED, self.enable_clipping
        )
        folder.value("x", self.center[0], self.set_x_value)
        folder.value("y", self.center[1], self.set_y_value)
        folder.value("z", self.center[2], self.set_z_value)
        folder.value("nx", self.normal[0], self.set_nx_value)
        folder.value("ny", self.normal[1], self.set_ny_value)
        folder.value("nz", self.normal[2], self.set_nz_value)

    def render(self, encoder):
        pass

    def enable_clipping(self, value):
        self.mode = self.Mode.PLANE if value else self.Mode.DISABLED
        self.update(time.time())
        for cb in self.callbacks:
            cb()

    def set_x_value(self, value):
        self.center[0] = value
        self.update(time.time())
        for cb in self.callbacks:
            cb()

    def set_y_value(self, value):
        self.center[1] = value
        self.update(time.time())
        for cb in self.callbacks:
            cb()

    def set_z_value(self, value):
        self.center[2] = value
        self.update(time.time())
        for cb in self.callbacks:
            cb()

    def set_nx_value(self, value):
        self.normal[0] = value
        self.update(time.time())
        for cb in self.callbacks:
            cb()

    def set_ny_value(self, value):
        self.normal[1] = value
        self.update(time.time())
        for cb in self.callbacks:
            cb()

    def set_nz_value(self, value):
        self.normal[2] = value
        self.update(time.time())
        for cb in self.callbacks:
            cb()
