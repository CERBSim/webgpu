
from webgpu.render_object import BaseRenderObject
from webgpu.utils import read_shader_file
from .uniforms import UniformBase, ct

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
    def __init__(self, mode=Mode.DISABLED,
                 center=[0.,0.,0.], normal=[0.,1.,0.], radius=1.):
        self.mode = mode
        self.center = center
        self.normal = normal
        self.radius = radius

    def update(self, pnt=None, normal=None, mode=None):
        if pnt is not None:
            self.center = pnt
        if normal is not None:
            self.normal = normal
        if mode is not None:
            self.mode = mode
        if not hasattr(self, "uniforms"):
            self.uniforms = ClippingUniforms(self.device)
        import numpy as np
        c, n = (np.array(self.center, dtype=np.float32),
                np.array(self.normal, dtype=np.float32))
        if np.linalg.norm(n) == 0:
            n = np.array([0., 0., -1.], dtype=np.float32)
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
        self.uniforms._buffer.destroy()

    def add_options_to_gui(self, gui):
        print("add options to gui")
        folder = gui.folder("Clipping", closed=True)
        folder.checkbox("enabled", self.mode != self.Mode.DISABLED,
                        enable_clipping, self)
        folder.value("x", self.center[0], set_x_value, self)
        folder.value("y", self.center[1], set_y_value, self)
        folder.value("z", self.center[2], set_z_value, self)
        folder.value("nx", self.normal[0], set_nx_value, self)
        folder.value("ny", self.normal[1], set_ny_value, self)
        folder.value("nz", self.normal[2], set_nz_value, self)

    def render(self, encoder):
        pass

def enable_clipping(me, value):
    me.mode = me.Mode.PLANE if value else me.Mode.DISABLED
    me.update()

def set_x_value(me, value):
    me.center[0] = value
    me.update()

def set_y_value(me, value):
    me.center[1] = value
    me.update()

def set_z_value(me, value):
    me.center[2] = value
    me.update()

def set_nx_value(me, value):
    me.normal[0] = value
    me.update()

def set_ny_value(me, value):
    me.normal[1] = value
    me.update()

def set_nz_value(me, value):
    me.normal[2] = value
    me.update()
