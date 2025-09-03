import numpy as np

from .uniforms import BaseBinding, Binding, UniformBase, ct
from .utils import read_shader_file


class CameraUniforms(UniformBase):
    """Uniforms class, derived from ctypes.Structure to ensure correct memory layout"""

    _binding = Binding.CAMERA

    _fields_ = [
        ("model_view", ct.c_float * 16),
        ("model_view_projection", ct.c_float * 16),
        ("rot_mat", ct.c_float * 16),
        ("normal_mat", ct.c_float * 16),
        ("aspect", ct.c_float),
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("padding", ct.c_uint32),
    ]


class Transform:
    def __init__(self):
        self._mat = np.identity(4)
        self._center = np.zeros(3)

    def init(self, pmin, pmax):
        center = 0.5 * (pmin + pmax)
        self._center = center
        scale = 2 / np.linalg.norm(pmax - pmin)
        self._mat = np.identity(4)
        self.translate(-center[0], -center[1], -center[2])
        self.scale(scale)
        if not (pmin[2] == 0 and pmax[2] == 0):
            self.rotate(270, 0)
            self.rotate(0, -20)
            self.rotate(20, 0)

    def translate(self, dx=0.0, dy=0.0, dz=0.0):
        if isinstance(dx, (list, tuple, np.ndarray)) and len(dx) == 3:
            dx, dy, dz = dx
        translation = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])
        self._mat = translation @ self._mat

    def scale(self, s, center=None):
        with self._centering(center):
            self._mat = (
                np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]]) @ self._mat
            )

    def rotate(self, ang_x, ang_y=0, center=None):

        rx = np.radians(ang_x)
        cx = np.cos(rx)
        sx = np.sin(rx)

        rotation_x = np.array(
            [
                [1, 0, 0, 0],
                [0, cx, -sx, 0],
                [0, sx, cx, 0],
                [0, 0, 0, 1],
            ]
        )

        ry = np.radians(ang_y)
        cy = np.cos(ry)
        sy = np.sin(ry)
        rotation_y = np.array(
            [
                [cy, 0, sy, 0],
                [0, 1, 0, 0],
                [-sy, 0, cy, 0],
                [0, 0, 0, 1],
            ]
        )

        with self._centering(center):
            self._mat = rotation_x @ rotation_y @ self._mat

    def set_center(self, center):
        center = np.array(center)
        self.translate(-self.map_point(center))
        self._center = center

    @property
    def mat(self):
        return self._mat

    def map_point(self, point):
        p = np.array([*point, 1.0])
        p = self._mat @ p
        return p[0:3] / p[3]

    class _CenteringContext:
        def __init__(self, transform, center):
            self.transform = transform
            center = transform._center if center is None else center
            self.center = transform.map_point(center)

        def __enter__(self):
            self.transform.translate(-self.center[0], -self.center[1], -self.center[2])

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.transform.translate(self.center[0], self.center[1], self.center[2])

    def _centering(self, center):
        return self._CenteringContext(self, center)


class Camera:
    def __init__(self):
        self.uniforms = None
        self.canvas = None
        self.transform = Transform()
        self._render_function = None
        self._get_position_function = None
        self._is_moving = False
        self._is_rotating = False

    def __setstate__(self, state):
        self.transform = state["transform"]
        self.canvas = None
        self.uniforms = None
        self._render_function = None
        self._get_position_function = None
        self._is_moving = False
        self._is_rotating = False

    def __getstate__(self):
        return {"transform": self.transform}

    def reset(self, pmin, pmax):
        self.transform.init(pmin, pmax)
        self._update_uniforms()

    def set_canvas(self, canvas):
        self.canvas = canvas
        canvas.on_resize(self._update_uniforms)
        if self.uniforms is None:
            self.uniforms = CameraUniforms()
        self._update_uniforms()

    def get_bindings(self) -> list[BaseBinding]:
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("camera.wgsl")

    def __del__(self):
        del self.uniforms

    def register_callbacks(self, input_handler, redraw_function, get_position_function=None):
        self._render_function = redraw_function
        self._get_position_function = get_position_function
        input_handler.on_mousedown(self._on_mousedown)
        input_handler.on_mouseup(self._on_mouseup)
        input_handler.on_mouseout(self._on_mouseup)
        input_handler.on_mousemove(self._on_mousemove)
        input_handler.on_dblclick(self._on_dblclick)
        input_handler.on_wheel(self._on_wheel)

    def unregister_callbacks(self, input_handler):
        input_handler.unregister("mousedown", self._on_mousedown)
        input_handler.unregister("mouseup", self._on_mouseup)
        input_handler.unregister("mouseout", self._on_mouseup)
        input_handler.unregister("mousemove", self._on_mousemove)
        input_handler.unregister("dblclick", self._on_dblclick)
        input_handler.unregister("wheel", self._on_wheel)

    def _on_dblclick(self, ev):
        p = self._get_event_position(ev["canvasX"], ev["canvasY"])
        if p is not None:
            self.transform.set_center(p)
            self._render()

    def _on_mousedown(self, ev):
        if ev["button"] == 0:
            self._is_rotating = True
        if ev["button"] == 1:
            self._is_moving = True

    def _on_mouseup(self, _):
        self._is_moving = False
        self._is_rotating = False
        self._is_zooming = False

    def _on_wheel(self, ev):
        p = self._get_event_position(ev["canvasX"], ev["canvasY"])
        self.transform.scale(1 - ev["deltaY"] / 1000, p)
        self._render()
        if hasattr(ev, "preventDefault"):
            ev.preventDefault()

    def _on_mousemove(self, ev):
        if self._is_rotating:
            s = 0.3
            self.transform.rotate(s * ev["movementY"], s * ev["movementX"])
            self._render()
        if self._is_moving:
            s = 0.01
            self.transform.translate(s * ev["movementX"], -s * ev["movementY"])
            self._render()

    def _render(self):
        self._update_uniforms()
        if self._render_function:
            self._render_function()

    def _get_event_position(self, x, y):
        if self._get_position_function:
            return self._get_position_function(x, y)
        return None

    def _update_uniforms(self):
        if self.canvas is None:
            return
        if self.uniforms is None:
            self.uniforms = CameraUniforms()
        near = 0.1
        far = 10
        fov = 45
        if self.canvas.height == 0:
            aspect = 800 / 600
        else:
            aspect = self.canvas.width / self.canvas.height

        zoom = 1.0
        top = near * (np.tan(np.radians(fov) / 2)) * zoom
        height = 2 * top
        width = aspect * height
        left = -0.5 * width
        right = left + width
        bottom = top - height

        x = 2 * near / (right - left)
        y = 2 * near / (top - bottom)

        a = (right + left) / (right - left)
        b = (top + bottom) / (top - bottom)

        c = -far / (far - near)
        d = (-far * near) / (far - near)

        proj_mat = np.array(
            [
                [x, 0, a, 0],
                [0, y, b, 0],
                [0, 0, c, d],
                [0, 0, -1, 0],
            ]
        )

        view_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -3], [0, 0, 0, 1]])
        model_view = view_mat @ self.transform.mat
        model_view_proj = proj_mat @ model_view
        normal_mat = np.linalg.inv(model_view)
        self.model_view_proj = model_view_proj

        self.uniforms.aspect = aspect
        self.uniforms.model_view[:] = model_view.transpose().flatten()
        self.uniforms.model_view_projection[:] = model_view_proj.transpose().flatten()
        self.uniforms.normal_mat[:] = normal_mat.flatten()
        # self.uniforms.rot_mat[:] = self.transform._rot_mat.flatten()
        self.uniforms.width = self.canvas.width
        self.uniforms.height = self.canvas.height
        self.uniforms.update_buffer()
