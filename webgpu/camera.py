import numpy as np

from .uniforms import BaseBinding, Binding, UniformBase, ct
from .utils import read_shader_file


class CameraUniforms(UniformBase):
    """Uniforms class, derived from ctypes.Structure to ensure correct memory layout"""

    _binding = Binding.CAMERA

    _fields_ = [
        ("model_view", ct.c_float * 16),
        ("model_view_projection", ct.c_float * 16),
        ("normal_mat", ct.c_float * 16),
        ("aspect", ct.c_float),
        ("padding", ct.c_uint32 * 3),
    ]


class Transform:
    def __init__(self):
        self._mat = np.identity(4)
        self._rot_mat = np.identity(4)
        self._center = (0.5, 0.5, 0)
        self._scale = 1

    def translate(self, dx=0.0, dy=0.0, dz=0.0):
        translation = np.array(
            [[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]]
        )
        self._mat = translation @ self._mat

    def scale(self, s):
        self._scale *= s

    def rotate(self, ang_x, ang_y=0):
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

        self._rot_mat = rotation_x @ rotation_y @ self._rot_mat

    @property
    def mat(self):
        return self._mat @ self._rot_mat @ self._scale_mat @ self._center_mat

    @property
    def normal_mat(self):
        return self._mat @ self._rot_mat @ self._scale_mat @ self._center_mat

    @property
    def _center_mat(self):
        cx, cy, cz = self._center
        return np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 1, -cz], [0, 0, 0, 1]])

    @property
    def _scale_mat(self):
        s = self._scale
        return np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]])


class Camera:
    def __init__(self, canvas):
        self.canvas = canvas
        self.uniforms = CameraUniforms(canvas.device)
        self.transform = Transform()
        self._render_function = None
        self._is_moving = False
        self._is_rotating = False

        canvas.on_resize(self._update_uniforms)

    def get_bindings(self) -> list[BaseBinding]:
        return self.uniforms.get_bindings()

    def get_shader_code(self):
        return read_shader_file("camera.wgsl", __file__)

    def __del__(self):
        del self.uniforms

    def register_callbacks(self, input_handler, redraw_function):
        self._render_function = redraw_function
        input_handler.on_mousedown(self._on_mousedown)
        input_handler.on_mouseup(self._on_mouseup)
        input_handler.on_mouseout(self._on_mouseup)
        input_handler.on_mousemove(self._on_mousemove)
        input_handler.on_wheel(self._on_wheel)

    def unregister_callbacks(self, input_handler):
        input_handler.unregister("mousedown", self._on_mousedown)
        input_handler.unregister("mouseup", self._on_mouseup)
        input_handler.unregister("mouseout", self._on_mouseup)
        input_handler.unregister("mousemove", self._on_mousemove)
        input_handler.unregister("wheel", self._on_wheel)

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
        self.transform.scale(1 - ev["deltaY"] / 1000)
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

    def _update_uniforms(self):
        near = 0.1
        far = 10
        fov = 45
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

        self.uniforms.model_view[:] = model_view.transpose().flatten()
        self.uniforms.model_view_projection[:] = model_view_proj.transpose().flatten()
        self.uniforms.normal_mat[:] = normal_mat.flatten()
        self.uniforms.update_buffer()
