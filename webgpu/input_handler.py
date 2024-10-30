import js
import numpy as np

from .utils import to_js


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
    def _center_mat(self):
        cx, cy, cz = self._center
        return np.array([[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 1, -cz], [0, 0, 0, 1]])

    @property
    def _scale_mat(self):
        s = self._scale
        return np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]])


class InputHandler:
    def __init__(self, canvas, uniforms, render_function=None):
        self.canvas = canvas
        self.uniforms = uniforms
        self.render_function = render_function
        self._is_moving = False
        self._is_rotating = False

        self._callbacks = {}
        self.register_callbacks()
        self.transform = Transform()

        self.transform.scale(2)

        self._update_uniforms()

    def _update_uniforms(self):
        near = 0.1
        far = 10
        fov = 45
        aspect = 1.0

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

        mat = proj_mat @ view_mat @ self.transform.mat
        mat = mat.transpose()
        mat = mat.flatten()
        for i in range(16):
            self.uniforms.mat[i] = mat[i]

    def _render(self):
        self._update_uniforms()
        if self.render_function:
            js.requestAnimationFrame(self.render_function)

    def on_mousedown(self, ev):
        if ev.button == 0:
            self._is_rotating = True
        if ev.button == 1:
            self._is_moving = True

    def on_mouseup(self, _):
        global _is_moving
        self._is_moving = False
        self._is_rotating = False
        self._is_zooming = False

    def on_mousewheel(self, ev):
        self.transform.scale(1 - ev.deltaY / 1000)
        self._render()

    def on_mousemove(self, ev):
        if self._is_rotating:
            s = 0.3
            self.transform.rotate(s * ev.movementY, s * ev.movementX)
            self._render()
        if self._is_moving:
            s = 0.01
            self.transform.translate(s * ev.movementX, -s * ev.movementY)
            self._render()

            if self.render_function:
                js.requestAnimationFrame(self.render_function)

    def unregister_callbacks(self):
        for event in self._callbacks:
            for func in self._callbacks[event]:
                self.canvas.removeEventListener(event, func)
                func.destroy()
        self._callbacks = {}

    def on(self, event, func):
        from pyodide.ffi import create_proxy

        func = create_proxy(func)
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(func)
        self.canvas.addEventListener(event, func, to_js({"capture": True}))

    def register_callbacks(self):
        self.unregister_callbacks()
        self.on("mousedown", self.on_mousedown)
        self.on("mouseup", self.on_mouseup)
        self.on("mousemove", self.on_mousemove)
        self.on("wheel", self.on_mousewheel)

    def __del__(self):
        self.unregister_callbacks()
        if self.render_function:
            js.cancelAnimationFrame(self.render_function)
            self.render_function.destroy()
            self.render_function = None
