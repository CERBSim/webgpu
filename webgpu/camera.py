import numpy as np

from .uniforms import BaseBinding, Binding, UniformBase, ct
from .utils import read_shader_file, Lock


class CameraUniforms(UniformBase):
    """Uniforms class, derived from ctypes.Structure to ensure correct memory layout"""

    _binding = Binding.CAMERA

    _fields_ = [
        ("view", ct.c_float * 16),
        ("model_view", ct.c_float * 16),
        ("model_view_projection", ct.c_float * 16),
        ("rot_mat", ct.c_float * 16),
        ("normal_mat", ct.c_float * 16),
        ("aspect", ct.c_float),
        ("width", ct.c_uint32),
        ("height", ct.c_uint32),
        ("padding", ct.c_uint32),
    ]

    def update(self, transform, canvas):
        """Recompute projection/model-view matrices from transform and canvas dimensions.

        Returns (model_view_proj, model_view) matrices, or (None, None) if canvas is unavailable.
        """
        if canvas is None or canvas.height == 0:
            return None, None

        near = 0.1
        far = 10
        fov = 45
        aspect = canvas.width / canvas.height

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
        model_view = view_mat @ transform.mat
        model_view_proj = proj_mat @ model_view
        normal_mat = np.linalg.inv(model_view)

        self.aspect = aspect
        self.view[:] = view_mat.transpose().flatten()
        self.model_view[:] = model_view.transpose().flatten()
        self.model_view_projection[:] = model_view_proj.transpose().flatten()
        self.normal_mat[:] = normal_mat.flatten()

        # Extract pure rotation from model_view (upper-left 3x3, normalize columns to remove scale)
        mv33 = model_view[:3, :3]
        col_norms = np.linalg.norm(mv33, axis=0)
        col_norms[col_norms == 0] = 1.0
        rot33 = mv33 / col_norms
        rot_mat4 = np.identity(4)
        rot_mat4[:3, :3] = rot33
        self.rot_mat[:] = rot_mat4.transpose().flatten()
        self.width = canvas.width
        self.height = canvas.height
        self.update_buffer()

        return model_view_proj, model_view


class Transform:
    """3D transform with translation/rotation/scale around a configurable center."""

    def __init__(self):
        self._mat = np.identity(4)
        self._center = np.zeros(3)

    def copy(self):
        t = Transform()
        t._mat = self._mat.copy()
        t._center = self._center.copy()
        return t

    def init(self, pmin, pmax):
        """Initialize the transform to frame the axis-aligned box [pmin, pmax]."""
        center = 0.5 * (pmin + pmax)
        self._center = center
        scale = 2 / np.linalg.norm(pmax - pmin)
        self._mat = np.identity(4)
        self.translate(-center[0], -center[1], -center[2])
        self.scale(scale)
        if not (abs(pmin[2]) < 1e-12 and abs(pmax[2]) < 1e-12):
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

    def reset_xy(self, flip: bool = False):
        """Reset to a view looking along +Z onto the XY plane, optionally flipped."""
        s = np.linalg.norm(self._mat[:3, :3], axis=0)[0]  # current uniform scale
        self._mat = np.identity(4)
        self.translate(-self._center[0], -self._center[1], -self._center[2])
        self.scale(s)
        if flip:
            self.rotate(0, 180)

    def reset_xz(self, flip: bool = False):
        """Reset to a view looking along +Y onto the XZ plane, optionally flipped."""
        self.reset_xy()
        self.rotate(-90, 0)
        if flip:
            self.rotate(0, 180)

    def reset_yz(self, flip: bool = False):
        """Reset to a view looking along +X onto the YZ plane, optionally flipped."""
        self.reset_xy()
        self.rotate(-90, 0)
        self.rotate(0, -90)
        if flip:
            self.rotate(0, 180)

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
    """Interactive camera that holds a transform and notifies registered observers on change."""

    def __init__(self):
        self.transform = Transform()
        self._observers = []
        self._observers_lock = Lock()
        self._is_moving = False
        self._is_rotating = False
        self._registered_handlers = {}

    def __setstate__(self, state):
        """Restore pickled camera state (only the transform)."""
        self.transform = state["transform"]
        self._observers = []
        self._observers_lock = Lock()
        self._is_moving = False
        self._is_rotating = False
        self._registered_handlers = {}

    def __getstate__(self):
        """Return a minimal picklable representation of the camera."""
        return {"transform": self.transform}

    def register_observer(self, callback):
        """Register a callback to be called when the camera transform changes.
        
        Idempotent: registering the same callback twice has no additional effect.
        """
        with self._observers_lock:
            if callback not in self._observers:
                self._observers.append(callback)
                print(f"[Camera] register_observer: now {len(self._observers)} observers")

    def unregister_observer(self, callback):
        """Remove a previously registered observer callback."""
        with self._observers_lock:
            old_len = len(self._observers)
            self._observers = [cb for cb in self._observers if cb is not callback]
            if len(self._observers) != old_len:
                print(f"[Camera] unregister_observer: now {len(self._observers)} observers")

    def _notify_observers(self):
        """Notify all registered observers that the transform has changed."""
        with self._observers_lock:
            observers = list(self._observers)
        for cb in observers:
            cb()

    def reset(self, pmin, pmax):
        """Fit the camera to the axis-aligned box [pmin, pmax] and notify observers."""
        self.transform.init(pmin, pmax)
        self._notify_observers()

    def reset_xy(self, flip: bool = False):
        """Reset to a top-down XY view and notify observers."""
        self.transform.reset_xy(flip)
        self._notify_observers()

    def reset_xz(self, flip: bool = False):
        """Reset to an XZ view and notify observers."""
        self.transform.reset_xz(flip)
        self._notify_observers()

    def reset_yz(self, flip: bool = False):
        """Reset to a YZ view and notify observers."""
        self.transform.reset_yz(flip)
        self._notify_observers()

    def get_shader_code(self):
        return read_shader_file("camera.wgsl")

    def register_callbacks(self, input_handler, get_position_fn=None):
        """Register mouse/wheel handlers on the given input_handler.
        
        Idempotent: unregisters existing handlers for this input_handler first.
        get_position_fn is used for dblclick center-on-point (per-canvas).
        """
        # Unregister first to avoid duplicates
        self.unregister_callbacks(input_handler)

        def on_dblclick(ev):
            if get_position_fn:
                p = get_position_fn(ev["canvasX"], ev["canvasY"])
                if p is not None:
                    self.transform.set_center(p)
                    self._notify_observers()

        handlers = {
            'mousedown': self._on_mousedown,
            'mouseup': self._on_mouseup,
            'mouseout': self._on_mouseup,
            'mousemove': self._on_mousemove,
            'wheel': self._on_wheel,
            'dblclick': on_dblclick,
        }
        self._registered_handlers[id(input_handler)] = handlers

        input_handler.on_mousedown(handlers['mousedown'], ctrl=False, shift=False, alt=False)
        input_handler.on_mouseup(handlers['mouseup'], ctrl=False, shift=False, alt=False)
        input_handler.on_mouseout(handlers['mouseout'], ctrl=False, shift=False, alt=False)
        input_handler.on_mousemove(handlers['mousemove'], ctrl=False, shift=False, alt=False)
        input_handler.on_dblclick(handlers['dblclick'], ctrl=False, shift=False, alt=False)
        input_handler.on_wheel(handlers['wheel'], ctrl=False, shift=False, alt=False)

    def unregister_callbacks(self, input_handler):
        """Remove previously registered handlers from the given input_handler."""
        key = id(input_handler)
        handlers = self._registered_handlers.pop(key, None)
        if handlers is None:
            return
        input_handler.unregister("mousedown", handlers['mousedown'])
        input_handler.unregister("mouseup", handlers['mouseup'])
        input_handler.unregister("mouseout", handlers['mouseout'])
        input_handler.unregister("mousemove", handlers['mousemove'])
        input_handler.unregister("dblclick", handlers['dblclick'])
        input_handler.unregister("wheel", handlers['wheel'])

    def _on_mousedown(self, ev):
        if ev["button"] == 0:
            self._is_rotating = True
        if ev["button"] == 1:
            self._is_moving = True

    def _on_mouseup(self, _):
        self._is_moving = False
        self._is_rotating = False

    def _on_wheel(self, ev):
        self.transform.scale(1 - ev["deltaY"] / 1000, self.transform._center)
        self._notify_observers()
        if hasattr(ev, "preventDefault"):
            ev.preventDefault()

    def _on_mousemove(self, ev):
        if self._is_rotating:
            s = 0.3
            self.transform.rotate(s * ev["movementY"], s * ev["movementX"])
            self._notify_observers()
        elif self._is_moving:
            s = 0.01
            self.transform.translate(s * ev["movementX"], -s * ev["movementY"])
            self._notify_observers()
