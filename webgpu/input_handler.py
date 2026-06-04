import math
from typing import Callable
from .utils import Lock


class InputHandler:
    _js_handlers: dict

    class Modifiers:
        def __init__(
            self, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
        ):
            self.alt = alt
            self.shift = shift
            self.ctrl = ctrl

        def get_set(self):
            s = set()
            if self.alt is not None:
                s.add("alt" + str(self.alt))
            if self.shift is not None:
                s.add("shift" + str(self.shift))
            if self.ctrl is not None:
                s.add("ctrl" + str(self.ctrl))
            return s

    def __init__(self):
        self._mutex = Lock(True)
        self._callbacks = {}
        self._js_handlers = {}
        self._is_mousedown = False
        self._is_moving = False
        # When True, the JS engine owns the DOM listeners and forwards events
        # via handle_engine_event(); we attach no DOM handlers ourselves.
        self._engine_mode = False

        self.html_canvas = None

        # Touch state tracking
        self._touches = {}  # identifier -> (x, y)
        self._prev_touch_dist = None
        self._prev_touch_centroid = None
        self._touch_is_down = False
        self._last_single_touch_pos = None

        self.on_mousedown(self.__on_mousedown, None, None, None)
        self.on_mouseup(self.__on_mouseup, None, None, None)
        self.on_mousemove(self.__on_mousemove, None, None, None)

    def set_canvas(self, html_canvas):
        if self.html_canvas:
            self.unregister_callbacks()
        self.html_canvas = html_canvas
        if self.html_canvas:
            self.register_callbacks()

    def set_engine_mode(self, enabled: bool):
        """Switch between JS-engine-owned input and the legacy DOM-handler path."""
        self._engine_mode = enabled
        if enabled:
            self.unregister_callbacks()
        elif self.html_canvas is not None:
            self.register_callbacks()

    def handle_engine_event(self, event):
        """Dispatch an event forwarded from the JS engine to Python callbacks."""
        try:
            import pyodide.ffi

            if isinstance(event, pyodide.ffi.JsProxy):
                event = event.to_py()
        except ImportError:
            pass
        etype = event.get("type")
        if etype is not None:
            self.emit(etype, event)

    def __on_mousedown(self, ev):
        self._is_mousedown = True
        self._is_moving = False

    def __on_mouseup(self, ev):
        self._is_mousedown = False

        if not self._is_moving:
            self.emit("click", ev)

    def __on_mousemove(self, ev):
        self._is_moving = True
        if "buttons" in ev and ev["buttons"] != 0:
            self.emit("drag", ev)

    def on(
        self,
        event: str,
        func: Callable,
        alt: bool | None = None,
        shift: bool | None = None,
        ctrl: bool | None = None,
    ):
        if event not in self._callbacks:
            self._callbacks[event] = []

        mod_set = self.Modifiers(alt, shift, ctrl).get_set()
        self._callbacks[event].append((func, mod_set))

    def unregister(self, event, func: Callable):
        if event in self._callbacks:
            for f, mod in self._callbacks[event]:
                if f == func:
                    self._callbacks[event].remove((f, mod))

    def emit(self, event: str, ev: dict, *args):
        mod_set = self.Modifiers(
            ev.get("altKey", False), ev.get("shiftKey", False), ev.get("ctrlKey", False)
        ).get_set()
        if event in self._callbacks:
            for func, mod in self._callbacks[event]:
                if mod.issubset(mod_set):
                    func(ev, *args)

    def on_dblclick(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("dblclick", func, alt, shift, ctrl)

    def on_click(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("click", func, alt, shift, ctrl)

    def on_mousedown(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("mousedown", func, alt, shift, ctrl)

    def on_mouseup(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("mouseup", func, alt, shift, ctrl)

    def on_mouseout(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("mouseout", func, alt, shift, ctrl)

    def on_wheel(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("wheel", func, alt, shift, ctrl)

    def on_mousemove(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("mousemove", func, alt, shift, ctrl)

    def on_drag(
        self, func, alt: bool | None = None, shift: bool | None = None, ctrl: bool | None = None
    ):
        self.on("drag", func, alt, shift, ctrl)

    def unregister_callbacks(self):
        if self.html_canvas is not None:
            with self._mutex:
                for event, _ in self._js_handlers.items():
                    self.html_canvas["on" + event] = None
                self._js_handlers = {}

    def _handle_js_event(self, event_type):
        def wrapper(event):
            if self._engine_mode:
                return
            try:
                import pyodide.ffi

                if isinstance(event, pyodide.ffi.JsProxy):
                    ev = {}
                    for key in dir(event):
                        ev[key] = getattr(event, key)
                    # Extract touch data from JsProxy for touch events
                    if event_type.startswith("touch"):
                        ev["touches"] = self._extract_touch_list(event.touches)
                        ev["changedTouches"] = self._extract_touch_list(
                            event.changedTouches
                        )
                    event = ev
            except ImportError:
                pass

            if event_type.startswith("touch"):
                self._process_touch(event_type, event)
            elif event_type in self._callbacks:
                self.emit(event_type, event)

        return wrapper

    @staticmethod
    def _extract_touch_list(touch_list):
        """Convert a JsProxy TouchList to a Python list of dicts."""
        result = []
        try:
            for i in range(touch_list.length):
                t = touch_list.item(i)
                result.append(
                    {
                        "identifier": t.identifier,
                        "clientX": t.clientX,
                        "clientY": t.clientY,
                    }
                )
        except Exception:
            pass
        return result

    def _process_touch(self, event_type, ev):
        """Process touch events and synthesize mouse/wheel events."""
        touches_data = ev.get("touches", [])
        changed_data = ev.get("changedTouches", [])

        if event_type == "touchstart":
            # Update tracked touches
            for t in changed_data:
                self._touches[t["identifier"]] = (t["clientX"], t["clientY"])

            if len(self._touches) == 1:
                # Single finger down → emit mousedown with button=0
                t = changed_data[0]
                synthetic = {
                    "button": 0,
                    "buttons": 1,
                    "x": t["clientX"],
                    "y": t["clientY"],
                    "movementX": 0,
                    "movementY": 0,
                    "altKey": False,
                    "shiftKey": False,
                    "ctrlKey": False,
                }
                self._touch_is_down = True
                self._last_single_touch_pos = (t["clientX"], t["clientY"])
                self.emit("mousedown", synthetic)
            elif len(self._touches) == 2:
                # Second finger down → initialize pinch/pan state
                points = list(self._touches.values())
                self._prev_touch_dist = self._touch_distance(points[0], points[1])
                self._prev_touch_centroid = self._touch_centroid(points[0], points[1])
                # Cancel single-finger rotation by emitting mouseup, then
                # emit mousedown with button=1 to start panning mode
                if self._touch_is_down:
                    synthetic = {
                        "button": 0,
                        "buttons": 0,
                        "x": 0,
                        "y": 0,
                        "movementX": 0,
                        "movementY": 0,
                        "altKey": False,
                        "shiftKey": False,
                        "ctrlKey": False,
                    }
                    self.emit("mouseup", synthetic)
                    self._touch_is_down = False
                # Emit mousedown with button=1 (middle) to activate panning
                centroid = self._touch_centroid(points[0], points[1])
                pan_start = {
                    "button": 1,
                    "buttons": 4,
                    "x": centroid[0],
                    "y": centroid[1],
                    "movementX": 0,
                    "movementY": 0,
                    "altKey": False,
                    "shiftKey": False,
                    "ctrlKey": False,
                }
                self.emit("mousedown", pan_start)

        elif event_type == "touchmove":
            # Update tracked touches with current positions
            for t in touches_data:
                if t["identifier"] in self._touches:
                    self._touches[t["identifier"]] = (t["clientX"], t["clientY"])

            if len(self._touches) == 1:
                # Single finger move → emit mousemove for rotation
                t = touches_data[0]
                cur = (t["clientX"], t["clientY"])

                prev_pos = getattr(self, "_last_single_touch_pos", None)
                self._last_single_touch_pos = cur
                if prev_pos is not None:
                    dx = cur[0] - prev_pos[0]
                    dy = cur[1] - prev_pos[1]
                else:
                    dx = 0
                    dy = 0

                synthetic = {
                    "button": 0,
                    "buttons": 1,
                    "x": cur[0],
                    "y": cur[1],
                    "movementX": dx,
                    "movementY": dy,
                    "altKey": False,
                    "shiftKey": False,
                    "ctrlKey": False,
                }
                self.emit("mousemove", synthetic)

            elif len(self._touches) >= 2:
                # Two-finger gesture → pinch zoom + pan
                points = list(self._touches.values())
                p0, p1 = points[0], points[1]

                # Pinch: emit wheel event
                dist = self._touch_distance(p0, p1)
                if self._prev_touch_dist is not None and self._prev_touch_dist > 0:
                    delta = self._prev_touch_dist - dist
                    if abs(delta) > 0.5:
                        wheel_ev = {
                            "button": 0,
                            "buttons": 0,
                            "x": (p0[0] + p1[0]) / 2,
                            "y": (p0[1] + p1[1]) / 2,
                            "deltaX": 0,
                            "deltaY": delta * 2,
                            "deltaMode": 0,
                            "movementX": 0,
                            "movementY": 0,
                            "altKey": False,
                            "shiftKey": False,
                            "ctrlKey": False,
                        }
                        self.emit("wheel", wheel_ev)
                self._prev_touch_dist = dist

                # Pan: emit mousemove with middle button (buttons=4)
                centroid = self._touch_centroid(p0, p1)
                if self._prev_touch_centroid is not None:
                    dx = centroid[0] - self._prev_touch_centroid[0]
                    dy = centroid[1] - self._prev_touch_centroid[1]
                    if abs(dx) > 0.5 or abs(dy) > 0.5:
                        pan_ev = {
                            "button": 1,
                            "buttons": 4,
                            "x": centroid[0],
                            "y": centroid[1],
                            "movementX": dx,
                            "movementY": dy,
                            "altKey": False,
                            "shiftKey": False,
                            "ctrlKey": False,
                        }
                        self.emit("mousemove", pan_ev)
                        self.emit("drag", pan_ev)
                self._prev_touch_centroid = centroid

        elif event_type in ("touchend", "touchcancel"):
            # Remove ended touches
            for t in changed_data:
                self._touches.pop(t["identifier"], None)

            if len(self._touches) == 0:
                # All fingers lifted → end any active gesture
                end_ev = {
                    "button": 0,
                    "buttons": 0,
                    "x": changed_data[0]["clientX"] if changed_data else 0,
                    "y": changed_data[0]["clientY"] if changed_data else 0,
                    "movementX": 0,
                    "movementY": 0,
                    "altKey": False,
                    "shiftKey": False,
                    "ctrlKey": False,
                }
                self.emit("mouseup", end_ev)
                self._touch_is_down = False
                self._prev_touch_dist = None
                self._prev_touch_centroid = None
                self._last_single_touch_pos = None
            elif len(self._touches) == 1:
                # Went from 2 fingers to 1 → end pan, restart rotation
                # First emit mouseup to end panning
                end_ev = {
                    "button": 1,
                    "buttons": 0,
                    "x": 0,
                    "y": 0,
                    "movementX": 0,
                    "movementY": 0,
                    "altKey": False,
                    "shiftKey": False,
                    "ctrlKey": False,
                }
                self.emit("mouseup", end_ev)
                self._prev_touch_dist = None
                self._prev_touch_centroid = None
                remaining = list(self._touches.values())[0]
                self._last_single_touch_pos = remaining
                # Re-emit mousedown for single-finger rotation
                synthetic = {
                    "button": 0,
                    "buttons": 1,
                    "x": remaining[0],
                    "y": remaining[1],
                    "movementX": 0,
                    "movementY": 0,
                    "altKey": False,
                    "shiftKey": False,
                    "ctrlKey": False,
                }
                self._touch_is_down = True
                self.emit("mousedown", synthetic)

    @staticmethod
    def _touch_distance(p0, p1):
        """Euclidean distance between two touch points."""
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _touch_centroid(p0, p1):
        """Midpoint between two touch points."""
        return ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)

    def register_callbacks(self):
        if self._engine_mode:
            return

        from .platform import create_event_handler

        # Prevent default touch actions (scrolling/zooming) on the canvas
        try:
            self.html_canvas.style.touchAction = "none"
        except Exception:
            pass

        for event in [
            "mousedown",
            "mouseup",
            "mousemove",
            "wheel",
            "mouseout",
            "dblclick",
            "touchstart",
            "touchmove",
            "touchend",
            "touchcancel",
        ]:
            js_handler = create_event_handler(
                self._handle_js_event(event),
                prevent_default=True,
                stop_propagation=event not in ["mousemove", "mouseout", "touchmove"],
            )
            self.html_canvas["on" + event] = js_handler
            self._js_handlers[event] = js_handler

    def __del__(self):
        self.unregister_callbacks()
