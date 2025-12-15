from typing import Callable
from .utils import Lock


class InputHandler:
    _js_handlers: dict

    class Modifiers:
        def __init__(
            self, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
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

        self.html_canvas = None

        self.on_mousedown(self.__on_mousedown, None, None, None)
        self.on_mouseup(self.__on_mouseup, None, None, None)
        self.on_mousemove(self.__on_mousemove, None, None, None)

    def set_canvas(self, html_canvas):
        if self.html_canvas:
            self.unregister_callbacks()
        self.html_canvas = html_canvas
        if self.html_canvas:
            self.register_callbacks()

    def __on_mousedown(self, ev):
        self._is_mousedown = True
        self._is_moving = False
        self._mouse_button_down = ev.get("button", 0)

    def __on_mouseup(self, ev):
        self._is_mousedown = False

        if not self._is_moving:
            self.emit("click", ev)

    def __on_mousemove(self, ev):
        self._is_moving = True
        if self._is_mousedown:
            ev["button"] = self._mouse_button_down
            self.emit("drag", ev)

    def on(
        self,
        event: str,
        func: Callable,
        alt: bool | None = False,
        shift: bool | None = False,
        ctrl: bool | None = False,
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
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
    ):
        self.on("dblclick", func, alt, shift, ctrl)

    def on_click(
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
    ):
        self.on("click", func, alt, shift, ctrl)

    def on_mousedown(
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
    ):
        self.on("mousedown", func, alt, shift, ctrl)

    def on_mouseup(
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
    ):
        self.on("mouseup", func, alt, shift, ctrl)

    def on_mouseout(
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
    ):
        self.on("mouseout", func, alt, shift, ctrl)

    def on_wheel(
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
    ):
        self.on("wheel", func, alt, shift, ctrl)

    def on_mousemove(
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
    ):
        self.on("mousemove", func, alt, shift, ctrl)

    def on_drag(
        self, func, alt: bool | None = False, shift: bool | None = False, ctrl: bool | None = False
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
            if event_type in self._callbacks:
                try:
                    import pyodide.ffi

                    if isinstance(event, pyodide.ffi.JsProxy):
                        ev = {}
                        for key in dir(event):
                            ev[key] = getattr(event, key)
                        event = ev
                except ImportError:
                    pass

                self.emit(event_type, event)

        return wrapper

    def register_callbacks(self):
        from .platform import create_event_handler

        for event in ["mousedown", "mouseup", "mousemove", "wheel", "mouseout", "dblclick"]:
            js_handler = create_event_handler(
                self._handle_js_event(event),
                prevent_default=True,
                stop_propagation=event not in ["mousemove", "mouseout"],
            )
            self.html_canvas["on" + event] = js_handler
            self._js_handlers[event] = js_handler

    def __del__(self):
        self.unregister_callbacks()
