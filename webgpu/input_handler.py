from typing import Callable

from .platform import is_pyodide
from .utils import to_js
import threading


class InputHandler:
    def __init__(self):
        self._mutex = threading.Lock()
        self._callbacks = {}

        self.html_canvas = None

        self._callbacks = {}

        self.on_mousedown(self.__on_mousedown)
        self.on_mouseup(self.__on_mouseup)
        self.on_mousemove(self.__on_mousemove)

    def set_canvas(self, html_canvas):
        self.html_canvas = html_canvas
        self.register_callbacks()

    def __on_mousedown(self, _):
        self._is_mousedown = True
        self._is_moving = False

    def __on_mouseup(self, ev):
        self._is_mousedown = False

        if not self._is_moving:
            self.emit("click", ev)

    def __on_mousemove(self, ev):
        self._is_moving = True

    def on(self, event: str, func: Callable):
        if event not in self._callbacks:
            self._callbacks[event] = []

        self._callbacks[event].append(func)

    def unregister(self, event, func: Callable):
        if event in self._callbacks:
            self._callbacks[event].remove(func)

    def emit(self, event: str, *args):
        if event in self._callbacks:
            for func in self._callbacks[event]:
                func(*args)

    def on_click(self, func):
        self.on("click", func)

    def on_mousedown(self, func):
        self.on("mousedown", func)

    def on_mouseup(self, func):
        self.on("mouseup", func)

    def on_mouseout(self, func):
        self.on("mouseout", func)

    def on_wheel(self, func):
        self.on("wheel", func)

    def on_mousemove(self, func):
        self.on("mousemove", func)

    def unregister_callbacks(self):
        with self._mutex:
            for event in self._callbacks:
                for func in self._callbacks[event]:
                    self.html_canvas.removeEventListener(event, func)
                    if is_pyodide:
                        func.destroy()
            self._callbacks = {}

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

                if "x" in event and "y" in event:
                    rect = self.html_canvas.getBoundingClientRect()
                    event["canvasX"] = event["x"] - int(rect.x)
                    event["canvasY"] = event["y"] - int(rect.y)
                self.emit(event_type, event)

        return wrapper

    def register_callbacks(self):
        from .platform import create_proxy

        options = to_js({"capture": True})
        for event in ["mousedown", "mouseup", "mousemove", "wheel", "mouseout"]:
            js_handler = create_proxy(self._handle_js_event(event))
            self.html_canvas.addEventListener(event, js_handler, options)

    def __del__(self):
        self.unregister_callbacks()
