from typing import Callable

from .utils import to_js
from .platform import is_pyodide


class InputHandler:
    def __init__(self, html_canvas):
        self._callbacks = {}

        self.html_canvas = html_canvas

        self._callbacks = {}
        self.register_callbacks()

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
                self.emit(event_type, event)

        return wrapper

    def register_callbacks(self):
        from .webgpu_api import create_proxy

        self.unregister_callbacks()
        options = to_js({"capture": True})
        for event in ["mousedown", "mouseup", "mousemove", "wheel", "mouseout"]:
            js_handler = create_proxy(self._handle_js_event(event))
            self.html_canvas.addEventListener(event, js_handler, options)

    def __del__(self):
        self.unregister_callbacks()
