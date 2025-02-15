from .utils import to_js
from typing import Callable


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

    def on_mousewheel(self, func):
        self.on("mousewheel", func)

    def on_mousemove(self, func):
        self.on("mousemove", func)

    def unregister_callbacks(self):
        for event in self._callbacks:
            for func in self._callbacks[event]:
                self.html_canvas.removeEventListener(event, func)
                func.destroy()
        self._callbacks = {}

    def _handle_js_event(self, event):
        if event.type in self._callbacks:
            self.emit(event.type, event)

    def register_callbacks(self):
        from pyodide.ffi import create_proxy

        self.unregister_callbacks()
        js_handler = create_proxy(self._handle_js_event)
        options = to_js({"capture": True})
        for event in ["mousedown", "mouseup", "mousemove", "mousewheel", "mouseout"]:
            self.html_canvas.addEventListener(event, js_handler, options)

    def __del__(self):
        self.unregister_callbacks()
