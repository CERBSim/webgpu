from .utils import _is_pyodide
from .render_object import _render_objects, RenderObject
import uuid
from typing import Callable


class LilGUI:
    def __init__(self, canvas_id, scene_id):
        self.canvas_id = canvas_id
        self.scene_id = scene_id

    def slider(
        self,
        value: float,
        func: Callable[[RenderObject, float], None],
        objects: list[RenderObject] | RenderObject = [],
        min=0.0,
        max=1.0,
        label="Slider",
    ):
        self._connect(
            func,
            option={
                "type": "slider",
                "min": min,
                "max": max,
                "value": value,
                "label": label,
            },
            render_objects=objects,
        )

    def _connect(
        self, func, option, render_objects: list[RenderObject] | RenderObject = []
    ):
        if not isinstance(render_objects, list):
            render_objects = [render_objects]
        if _is_pyodide:

            def _func(*args):
                from webgpu.scene import redraw_scene

                func(*args)
                redraw_scene(self.scene_id)

            create_gui_option(self.canvas_id, option, _func)
        else:
            from .jupyter import _encode_function, _encode_data, run_code_in_pyodide

            render_objects = [str(obj._id) for obj in render_objects]
            run_code_in_pyodide(
                f"import webgpu.lilgui; webgpu.lilgui._receive('{self.canvas_id}', '{self.scene_id}', {option}, {render_objects}, '{_encode_data(_encode_function(func))}')"
            )


def _receive(canvas_id, scene_id, option, render_objects: list[str], func: str):
    assert _is_pyodide

    def _func(*args):
        from webgpu.jupyter import _decode_function, _decode_data
        from webgpu.scene import redraw_scene

        ro = [_render_objects[uuid.UUID(obj_id)] for obj_id in render_objects]
        if len(ro) == 1:
            ro = ro[0]
        f = _decode_function(_decode_data(func))
        f(ro, *args)
        redraw_scene(scene_id)

    create_gui_option(canvas_id, option, _func)


def create_gui_option(canvas_id, option, f):
    import pyodide.ffi
    import js

    gui = getattr(js.lil_guis, canvas_id)
    if option["type"] == "slider":
        slider = pyodide.ffi.to_js({option["label"]: option["value"]})
        gui.add(slider, option["label"], option["min"], option["max"]).onChange(
            pyodide.ffi.create_proxy(f)
        )
