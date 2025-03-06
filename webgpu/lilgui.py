import uuid
from typing import Callable

from .render_object import RenderObject, _render_objects
from .utils import _is_pyodide


class Folder:
    def __init__(self, label: str | None, canvas_id, scene_id):
        self.label = label
        self._id = str(uuid.uuid4())
        self.canvas_id = canvas_id
        self.scene_id = scene_id

    def folder(self, label: str, closed=False):
        folder = Folder(label, self.canvas_id, self.scene_id)
        self._connect(
            None,
            option={
                "type": "folder",
                "label": label,
                "folder_id": folder._id,
                "closed": closed,
            },
        )
        return folder

    def checkbox(
        self,
        label: str,
        value: bool,
        func: Callable[[RenderObject, bool], None],
        objects: list[RenderObject] | RenderObject = [],
    ):
        self._connect(
            func,
            option={"type": "checkbox", "value": value, "label": label},
            render_objects=objects,
        )

    def value(
        self,
        label: str,
        value: object,
        func: Callable[[RenderObject, object], None],
        objects: list[RenderObject] | RenderObject = [],
    ):
        self._connect(
            func,
            option={"type": "value", "value": value, "label": label},
            render_objects=objects,
        )

    def dropdown(
        self,
        values: dict[str, object],
        func: Callable[[RenderObject, object], None],
        value: str | None = None,
        objects: list[RenderObject] | RenderObject = [],
        label="Dropdown",
    ):
        if value is None:
            value = list(values.keys())[0]
        self._connect(
            func,
            option={
                "type": "dropdown",
                "values": values,
                "label": label,
                "value": value,
            },
            render_objects=objects,
        )

    def slider(
        self,
        value: float,
        func: Callable[[RenderObject, float], None],
        objects: list[RenderObject] | RenderObject = [],
        min=0.0,
        max=1.0,
        step=None,
        label="Slider",
    ):
        if step is None:
            step = (max - min) / 100
        self._connect(
            func,
            option={
                "type": "slider",
                "min": min,
                "max": max,
                "value": value,
                "label": label,
                "step": step,
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

            create_gui_option(self.canvas_id, option, _func, self._id)
        else:
            from .jupyter import (_encode_data, _encode_function,
                                  run_code_in_pyodide)

            render_objects = [str(obj._id) for obj in render_objects]
            run_code_in_pyodide(
                f"import webgpu.lilgui; webgpu.lilgui._receive('{self.canvas_id}', '{self.scene_id}', {option}, {render_objects}, '{_encode_data(_encode_function(func) if func is not None else None)}', '{self._id}')"
            )


class LilGUI(Folder):
    def __init__(self, canvas_id, scene_id):
        super().__init__(None, canvas_id, scene_id)


def _receive(
    canvas_id, scene_id, option, render_objects: list[str], func: str, folder_id: str
):
    assert _is_pyodide

    def _func(*args):
        from webgpu.jupyter import _decode_data, _decode_function
        from webgpu.scene import redraw_scene

        ro = [_render_objects[str(uuid.UUID(obj_id))] for obj_id in render_objects]
        if len(ro) == 1:
            ro = ro[0]
        f = _decode_function(_decode_data(func))
        f(ro, *args)
        redraw_scene(scene_id)

    create_gui_option(canvas_id, option, _func, folder_id)


def create_gui_option(canvas_id, option, f, folder_id):
    import js
    import pyodide.ffi

    gui = getattr(js.lil_guis, canvas_id)
    if not hasattr(gui, "my_gui_folders"):
        gui.my_gui_folders = {folder_id: gui}
    gui = gui.my_gui_folders[folder_id]
    if option["type"] == "slider":
        slider = pyodide.ffi.to_js({option["label"]: option["value"]})
        gui.add(
            slider, option["label"], option["min"], option["max"], option["step"]
        ).onChange(pyodide.ffi.create_proxy(f))
    if option["type"] == "dropdown":
        dropdown = pyodide.ffi.to_js({option["label"]: option["value"]})
        gui.add(
            dropdown, option["label"], pyodide.ffi.to_js(option["values"])
        ).onChange(pyodide.ffi.create_proxy(f))

    if option["type"] == "value":
        val = pyodide.ffi.to_js({option["label"]: option["value"]})
        gui.add(val, option["label"]).onChange(pyodide.ffi.create_proxy(f))

    if option["type"] == "checkbox":
        checkbox = pyodide.ffi.to_js({option["label"]: option["value"]})
        gui.add(checkbox, option["label"]).onChange(pyodide.ffi.create_proxy(f))

    if option["type"] == "folder":
        folder = gui.addFolder(option["label"])
        gui.my_gui_folders[option["folder_id"]] = folder
        if option["closed"]:
            folder.close()
