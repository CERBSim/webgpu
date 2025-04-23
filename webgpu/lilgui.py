from typing import Callable

from . import platform


class Folder:
    def __init__(self, label: str | None, container, scene):
        self.label = label
        self.container = container
        self.scene = scene
        self.gui = None

    def folder(self, label: str, closed=False):
        folder = Folder(label, self.container, self.scene)
        folder.gui = self.gui.addFolder(label)
        if closed:
            folder.gui.close()
        return folder

    def add(self, label: str, value, func: Callable, *args):
        def f(*args):
            func(*args)
            self.scene.render()

        return self.gui.add({label: value}, label, *args).onChange(
            platform.create_proxy(f)
        )

    def checkbox(
        self,
        label: str,
        value: bool,
        func: Callable[[bool], None],
    ):
        return self.add(label, value, func)

    def value(
        self,
        label: str,
        value: object,
        func: Callable[[object], None],
    ):
        return self.add(label, value, func)

    def dropdown(
        self,
        values: dict[str, object],
        func: Callable[[object], None],
        value: str | None = None,
        label="Dropdown",
    ):
        if value is None:
            value = list(values.keys())[0]

        return self.add(label, value, func, values)

    def slider(
        self,
        value: float,
        func: Callable[[float], None],
        min=0.0,
        max=1.0,
        step=None,
        label="Slider",
    ):
        if step is None:
            step = (max - min) / 100

        return self.add(label, value, func, min, max, step)


class LilGUI(Folder):
    def __init__(self, container, scene):
        super().__init__(None, container, scene)
        self.gui = platform.js.createLilGUI({"container": container})
