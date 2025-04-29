import uuid
from typing import Callable

from .camera import Camera
from .canvas import Canvas
from .light import Light
from .utils import BaseBinding, is_pyodide, create_bind_group, get_device
from .webgpu_api import (
    CommandEncoder,
    CompareFunction,
    DepthStencilState,
    Device,
    FragmentState,
    PrimitiveState,
    PrimitiveTopology,
    VertexState,
)


class RenderOptions:
    viewport: tuple[int, int, int, int, float, float]
    canvas: Canvas

    def __init__(self, canvas):
        self.canvas = canvas
        self.light = Light(self.device)
        self.camera = Camera(canvas)

    @property
    def device(self) -> Device:
        return self.canvas.device

    def update_buffers(self):
        self.camera._update_uniforms()

    def get_bindings(self):
        return [
            *self.light.get_bindings(),
            *self.camera.get_bindings(),
        ]

    def begin_render_pass(self, command_encoder: CommandEncoder, **kwargs):
        load_op = command_encoder.getLoadOp()

        render_pass_encoder = command_encoder.beginRenderPass(
            self.canvas.color_attachments(load_op),
            self.canvas.depth_stencil_attachment(load_op),
            **kwargs,
        )

        render_pass_encoder.setViewport(
            0, 0, self.canvas.width, self.canvas.height, 0.0, 1.0
        )

        return render_pass_encoder


class BaseRenderObject:
    options: RenderOptions
    label: str = ""
    _timestamp: float = -1
    active: bool = True

    def __init__(self, label=None):
        if label is None:
            self.label = self.__class__.__name__
        else:
            self.label = label

    def get_bounding_box(self) -> tuple[list[float], list[float]] | None:
        return ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    def update(self, timestamp):
        if timestamp == self._timestamp:
            return
        self._timestamp = timestamp
        self.create_render_pipeline()

    @property
    def device(self) -> Device:
        return get_device()

    @property
    def canvas(self) -> Canvas:
        return self.options.canvas

    def create_render_pipeline(self) -> None:
        raise NotImplementedError

    def render(self, encoder: CommandEncoder):
        raise NotImplementedError

    def get_bindings(self) -> list[BaseBinding]:
        raise NotImplementedError

    def get_shader_code(self) -> str:
        raise NotImplementedError

    def add_options_to_gui(self, gui):
        pass


class MultipleRenderObject(BaseRenderObject):
    def __init__(self, render_objects):
        self.render_objects = render_objects

    def update(self, timestamp):
        for r in self.render_objects:
            r.options = self.options
            r.update(timestamp=timestamp)

    def redraw(self, timestamp=None):
        for r in self.render_objects:
            r.redraw(timestamp=timestamp)

    def render(self, encoder):
        for r in self.render_objects:
            r.render(encoder)


class RenderObject(BaseRenderObject):
    """Base class for render objects"""

    n_vertices: int = 0
    n_instances: int = 1
    topology: PrimitiveTopology = PrimitiveTopology.triangle_list
    depthBias: int = 0
    vertex_entry_point: str = "vertex_main"
    fragment_entry_point: str = "fragment_main"

    def create_render_pipeline(self) -> None:
        shader_module = self.device.createShaderModule(self.get_shader_code())
        layout, self.group = create_bind_group(self.device, self.get_bindings())
        self.pipeline = self.device.createRenderPipeline(
            self.device.createPipelineLayout([layout]),
            vertex=VertexState(
                module=shader_module, entryPoint=self.vertex_entry_point
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint=self.fragment_entry_point,
                targets=[self.options.canvas.color_target],
            ),
            primitive=PrimitiveState(topology=self.topology),
            depthStencil=DepthStencilState(
                format=self.options.canvas.depth_format,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                depthBias=self.depthBias,
            ),
            multisample=self.options.canvas.multisample,
        )

    def render(self, encoder: CommandEncoder) -> None:
        render_pass = self.options.begin_render_pass(encoder)
        render_pass.setPipeline(self.pipeline)
        render_pass.setBindGroup(0, self.group)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()
