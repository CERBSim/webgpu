from typing import Callable

from .camera import Camera
from .canvas import Canvas
from .light import Light
from .utils import BaseBinding, create_bind_group, get_device, preprocess_shader_code
from .webgpu_api import (
    Buffer,
    CommandEncoder,
    CompareFunction,
    DepthStencilState,
    Device,
    FragmentState,
    PrimitiveState,
    PrimitiveTopology,
    VertexBufferLayout,
    VertexState,
)


class RenderOptions:
    viewport: tuple[int, int, int, int, float, float]
    canvas: Canvas
    command_encoder: CommandEncoder
    timestamp: float

    def __init__(self, canvas):
        self.canvas = canvas
        self.light = Light()
        self.camera = Camera(canvas)

    @property
    def device(self) -> Device:
        return get_device()

    def update_buffers(self):
        self.camera._update_uniforms()
        self.light._update_uniforms()

    def get_bindings(self):
        return [
            *self.light.get_bindings(),
            *self.camera.get_bindings(),
        ]

    def begin_render_pass(self, **kwargs):
        load_op = self.command_encoder.getLoadOp()

        render_pass_encoder = self.command_encoder.beginRenderPass(
            self.canvas.color_attachments(load_op),
            self.canvas.depth_stencil_attachment(load_op),
            **kwargs,
        )

        render_pass_encoder.setViewport(0, 0, self.canvas.width, self.canvas.height, 0.0, 1.0)

        return render_pass_encoder


def check_timestamp(callback: Callable):
    """Decorator to handle updates for render objects. The function is only called if the timestamp has changed."""

    def wrapper(self, options, *args, **kwargs):
        if options.timestamp == self._timestamp:
            return
        callback(self, options, *args, **kwargs)
        self._timestamp = options.timestamp

    return wrapper


class BaseRenderer:
    label: str = ""
    _timestamp: float = -1
    active: bool = True
    shader_defines: dict[str, str] = None

    def __init__(self, label=None):
        self.shader_defines = {}
        if label is None:
            self.label = self.__class__.__name__
        else:
            self.label = label

    def get_bounding_box(self) -> tuple[list[float], list[float]] | None:
        return None

    def update(self, options: RenderOptions) -> None:
        pass

    @check_timestamp
    def _update_and_create_render_pipeline(self, options: RenderOptions) -> None:
        self.update(options)
        self.create_render_pipeline(options)

    @property
    def device(self) -> Device:
        return get_device()

    def create_render_pipeline(self, options: RenderOptions) -> None:
        pass

    def render(self, options: RenderOptions) -> None:
        raise NotImplementedError

    def get_bindings(self) -> list[BaseBinding]:
        return []

    def get_shader_code(self) -> str:
        raise NotImplementedError

    def _get_preprocessed_shader_code(self) -> str:
        return preprocess_shader_code(self.get_shader_code(), defines=self.shader_defines)

    def add_options_to_gui(self, gui):
        pass

    def set_needs_update(self) -> None:
        self._timestamp = -1

    @property
    def needs_update(self) -> bool:
        return self._timestamp == -1


class MultipleRenderer(BaseRenderer):
    def __init__(self, render_objects):
        super().__init__()
        self.render_objects = render_objects

    def update(self, options: RenderOptions) -> None:
        for r in self.render_objects:
            r.update(options)

    @check_timestamp
    def _update_and_create_render_pipeline(self, options: RenderOptions) -> None:
        self.update(options)
        for r in self.render_objects:
            r.create_render_pipeline(options)

    def render(self, options: RenderOptions) -> None:
        for r in self.render_objects:
            r.render(options)

    def set_needs_update(self) -> None:
        super().set_needs_update()
        for r in self.render_objects:
            r.set_needs_update()

    @property
    def needs_update(self) -> bool:
        ret = self._timestamp == -1
        for r in self.render_objects:
            ret = ret or r.needs_update

        return ret


class Renderer(BaseRenderer):
    """Base class for renderer classes"""

    n_vertices: int = 0
    n_instances: int = 1
    topology: PrimitiveTopology = PrimitiveTopology.triangle_list
    depthBias: int = 0
    vertex_entry_point: str = "vertex_main"
    fragment_entry_point: str = "fragment_main"
    vertex_buffer_layouts: list[VertexBufferLayout] = []
    vertex_buffers: list[Buffer] = []

    def create_render_pipeline(self, options: RenderOptions) -> None:
        shader_module = self.device.createShaderModule(self._get_preprocessed_shader_code())
        layout, self.group = create_bind_group(
            self.device, options.get_bindings() + self.get_bindings()
        )
        self.pipeline = self.device.createRenderPipeline(
            self.device.createPipelineLayout([layout]),
            vertex=VertexState(
                module=shader_module,
                entryPoint=self.vertex_entry_point,
                buffers=self.vertex_buffer_layouts,
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint=self.fragment_entry_point,
                targets=[options.canvas.color_target],
            ),
            primitive=PrimitiveState(topology=self.topology),
            depthStencil=DepthStencilState(
                format=options.canvas.depth_format,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                depthBias=self.depthBias,
            ),
            multisample=options.canvas.multisample,
        )

    def render(self, options: RenderOptions) -> None:
        render_pass = options.begin_render_pass()
        render_pass.setPipeline(self.pipeline)
        render_pass.setBindGroup(0, self.group)
        for i, vertex_buffer in enumerate(self.vertex_buffers):
            render_pass.setVertexBuffer(i, vertex_buffer)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()
