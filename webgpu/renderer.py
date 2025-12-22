import itertools
from typing import Callable

import numpy as np

from .camera import Camera
from .canvas import Canvas
from .light import Light
from .utils import (
    BaseBinding,
    buffer_from_array,
    create_bind_group,
    get_device,
    preprocess_shader_code,
)
from .webgpu_api import (
    Buffer,
    CommandEncoder,
    CompareFunction,
    DepthStencilState,
    Device,
    FragmentState,
    MultisampleState,
    PrimitiveState,
    PrimitiveTopology,
    VertexBufferLayout,
    VertexState,
)


class SelectEvent:
    x: int
    y: int
    data: bytes

    def __init__(self, x: int, y: int, data: bytes):
        self.x = x
        self.y = y
        self.data = data
        self.z = np.frombuffer(self.data[4:8], dtype=np.float32)[0]
        self.user_data = data[8:]

    def __repr__(self):
        return f"SelectEvent(x={self.x}, y={self.y}, data={self.data})"

    @property
    def uint32(self):
        return np.frombuffer(self.user_data, dtype=np.uint32)

    @property
    def float32(self):
        return np.frombuffer(self.user_data, dtype=np.float32)

    @property
    def obj_id(self):
        return np.frombuffer(self.data[:4], dtype=np.uint32)[0]

    def calculate_position(self, camera: Camera):
        x = self.x / camera.canvas.width * 2 - 1
        y = 1 - self.y / camera.canvas.height * 2
        z = self.z

        mat = camera.model_view_proj
        inv_mat = np.linalg.inv(mat)
        ndc = np.array([x, y, z, 1.0], dtype=np.float32)
        pos = inv_mat @ ndc
        return pos[:3] / pos[3]


class RenderOptions:
    viewport: tuple[int, int, int, int, float, float]
    canvas: Canvas
    command_encoder: CommandEncoder
    timestamp: float

    def __init__(self, camera: Camera, light: Light):
        self.light = light
        self.camera = camera

    def set_canvas(self, canvas: Canvas):
        self.canvas = canvas
        self.camera.set_canvas(canvas)

    @property
    def device(self) -> Device:
        return get_device()

    def update_buffers(self):
        self.camera._update_uniforms()
        self.light.update(self)

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
        # render_pass_encoder.setViewport(100, 100, 88, 99, 0.0, 1.0)
        # render_pass_encoder.setScissorRect(100, 100, 88, 99)

        return render_pass_encoder

    def begin_select_pass(self, x, y, **kwargs):
        load_op = self.command_encoder.getLoadOp()

        render_pass_encoder = self.command_encoder.beginRenderPass(
            self.canvas.select_attachments(load_op),
            self.canvas.select_depth_stencil_attachment(load_op),
            **kwargs,
        )

        # w = self.canvas.select_texture.width
        # h = self.canvas.select_texture.height
        # x0 = x - w // 2
        # y0 = y - h // 2
        # print('viewport', x0, y0, w, h)
        # render_pass_encoder.setViewport(x0, y0, w, h, 0.0, 1.0)
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


_id_counter = itertools.count(1)


class BaseRenderer:
    label: str = ""
    _timestamp: float = -1
    active: bool = True
    shader_defines: dict[str, str] = None
    _id = None
    _on_select: list[Callable[[SelectEvent], None]]

    def __init__(self, label=None):
        self._id = next(_id_counter)
        self.shader_defines = {}
        self._on_select = []
        if label is None:
            self.label = self.__class__.__name__
        else:
            self.label = label

        self._have_pipeline = False

    def get_bounding_box(self) -> tuple[list[float], list[float]] | None:
        return None

    def update(self, options: RenderOptions) -> None:
        pass

    def all_renderer(self):
        return [self]

    @property
    def on_select_set(self):
        return bool(self._on_select)

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

    def _get_preprocessed_shader_code(self, defines: dict | None = None) -> str:
        defines = defines or {}
        return preprocess_shader_code(
            self.get_shader_code(),
            defines=self.shader_defines | defines | {"RENDER_OBJECT_ID": str(self._id)},
        )

    def add_options_to_gui(self, gui):
        pass

    def set_needs_update(self) -> None:
        self._timestamp = -1

    @property
    def needs_update(self) -> bool:
        return self._timestamp == -1

    def select(self, options: RenderOptions, x: int, y: int) -> None:
        pass

    def on_select(self, callback):
        self._on_select.append(callback)

    def _handle_on_select(self, ev: SelectEvent):
        for callback in self._on_select:
            callback(ev)


class MultipleRenderer(BaseRenderer):
    def __init__(self, render_objects):
        super().__init__()
        self.render_objects = render_objects

    def update(self, options: RenderOptions) -> None:
        for r in self.render_objects:
            r.update(options)

    def all_renderer(self):
        return [self] + self.render_objects

    @check_timestamp
    def _update_and_create_render_pipeline(self, options: RenderOptions):
        # Let the subclass update itself and its children
        self.update(options)

        # Children are part of the same logical object, so once this
        # group is updated for this timestamp, they should be too.
        for r in self.render_objects:
            # Mark child as up-to-date for this options.timestamp
            # since check_timestamp won't be called on them directly.
            if hasattr(r, "_timestamp") and hasattr(options, "timestamp"):
                r._timestamp = options.timestamp
            r.create_render_pipeline(options)

    def render(self, options: RenderOptions) -> None:
        for r in self.render_objects:
            if r.active:
                r.render(options)

    def select(self, options: RenderOptions, x: int, y: int) -> None:
        for r in self.render_objects:
            if r.active:
                r.select(options, x, y)

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

    @property
    def on_select_set(self):
        return any(r.on_select_set for r in self.render_objects)


class Renderer(BaseRenderer):
    """Base class for renderer classes"""

    n_vertices: int = 0
    n_instances: int = 1
    topology: PrimitiveTopology = PrimitiveTopology.triangle_list
    depthBias: int = 0
    depthBiasSlopeScale: int = 0
    vertex_entry_point: str = "vertex_main"
    fragment_entry_point: str = "fragment_main"
    select_entry_point: str = "fragment_select_default"
    vertex_buffer_layouts: list[VertexBufferLayout] = []
    vertex_buffers: list[Buffer] = []

    _last_bindings: list[BaseBinding] = []

    def create_render_pipeline(self, options: RenderOptions) -> None:
        bindings = options.get_bindings() + self.get_bindings()

        if bindings == self._last_bindings:
            return

        shader_module = self.device.createShaderModule(self._get_preprocessed_shader_code())
        layout, self.group = create_bind_group(
            self.device, options.get_bindings() + self.get_bindings()
        )
        pipeline_layout = self.device.createPipelineLayout([layout])
        vertex_state = VertexState(
            module=shader_module,
            entryPoint=self.vertex_entry_point,
            buffers=self.vertex_buffer_layouts,
        )
        depth_stencil = DepthStencilState(
            format=options.canvas.depth_format,
            depthWriteEnabled=True,
            depthCompare=CompareFunction.less,
            depthBias=self.depthBias,
            depthBiasSlopeScale=self.depthBiasSlopeScale,
        )
        self.pipeline = self.device.createRenderPipeline(
            pipeline_layout,
            vertex=vertex_state,
            fragment=FragmentState(
                module=shader_module,
                entryPoint=self.fragment_entry_point,
                targets=[options.canvas.color_target],
            ),
            primitive=PrimitiveState(topology=self.topology),
            depthStencil=depth_stencil,
            multisample=options.canvas.multisample,
            label=self.label,
        )

        if self.select_entry_point:
            select_shader_module = self.device.createShaderModule(
                self._get_preprocessed_shader_code({"SELECT_PIPELINE": "1"})
            )
            vertex_state.module = select_shader_module
            self._select_pipeline = self.device.createRenderPipeline(
                pipeline_layout,
                vertex=vertex_state,
                fragment=FragmentState(
                    module=select_shader_module,
                    entryPoint=self.select_entry_point,
                    targets=[options.canvas.select_target],
                ),
                primitive=PrimitiveState(topology=self.topology),
                depthStencil=depth_stencil,
                multisample=MultisampleState(),
                label=self.label + " (select)",
            )
        else:
            self._select_pipeline = None

        self._last_bindings = bindings

    def render(self, options: RenderOptions) -> None:
        render_pass = options.begin_render_pass()
        render_pass.setPipeline(self.pipeline)
        render_pass.setBindGroup(0, self.group)
        for i, vertex_buffer in enumerate(self.vertex_buffers):
            render_pass.setVertexBuffer(i, vertex_buffer)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()

    def select(self, options: RenderOptions, x: int, y: int) -> None:
        if not self._select_pipeline:
            return
        render_pass = options.begin_select_pass(x, y)
        render_pass.setPipeline(self._select_pipeline)
        render_pass.setBindGroup(0, self.group)
        for i, vertex_buffer in enumerate(self.vertex_buffers):
            render_pass.setVertexBuffer(i, vertex_buffer)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()
