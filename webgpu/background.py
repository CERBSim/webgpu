"""Generic rounded-rectangle background overlay renderer."""

import ctypes as ct

from .renderer import Renderer, RenderOptions
from .uniforms import UniformBase
from .utils import (
    create_bind_group,
    get_device,
    read_shader_file,
)
from .webgpu_api import (
    BlendComponent,
    BlendFactor,
    BlendOperation,
    BlendState,
    ColorTargetState,
    CompareFunction,
    DepthStencilState,
    FragmentState,
    PrimitiveState,
    VertexState,
)


_BINDING = 50


class BackgroundUniforms(UniformBase):
    _binding = _BINDING
    _fields_ = [
        ("position", ct.c_float * 2),
        ("width", ct.c_float),
        ("height", ct.c_float),
        ("bg_color", ct.c_float * 3),
        ("_pad", ct.c_float),
    ]


class Background(Renderer):
    """Semi-transparent rounded-rectangle background overlay.

    Place this before other renderers in a MultipleRenderer to provide
    a readable backdrop behind text or UI elements.

    @param position: (x, y) top-left corner in NDC
    @param width: width in NDC
    @param height: height in NDC (of the content area, padding is added automatically)
    """
    vertex_entry_point: str = "background_vertex"
    fragment_entry_point: str = "background_fragment"
    select_entry_point: str = ""
    n_vertices: int = 6
    n_instances: int = 1

    def __init__(self, position=(0, 0), width=1, height=0.05):
        super().__init__()
        self._position = position
        self._width = width
        self._height = height
        self._bg_color = (1.0, 1.0, 1.0)
        self.uniforms = None

    @property
    def bg_color(self):
        return self._bg_color

    @bg_color.setter
    def bg_color(self, value):
        self._bg_color = tuple(value[:3])
        if self.uniforms is not None:
            self.uniforms.bg_color = self._bg_color
            self.uniforms.update_buffer()
        self.set_needs_update()

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value
        if self.uniforms is not None:
            self.uniforms.position = value
            self.uniforms.update_buffer()
        self.set_needs_update()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        if self.uniforms is not None:
            self.uniforms.width = value
            self.uniforms.update_buffer()
        self.set_needs_update()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        if self.uniforms is not None:
            self.uniforms.height = value
            self.uniforms.update_buffer()
        self.set_needs_update()

    def get_shader_code(self):
        return read_shader_file("background.wgsl")

    def get_bindings(self):
        return self.uniforms.get_bindings()

    def update(self, options: RenderOptions):
        if self.uniforms is None:
            self.uniforms = BackgroundUniforms()
            self.uniforms.position = self.position
            self.uniforms.width = self.width
            self.uniforms.height = self.height
            self.uniforms.bg_color = self._bg_color
        self.uniforms.update_buffer()

    def create_render_pipeline(self, options: RenderOptions) -> None:
        bindings = options.get_bindings() + self.get_bindings()

        if bindings == self._last_bindings:
            return

        layout, self.group = create_bind_group(
            self.device, options.get_bindings() + self.get_bindings()
        )
        pipeline_layout = self.device.createPipelineLayout([layout])

        depth_stencil = DepthStencilState(
            format=options.canvas.depth_format,
            depthWriteEnabled=False,
            depthCompare=CompareFunction.always,
        )

        bg_color_target = ColorTargetState(
            format=options.canvas.format,
            blend=BlendState(
                color=BlendComponent(
                    srcFactor=BlendFactor.src_alpha,
                    dstFactor=BlendFactor.one_minus_src_alpha,
                    operation=BlendOperation.add,
                ),
                alpha=BlendComponent(
                    srcFactor=BlendFactor.one,
                    dstFactor=BlendFactor.one_minus_src_alpha,
                    operation=BlendOperation.add,
                ),
            ),
        )

        shader_module = self.device.createShaderModule(self._get_preprocessed_shader_code())
        self.pipeline = self.device.createRenderPipeline(
            pipeline_layout,
            vertex=VertexState(
                module=shader_module,
                entryPoint=self.vertex_entry_point,
                buffers=[],
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint=self.fragment_entry_point,
                targets=[bg_color_target],
            ),
            primitive=PrimitiveState(topology=self.topology),
            depthStencil=depth_stencil,
            multisample=options.canvas.multisample,
            label="Background",
        )
        self._select_pipeline = None
        self._transparent_pipeline = None
        self._last_bindings = bindings

    def get_export_descriptor(self, options, buffer_registry):
        desc = super().get_export_descriptor(options, buffer_registry)
        desc.depth_write = False
        desc.pass_type = "transparent"
        return desc

    def get_theme_buffer_id(self, registry):
        """Return the buffer id for theme color updates (bg_color at offset 16)."""
        if self.uniforms is None or self.uniforms._buffer is None:
            return None
        key = id(self.uniforms._buffer)
        if key in registry._buffers:
            return registry._buffers[key][0]
        return None
