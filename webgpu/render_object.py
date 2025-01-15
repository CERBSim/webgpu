
from .gpu import WebGPU
from .utils import BaseBinding, create_bind_group
from .webgpu_api import (CommandEncoder, Device,
        CompareFunction,
        DepthStencilState,
        FragmentState,
        PrimitiveState,
        PrimitiveTopology,
        VertexState,
        )

class BaseRenderObject:
    gpu: WebGPU
    label: str = ""
    def __init__(self, gpu, label=None):
        self.gpu = gpu
        if label is None:
            self.label = self.__class__.__name__
        else:
            self.label = label
        self.create_render_pipeline()

    def create_render_pipeline(self) -> None:
        raise NotImplementedError

    def render(self, encoder: CommandEncoder):
        raise NotImplementedError

    def get_bindings(self) -> list[BaseBinding]:
        raise NotImplementedError

    def get_shader_code(self) -> str:
        raise NotImplementedError

    def get_bounding_box(self):
        return ((0,0,0), (1,1,1))

    @property
    def device(self) -> Device:
        return self.gpu.device

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
        layout, self.group = create_bind_group(
            self.device, self.get_bindings())
        self.pipeline = self.device.createRenderPipeline(
        self.device.createPipelineLayout([layout]),
        vertex=VertexState(module=shader_module,
                           entryPoint=self.vertex_entry_point),
        fragment=FragmentState(
            module=shader_module,
            entryPoint=self.fragment_entry_point,
            targets=[self.gpu.color_target],
        ),
        primitive=PrimitiveState(topology=self.topology),
        depthStencil=DepthStencilState(
            format=self.gpu.depth_format,
            depthWriteEnabled=True,
            depthCompare=CompareFunction.less,
            depthBias=self.depthBias
        ),
        multisample=self.gpu.multisample,
        )

    def render(self, encoder: CommandEncoder) -> None:
        render_pass = self.gpu.begin_render_pass(encoder)
        render_pass.setPipeline(self.pipeline)
        render_pass.setBindGroup(0, self.group)
        render_pass.draw(self.n_vertices, self.n_instances)
        render_pass.end()

