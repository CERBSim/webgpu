import math
import typing
from enum import Enum

import ngsolve as ngs
import ngsolve.webgui
import numpy as np

from .gpu import WebGPU
from .uniforms import Binding
from .utils import (
    BaseBinding,
    BufferBinding,
    ShaderStage,
    TextureBinding,
    create_bind_group,
    decode_bytes,
    encode_bytes,
)
from .webgpu_api import *


class _eltype:
    dim: int
    primitive_topology: PrimitiveTopology
    num_vertices_per_primitive: int

    def __init__(self, dim, primitive_topology, num_vertices_per_primitive):
        self.dim = dim
        self.primitive_topology = primitive_topology
        self.num_vertices_per_primitive = num_vertices_per_primitive


class ElType(Enum):
    POINT = _eltype(0, PrimitiveTopology.point_list, 1)
    SEG = _eltype(1, PrimitiveTopology.line_list, 2)
    TRIG = _eltype(2, PrimitiveTopology.triangle_list, 3)
    QUAD = _eltype(2, PrimitiveTopology.triangle_list, 2 * 3)
    TET = _eltype(3, PrimitiveTopology.triangle_list, 4 * 3)
    HEX = _eltype(3, PrimitiveTopology.triangle_list, 6 * 2 * 3)
    PRISM = _eltype(3, PrimitiveTopology.triangle_list, 2 * 3 + 3 * 2 * 3)
    PYRAMID = _eltype(3, PrimitiveTopology.triangle_list, 4 + 2 * 3)

    @staticmethod
    def from_dim_np(dim: int, np: int):
        if dim == 2:
            if np == 3:
                return ElType.TRIG
            if np == 4:
                return ElType.QUAD
        if dim == 3:
            if np == 4:
                return ElType.TET
            if np == 8:
                return ElType.HEX
            if np == 6:
                return ElType.PRISM
            if np == 5:
                return ElType.PYRAMID
        raise ValueError(f"Unsupported element type dim={dim} np={np}")


ElTypes2D = [ElType.TRIG, ElType.QUAD]
ElTypes3D = [ElType.TET, ElType.HEX, ElType.PRISM, ElType.PYRAMID]


class RenderObject:
    """Base class for render objects"""

    data: typing.Any
    gpu: WebGPU
    label: str = ""

    def __init__(self, gpu, label=None):
        self.gpu = gpu

        if label is None:
            self.label = self.__class__.__name__
        else:
            self.label = label

        self.on_resize()

    def render(self, encoder: CommandEncoder):
        raise NotImplementedError

    def on_resize(self):
        pass

    def get_bindings(self) -> list[BaseBinding]:
        raise NotImplementedError

    def create_bind_group(self):
        return create_bind_group(self.device, self.get_bindings(), self.label)

    @property
    def device(self) -> Device:
        return self.gpu.device


class DataRenderObject(RenderObject):
    """Base class for render objects that use a "data" object, like MeshData"""

    _buffers: dict = {}

    def __init__(self, gpu, data, label=None):
        super().__init__(gpu, label=label)
        self.update_data(data)
        print("init", self.label)

    def update_data(self, data):
        print("update", self.label)
        self.data = data
        self._buffers = data.get_buffers(self.device)
        self._create_pipelines()

    def _create_pipelines(self):
        raise NotImplementedError


class MeshRenderObject(DataRenderObject):
    """Use "trigs" and "trig_function_values" buffers to render a function on a mesh"""

    def get_bindings(self):
        return [
            *self.gpu.get_bindings(),
            BufferBinding(Binding.TRIGS, self._buffers["trigs"]),
            BufferBinding(
                Binding.TRIG_FUNCTION_VALUES, self._buffers["trig_function_values"]
            ),
        ]

    def _create_pipelines(self):
        bind_layout, self._bind_group = self.create_bind_group()
        shader_module = self.gpu.shader_module

        self._pipeline = self.device.createRenderPipeline(
            self.device.createPipelineLayout([bind_layout], self.label),
            vertex=VertexState(
                module=shader_module,
                entryPoint="vertexTrigP1",
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentTrig",
                targets=[ColorTargetState(format=self.gpu.format)],
            ),
            primitive=PrimitiveState(
                topology=PrimitiveTopology.triangle_list,
            ),
            depthStencil=DepthStencilState(
                format=TextureFormat.depth24plus,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                # shift trigs behind to ensure that edges are rendered properly
                depthBias=1,
                depthBiasSlopeScale=1.0,
            ),
        )

    def render(self, encoder: CommandEncoder):
        render_pass = self.gpu.begin_render_pass(encoder, label=self.label)
        render_pass.setBindGroup(0, self._bind_group)
        render_pass.setPipeline(self._pipeline)
        render_pass.draw(3, self.data.num_trigs)
        render_pass.end()


class MeshRenderObjectIndexed(MeshRenderObject):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh"""

    def get_bindings(self):
        return [
            *self.gpu.get_bindings(),
            BufferBinding(
                Binding.TRIG_FUNCTION_VALUES, self._buffers["trig_function_values"]
            ),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            BufferBinding(Binding.TRIGS_INDEX, self._buffers["trigs_index"]),
        ]

    def _create_pipelines(self):
        bind_layout, self._bind_group = self.create_bind_group()
        shader_module = self.gpu.shader_module

        self._pipeline = self.device.createRenderPipeline(
            self.device.createPipelineLayout([bind_layout], self.label),
            vertex=VertexState(
                module=shader_module,
                entryPoint="vertexTrigP1Indexed",
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentTrig",
                targets=[ColorTargetState(format=self.gpu.format)],
            ),
            primitive=PrimitiveState(
                topology=PrimitiveTopology.triangle_list,
            ),
            depthStencil=DepthStencilState(
                format=TextureFormat.depth24plus,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                # shift trigs behind to ensure that edges are rendered properly
                depthBias=1,
                depthBiasSlopeScale=1.0,
            ),
        )


class MeshRenderObjectDeferred(DataRenderObject):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh in two render passes
    The first pass renders the trig indices and barycentric coordinates to a g-buffer texture.
    The second pass renders the trigs using the g-buffer texture to evaluate the function value in each pixel of the frame buffer.

    This approach is especialy more efficient if function evaluation is expensive (high order) and many triangles overlap,
    because the function values are only evaluated for the pixels that are visible.
    """

    _g_buffer_format: TextureFormat = TextureFormat.rgba32float
    _g_buffer: Texture

    def on_resize(self):
        # texture to store g-buffer (trig index and barycentric coordinates)

        self._g_buffer = self.device.createTexture(
            size=[self.gpu.canvas.width, self.gpu.canvas.height, 1],
            usage=TextureUsage.TEXTURE_BINDING | TextureUsage.RENDER_ATTACHMENT,
            format=self._g_buffer_format,
            label="gBufferLam",
        )

    def get_bindings_pass1(self):
        return [
            *self.gpu.get_bindings(),
            BufferBinding(
                Binding.TRIG_FUNCTION_VALUES, self._buffers["trig_function_values"]
            ),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            BufferBinding(Binding.TRIGS_INDEX, self._buffers["trigs_index"]),
        ]

    def get_bindings_pass2(self):
        return [
            *self.get_bindings_pass1(),
            TextureBinding(
                Binding.GBUFFERLAM,
                self._g_buffer,
                sample_type="unfilterable-float",
                dim=2,
            ),
        ]

    def _create_pipelines(self):
        device = self.device
        label1 = self.label + " pass 1"
        bind_layout_pass1, self._bind_group_pass1 = create_bind_group(
            device, self.get_bindings_pass1(), label1
        )
        shader_module = self.gpu.shader_module

        self._pipeline_pass1 = self.device.createRenderPipeline(
            label=label1,
            layout=self.device.createPipelineLayout([bind_layout_pass1]),
            vertex=VertexState(
                module=shader_module,
                entryPoint="vertexTrigP1Indexed",
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentTrigToGBuffer",
                targets=[ColorTargetState(format=self._g_buffer_format)],
            ),
            primitive=PrimitiveState(
                topology=PrimitiveTopology.triangle_list,
            ),
            depthStencil=DepthStencilState(
                format=TextureFormat.depth24plus,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
                # shift trigs behind to ensure that edges are rendered properly
                depthBias=1,
                depthBiasSlopeScale=1.0,
            ),
        )

        label2 = self.label + " pass 2"

        bind_layout_pass2, self._bind_group_pass2 = create_bind_group(
            device, self.get_bindings_pass2(), label2
        )

        self._pipeline_pass2 = self.device.createRenderPipeline(
            label=label2,
            layout=self.device.createPipelineLayout([bind_layout_pass2]),
            vertex=VertexState(
                module=shader_module,
                entryPoint="vertexDeferred",
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentDeferred",
                targets=[ColorTargetState(format=self.gpu.format)],
            ),
            primitive=PrimitiveState(
                topology=PrimitiveTopology.triangle_strip,
            ),
            depthStencil=DepthStencilState(
                format=TextureFormat.depth24plus,
                depthWriteEnabled=False,
                depthCompare=CompareFunction.always,
            ),
        )

    def render(self, encoder: CommandEncoder):
        loadOp = encoder.getLoadOp()
        pass1 = encoder.beginRenderPass(
            colorAttachments=[
                RenderPassColorAttachment(
                    self._g_buffer.createView(),
                    clearValue=Color(0, -1, -1, -1),
                    loadOp=loadOp,
                )
            ],
            depthStencilAttachment=RenderPassDepthStencilAttachment(
                self.gpu.depth_texture.createView(),
                depthClearValue=1.0,
                depthLoadOp=loadOp,
            ),
            label=self.label + " pass 1",
        )

        pass1.setViewport(0, 0, self.gpu.canvas.width, self.gpu.canvas.height, 0.0, 1.0)
        pass1.setBindGroup(0, self._bind_group_pass1)
        pass1.setPipeline(self._pipeline_pass1)
        pass1.draw(3, self.data.num_trigs)
        pass1.end()

        pass2 = self.gpu.begin_render_pass(encoder, label=self.label + " pass 2")
        pass2.setBindGroup(0, self._bind_group_pass2)
        pass2.setViewport(0, 0, self.gpu.canvas.width, self.gpu.canvas.height, 0.0, 1.0)
        pass2.setPipeline(self._pipeline_pass2)
        pass2.draw(4)
        pass2.end()


class Mesh3dElementsRenderObject(DataRenderObject):
    def get_bindings(self):
        bindings = [
            *self.gpu.get_bindings(),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
        ]

        for eltype in ElType:
            if self.data.num_els[eltype.name]:
                bindings.append(
                    BufferBinding(
                        getattr(Binding, eltype.name), self._buffers[eltype.name]
                    )
                )

        return bindings

    def _create_pipelines(self):
        bind_layout, self._bind_group = self.create_bind_group()
        shader_module = self.gpu.shader_module

        self._pipelines = {}
        for eltype in ElType:

            if self.data.num_els[eltype.name] == 0:
                continue
            el_name = eltype.name.capitalize()

            self._pipelines[el_name.upper()] = self.device.createRenderPipeline(
                label=f"{self.label}:{el_name}",
                layout=self.device.createPipelineLayout([bind_layout]),
                vertex=VertexState(
                    module=shader_module,
                    entryPoint=f"vertexMesh{el_name}",
                ),
                fragment=FragmentState(
                    module=shader_module,
                    entryPoint="fragmentMesh",
                    targets=[ColorTargetState(format=self.gpu.format)],
                ),
                primitive=PrimitiveState(
                    topology=eltype.value.primitive_topology,
                ),
                depthStencil=DepthStencilState(
                    **self.gpu.depth_stencil,
                    # shift trigs behind to ensure that edges are rendered properly
                    depthBias=1.0,
                    depthBiasSlopeScale=1,
                ),
            )

    def render(self, encoder):
        render_pass = self.gpu.begin_render_pass(encoder)
        render_pass.setBindGroup(0, self._bind_group)
        for name, pipeline in self._pipelines.items():
            eltype = ElType[name].value
            render_pass.setPipeline(pipeline)
            render_pass.draw(eltype.num_vertices_per_primitive, self.data.num_els[name])
        render_pass.end()


def _get_bernstein_matrix_trig(n, intrule):
    """Create inverse vandermonde matrix for the Bernstein basis functions on a triangle of degree n and given integration points"""
    ndtrig = int((n + 1) * (n + 2) / 2)

    mat = ngs.Matrix(ndtrig, ndtrig)
    fac_n = math.factorial(n)
    for row, ip in enumerate(intrule):
        col = 0
        x = ip.point[0]
        y = ip.point[1]
        z = 1.0 - x - y
        for i in range(n + 1):
            factor = fac_n / math.factorial(i) * x**i
            for j in range(n + 1 - i):
                k = n - i - j
                factor2 = 1.0 / (math.factorial(j) * math.factorial(k))
                mat[row, col] = factor * factor2 * y**j * z**k
                col += 1
    return mat


class MeshData:
    vertices: bytes
    trigs: bytes
    trigs_index: bytes
    edges: bytes
    trig_function_values: bytes
    tests: bytes

    num_trigs: int
    num_verts: int
    num_edges: int
    num_tets: int
    func_dim: int
    num_elements: dict[str, int]
    elements: dict[str, bytes]

    _buffers: dict = {}

    __BUFFER_NAMES = [
        "vertices",
        "trigs",
        "trigs_index",
        "edges",
        "trig_function_values",
    ]
    __INT_NAMES = ["num_trigs", "num_verts", "num_edges", "func_dim"]

    def load(self, data: dict):
        for name in self.__BUFFER_NAMES:
            setattr(self, name, decode_bytes(data.get(name, "")))

        for name in self.__INT_NAMES:
            setattr(self, name, data.get(name, 0))

    def dump(self):
        data = {}
        for name in self.__BUFFER_NAMES:
            data[name] = encode_bytes(getattr(self, name))

        for name in self.__INT_NAMES:
            data[name] = getattr(self, name)

        return data

    def __init__(self, region_or_mesh=None, cf=None, order=1):
        # TODO: implement other element types than triangles
        # TODO: handle region correctly to draw only part of the mesh
        # TODO: set up proper index buffer - it's currently slow and wrong (due to ngsolve vertex numbering)
        for name in self.__BUFFER_NAMES:
            setattr(self, name, b"")
        for name in self.__INT_NAMES:
            setattr(self, name, 0)

        if region_or_mesh is None:
            return

        if isinstance(region_or_mesh, ngs.Region):
            mesh = region_or_mesh.mesh
            region = region_or_mesh
        else:
            mesh = region_or_mesh
            region = mesh.Region(ngs.VOL)

        region_2d = region.Boundaries() if mesh.dim == 3 else region

        self.num_verts = len(mesh.vertices)
        vertices = np.zeros((self.num_verts, 3), dtype=np.float32)
        for i, v in enumerate(mesh.vertices):
            if len(v.point) == 2:
                vertices[i, :2] = v.point
            else:
                vertices[i, :] = v.point

        self.vertices = vertices.tobytes()

        self.num_trigs = len(mesh.ngmesh.Elements2D())

        # du to vertex numer ordering in ngsolve, we need to store the points multiple times
        points = evaluate_cf(ngs.CF((ngs.x, ngs.y, ngs.z)), region_2d, order=1)[2:]

        trigs_index = np.zeros((self.num_trigs, 3), dtype=np.uint32)
        for i, el in enumerate(mesh.ngmesh.Elements2D()):
            trigs_index[i, :] = [p.nr - 1 for p in el.vertices[:3]]
        self.trigs_index = trigs_index.tobytes()

        edge_points = points.reshape(-1, 3, 3)
        edges = np.zeros((self.num_trigs, 3, 2, 3), dtype=np.float32)
        for i in range(3):
            edges[:, i, 0, :] = edge_points[:, i, :]
            edges[:, i, 1, :] = edge_points[:, (i + 1) % 3, :]

        self.edges = edges.flatten().tobytes()
        trigs = np.zeros(
            self.num_trigs,
            dtype=[
                ("p", np.float32, 9),  # 3 vec3<f32> (each 4 floats due to padding)
                ("index", np.int32),  # index (i32)
            ],
        )
        trigs["p"] = points.flatten().reshape(-1, 9)
        trigs["index"] = [1] * self.num_trigs
        self.trigs = trigs.tobytes()

        if cf is not None:
            self.trig_function_values = evaluate_cf(cf, region, order).tobytes()
            self.func_dim = cf.dim

        self.num_els = {eltype.name: 0 for eltype in ElType}
        self.elements = {eltype.name: [] for eltype in ElType}

        for i, el in enumerate(mesh.ngmesh.Elements3D()):
            eltype = ElType.from_dim_np(3, len(el.vertices))
            data = [p.nr - 1 for p in el.vertices]
            data.append(el.index)
            data.append(i)
            self.elements[eltype.name].append(data)
            self.num_els[eltype.name] += 1

        for eltype in self.elements:
            self.elements[eltype] = np.array(
                self.elements[eltype], dtype=np.uint32
            ).tobytes()

    def get_buffers(self, device: Device):
        if not self._buffers:
            data = {}
            for name in self.__BUFFER_NAMES:
                b = getattr(self, name)
                if b:
                    data[name] = b

            for eltype in self.elements:
                data[eltype] = self.elements[eltype]

            buffers = {}
            for key in data:
                d = data[key]
                buffer = device.createBuffer(
                    size=len(d), usage=BufferUsage.STORAGE | BufferUsage.COPY_DST
                )
                device.queue.writeBuffer(buffer, 0, d)
                buffers[key] = buffer

            self._buffers = buffers
        return self._buffers

    def __del__(self):
        for buf in self._buffers.values():
            buf.destroy()


def evaluate_cf(cf, region, order):
    """Evaluate a coefficient function on a mesh and returns the values as a flat array, ready to copy to the GPU as storage buffer.
    The first two entries are the function dimension and the polynomial order of the stored values.
    """
    comps = cf.dim
    int_points = ngsolve.webgui._make_trig(order)
    intrule = ngs.IntegrationRule(
        int_points,
        [
            0,
        ]
        * len(int_points),
    )
    ibmat = _get_bernstein_matrix_trig(order, intrule).I

    ndof = ibmat.h

    pts = region.mesh.MapToAllElements(
        {ngs.ET.TRIG: intrule, ngs.ET.QUAD: intrule}, region
    )
    pmat = cf(pts)
    pmat = pmat.reshape(-1, ndof, comps)

    values = np.zeros((ndof, pmat.shape[0], comps), dtype=np.float32)
    for i in range(comps):
        ngsmat = ngs.Matrix(pmat[:, :, i].transpose())
        values[:, :, i] = ibmat * ngsmat

    values = values.transpose((1, 0, 2)).flatten()
    ret = np.concatenate(([np.float32(cf.dim), np.float32(order)], values))
    return ret


def create_testing_square_mesh(gpu, n):
    device = gpu.device
    # launch compute shader
    n = math.ceil(n / 16) * 16
    n_trigs = 2 * n * n
    if n_trigs >= 1e5:
        print(f"Creating {n_trigs//1000} K trigs")
    else:
        print(f"Creating {n_trigs} trigs")
    trig_size = 4 * n_trigs * 10
    value_size = 4 * (3 * n_trigs + 2)
    index_size = 4 * (3 * n_trigs)
    vertex_size = 4 * 3 * (n + 1) * (n + 1)
    print(f"trig size {trig_size/1024/1024:.2f} MB")
    print(f"vals size {value_size/1024/1024:.2f} MB")
    print(f"index size {index_size/1024/1024:.2f} MB")
    print(f"vertex size {index_size/1024/1024:.2f} MB")

    trigs_buffer = device.create_buffer(trig_size)
    function_buffer = device.create_buffer(value_size)
    index_buffer = device.create_buffer(index_size)
    vertex_buffer = device.create_buffer(vertex_size)

    buffers = {
        "trigs": trigs_buffer,
        "trig_function_values": function_buffer,
        "vertices": vertex_buffer,
        "trigs_index": index_buffer,
    }

    shader_module = gpu.shader_module

    bindings = []
    for name in ["trigs", "trig_function_values", "vertices", "trigs_index"]:
        binding = getattr(Binding, name.upper())
        bindings.append(
            BufferBinding(
                binding,
                buffers[name],
                read_only=False,
                visibility=ShaderStage.COMPUTE,
            )
        )

    layout, group = create_bind_group(device, bindings, "create_test_mesh")

    pipeline = device.create_compute_pipeline(
        layout,
        {
            "label": "create_test_mesh",
            "layout": device.create_pipeline_layout(layout, "create_test_mesh"),
            "compute": {"module": shader_module, "entryPoint": "create_mesh"},
        },
    )

    command_encoder = gpu.native_device.createCommandEncoder()
    pass_encoder = command_encoder.beginComputePass()
    pass_encoder.setPipeline(pipeline)
    pass_encoder.setBindGroup(0, group)

    pass_encoder.dispatchWorkgroups(n // 16, 1, 1)
    pass_encoder.end()
    gpu.native_device.queue.submit([command_encoder.finish()])

    data = MeshData()
    data._buffers = buffers
    data.num_trigs = n_trigs
    data.num_verts = (n + 1) * (n + 1)
    data.func_dim = 1
    return data


class PointNumbersRenderObject(DataRenderObject):
    """Render a point numbers of a mesh"""

    _buffers: dict = {}

    def __init__(self, gpu, data, font_size=20, label=None):
        self.n_digits = 6
        self.gpu = gpu
        self.set_font_size(font_size)
        super().__init__(gpu, data, label=label)

    def get_bindings(self):
        return [
            *self.gpu.get_bindings(),
            TextureBinding(Binding.FONT_TEXTURE, self._texture, dim=2),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
        ]

    def _create_pipelines(self):
        bind_layout, self._bind_group = create_bind_group(
            self.device, self.get_bindings(), "PointNumbersRenderObject"
        )
        shader_module = self.gpu.shader_module

        self._pipeline = self.device.createRenderPipeline(
            label=self.label,
            layout=self.device.createPipelineLayout([bind_layout]),
            vertex=VertexState(
                module=shader_module,
                entryPoint="vertexPointNumber",
            ),
            fragment=FragmentState(
                module=shader_module,
                entryPoint="fragmentText",
                targets=[
                    ColorTargetState(
                        format=self.gpu.format,
                        blend=BlendState(
                            color=BlendComponent(
                                srcFactor=BlendFactor.one,
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
                ],
            ),
            primitive=PrimitiveState(
                topology=PrimitiveTopology.triangle_list,
            ),
            depthStencil=DepthStencilState(
                format=TextureFormat.depth24plus,
                depthWriteEnabled=True,
                depthCompare=CompareFunction.less,
            ),
        )

    def render(self, encoder):
        render_pass = self.gpu.begin_render_pass(encoder)
        render_pass.setBindGroup(0, self._bind_group)
        render_pass.setPipeline(self._pipeline)
        render_pass.draw(self.n_digits * 6, self.data.num_verts, 0, 0)
        render_pass.end()

    def set_font_size(self, font_size: int):
        from .font import create_font_texture

        self._texture = create_font_texture(self.gpu.device, font_size)
        char_width = self._texture.width // (127 - 32)
        char_height = self._texture.height
        self.gpu.u_font.width = char_width
        self.gpu.u_font.height = char_height
        if self._buffers:
            self._create_pipeline()
