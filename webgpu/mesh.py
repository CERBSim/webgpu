import math
import typing

import ngsolve as ngs
import ngsolve.webgui
import numpy as np

from .uniforms import Binding
from .utils import (
    BufferBinding,
    Device,
    ShaderStage,
    TextureBinding,
    decode_bytes,
    encode_bytes,
    to_js,
)


class RenderObject:
    """Base class for render objects"""

    data: typing.Any
    _buffers: dict = {}

    def __init__(self, gpu, data):
        self.gpu = gpu
        self.device = Device(gpu.device)
        self.update_data(data)

    def update_data(self, data):
        self.data = data
        self._buffers = data.get_buffers(self.device)
        self._create_pipelines()

    def _create_pipelines(self):
        pass

    def render(self, encoder):
        pass


class MeshRenderObject(RenderObject):
    """Use "trigs" and "trig_function_values" buffers to render a function on a mesh"""

    def get_bindings(self):
        return [
            *self.gpu.uniforms.get_bindings(),
            *self.gpu.colormap.get_bindings(),
            BufferBinding(Binding.TRIGS, self._buffers["trigs"]),
            BufferBinding(
                Binding.TRIG_FUNCTION_VALUES, self._buffers["trig_function_values"]
            ),
        ]

    def _create_pipelines(self):
        bind_layout, self._bind_group = self.device.create_bind_group(
            self.get_bindings(), "MeshRenderObject"
        )
        pipeline_layout = self.device.create_pipeline_layout(bind_layout)
        shader_module = self.device.compile_files("shader.wgsl", "eval.wgsl")
        self._pipeline = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "label": "MeshRenderObject",
                    "layout": pipeline_layout,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexTrigP1",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentTrig",
                        "targets": [{"format": self.gpu.format}],
                    },
                    "primitive": {
                        "topology": "triangle-list",
                        "cullMode": "none",
                        "frontFace": "ccw",
                    },
                    "depthStencil": {
                        **self.gpu.depth_stencil,
                        # shift trigs behind to ensure that edges are rendered properly
                        "depthBias": 1.0,
                        "depthBiasSlopeScale": 1,
                    },
                }
            )
        )

    def render(self, encoder):
        render_pass = self.gpu.begin_render_pass(encoder)
        render_pass.setBindGroup(0, self._bind_group)
        render_pass.setPipeline(self._pipeline)
        render_pass.draw(3, self.data.num_trigs, 0, 0)
        render_pass.end()


class MeshRenderObjectIndexed(RenderObject):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh"""

    def get_bindings(self):
        return [
            *self.gpu.uniforms.get_bindings(),
            *self.gpu.colormap.get_bindings(),
            BufferBinding(
                Binding.TRIG_FUNCTION_VALUES, self._buffers["trig_function_values"]
            ),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
            BufferBinding(Binding.TRIGS_INDEX, self._buffers["trigs_index"]),
        ]

    def _create_pipelines(self):
        bind_layout, self._bind_group = self.device.create_bind_group(
            self.get_bindings(), "MeshRenderObject"
        )
        pipeline_layout = self.device.create_pipeline_layout(bind_layout)
        shader_module = self.device.compile_files("shader.wgsl", "eval.wgsl")
        self._pipeline = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "label": "MeshRenderObjectIndexed",
                    "layout": pipeline_layout,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexTrigP1Indexed",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentTrig",
                        "targets": [{"format": self.gpu.format}],
                    },
                    "primitive": {
                        "topology": "triangle-list",
                        "cullMode": "none",
                        "frontFace": "ccw",
                    },
                    "depthStencil": {
                        **self.gpu.depth_stencil,
                        # shift trigs behind to ensure that edges are rendered properly
                        "depthBias": 1.0,
                        "depthBiasSlopeScale": 1,
                    },
                }
            )
        )

    def render(self, encoder):
        render_pass = self.gpu.begin_render_pass(encoder)
        render_pass.setBindGroup(0, self._bind_group)
        render_pass.setPipeline(self._pipeline)
        render_pass.draw(3, self.data.num_trigs)
        render_pass.end()


class MeshRenderObjectDeferred(RenderObject):
    """Use "vertices", "index" and "trig_function_values" buffers to render a mesh in two render passes
    The first pass renders the trig indices and barycentric coordinates to a g-buffer texture.
    The second pass renders the trigs using the g-buffer texture to evaluate the function value in each pixel of the frame buffer.

    This approach is especialy more efficient if function evaluation is expensive (high order) and many triangles overlap,
    because the function values are only evaluated for the pixels that are visible.
    """

    _g_buffer_format = "rgba32float"

    def __init__(self, gpu, data):
        import js

        # texture to store g-buffer (trig index and barycentric coordinates)
        self._g_buffer = gpu.device.createTexture(
            to_js(
                {
                    "label": "gBufferLam",
                    "size": [gpu.canvas.width, gpu.canvas.height],
                    "usage": js.GPUTextureUsage.RENDER_ATTACHMENT
                    | js.GPUTextureUsage.TEXTURE_BINDING,
                    "format": self._g_buffer_format,
                }
            )
        )

        super().__init__(gpu, data)

    def get_bindings_pass1(self):
        return [
            *self.gpu.uniforms.get_bindings(),
            *self.gpu.colormap.get_bindings(),
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
        bind_layout_pass1, self._bind_group_pass1 = self.device.create_bind_group(
            self.get_bindings_pass1(), "MeshRenderObjectDeferredPass1"
        )
        pipeline_layout_pass1 = self.device.create_pipeline_layout(bind_layout_pass1)
        shader_module = self.device.compile_files("shader.wgsl", "eval.wgsl")
        self._pipeline_pass1 = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "label": "MeshRenderObjectDeferredPass1",
                    "layout": pipeline_layout_pass1,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexTrigP1Indexed",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentTrigToGBuffer",
                        "targets": [{"format": self._g_buffer_format}],
                    },
                    "targets": [{"format": self._g_buffer_format}],
                    "primitive": {
                        "topology": "triangle-list",
                        "cullMode": "none",
                        "frontFace": "ccw",
                    },
                    "depthStencil": {
                        **self.gpu.depth_stencil,
                        # shift trigs behind to ensure that edges are rendered properly
                        "depthBias": 1.0,
                        "depthBiasSlopeScale": 1,
                    },
                }
            )
        )

        bind_layout_pass2, self._bind_group_pass2 = self.device.create_bind_group(
            self.get_bindings_pass2(),
            "mesh_object_deferred_pass2",
        )

        deferred_pipeline_layout = self.device.create_pipeline_layout(bind_layout_pass2)

        self._pipeline_pass2 = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "label": "trigs_deferred",
                    "layout": deferred_pipeline_layout,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexDeferred",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentDeferred",
                        "targets": [{"format": self.gpu.format}],
                    },
                    "primitive": {
                        "topology": "triangle-strip",
                        "cullMode": "none",
                        "frontFace": "ccw",
                    },
                    "depthStencil": {
                        **self.gpu.depth_stencil,
                        "depthWriteEnabled": False,
                        "depthCompare": "always",
                        # shift trigs behind to ensure that edges are rendered properly
                        # "depthBias": 1.0,
                        # "depthBiasSlopeScale": 1,
                    },
                }
            )
        )

    def render(self, encoder):
        load_op = "clear" if self.gpu._is_first_render_pass else "load"
        pass1_options = {
            "colorAttachments": [
                {
                    "view": self._g_buffer.createView(),
                    "clearValue": {"r": 0, "g": -1, "b": -1, "a": -1},
                    "loadOp": "clear",
                    "storeOp": "store",
                }
            ],
            "depthStencilAttachment": {
                "view": self.gpu.depth_texture.createView(
                    to_js({"format": self.gpu.depth_format, "aspect": "all"})
                ),
                "depthLoadOp": "clear",
                "depthStoreOp": "store",
                "depthClearValue": 1.0,
            },
        }
        pass1 = self.gpu.begin_render_pass(encoder, pass1_options)
        pass1.setViewport(0, 0, self.gpu.canvas.width, self.gpu.canvas.height, 0.0, 1.0)
        pass1.setBindGroup(0, self._bind_group_pass1)
        pass1.setPipeline(self._pipeline_pass1)
        pass1.draw(3, self.data.num_trigs)
        pass1.end()

        pass2_options = {
            "colorAttachments": [
                {
                    "view": self.gpu.context.getCurrentTexture().createView(),
                    "clearValue": {"r": 1, "g": 1, "b": 1, "a": 1},
                    "loadOp": load_op,
                    "storeOp": "store",
                }
            ],
            "depthStencilAttachment": {
                "view": self.gpu.depth_texture.createView(
                    to_js({"format": self.gpu.depth_format, "aspect": "all"})
                ),
                # "depthReadOnly": True,
                "depthLoadOp": load_op,
                "depthStoreOp": "store",
                "depthClearValue": 1.0,
            },
        }
        pass2 = self.gpu.begin_render_pass(encoder, pass2_options)
        pass2.setBindGroup(0, self._bind_group_pass2)
        pass2.setViewport(0, 0, self.gpu.canvas.width, self.gpu.canvas.height, 0.0, 1.0)
        pass2.setPipeline(self._pipeline_pass2)
        pass2.draw(4)
        pass2.end()


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

    num_trigs: int
    num_verts: int
    num_edges: int
    func_dim: int

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
            trigs_index[i, :] = [p.nr - 1 for p in el.vertices]
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

    def get_buffers(self, device):
        if not self._buffers:
            data = {}
            for name in self.__BUFFER_NAMES:
                b = getattr(self, name)
                if b:
                    data[name] = b

            self._buffers = device.data_to_buffers(data)
        return self._buffers

    def __del__(self):
        for buf in self._buffers.values():
            buf.destroy()


def create_function_value_buffers(device, cf, region, order):
    import js

    """Evaluate a coefficient function on a mesh and create GPU buffer with the values,
    returns a dictionary with the buffer as value and the name/element type as key"""
    # TODO: implement other element types than triangles
    values = evaluate_cf(cf, region, order)
    data = js.Uint8Array.new(values.tobytes())
    buffer = device.createBuffer(
        to_js(
            {
                "size": data.length,
                "usage": js.GPUBufferUsage.STORAGE | js.GPUBufferUsage.COPY_DST,
            }
        )
    )
    device.queue.writeBuffer(buffer, 0, data)
    return {"trig_function_values": buffer}


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
    device = Device(gpu.device)
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

    shader_module = device.compile_files("compute.wgsl")

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

    layout, group = device.create_bind_group(bindings, "create_test_mesh")

    pipeline = gpu.device.createComputePipeline(
        to_js(
            {
                "label": "create_test_mesh",
                "layout": device.create_pipeline_layout(layout, "create_test_mesh"),
                "compute": {"module": shader_module, "entryPoint": "create_mesh"},
            }
        )
    )

    command_encoder = gpu.device.createCommandEncoder()
    pass_encoder = command_encoder.beginComputePass()
    pass_encoder.setPipeline(pipeline)
    pass_encoder.setBindGroup(0, group)

    pass_encoder.dispatchWorkgroups(n // 16, 1, 1)
    pass_encoder.end()
    gpu.device.queue.submit([command_encoder.finish()])

    data = MeshData()
    data._buffers = buffers
    data.num_trigs = n_trigs
    data.num_verts = (n + 1) * (n + 1)
    data.func_dim = 1
    return data


class PointNumbersRenderObject:
    """Render a point numbers of a mesh"""

    _buffers: dict = {}

    def __init__(self, gpu, data, font_size=20):
        self._buffers = {}
        self._texture = None

        self.gpu = gpu
        self.device = Device(gpu.device)
        self.n_digits = 6
        self.set_font_size(font_size)
        self.update_data(data)

    def update_data(self, data):
        self.data = data
        self._buffers = self.data.get_buffers(self.device)
        self._create_pipeline()

    def get_bindings(self):
        return [
            *self.gpu.uniforms.get_bindings(),
            TextureBinding(Binding.FONT_TEXTURE, self._texture, dim=2),
            BufferBinding(Binding.VERTICES, self._buffers["vertices"]),
        ]

    def _create_pipeline(self):
        bind_layout, self._bind_group = self.device.create_bind_group(
            self.get_bindings(), "PointNumbersRenderObject"
        )
        pipeline_layout = self.device.create_pipeline_layout(bind_layout)
        shader_module = self.device.compile_files("shader.wgsl", "eval.wgsl")
        self._pipeline = self.gpu.device.createRenderPipeline(
            to_js(
                {
                    "label": "PointNumbersRenderObject",
                    "layout": pipeline_layout,
                    "vertex": {
                        "module": shader_module,
                        "entryPoint": "mainVertexPointNumber",
                    },
                    "fragment": {
                        "module": shader_module,
                        "entryPoint": "mainFragmentText",
                        "targets": [
                            {
                                "format": self.gpu.format,
                                "blend": {
                                    "color": {
                                        "operation": "add",
                                        "srcFactor": "one",
                                        "dstFactor": "one-minus-src-alpha",
                                    },
                                    "alpha": {
                                        "operation": "add",
                                        "srcFactor": "one",
                                        "dstFactor": "one-minus-src-alpha",
                                    },
                                },
                            }
                        ],
                    },
                    "primitive": {
                        "topology": "triangle-list",
                        "cullMode": "none",
                        "frontFace": "ccw",
                    },
                    "depthStencil": self.gpu.depth_stencil,
                }
            )
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
        self.gpu.uniforms.font_width = char_width
        self.gpu.uniforms.font_height = char_height
        if self._buffers:
            self._create_pipeline()
