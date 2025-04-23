import sys
from dataclasses import dataclass, field
from enum import Enum, IntFlag

from . import platform
from .platform import is_pyodide, JsProxy, toJS, create_proxy, JsPromise


class BaseWebGPUHandle:
    handle: JsProxy

    def __init__(self, handle):
        self.handle = handle

    def destroy(self):
        pass

    def __del__(self):
        pass
        # self.destroy()


def fromJS(obj):
    if type(obj) in [str, int, float, bool]:
        return obj
    return dict(obj)


class BaseWebGPUObject:
    def toJS(self):
        return toJS(self.__dict__)


class Sampler(BaseWebGPUHandle):
    pass


class BindGroup(BaseWebGPUHandle):
    pass


class BindGroupLayout(BaseWebGPUHandle):
    pass


class CommandBuffer(BaseWebGPUHandle):
    pass


class PipelineLayout(BaseWebGPUHandle):
    pass


class RenderBundle(BaseWebGPUHandle):
    pass


class TextureView(BaseWebGPUHandle):
    pass


class AutoLayoutMode(str, Enum):
    auto = "auto"


class AdapterType(str, Enum):
    discrete_GPU = "discrete_GPU"
    integrated_GPU = "integrated_GPU"
    CPU = "CPU"
    unknown = "unknown"


class AddressMode(str, Enum):
    undefined = "undefined"
    clamp_to_edge = "clamp-to-edge"
    repeat = "repeat"
    mirror_repeat = "mirror-repeat"


class BackendType(str, Enum):
    undefined = "undefined"
    null = "null"
    WebGPU = "WebGPU"
    D3D11 = "D3D11"
    D3D12 = "D3D12"
    metal = "metal"
    vulkan = "vulkan"
    openGL = "openGL"
    openGLES = "openGLES"


class BlendFactor(str, Enum):
    undefined = "undefined"
    zero = "zero"
    one = "one"
    src = "src"
    one_minus_src = "one-minus-src"
    src_alpha = "src-alpha"
    one_minus_src_alpha = "one-minus-src-alpha"
    dst = "dst"
    one_minus_dst = "one-minus-dst"
    dst_alpha = "dst-alpha"
    one_minus_dst_alpha = "one-minus-dst-alpha"
    src_alpha_saturated = "src-alpha-saturated"
    constant = "constant"
    one_minus_constant = "one-minus-constant"
    src1 = "src1"
    one_minus_src1 = "one-minus-src1"
    src1_alpha = "src1-alpha"
    one_minus_src1_alpha = "one-minus-src1-alpha"


class BlendOperation(str, Enum):
    undefined = "undefined"
    add = "add"
    subtract = "subtract"
    reverse_subtract = "reverse-subtract"
    min = "min"
    max = "max"


class BufferBindingType(str, Enum):
    binding_not_used = "binding_not_used"
    undefined = "undefined"
    uniform = "uniform"
    storage = "storage"
    read_only_storage = "read_only_storage"


class BufferMapState(str, Enum):
    unmapped = "unmapped"
    pending = "pending"
    mapped = "mapped"


class CallbackMode(str, Enum):
    wait_any_only = "wait_any_only"
    allow_process_events = "allow_process_events"
    allow_spontaneous = "allow_spontaneous"


class CompareFunction(str, Enum):
    undefined = "undefined"
    never = "never"
    less = "less"
    equal = "equal"
    less_equal = "less-equal"
    greater = "greater"
    not_equal = "not-equal"
    greater_equal = "greater-equal"
    always = "always"


class CompilationInfoRequestStatus(str, Enum):
    success = "success"
    instance_dropped = "instance_dropped"
    error = "error"


class CompilationMessageType(str, Enum):
    error = "error"
    warning = "warning"
    info = "info"


class CompositeAlphaMode(str, Enum):
    auto = "auto"
    opaque = "opaque"
    premultiplied = "premultiplied"
    unpremultiplied = "unpremultiplied"
    inherit = "inherit"


class CullMode(str, Enum):
    none = "none"
    front = "front"
    back = "back"


class DeviceLostReason(str, Enum):
    unknown = "unknown"
    destroyed = "destroyed"
    instance_dropped = "instance_dropped"
    failed_creation = "failed_creation"


class ErrorFilter(str, Enum):
    validation = "validation"
    out_of_memory = "out-of-memory"
    internal = "internal"


class FeatureLevel(str, Enum):
    compatibility = "compatibility"
    core = "core"


class FeatureName(str):
    undefined = "undefined"
    depth_clip_control = "depth-clip-control"
    depth32_float_stencil8 = "depth32float-stencil8"
    timestamp_query = "timestamp-query"
    texture_compression_BC = "texture-compression-BC"
    texture_compression_BC_sliced_3D = "texture-compression-BC-sliced-3D"
    texture_compression_ETC2 = "texture-compression-ETC2"
    texture_compression_ASTC = "texture-compression-ASTC"
    texture_compression_ASTC_sliced_3D = "texture-compression-ASTC-sliced-3D"
    indirect_first_instance = "indirect-first-instance"
    shader_f16 = "shader-f16"
    rg11b10ufloat_renderable = "rg11b10ufloat-renderable"
    bgra8unorm_storage = "bgra8unorm-storage"
    float32_filterable = "float32-filterable"
    float32_blendable = "float32-blendable"
    clip_distances = "clip-distances"
    dual_source_blending = "dual-source-blending"


class FilterMode(str, Enum):
    nearest = "nearest"
    linear = "linear"


class FrontFace(str, Enum):
    CCW = "ccw"
    CW = "cw"


class IndexFormat(str, Enum):
    uint16 = "uint16"
    uint32 = "uint32"


class LoadOp(str, Enum):
    load = "load"
    clear = "clear"


class MipmapFilterMode(str, Enum):
    nearest = "nearest"
    linear = "linear"


class PowerPreference(str, Enum):
    low_power = "low-power"
    high_performance = "high-performance"


class PresentMode(str, Enum):
    undefined = "undefined"
    fifo = "fifo"
    fifo_relaxed = "fifo_relaxed"
    immediate = "immediate"
    mailbox = "mailbox"


class PrimitiveTopology(str, Enum):
    point_list = "point-list"
    line_list = "line-list"
    line_strip = "line-strip"
    triangle_list = "triangle-list"
    triangle_strip = "triangle-strip"


class QueryType(str, Enum):
    occlusion = "occlusion"
    timestamp = "timestamp"


class QuerySet(BaseWebGPUHandle):
    @property
    def type(self) -> QueryType:
        return self.handle.type()

    @property
    def count(self) -> int:
        return self.handle.count()

    def destroy(self) -> None:
        return self.handle.destroy()


class SamplerBindingType(str, Enum):
    filtering = "filtering"
    non_filtering = "non-filtering"
    comparison = "comparison"


class Status(str, Enum):
    success = "success"
    error = "error"


class StencilOperation(str, Enum):
    undefined = "undefined"
    keep = "keep"
    zero = "zero"
    replace = "replace"
    invert = "invert"
    increment_clamp = "increment-clamp"
    decrement_clamp = "decrement-clamp"
    increment_wrap = "increment-wrap"
    decrement_wrap = "decrement-wrap"


class StorageTextureAccess(str, Enum):
    write_only = "write-only"
    read_only = "read-ony"
    read_write = "read-write"


class StoreOp(str, Enum):
    store = "store"
    discard = "discard"


class TextureAspect(str, Enum):
    all = "all"
    stencil_only = "stencil-only"
    depth_only = "depth-only"


def TextureDimensionInt2Str(dim: int):
    return ["1d", "2d", "3d"][dim - 1]


class TextureFormat(str, Enum):
    # 8-bit formats
    r8unorm = "r8unorm"
    r8snorm = "r8snorm"
    r8uint = "r8uint"
    r8sint = "r8sint"

    # 16-bit formats
    r16uint = "r16uint"
    r16sint = "r16sint"
    r16float = "r16float"
    rg8unorm = "rg8unorm"
    rg8snorm = "rg8snorm"
    rg8uint = "rg8uint"
    rg8sint = "rg8sint"

    # 32-bit formats
    r32uint = "r32uint"
    r32sint = "r32sint"
    r32float = "r32float"
    rg16uint = "rg16uint"
    rg16sint = "rg16sint"
    rg16float = "rg16float"
    rgba8unorm = "rgba8unorm"
    rgba8unorm_srgb = "rgba8unorm-srgb"
    rgba8snorm = "rgba8snorm"
    rgba8uint = "rgba8uint"
    rgba8sint = "rgba8sint"
    bgra8unorm = "bgra8unorm"
    bgra8unorm_srgb = "bgra8unorm-srgb"
    # Packed 32-bit formats
    rgb9e5ufloat = "rgb9e5ufloat"
    rgb10a2uint = "rgb10a2uint"
    rgb10a2unorm = "rgb10a2unorm"
    rg11b10ufloat = "rg11b10ufloat"

    # 64-bit formats
    rg32uint = "rg32uint"
    rg32sint = "rg32sint"
    rg32float = "rg32float"
    rgba16uint = "rgba16uint"
    rgba16sint = "rgba16sint"
    rgba16float = "rgba16float"

    # 128-bit formats
    rgba32uint = "rgba32uint"
    rgba32sint = "rgba32sint"
    rgba32float = "rgba32float"

    # Depth/stencil formats
    stencil8 = "stencil8"
    depth16unorm = "depth16unorm"
    depth24plus = "depth24plus"
    depth24plus_stencil8 = "depth24plus-stencil8"
    depth32float = "depth32float"

    # "depth32float-stencil8" feature
    depth32float_stencil8 = "depth32float-stencil8"

    # BC compressed formats usable if "texture-compression-bc" is both
    # supported by the device/user agent and enabled in requestDevice.
    bc1_rgba_unorm = "bc1-rgba-unorm"
    bc1_rgba_unorm_srgb = "bc1-rgba-unorm-srgb"
    bc2_rgba_unorm = "bc2-rgba-unorm"
    bc2_rgba_unorm_srgb = "bc2-rgba-unorm-srgb"
    bc3_rgba_unorm = "bc3-rgba-unorm"
    bc3_rgba_unorm_srgb = "bc3-rgba-unorm-srgb"
    bc4_r_unorm = "bc4-r-unorm"
    bc4_r_snorm = "bc4-r-snorm"
    bc5_rg_unorm = "bc5-rg-unorm"
    bc5_rg_snorm = "bc5-rg-snorm"
    bc6h_rgb_ufloat = "bc6h-rgb-ufloat"
    bc6h_rgb_float = "bc6h-rgb-float"
    bc7_rgba_unorm = "bc7-rgba-unorm"
    bc7_rgba_unorm_srgb = "bc7-rgba-unorm-srgb"

    # ETC2 compressed formats usable if "texture-compression-etc2" is both
    # supported by the device/user agent and enabled in requestDevice.
    etc2_rgb8unorm = "etc2-rgb8unorm"
    etc2_rgb8unorm_srgb = "etc2-rgb8unorm-srgb"
    etc2_rgb8a1unorm = "etc2-rgb8a1unorm"
    etc2_rgb8a1unorm_srgb = "etc2-rgb8a1unorm-srgb"
    etc2_rgba8unorm = "etc2-rgba8unorm"
    etc2_rgba8unorm_srgb = "etc2-rgba8unorm-srgb"
    eac_r11unorm = "eac-r11unorm"
    eac_r11snorm = "eac-r11snorm"
    eac_rg11unorm = "eac-rg11unorm"
    eac_rg11snorm = "eac-rg11snorm"

    # ASTC compressed formats usable if "texture-compression-astc" is both
    # supported by the device/user agent and enabled in requestDevice.
    astc_4x4_unorm = "astc-4x4-unorm"
    astc_4x4_unorm_srgb = "astc-4x4-unorm-srgb"
    astc_5x4_unorm = "astc-5x4-unorm"
    astc_5x4_unorm_srgb = "astc-5x4-unorm-srgb"
    astc_5x5_unorm = "astc-5x5-unorm"
    astc_5x5_unorm_srgb = "astc-5x5-unorm-srgb"
    astc_6x5_unorm = "astc-6x5-unorm"
    astc_6x5_unorm_srgb = "astc-6x5-unorm-srgb"
    astc_6x6_unorm = "astc-6x6-unorm"
    astc_6x6_unorm_srgb = "astc-6x6-unorm-srgb"
    astc_8x5_unorm = "astc-8x5-unorm"
    astc_8x5_unorm_srgb = "astc-8x5-unorm-srgb"
    astc_8x6_unorm = "astc-8x6-unorm"
    astc_8x6_unorm_srgb = "astc-8x6-unorm-srgb"
    astc_8x8_unorm = "astc-8x8-unorm"
    astc_8x8_unorm_srgb = "astc-8x8-unorm-srgb"
    astc_10x5_unorm = "astc-10x5-unorm"
    astc_10x5_unorm_srgb = "astc-10x5-unorm-srgb"
    astc_10x6_unorm = "astc-10x6-unorm"
    astc_10x6_unorm_srgb = "astc-10x6-unorm-srgb"
    astc_10x8_unorm = "astc-10x8-unorm"
    astc_10x8_unorm_srgb = "astc-10x8-unorm-srgb"
    astc_10x10_unorm = "astc-10x10-unorm"
    astc_10x10_unorm_srgb = "astc-10x10-unorm-srgb"
    astc_12x10_unorm = "astc-12x10-unorm"
    astc_12x10_unorm_srgb = "astc-12x10-unorm-srgb"
    astc_12x12_unorm = "astc-12x12-unorm"
    astc_12x12_unorm_srgb = "astc-12x12-unorm-srgb"


class TextureSampleType(str, Enum):
    float = "float"
    unfilterable_float = "unfilterable_float"
    depth = "depth"
    sint = "sint"
    uint = "uint"


class VertexFormat(str, Enum):
    uint8 = "uint8"
    uint8x2 = "uint8x2"
    uint8x4 = "uint8x4"
    sint8 = "sint8"
    sint8x2 = "sint8x2"
    sint8x4 = "sint8x4"
    unorm8 = "unorm8"
    unorm8x2 = "unorm8x2"
    unorm8x4 = "unorm8x4"
    snorm8 = "snorm8"
    snorm8x2 = "snorm8x2"
    snorm8x4 = "snorm8x4"
    uint16 = "uint16"
    uint16x2 = "uint16x2"
    uint16x4 = "uint16x4"
    sint16 = "sint16"
    sint16x2 = "sint16x2"
    sint16x4 = "sint16x4"
    unorm16 = "unorm16"
    unorm16x2 = "unorm16x2"
    unorm16x4 = "unorm16x4"
    snorm16 = "snorm16"
    snorm16x2 = "snorm16x2"
    snorm16x4 = "snorm16x4"
    float16 = "float16"
    float16x2 = "float16x2"
    float16x4 = "float16x4"
    float32 = "float32"
    float32x2 = "float32x2"
    float32x3 = "float32x3"
    float32x4 = "float32x4"
    uint32 = "uint32"
    uint32x2 = "uint32x2"
    uint32x3 = "uint32x3"
    uint32x4 = "uint32x4"
    sint32 = "sint32"
    sint32x2 = "sint32x2"
    sint32x3 = "sint32x3"
    sint32x4 = "sint32x4"
    unorm10__10__10__2 = "unorm10__10__10__2"
    unorm8x4_B_G_R_A = "unorm8x4_B_G_R_A"


class VertexStepMode(str, Enum):
    vertex = "vertex"
    instance = "instance"


class BufferUsage(IntFlag):
    NONE = 0
    MAP_READ = 0x01
    MAP_WRITE = 0x02
    COPY_SRC = 0x04
    COPY_DST = 0x08
    INDEX = 0x10
    VERTEX = 0x20
    UNIFORM = 0x40
    STORAGE = 0x80
    INDIRECT = 0x100
    QUERY_RESOLVE = 0x200


class ColorWriteMask(IntFlag):
    NONE = 0
    RED = 1
    GREEN = 2
    BLUE = 4
    ALPHA = 8
    ALL = 15


class MapMode(IntFlag):
    NONE = 0
    READ = 1
    WRITE = 2


class ShaderStage(IntFlag):
    NONE = 0
    VERTEX = 1
    FRAGMENT = 2
    COMPUTE = 4
    ALL = 7


class TextureUsage(IntFlag):
    NONE = 0
    COPY_SRC = 1
    COPY_DST = 2
    TEXTURE_BINDING = 4
    STORAGE_BINDING = 8
    RENDER_ATTACHMENT = 16


@dataclass
class AdapterInfo(BaseWebGPUObject):
    vendor: str = ""
    architecture: str = ""
    device: str = ""
    description: str = ""


@dataclass
class BindGroupEntry(BaseWebGPUObject):
    binding: int
    resource: "Sampler | TextureView | Buffer"


@dataclass
class BindGroupDescriptor(BaseWebGPUObject):
    layout: BindGroupLayout
    entries: list[BindGroupEntry] = field(default_factory=list)
    label: str = ""


@dataclass
class BindGroupLayoutDescriptor(BaseWebGPUObject):
    label: str = ""
    entries: list["BindGroupLayoutEntry"] = field(default_factory=list)


@dataclass
class BindGroupLayoutEntry(BaseWebGPUObject):
    binding: int = 0
    visibility: ShaderStage = ShaderStage.NONE
    buffer: "BufferBindingLayout | None" = None
    sampler: "SamplerBindingLayout | None" = None
    texture: "TextureBindingLayout | None" = None
    storageTexture: "StorageTextureBindingLayout | None" = None


@dataclass
class BlendComponent(BaseWebGPUObject):
    operation: BlendOperation = BlendOperation.add
    srcFactor: BlendFactor = BlendFactor.one
    dstFactor: BlendFactor = BlendFactor.zero


@dataclass
class BlendState(BaseWebGPUObject):
    color: BlendComponent
    alpha: BlendComponent


@dataclass
class BufferBindingLayout(BaseWebGPUObject):
    type: BufferBindingType = BufferBindingType.uniform
    hasDynamicOffset: bool = False
    minBindingSize: int = 0


@dataclass
class BufferDescriptor(BaseWebGPUObject):
    size: int
    usage: BufferUsage
    mappedAtCreation: bool = False
    label: str = ""


@dataclass
class Color(BaseWebGPUObject):
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    a: float = 0.0


@dataclass
class ColorTargetState(BaseWebGPUObject):
    format: TextureFormat
    blend: BlendState | None = None
    writeMask: ColorWriteMask = ColorWriteMask.ALL


@dataclass
class CommandBufferDescriptor(BaseWebGPUObject):
    label: str = ""


@dataclass
class CommandEncoderDescriptor(BaseWebGPUObject):
    label: str = ""


@dataclass
class CompilationMessage(BaseWebGPUObject):
    message: str
    type: CompilationMessageType
    lineNum: int
    linePos: int
    offset: int
    length: int


@dataclass
class CompilationInfo(BaseWebGPUObject):
    messages: list[CompilationMessage] = field(default_factory=list)


@dataclass
class ComputePassDescriptor(BaseWebGPUObject):
    timestampWrites: "PassTimestampWrites"
    label: str = ""


@dataclass
class ComputeState(BaseWebGPUObject):
    module: "ShaderModule"
    entryPoint: str = ""


@dataclass
class ComputePipelineDescriptor(BaseWebGPUObject):
    layout: "PipelineLayout"
    compute: ComputeState
    label: str = ""


@dataclass
class StencilFaceState(BaseWebGPUObject):
    compare: "CompareFunction | None" = None
    failOp: "StencilOperation | None" = None
    depthFailOp: "StencilOperation | None" = None
    passOp: "StencilOperation | None" = None


@dataclass
class DepthStencilState(BaseWebGPUObject):
    format: TextureFormat
    depthWriteEnabled: bool
    depthCompare: CompareFunction
    stencilFront: StencilFaceState = field(default_factory=StencilFaceState)
    stencilBack: StencilFaceState = field(default_factory=StencilFaceState)
    stencilReadMask: int = 0xFFFFFFFF
    stencilWriteMask: int = 0xFFFFFFFF
    depthBias: int = 0
    depthBiasSlopeScale: float = 0.0
    depthBiasClamp: float = 0.0


@dataclass
class Limits(BaseWebGPUObject):
    maxTextureDimension1D: int | None = None
    maxTextureDimension2D: int | None = None
    maxTextureDimension3D: int | None = None
    maxTextureArrayLayers: int | None = None
    maxBindGroups: int | None = None
    maxBindGroupsPlusVertexBuffers: int | None = None
    maxBindingsPerBindGroup: int | None = None
    maxDynamicUniformBuffersPerPipelineLayout: int | None = None
    maxDynamicStorageBuffersPerPipelineLayout: int | None = None
    maxSampledTexturesPerShaderStage: int | None = None
    maxSamplersPerShaderStage: int | None = None
    maxStorageBuffersPerShaderStage: int | None = None
    maxStorageTexturesPerShaderStage: int | None = None
    maxUniformBuffersPerShaderStage: int | None = None
    maxUniformBufferBindingSize: int | None = None
    maxStorageBufferBindingSize: int | None = None
    minUniformBufferOffsetAlignment: int | None = None
    minStorageBufferOffsetAlignment: int | None = None
    maxVertexBuffers: int | None = None
    maxBufferSize: int | None = None
    maxVertexAttributes: int | None = None
    maxVertexBufferArrayStride: int | None = None
    maxInterStageShaderVariables: int | None = None
    maxColorAttachments: int | None = None
    maxColorAttachmentBytesPerSample: int | None = None
    maxComputeWorkgroupStorageSize: int | None = None
    maxComputeInvocationsPerWorkgroup: int | None = None
    maxComputeWorkgroupSizeX: int | None = None
    maxComputeWorkgroupSizeY: int | None = None
    maxComputeWorkgroupSizeZ: int | None = None
    maxComputeWorkgroupsPerDimension: int | None = None


@dataclass
class QueueDescriptor(BaseWebGPUObject):
    label: str = ""


@dataclass
class DeviceDescriptor(BaseWebGPUObject):
    requiredFeatures: list["FeatureName"] | None = None
    requiredLimits: Limits | None = None
    defaultQueue: QueueDescriptor | None = None
    label: str = ""


@dataclass
class Extent3d(BaseWebGPUObject):
    width: int = 0
    height: int = 0
    depthOrArrayLayers: int = 0


@dataclass
class FragmentState(BaseWebGPUObject):
    module: "ShaderModule | None" = None
    entryPoint: str = ""
    targets: list[ColorTargetState] = field(default_factory=list)


@dataclass
class MultisampleState(BaseWebGPUObject):
    count: int = 1
    mask: int = 0xFFFFFFFF
    alphaToCoverageEnabled: bool = False


@dataclass
class Origin3d(BaseWebGPUObject):
    x: int = 0
    y: int = 0
    z: int = 0


@dataclass
class PassTimestampWrites(BaseWebGPUObject):
    querySet: "QuerySet"
    beginningOfPassWriteIndex: int
    endOfPassWriteIndex: int


@dataclass
class PipelineLayoutDescriptor(BaseWebGPUObject):
    bindGroupLayouts: list["BindGroupLayout"] = field(default_factory=list)
    label: str = ""


@dataclass
class PrimitiveState(BaseWebGPUObject):
    topology: "PrimitiveTopology | None" = None
    stripIndexFormat: IndexFormat | None = None
    frontFace: FrontFace = FrontFace.CCW
    cullMode: CullMode = CullMode.none
    unclippedDepth: bool = False


@dataclass
class QuerySetDescriptor(BaseWebGPUObject):
    type: QueryType
    count: int
    label: str = ""


@dataclass
class RenderBundleDescriptor(BaseWebGPUObject):
    label: str = ""


@dataclass
class RenderBundleEncoderDescriptor(BaseWebGPUObject):
    colorFormats: list[TextureFormat]
    depthStencilFormat: TextureFormat
    sampleCount: int = 1
    depthReadOnly: bool = False
    stencilReadOnly: bool = False
    label: str = ""


@dataclass
class RenderPassColorAttachment(BaseWebGPUObject):
    view: TextureView
    resolveTarget: TextureView | None = None
    loadOp: LoadOp = LoadOp.load
    storeOp: StoreOp = StoreOp.store
    clearValue: Color = field(default_factory=Color)
    depthSlice: int | None = None


@dataclass
class RenderPassDepthStencilAttachment(BaseWebGPUObject):
    view: TextureView
    depthLoadOp: LoadOp = LoadOp.load
    depthStoreOp: StoreOp = StoreOp.store
    depthClearValue: float = 0.0
    depthReadOnly: bool = False
    stencilClearValue: int = 0
    stencilLoadOp: LoadOp | None = None
    stencilStoreOp: StoreOp | None = None
    stencilReadOnly: bool = False


@dataclass
class RenderPassDescriptor(BaseWebGPUObject):
    colorAttachments: list[RenderPassColorAttachment]
    depthStencilAttachment: RenderPassDepthStencilAttachment
    occlusionQuerySet: "QuerySet | None" = None
    timestampWrites: PassTimestampWrites | None = None
    label: str = ""


@dataclass
class RenderPipelineDescriptor(BaseWebGPUObject):
    layout: PipelineLayout | AutoLayoutMode
    vertex: "VertexState"
    fragment: "FragmentState"
    depthStencil: "DepthStencilState"
    primitive: PrimitiveState = field(default_factory=PrimitiveState)
    multisample: MultisampleState = field(default_factory=MultisampleState)
    label: str = ""


@dataclass
class RequestAdapterOptions(BaseWebGPUObject):
    featureLevel: FeatureLevel | None = None
    powerPreference: "PowerPreference | None" = None
    forceFallbackAdapter: bool = False
    xrCompatible: bool = False


async def requestAdapter(
    featureLevel: FeatureLevel | None = None,
    powerPreference: "PowerPreference | None" = None,
    forceFallbackAdapter: bool = False,
    xrCompatible: bool = False,
) -> "Adapter":
    if not platform.js.navigator.gpu:
        platform.js.alert("WebGPU is not supported")
        sys.exit(1)

    reqAdapter = platform.js.navigator.gpu.requestAdapter
    options = RequestAdapterOptions(
        featureLevel=featureLevel,
        powerPreference=powerPreference,
        forceFallbackAdapter=forceFallbackAdapter,
        xrCompatible=xrCompatible,
    ).toJS()
    print("requestAdapter", reqAdapter, options)
    handle = reqAdapter(options)
    try:
        handle = await handle
    except:
        pass
    if not handle:
        platform.js.alert("WebGPU is not supported")
        sys.exit(1)
    return Adapter(handle)


@dataclass
class SamplerBindingLayout(BaseWebGPUObject):
    type: SamplerBindingType = SamplerBindingType.filtering


@dataclass
class SamplerDescriptor(BaseWebGPUObject):
    label: str = ""
    addressModeU: AddressMode = AddressMode.clamp_to_edge
    addressModeV: AddressMode = AddressMode.clamp_to_edge
    addressModeW: AddressMode = AddressMode.clamp_to_edge
    magFilter: FilterMode = FilterMode.nearest
    minFilter: FilterMode = FilterMode.nearest
    mipmapFilter: MipmapFilterMode = MipmapFilterMode.nearest
    lodMinClamp: float = 0.0
    lodMaxClamp: float = 32
    compare: "CompareFunction | None" = None
    maxAnisotropy: int = 1


@dataclass
class ShaderModuleCompilationHint(BaseWebGPUObject):
    entryPoint: str
    layout: PipelineLayout | AutoLayoutMode


@dataclass
class ShaderModuleDescriptor(BaseWebGPUObject):
    code: str
    compilationHints: list["ShaderModuleCompilationHint"] = field(default_factory=list)
    label: str = ""


@dataclass
class StorageTextureBindingLayout(BaseWebGPUObject):
    format: TextureFormat
    access: StorageTextureAccess = StorageTextureAccess.write_only
    viewDimension: str = "2d"


@dataclass
class TexelCopyBufferLayout(BaseWebGPUObject):
    bytesPerRow: int
    offset: int = 0
    rowsPerImage: int | None = None


@dataclass
class TexelCopyBufferInfo(BaseWebGPUObject):
    layout: TexelCopyBufferLayout
    buffer: "Buffer"


@dataclass
class TexelCopyTextureInfo(BaseWebGPUObject):
    texture: "Texture"
    mipLevel: int = 0
    origin: Origin3d = field(default_factory=Origin3d)
    aspect: TextureAspect = TextureAspect.all


@dataclass
class TextureBindingLayout(BaseWebGPUObject):
    sampleType: TextureSampleType = TextureSampleType.float
    viewDimension: str = "2d"
    multisampled: bool = False


@dataclass
class TextureDescriptor(BaseWebGPUObject):
    size: list
    usage: TextureUsage
    format: TextureFormat
    sampleCount: int = 1
    dimension: str = "2d"
    mipLevelCount: int = 1
    viewFormats: list["TextureFormat"] | None = None
    label: str = ""


@dataclass
class TextureViewDescriptor(BaseWebGPUObject):
    format: TextureFormat
    dimension: str
    baseMipLevel: int = 0
    mipLevelCount: int = 1
    baseArrayLayer: int = 0
    arrayLayerCount: int = 0
    aspect: TextureAspect = TextureAspect.all
    usage: TextureUsage = TextureUsage.NONE
    label: str = ""


@dataclass
class VertexAttribute(BaseWebGPUObject):
    format: VertexFormat
    offset: int
    shaderLocation: int


@dataclass
class VertexBufferLayout(BaseWebGPUObject):
    arrayStride: int
    stepMode: VertexStepMode = VertexStepMode.vertex
    attributes: list["VertexAttribute"] = field(default_factory=list)


@dataclass
class VertexState(BaseWebGPUObject):
    module: "ShaderModule"
    entryPoint: str = ""
    buffers: list["VertexBufferLayout"] = field(default_factory=list)


class Adapter(BaseWebGPUHandle):

    @property
    def limits(self) -> Limits:
        return self.handle.limits

    @property
    def features(self) -> list[FeatureName]:
        return [FeatureName(f) for f in self.handle.features]

    @property
    def info(self) -> AdapterInfo:
        return self.handle.info

    @property
    def isFallbackAdapter(self) -> bool:
        return self.handle.isFallbackAdapter

    async def requestDevice(
        self,
        requiredFeatures: list["FeatureName"] | None = None,
        requiredLimits: Limits | None = None,
        defaultQueue: QueueDescriptor | None = None,
        label: str = "",
    ) -> "Device":

        device = self.handle.requestDevice(
            DeviceDescriptor(
                requiredFeatures=requiredFeatures,
                requiredLimits=requiredLimits.toJS() if requiredLimits else None,
                defaultQueue=defaultQueue,
                label=label,
            ).toJS()
        )

        try:
            device = await device
        except:
            pass
        return Device(device)


class Buffer(BaseWebGPUHandle):
    async def mapAsync(self, mode: MapMode, offset: int = 0, size: int = 0) -> None:
        return await self.handle.mapAsync(mode, offset, size)

    def getMappedRange(self, offset: int = 0, size: int = 0) -> int:
        return self.handle.getMappedRange(offset, size)

    def getConstMappedRange(self, offset: int = 0, size: int = 0) -> int:
        return self.handle.getConstMappedRange(offset, size)

    @property
    def usage(self) -> BufferUsage:
        return self.handle.usage

    @property
    def size(self) -> int:
        return self.handle.size

    @property
    def mapState(self) -> BufferMapState:
        return self.handle.mapState

    def unmap(self) -> None:
        self.handle.unmap()

    def destroy(self) -> None:
        self.handle.destroy()

    def __del__(self):
        self.destroy()


class CommandEncoder(BaseWebGPUHandle):
    _first_render_pass: bool = True

    def __init__(self, handle):
        super().__init__(handle)
        self._first_render_pass = True

    def getLoadOp(self) -> LoadOp:
        if self._first_render_pass:
            self._first_render_pass = False
            return LoadOp.clear
        return LoadOp.load

    def finish(self, label: str = "") -> "CommandBuffer":
        return self.handle.finish(CommandBufferDescriptor(label=label).toJS())

    def beginComputePass(
        self, timestampWrites: PassTimestampWrites | None = None, label: str = ""
    ) -> "ComputePassEncoder":
        return self.handle.beginComputePass(
            ComputePassDescriptor(timestampWrites=timestampWrites, label=label).toJS()
        )

    def beginRenderPass(
        self,
        colorAttachments: list[RenderPassColorAttachment],
        depthStencilAttachment: RenderPassDepthStencilAttachment,
        occlusionQuerySet: "QuerySet | None" = None,
        timestampWrites: PassTimestampWrites | None = None,
        label: str = "",
    ) -> "RenderPassEncoder":

        return self.handle.beginRenderPass(
            RenderPassDescriptor(
                colorAttachments=colorAttachments,
                depthStencilAttachment=depthStencilAttachment,
                occlusionQuerySet=occlusionQuerySet,
                timestampWrites=timestampWrites,
                label=label,
            ).toJS()
        )

    def copyBufferToBuffer(
        self, source: Buffer, sourceOffset, destination: Buffer, destinationOffset, size
    ) -> None:
        return self.handle.copyBufferToBuffer(
            source.handle, sourceOffset, destination.handle, destinationOffset, size
        )

    def copyBufferToTexture(
        self,
        source: TexelCopyBufferInfo,
        destination: TexelCopyTextureInfo,
        copySize: list,
    ) -> None:
        return self.handle.copyBufferToTexture(
            source.toJS(), destination.toJS(), copySize
        )

    def copyTextureToBuffer(
        self,
        source: TexelCopyTextureInfo,
        destination: TexelCopyBufferInfo,
        copySize: list,
    ) -> None:
        return self.handle.copyTextureToBuffer(
            source.toJS(), destination.toJS(), copySize
        )

    def copyTextureToTexture(
        self,
        source: TexelCopyTextureInfo,
        destination: TexelCopyTextureInfo,
        copySize: list,
    ) -> None:
        return self.handle.copyTextureToTexture(
            source.toJS(), destination.toJS(), copySize
        )

    def clearBuffer(self, buffer: Buffer, offset: int = 0, size: int = 0) -> None:
        return self.handle.clearBuffer(buffer.handle, offset, size)

    def insertDebugMarker(self, markerLabel: str = "") -> None:
        return self.handle.insertDebugMarker(markerLabel)

    def popDebugGroup(self) -> None:
        return self.handle.popDebugGroup()

    def pushDebugGroup(self, groupLabel: str = "") -> None:
        return self.handle.pushDebugGroup(groupLabel)

    def resolveQuerySet(
        self,
        querySet: "QuerySet",
        firstQuery: int,
        queryCount: int,
        destination: Buffer,
        destinationOffset: int = 0,
    ) -> None:
        return self.handle.resolveQuerySet(
            querySet, firstQuery, queryCount, destination.handle, destinationOffset
        )


class ComputePassEncoder(BaseWebGPUHandle):
    def insertDebugMarker(self, markerLabel: str = "") -> None:
        return self.handle.insertDebugMarker(markerLabel)

    def popDebugGroup(self) -> None:
        return self.handle.popDebugGroup()

    def pushDebugGroup(self, groupLabel: str = "") -> None:
        return self.handle.pushDebugGroup(groupLabel)

    def setPipeline(self, pipeline: "ComputePipeline | None" = None) -> None:
        return self.handle.setPipeline(pipeline)

    def setBindGroup(
        self,
        index: int,
        bindGroup: BindGroup,
        dynamicOffsets: list[int] = [],
    ) -> None:
        return self.handle.setBindGroup(index, bindGroup, dynamicOffsets)

    def dispatchWorkgroups(
        self,
        workgroupCountX: int,
        workgroupCountY: int = 0,
        workgroupCountZ: int = 0,
    ) -> None:
        return self.handle.dispatchWorkgroups(
            workgroupCountX, workgroupCountY, workgroupCountZ
        )

    def dispatchWorkgroupsIndirect(
        self, indirectBuffer: Buffer, indirectOffset: int = 0
    ) -> None:
        return self.handle.dispatchWorkgroupsIndirect(
            indirectBuffer.handle, indirectOffset
        )

    def end(self) -> None:
        return self.handle.end()


class ComputePipeline(BaseWebGPUHandle):
    def getBindGroupLayout(self, groupIndex: int = 0) -> BindGroupLayout:
        return self.handle.getBindGroupLayout(groupIndex)


class Device(BaseWebGPUHandle):
    def createBindGroup(
        self,
        layout: "BindGroupLayout",
        entries: list["BindGroupEntry"] = field(default_factory=list),
        label: str = "",
    ) -> BindGroup:

        return self.handle.createBindGroup(
            BindGroupDescriptor(
                layout=layout,
                entries=entries,
                label=label,
            ).toJS()
        )

    def createBindGroupLayout(
        self,
        entries: list["BindGroupLayoutEntry"] = field(default_factory=list),
        label: str = "",
    ) -> "BindGroupLayout":
        return self.handle.createBindGroupLayout(
            BindGroupLayoutDescriptor(entries=entries, label=label).toJS()
        )

    def createBuffer(
        self,
        size: int,
        usage: BufferUsage,
        mappedAtCreation: bool = False,
        label: str = "",
    ) -> Buffer:
        return Buffer(
            self.handle.createBuffer(
                BufferDescriptor(
                    size=size,
                    usage=usage,
                    mappedAtCreation=mappedAtCreation,
                    label=label,
                ).toJS()
            )
        )

    def createCommandEncoder(
        self,
        label: str = "",
    ) -> CommandEncoder:
        return CommandEncoder(
            self.handle.createCommandEncoder(
                CommandEncoderDescriptor(label=label).toJS()
            )
        )

    async def createComputePipelineAsync(
        self,
        layout: "PipelineLayout",
        compute: ComputeState,
        label: str = "",
    ) -> ComputePipeline:
        return await self.handle.createComputePipelineAsync(
            ComputePipelineDescriptor(
                layout=layout,
                compute=compute,
                label=label,
            ).toJS()
        )

    def createComputePipeline(
        self,
        layout: "PipelineLayout",
        compute: ComputeState,
        label: str = "",
    ) -> ComputePipeline:
        return self.handle.createComputePipeline(
            ComputePipelineDescriptor(
                layout=layout,
                compute=compute,
                label=label,
            ).toJS()
        )

    def createPipelineLayout(
        self,
        bindGroupLayouts: list[BindGroupLayout] = [],
        label: str = "",
    ) -> PipelineLayout:
        return self.handle.createPipelineLayout(
            PipelineLayoutDescriptor(
                bindGroupLayouts=bindGroupLayouts, label=label
            ).toJS()
        )

    def createQuerySet(self, count: int, label: str = "") -> "QuerySet":
        return self.handle.createQuerySet(
            QuerySetDescriptor(
                type=QueryType.occlusion, count=count, label=label
            ).toJS()
        )

    def createRenderPipeline(
        self,
        layout: PipelineLayout | AutoLayoutMode,
        vertex: VertexState,
        fragment: FragmentState,
        depthStencil: DepthStencilState,
        primitive: PrimitiveState = field(default_factory=PrimitiveState),
        multisample: MultisampleState = field(default_factory=MultisampleState),
        label: str = "",
    ) -> None:
        return self.handle.createRenderPipeline(
            RenderPipelineDescriptor(
                layout=layout,
                vertex=vertex,
                fragment=fragment,
                depthStencil=depthStencil,
                primitive=primitive,
                multisample=multisample,
                label=label,
            ).toJS()
        )

    async def createRenderPipelineAsync(
        self,
        layout: "PipelineLayout",
        vertex: "VertexState",
        fragment: "FragmentState",
        depthStencil: "DepthStencilState",
        primitive: "PrimitiveState" = field(default_factory=PrimitiveState),
        multisample: "MultisampleState" = field(default_factory=MultisampleState),
        label: str = "",
    ) -> None:
        return await self.handle.createRenderPipelineAsync(
            RenderPipelineDescriptor(
                layout=layout,
                vertex=vertex,
                fragment=fragment,
                depthStencil=depthStencil,
                primitive=primitive,
                multisample=multisample,
                label=label,
            ).toJS()
        )

    def createRenderBundleEncoder(self, label: str = "") -> "RenderBundleEncoder":
        return self.handle.createRenderBundleEncoder(
            RenderBundleDescriptor(label=label).toJS()
        )

    def createSampler(
        self,
        addressModeU: AddressMode = AddressMode.clamp_to_edge,
        addressModeV: AddressMode = AddressMode.clamp_to_edge,
        addressModeW: AddressMode = AddressMode.clamp_to_edge,
        magFilter: FilterMode = FilterMode.nearest,
        minFilter: FilterMode = FilterMode.nearest,
        mipmapFilter: MipmapFilterMode = MipmapFilterMode.nearest,
        lodMinClamp: float = 0.0,
        lodMaxClamp: float = 32,
        compare: "CompareFunction | None" = None,
        maxAnisotropy: int = 1,
        label: str = "",
    ):
        return self.handle.createSampler(
            SamplerDescriptor(
                addressModeU=addressModeU,
                addressModeV=addressModeV,
                addressModeW=addressModeW,
                magFilter=magFilter,
                minFilter=minFilter,
                mipmapFilter=mipmapFilter,
                lodMinClamp=lodMinClamp,
                lodMaxClamp=lodMaxClamp,
                compare=compare,
                maxAnisotropy=maxAnisotropy,
                label=label,
            ).toJS()
        )

    def createShaderModule(
        self,
        code: str,
        compilationHints: list[ShaderModuleCompilationHint] = [],
        label: str = "",
    ) -> "ShaderModule":
        return self.handle.createShaderModule(
            ShaderModuleDescriptor(
                code=code,
                compilationHints=compilationHints,
                label=label,
            ).toJS()
        )

    def createTexture(
        self,
        size: list,
        usage: TextureUsage,
        format: TextureFormat,
        sampleCount: int = 1,
        dimension: str = "2d",
        mipLevelCount: int = 1,
        viewFormats: list["TextureFormat"] | None = None,
        label: str = "",
    ) -> "Texture":
        return self.handle.createTexture(
            TextureDescriptor(
                size=size,
                usage=usage,
                format=format,
                sampleCount=sampleCount,
                dimension=dimension,
                mipLevelCount=mipLevelCount,
                viewFormats=viewFormats,
                label=label,
            ).toJS()
        )

    def destroy(self) -> None:
        return self.handle.destroy()

    def __del__(self):
        self.destroy()

    @property
    def limits(self) -> Limits:
        return self.handle.limits

    @property
    def features(self) -> list[FeatureName]:
        return [FeatureName(f) for f in self.handle.features]

    @property
    def adapterInfo(self) -> AdapterInfo:
        return self.handle.adapterInfo

    @property
    def queue(self) -> "Queue":
        return Queue(self.handle.queue)

    def pushErrorScope(self, filter: ErrorFilter) -> None:
        return self.handle.pushErrorScope(filter)

    def popErrorScope(self) -> None:
        return self.handle.popErrorScope()


class Queue(BaseWebGPUHandle):
    def submit(self, commands: list[CommandBuffer] = []) -> None:
        return self.handle.submit(commands)

    def onSubmittedWorkDone(self) -> JsPromise:
        return self.handle.onSubmittedWorkDone()

    def writeBuffer(
        self,
        buffer: Buffer,
        bufferOffset: int,
        data: bytes,
        dataOffset: int = 0,
        size: int | None = None,
    ):
        self.handle.writeBuffer(
            buffer.handle,
            bufferOffset,
            toJS(memoryview(data)),
            dataOffset,
            size,
        )

    def writeTexture(
        self,
        destination: TexelCopyTextureInfo,
        data: bytes,
        dataLayout: TexelCopyBufferLayout,
        size: list,
    ) -> None:
        return self.handle.writeTexture(
            destination.toJS(),
            toJS(memoryview(bytes(data))),
            dataLayout.toJS(),
            size,
        )


class RenderBundleEncoder(BaseWebGPUHandle):
    def setPipeline(self, pipeline: "RenderPipeline | None" = None) -> None:
        self.handle.setPipeline(pipeline)

    def setBindGroup(
        self,
        groupIndex: int = 0,
        group: "BindGroup | None" = None,
        dynamicOffsets: list[int] = [],
    ) -> None:
        return self.handle.setBindGroup(groupIndex, group, dynamicOffsets)

    def draw(
        self,
        vertexCount: int = 0,
        instanceCount: int = 0,
        firstVertex: int = 0,
        firstInstance: int = 0,
    ) -> None:
        return self.handle.draw(vertexCount, instanceCount, firstVertex, firstInstance)

    def drawIndexed(
        self,
        indexCount: int = 0,
        instanceCount: int = 0,
        firstIndex: int = 0,
        baseVertex: int = 0,
        firstInstance: int = 0,
    ) -> None:
        return self.handle.drawIndexed(
            indexCount, instanceCount, firstIndex, baseVertex, firstInstance
        )

    def drawIndirect(self, indirectBuffer: Buffer, indirectOffset: int = 0) -> None:
        return self.handle.drawIndirect(indirectBuffer.handle, indirectOffset)

    def drawIndexedIndirect(
        self, indirectBuffer: Buffer, indirectOffset: int = 0
    ) -> None:
        return self.handle.drawIndexedIndirect(indirectBuffer.handle, indirectOffset)

    def insertDebugMarker(self, markerLabel: str = "") -> None:
        return self.handle.insertDebugMarker(markerLabel)

    def popDebugGroup(self) -> None:
        return self.handle.popDebugGroup()

    def pushDebugGroup(self, groupLabel: str = "") -> None:
        return self.handle.pushDebugGroup(groupLabel)

    def setVertexBuffer(
        self,
        slot: int = 0,
        buffer: "Buffer | None" = None,
        offset: int = 0,
        size: int = 0,
    ) -> None:
        return self.handle.setVertexBuffer(
            slot, None if buffer is None else buffer.handle, offset, size
        )

    def setIndexBuffer(
        self,
        buffer: "Buffer | None" = None,
        format: "IndexFormat | None" = None,
        offset: int = 0,
        size: int = 0,
    ) -> None:
        return self.handle.setIndexBuffer(
            None if buffer is None else buffer.handle, format, offset, size
        )

    def finish(
        self,
        label: str = "",
    ) -> "RenderBundle":
        return self.handle.finish(RenderBundleDescriptor(label=label).toJS())


class RenderPassEncoder(BaseWebGPUHandle):
    def setPipeline(self, pipeline: "RenderPipeline | None" = None) -> None:
        return self.handle.setPipeline(pipeline)

    def setBindGroup(
        self,
        groupIndex: int = 0,
        group: "BindGroup | None" = None,
        dynamicOffsets: list[int] = [],
    ) -> None:
        return self.handle.setBindGroup(groupIndex, group, dynamicOffsets)

    def draw(
        self,
        vertexCount: int = 0,
        instanceCount: int = 0,
        firstVertex: int = 0,
        firstInstance: int = 0,
    ) -> None:
        return self.handle.draw(vertexCount, instanceCount, firstVertex, firstInstance)

    def drawIndexed(
        self,
        indexCount: int = 0,
        instanceCount: int = 0,
        firstIndex: int = 0,
        baseVertex: int = 0,
        firstInstance: int = 0,
    ) -> None:
        return self.handle.drawIndexed(
            indexCount, instanceCount, firstIndex, baseVertex, firstInstance
        )

    def drawIndirect(self, indirectBuffer: Buffer, indirectOffset: int = 0) -> None:
        return self.handle.drawIndirect(indirectBuffer.handle, indirectOffset)

    def drawIndexedIndirect(
        self, indirectBuffer: Buffer, indirectOffset: int = 0
    ) -> None:
        return self.handle.drawIndexedIndirect(indirectBuffer.handle, indirectOffset)

    def executeBundles(self, bundles: list["RenderBundle"] = []) -> None:
        return self.handle.executeBundles(bundles)

    def insertDebugMarker(self, markerLabel: str = "") -> None:
        return self.handle.insertDebugMarker(markerLabel)

    def popDebugGroup(self) -> None:
        return self.handle.popDebugGroup()

    def pushDebugGroup(self, groupLabel: str = "") -> None:
        return self.handle.pushDebugGroup(groupLabel)

    def setStencilReference(self, reference: int = 0) -> None:
        return self.handle.setStencilReference(reference)

    def setBlendConstant(self, color: "Color | None" = None) -> None:
        return self.handle.setBlendConstant(color)

    def setViewport(
        self,
        x: float = 0.0,
        y: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
        minDepth: float = 0.0,
        maxDepth: float = 0.0,
    ) -> None:
        return self.handle.setViewport(x, y, width, height, minDepth, maxDepth)

    def setScissorRect(
        self, x: int = 0, y: int = 0, width: int = 0, height: int = 0
    ) -> None:
        return self.handle.setScissorRect(x, y, width, height)

    def setVertexBuffer(
        self,
        slot: int = 0,
        buffer: "Buffer | None" = None,
        offset: int = 0,
        size: int = 0,
    ) -> None:
        return self.handle.setVertexBuffer(
            slot, None if buffer is None else buffer.handle, offset, size
        )

    def setIndexBuffer(
        self,
        buffer: "Buffer | None" = None,
        format: "IndexFormat | None" = None,
        offset: int = 0,
        size: int = 0,
    ) -> None:
        return self.handle.setIndexBuffer(
            None if buffer is None else buffer.handle, format, offset, size
        )

    def beginOcclusionQuery(self, queryIndex: int = 0) -> None:
        return self.handle.beginOcclusionQuery(queryIndex)

    def endOcclusionQuery(self) -> None:
        return self.handle.endOcclusionQuery()

    def end(self) -> None:
        return self.handle.end()


class RenderPipeline(BaseWebGPUHandle):
    def getBindGroupLayout(self, groupIndex: int = 0) -> "BindGroupLayout":
        return self.handle.getBindGroupLayout(groupIndex)


class ShaderModule(BaseWebGPUHandle):
    def getCompilationInfo(self) -> None:
        return self.handle.getCompilationInfo()


class Texture(BaseWebGPUHandle):
    def createView(
        self, descriptor: TextureViewDescriptor | None = None
    ) -> "TextureView":
        if descriptor is None:
            return TextureView(self.handle.createView())
        return TextureView(self.handle.createView(descriptor.toJS()))

    @property
    def width(self) -> int:
        return self.handle.width

    @property
    def height(self) -> int:
        return self.handle.height

    @property
    def depthOrArrayLayers(self) -> int:
        return self.handle.depthOrArrayLayers

    @property
    def mipLevelCount(self) -> int:
        return self.handle.mipLevelCount

    @property
    def sampleCount(self) -> int:
        return self.handle.sampleCount

    @property
    def dimension(self) -> str:
        return self.handle.dimension

    @property
    def format(self) -> "TextureFormat":
        return TextureFormat(self.handle.format())

    @property
    def usage(self) -> "TextureUsage":
        return TextureUsage(self.handle.usage())

    def destroy(self) -> None:
        print("destroy texture")
        return self.handle.destroy()

    def __del__(self):
        self.destroy()
