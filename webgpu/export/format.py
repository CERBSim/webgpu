from dataclasses import dataclass, field


@dataclass
class ExportBuffer:
    id: str
    data: bytes
    usage: str  # "storage" | "uniform" | "storage-write" | "indirect"
    size: int


@dataclass
class ExportTexture:
    id: str
    data: bytes  # raw pixel data
    width: int
    height: int
    format: str  # "rgba8unorm" etc
    sampler: dict  # {magFilter, minFilter, ...}


@dataclass
class ExportComputePass:
    id: str
    shader: str
    bindings: dict  # binding_number → buffer_id
    workgroups: int | list
    triggers: list  # buffer_ids or pass_ids
    reset_buffers: list  # buffer_ids to zero before dispatch
    # After dispatch: build drawIndirect args from atomic counter
    # { counter_id: str, indirect_id: str, vertex_count: int }
    indirect_setup: dict | None = None
    # Generic count-then-fill: run pass twice, reading the atomic counter
    # in between to resize the output buffer to the exact needed size.
    # {
    #   "counter_id": str,         # buffer holding the atomic<u32> counter
    #   "output_id": str,          # buffer to resize based on counter value
    #   "element_size": int,       # bytes per element (e.g. 64 for SubTrig)
    #   "indirect_id": str | None, # optional: indirect draw buffer to update
    #   "vertex_count": int | None,# vertices per instance for indirect draw
    # }
    count_then_fill: dict | None = None


@dataclass
class ExportRenderPass:
    id: str
    shader: str
    bindings: dict  # binding_number → buffer_id
    vertex_count: int
    instance_count: int
    draw_indirect: str | None = None
    topology: str = "triangle-list"
    depth_write: bool = True
    depth_bias: int = 0
    pass_type: str = "opaque"  # "opaque" | "transparent"
    vertex_entry_point: str = "vertex_main"
    fragment_entry_point: str = "fragment_main"
    vertex_buffers: list = field(default_factory=list)
    index_buffer_id: str | None = None
    index_format: str = "uint32"


@dataclass
class Interaction:
    type: str  # "gui" | "time_animation" | ...
    buffer_id: str
    config: dict = field(default_factory=dict)


# Deprecated alias for backward compatibility.
ExportInteraction = Interaction


@dataclass
class ExportScene:
    buffers: dict = field(default_factory=dict)  # id → ExportBuffer
    textures: dict = field(default_factory=dict)  # id → ExportTexture
    compute_passes: list = field(default_factory=list)
    render_passes: list = field(default_factory=list)
    interactions: list = field(default_factory=list)
    camera: dict = field(default_factory=dict)
    light: dict = field(default_factory=dict)
    theme: dict = field(default_factory=dict)


class BufferRegistry:
    def __init__(self, live: bool = False):
        self._buffers = {}  # id(proxy) → (buf_id, proxy, usage)
        self._textures = {}  # id(proxy) → (tex_id, proxy)
        self._samplers = {}  # id(proxy) → (sampler_id, proxy)
        self._counter = 0
        self._data = {}  # buf_id → bytes
        # When True, callers (renderers) should skip destroy/recreate tricks
        # that only make sense for the export blob (e.g. shrinking buffers so
        # the blob is small). The JS engine resizes them at runtime instead.
        self.live = live

        self.buffers = {}  # buf_id → ExportBuffer
        self.textures = {}  # tex_id → ExportTexture

    def _next_id(self, prefix="buf"):
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def register_bindings(self, bindings) -> dict:
        """Walk binding list, deduplicate, return {binding_nr: id_str}."""
        result = {}
        for b in bindings:
            result[b.nr] = self._register_binding(b)
        return result

    def _register_binding(self, binding):
        from ..utils import (
            BufferBinding,
            UniformBinding,
            TextureBinding,
            SamplerBinding,
            StorageTextureBinding,
        )

        if isinstance(binding, (BufferBinding, UniformBinding)):
            buf = binding._resource["buffer"]
            key = id(buf)
            if key not in self._buffers:
                usage = "uniform" if isinstance(binding, UniformBinding) else "storage"
                buf_id = self._next_id(usage)
                self._buffers[key] = (buf_id, buf, usage)
            return self._buffers[key][0]

        if isinstance(binding, TextureBinding):
            tex = binding.texture
            key = id(tex)
            if key not in self._textures:
                tex_id = self._next_id("tex")
                self._textures[key] = (tex_id, tex)
            return self._textures[key][0]

        if isinstance(binding, SamplerBinding):
            sampler = binding._resource
            key = id(sampler)
            if key not in self._samplers:
                sampler_id = self._next_id("sampler")
                self._samplers[key] = (sampler_id, sampler)
            return self._samplers[key][0]

        if isinstance(binding, StorageTextureBinding):
            tex_view = binding._resource
            key = id(tex_view)
            if key not in self._textures:
                tex_id = self._next_id("stex")
                self._textures[key] = (tex_id, tex_view)
            return self._textures[key][0]

        raise ValueError(f"Unknown binding type: {type(binding)}")

    def add_raw_buffer(self, prefix: str, data: bytes, usage: str = "frame") -> str:
        """Register a CPU-only blob (no GPU proxy) and return its id.

        Used by interactions that need to ship arbitrary binary data
        (e.g. animation frame snapshots) alongside the scene. Buffers
        registered this way bypass GPU allocation in the JS engine.
        """
        buf_id = self._next_id(prefix)
        self.buffers[buf_id] = ExportBuffer(
            id=buf_id, data=bytes(data), usage=usage, size=len(data),
        )
        return buf_id

    def get_id(self, buffer_proxy):
        """Get buffer_id for a known buffer proxy."""
        key = id(buffer_proxy)
        if key not in self._buffers:
            raise KeyError(f"Buffer not registered: {buffer_proxy}")
        return self._buffers[key][0]

    def set_buffer_data(self, buffer_id, data: bytes):
        """Set CPU-side data for a buffer (called by capture logic)."""
        self._data[buffer_id] = data

    def get_all_buffer_ids(self):
        """Return {buf_id: proxy} for all registered buffers."""
        return {buf_id: proxy for buf_id, proxy, _usage in self._buffers.values()}

    def get_all_buffer_info(self):
        """Return list of (buf_id, proxy, usage) tuples."""
        return list(self._buffers.values())
