import weakref as _weakref
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
    entry_point: str = "main"  # compute shader entry point
    # After dispatch: build drawIndirect args from atomic counter
    # { counter_id: str, indirect_id: str, vertex_count: int }
    indirect_setup: dict | None = None
    # Generic count-then-fill: run pass twice, reading the atomic counter
    # in between to resize the output buffer to the exact needed size.
    # {
    #   "counter_id": str,         # buffer holding the atomic<u32> counter
    #   "output_id": str,          # primary buffer to resize (drives the cap)
    #   "element_size": int,       # bytes per element (e.g. 64 for SubTrig)
    #   "indirect_id": str | None, # optional: indirect draw buffer to update
    #   "vertex_count": int | None,# vertices/indices per instance for indirect draw
    #   "indexed": bool,           # indirect buffer is the 5-u32 drawIndexedIndirect
    #                              #   layout (indexCount, instanceCount, firstIndex,
    #                              #   baseVertex, firstInstance) instead of 4-u32
    #   "siblings": [              # extra output buffers resized in lockstep with the
    #     {"id": str, "element_size": int}, ...  #   primary from the same counter, so
    #   ]                          #   the shader's arrayLength(&primary) gate bounds
    #                              #   writes into all of them (e.g. vector
    #                              #   positions/directions/values arrays)
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
    select_shader: str | None = None
    select_entry_point: str | None = None


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


class StableResourceIds:
    """Hands out string ids for GPU resources that stay stable across the live
    scene's repeated captures.

    The BufferRegistry is rebuilt on every capture and assigns ids in the order
    resources are encountered, so a buffer's id would change whenever the set or
    order of render objects changes. The live RenderEngine caches ids (e.g. the
    camera buffer) and owns buffers keyed by id (countThenFill outputs), so a
    shifting id corrupts those mappings — camera writes hit the wrong buffer,
    owned compute buffers diverge from their render bindings, etc. Keying by
    id(resource) with a weakref guards against address reuse after GC.
    """
    def __init__(self):
        self._ids = {}   # id(obj) -> (ref, id_str)
        self._counter = 0

    def get(self, obj, prefix):
        key = id(obj)
        ent = self._ids.get(key)
        if ent is not None and ent[0]() is obj:
            return ent[1]
        self._counter += 1
        sid = f"{prefix}_{self._counter}"
        try:
            ref = _weakref.ref(obj)
        except TypeError:
            ref = (lambda o=obj: o)  # not weak-referenceable; keep strong
        self._ids[key] = (ref, sid)
        return sid


class BufferRegistry:
    def __init__(self, live: bool = False, id_allocator: "StableResourceIds | None" = None):
        self._buffers = {}  # id(proxy) → (buf_id, proxy, usage)
        self._textures = {}  # id(proxy) → (tex_id, proxy)
        self._samplers = {}  # id(proxy) → (sampler_id, proxy)
        self._counter = 0
        self._data = {}  # buf_id → bytes
        # When True, callers (renderers) should skip destroy/recreate tricks
        # that only make sense for the export blob (e.g. shrinking buffers so
        # the blob is small). The JS engine resizes them at runtime instead.
        self.live = live
        # Persistent across captures (live mode) so resource ids stay stable.
        self._id_allocator = id_allocator

        self.buffers = {}  # buf_id → ExportBuffer
        self.textures = {}  # tex_id → ExportTexture

    def _next_id(self, prefix="buf"):
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _id_for(self, resource, prefix):
        if self._id_allocator is not None:
            return self._id_allocator.get(resource, prefix)
        return self._next_id(prefix)

    def register_buffer(self, buf, usage="storage"):
        """Register a buffer not reached via register_bindings (e.g. an indirect
        draw buffer) and return its id (stable across captures in live mode)."""
        key = id(buf)
        if key not in self._buffers:
            self._buffers[key] = (self._id_for(buf, usage), buf, usage)
        return self._buffers[key][0]

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
                buf_id = self._id_for(buf, usage)
                self._buffers[key] = (buf_id, buf, usage)
            return self._buffers[key][0]

        if isinstance(binding, TextureBinding):
            tex = binding.texture
            key = id(tex)
            if key not in self._textures:
                tex_id = self._id_for(tex, "tex")
                self._textures[key] = (tex_id, tex)
            return self._textures[key][0]

        if isinstance(binding, SamplerBinding):
            sampler = binding._resource
            key = id(sampler)
            if key not in self._samplers:
                sampler_id = self._id_for(sampler, "sampler")
                self._samplers[key] = (sampler_id, sampler)
            return self._samplers[key][0]

        if isinstance(binding, StorageTextureBinding):
            tex_view = binding._resource
            key = id(tex_view)
            if key not in self._textures:
                tex_id = self._id_for(tex_view, "stex")
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
