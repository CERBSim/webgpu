"""Capture an initialized Scene into ExportScene format."""

from .format import (
    ExportScene,
    ExportRenderPass,
    ExportComputePass,
    Interaction,
    ExportBuffer,
    ExportTexture,
    BufferRegistry,
)


def capture_scene(scene, live: bool = False) -> ExportScene:
    """Extract export data from a fully initialized Scene.

    If ``live=True``, skip GPU readback: ``ExportBuffer.data`` and
    ``ExportTexture.data`` are left empty. The registry still holds proxies
    which the caller can extract via :func:`build_live_resource_maps` to
    hand to ``RenderEngine.createLive``.
    """
    export, _registry = _capture(scene, live=live)
    return export


def capture_scene_live(scene):
    """Capture a Scene in live mode and return (ExportScene, BufferRegistry)."""
    return _capture(scene, live=True)


def _capture(scene, live: bool):
    from ..renderer import Renderer, MultipleRenderer

    registry = BufferRegistry(live=live)
    render_passes = []
    compute_passes = []

    options = scene.options

    for obj in scene.render_objects:
        if not obj.active:
            continue
        _capture_renderer(obj, options, registry, render_passes, compute_passes)

    # Sort: opaque first, transparent second
    opaque = [p for p in render_passes if p.pass_type == "opaque"]
    transparent = [p for p in render_passes if p.pass_type == "transparent"]
    render_passes = opaque + transparent

    _finalize_registry(registry, live=live)

    export = ExportScene(
        buffers=registry.buffers,
        textures=registry.textures,
        compute_passes=compute_passes,
        render_passes=render_passes,
        interactions=_detect_interactions(scene, registry),
        camera=_export_camera(options.camera, options, registry),
        light=_export_light(options.light, registry, live=live),
    )
    return export, registry


def build_live_resource_maps(registry: BufferRegistry):
    """Return (buffers, textures, samplers, frame_buffers) maps for live mode.

    ``buffers``/``textures``/``samplers`` map id -> JS proxy handle.
    ``frame_buffers`` maps id -> bytes for CPU-only blobs (animation frames).
    """
    from ..webgpu_api import BaseWebGPUHandle

    def _unwrap(obj):
        return obj.handle if isinstance(obj, BaseWebGPUHandle) else obj

    buffers = {bid: _unwrap(p) for bid, p, _u in registry._buffers.values()}
    textures = {tid: _unwrap(p) for tid, p in registry._textures.values()}
    samplers = {sid: _unwrap(p) for sid, p in registry._samplers.values()}

    # Collect CPU-only frame buffers (e.g. animation snapshots)
    frame_buffers = {}
    for buf_id, eb in registry.buffers.items():
        if eb.usage == "frame":
            frame_buffers[buf_id] = eb.data
    return buffers, textures, samplers, frame_buffers


def _capture_renderer(obj, options, registry, render_passes, compute_passes):
    """Recursively capture a renderer (handles MultipleRenderer)."""
    from ..renderer import Renderer, MultipleRenderer

    if isinstance(obj, MultipleRenderer):
        for child in obj.render_objects:
            if child.active:
                _capture_renderer(child, options, registry, render_passes, compute_passes)
    elif isinstance(obj, Renderer):
        if obj.n_vertices > 0 and obj.n_instances > 0:
            # Compute passes first (may set up indirect buffers needed by descriptor)
            compute_passes.extend(obj.get_export_compute_passes(options, registry))
            render_passes.append(obj.get_export_descriptor(options, registry))
        # Also capture sub-renderers stored in gpu_objects (e.g. Colorbar's Labels)
        if hasattr(obj, 'gpu_objects'):
            for attr in obj.gpu_objects:
                if isinstance(attr, Renderer) and attr is not obj:
                    if attr.n_vertices > 0 and attr.n_instances > 0:
                        render_passes.append(attr.get_export_descriptor(options, registry))
                        compute_passes.extend(attr.get_export_compute_passes(options, registry))


def _export_camera(camera, options, registry) -> dict:
    """Export camera state and identify the camera buffer."""
    t = camera.transform
    # The camera uniform buffer is in options._camera_uniforms
    camera_buf_id = None
    if hasattr(options, '_camera_uniforms') and options._camera_uniforms is not None:
        cu = options._camera_uniforms
        if hasattr(cu, '_buffer') and cu._buffer is not None:
            key = id(cu._buffer)
            if key in registry._buffers:
                camera_buf_id = registry._buffers[key][0]
    return {
        "matrix": t._mat.flatten().tolist(),
        "center": t._center.tolist() if hasattr(t._center, 'tolist') else list(t._center),
        "buffer_id": camera_buf_id,
    }


def _export_light(light, registry, live: bool = False) -> dict:
    """Export light uniform data."""
    light_buf_id = None
    if light.uniforms is not None:
        if hasattr(light.uniforms, '_buffer') and light.uniforms._buffer is not None:
            key = id(light.uniforms._buffer)
            if key in registry._buffers:
                light_buf_id = registry._buffers[key][0]
        if live:
            # Host (Python) already writes the light buffer; don't ship the bytes.
            return {"buffer_id": light_buf_id}
        return {"data": bytes(light.uniforms), "buffer_id": light_buf_id}
    return {}


def _finalize_registry(registry, live: bool = False):
    """Read GPU data for all registered buffers and textures.

    In ``live`` mode skip the GPU→CPU readback: only sizes and metadata are
    needed; the JS engine binds to the live proxies directly.
    """
    from ..utils import read_buffer, read_texture

    for buf_id, proxy, usage in registry._buffers.values():
        if live:
            data = b''
        else:
            data = bytes(read_buffer(proxy))
        registry.buffers[buf_id] = ExportBuffer(
            id=buf_id, data=data, usage=usage, size=proxy.size,
        )

    # Default sampler config (covers colormaps and most textures)
    default_sampler = {"magFilter": "linear", "minFilter": "linear"}

    for tex_id, proxy in list(registry._textures.values()):
        tex = proxy
        width = getattr(tex, 'width', 0)
        height = getattr(tex, 'height', 0)
        fmt = getattr(tex, 'format', 'rgba8unorm')
        if live or width == 0 or height == 0:
            data = b''
        else:
            tex_data = read_texture(tex)
            data = bytes(tex_data.tobytes())
        registry.textures[tex_id] = ExportTexture(
            id=tex_id, data=data, width=width, height=height,
            format=fmt, sampler=default_sampler,
        )


def _clipping_gui_interaction(buf_id, clip):
    """Emit a generic gui interaction for clipping plane controls."""
    from .format import Interaction
    cx, cy, cz = [float(v) for v in clip.center]
    radius = float(clip.radius)
    mode = int(clip.mode)
    # When clipping is disabled by default, use PLANE mode (1) when enabled via GUI
    enabled_mode = mode if mode != 0 else 1
    # The expr packs the full 48-byte ClippingUniforms struct:
    #   plane (vec4<f32>): nx, ny, nz, d
    #   sphere (vec4<f32>): cx, cy, cz, radius
    #   mode (u32) + padding (u32 * 3)
    pack_expr = (
        f"(() => {{"
        f" let Nx=nx,Ny=ny,Nz=nz;"
        f" const l=Math.sqrt(Nx*Nx+Ny*Ny+Nz*Nz);"
        f" if(l>1e-12){{Nx/=l;Ny/=l;Nz/=l;}}else{{Nx=0;Ny=0;Nz=-1;}}"
        f" const cx={cx}+Nx*offset,cy={cy}+Ny*offset,cz={cz}+Nz*offset;"
        f" const d=-(cx*Nx+cy*Ny+cz*Nz);"
        f" const ab=new ArrayBuffer(48);"
        f" const f=new Float32Array(ab),u=new Uint32Array(ab);"
        f" f[0]=Nx;f[1]=Ny;f[2]=Nz;f[3]=d;"
        f" f[4]=cx;f[5]=cy;f[6]=cz;f[7]={radius};"
        f" u[8]=enabled?{enabled_mode}:0;"
        f" return ab;}})()"
    )
    return Interaction(
        type="gui", buffer_id=buf_id, config={
            "label": "Clipping",
            "vars": {
                "nx": float(clip.normal[0]),
                "ny": float(clip.normal[1]),
                "nz": float(clip.normal[2]),
                "offset": float(clip.offset),
                "enabled": mode != 0,
            },
            "controls": [
                {"kind": "checkbox", "var": "enabled", "name": "Enabled"},
                {"kind": "slider", "var": "nx", "name": "Normal X", "min": -1, "max": 1, "step": 0.01},
                {"kind": "slider", "var": "ny", "name": "Normal Y", "min": -1, "max": 1, "step": 0.01},
                {"kind": "slider", "var": "nz", "name": "Normal Z", "min": -1, "max": 1, "step": 0.01},
                {"kind": "slider", "var": "offset", "name": "Offset", "min": -2, "max": 2, "step": 0.01},
            ],
            "writes": [
                {
                    "targets": [{"buffer_id": buf_id, "offset": 0}],
                    "expr": pack_expr,
                    "trigger": "*",
                },
            ],
        },
    )


def _colormap_gui_interaction(buf_id, cmap):
    """Emit a generic gui interaction for colormap range controls."""
    from .format import Interaction

    lo = float(cmap.minval)
    hi = float(cmap.maxval)

    # Derive sensible slider range and step from the actual data range.
    span = hi - lo if hi != lo else 1.0
    # Slider extends 2x beyond the data range in each direction.
    slider_min = lo - span
    slider_max = hi + span
    step = span / 100.0

    return Interaction(
        type="gui", buffer_id=buf_id, config={
            "label": "Colormap",
            "vars": {
                "min": lo,
                "max": hi,
            },
            "controls": [
                {"kind": "slider", "var": "min", "name": "Min",
                 "min": slider_min, "max": slider_max, "step": step},
                {"kind": "slider", "var": "max", "name": "Max",
                 "min": slider_min, "max": slider_max, "step": step},
            ],
            "writes": [
                {
                    "targets": [{"buffer_id": buf_id, "offset": 0, "dtype": "f32"}],
                    "expr": "min",
                    "trigger": "min",
                },
                {
                    "targets": [{"buffer_id": buf_id, "offset": 4, "dtype": "f32"}],
                    "expr": "max",
                    "trigger": "max",
                },
            ],
        },
    )


def _detect_interactions(scene, registry) -> list:
    """Detect interactive elements (clipping, colormap) in the scene."""
    from ..clipping import Clipping
    from ..colormap import Colormap
    from ..renderer import Renderer, MultipleRenderer

    interactions = []
    seen_buf_ids = set()

    def _walk(obj):
        if isinstance(obj, MultipleRenderer):
            for child in obj.render_objects:
                _walk(child)
            return
        if not isinstance(obj, Renderer):
            return

        # Check clipping
        clip = getattr(obj, '_clipping', None) or getattr(obj, 'clipping', None)
        if clip is None and hasattr(obj, 'gpu_objects'):
            clip = getattr(obj.gpu_objects, 'clipping', None)
        if isinstance(clip, Clipping) and getattr(clip, 'uniforms', None) is not None:
            buf = getattr(clip.uniforms, '_buffer', None)
            if buf is not None:
                key = id(buf)
                if key in registry._buffers:
                    buf_id = registry._buffers[key][0]
                    if buf_id not in seen_buf_ids:
                        seen_buf_ids.add(buf_id)
                        interactions.append(_clipping_gui_interaction(
                            buf_id, clip,
                        ))

        # Check colormap
        cmap = None
        if hasattr(obj, 'gpu_objects'):
            cmap = getattr(obj.gpu_objects, 'colormap', None)
        if isinstance(cmap, Colormap) and getattr(cmap, 'uniforms', None) is not None:
            buf = getattr(cmap.uniforms, '_buffer', None)
            if buf is not None:
                key = id(buf)
                if key in registry._buffers:
                    buf_id = registry._buffers[key][0]
                    if buf_id not in seen_buf_ids:
                        seen_buf_ids.add(buf_id)
                        interactions.append(_colormap_gui_interaction(
                            buf_id, cmap,
                        ))

    for obj in scene.render_objects:
        _walk(obj)

    # Renderer-level export hook: let renderers (e.g. Animation) emit their
    # own interactions after all GPU buffers have been registered.
    def _collect_from_renderers(obj):
        from ..renderer import Renderer, MultipleRenderer
        if isinstance(obj, MultipleRenderer):
            interactions.extend(obj.get_export_interactions(scene.options, registry))
            for child in obj.render_objects:
                _collect_from_renderers(child)
        elif isinstance(obj, Renderer):
            interactions.extend(obj.get_export_interactions(scene.options, registry))

    for obj in scene.render_objects:
        _collect_from_renderers(obj)

    # Collect GuiParam instances from renderers
    from ..gui_param import GuiParam
    seen_params = set()  # track by id
    param_interactions = []

    def _collect_params(obj):
        from ..renderer import Renderer, MultipleRenderer
        if isinstance(obj, MultipleRenderer):
            for child in obj.render_objects:
                _collect_params(child)
            return
        if not isinstance(obj, Renderer):
            return
        for param in getattr(obj, '_gui_params', []):
            if id(param) not in seen_params:
                seen_params.add(id(param))
                inter = param.export(registry)
                if inter is not None:
                    param_interactions.append(inter)

    for obj in scene.render_objects:
        _collect_params(obj)

    interactions = param_interactions + interactions

    # Deduplicate / merge interactions.  For ``gui`` interactions with the
    # same label, merge write targets so that a single control writes to all
    # relevant buffers.  For other types, drop exact duplicates.
    import json
    seen = set()
    gui_by_label = {}  # label -> Interaction (first occurrence)
    deduped = []
    for inter in interactions:
        if inter.type == "gui":
            label = inter.config.get("label")
            if label in gui_by_label:
                # Merge writes: match by (trigger, expr), extend targets
                first = gui_by_label[label]
                existing = {(w.get("trigger"), w.get("expr")): w for w in first.config["writes"]}
                for w in inter.config.get("writes", []):
                    key = (w.get("trigger"), w.get("expr"))
                    if key in existing:
                        seen_targets = {
                            (t.get("buffer_id"), t.get("offset"), t.get("dtype"))
                            for t in existing[key]["targets"]
                        }
                        for t in w["targets"]:
                            tk = (t.get("buffer_id"), t.get("offset"), t.get("dtype"))
                            if tk not in seen_targets:
                                existing[key]["targets"].append(t)
                                seen_targets.add(tk)
                    else:
                        first.config["writes"].append(w)
                        existing[key] = w
            else:
                gui_by_label[label] = inter
                deduped.append(inter)
        else:
            key = (inter.type, json.dumps(inter.config, sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(inter)
    return deduped
