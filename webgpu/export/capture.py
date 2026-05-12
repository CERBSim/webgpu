"""Capture an initialized Scene into ExportScene format."""

from .format import (
    ExportScene,
    ExportRenderPass,
    ExportComputePass,
    ExportInteraction,
    ExportBuffer,
    ExportTexture,
    BufferRegistry,
)


def capture_scene(scene) -> ExportScene:
    """Extract export data from a fully initialized Scene."""
    from ..renderer import Renderer, MultipleRenderer

    registry = BufferRegistry()
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

    _finalize_registry(registry)

    return ExportScene(
        buffers=registry.buffers,
        textures=registry.textures,
        compute_passes=compute_passes,
        render_passes=render_passes,
        interactions=_detect_interactions(scene, registry),
        camera=_export_camera(options.camera, options, registry),
        light=_export_light(options.light, registry),
    )


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


def _export_light(light, registry) -> dict:
    """Export light uniform data."""
    light_buf_id = None
    if light.uniforms is not None:
        if hasattr(light.uniforms, '_buffer') and light.uniforms._buffer is not None:
            key = id(light.uniforms._buffer)
            if key in registry._buffers:
                light_buf_id = registry._buffers[key][0]
        return {"data": bytes(light.uniforms), "buffer_id": light_buf_id}
    return {}


def _finalize_registry(registry):
    """Read GPU data for all registered buffers and textures."""
    from ..utils import read_buffer, read_texture

    for buf_id, proxy, usage in registry._buffers.values():
        data = read_buffer(proxy)
        registry.buffers[buf_id] = ExportBuffer(
            id=buf_id, data=bytes(data), usage=usage, size=proxy.size,
        )

    # Default sampler config (covers colormaps and most textures)
    default_sampler = {"magFilter": "linear", "minFilter": "linear"}

    for tex_id, proxy in list(registry._textures.values()):
        tex = proxy
        width = getattr(tex, 'width', 0)
        height = getattr(tex, 'height', 0)
        fmt = getattr(tex, 'format', 'rgba8unorm')
        if width > 0 and height > 0:
            tex_data = read_texture(tex)
            data = bytes(tex_data.tobytes())
        else:
            data = b''
        registry.textures[tex_id] = ExportTexture(
            id=tex_id, data=data, width=width, height=height,
            format=fmt, sampler=default_sampler,
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
            clip = obj.gpu_objects.clipping
        if isinstance(clip, Clipping) and getattr(clip, 'uniforms', None) is not None:
            buf = getattr(clip.uniforms, '_buffer', None)
            if buf is not None:
                key = id(buf)
                if key in registry._buffers:
                    buf_id = registry._buffers[key][0]
                    if buf_id not in seen_buf_ids:
                        seen_buf_ids.add(buf_id)
                        interactions.append(ExportInteraction(
                            type="clipping_plane", buffer_id=buf_id, config={
                                "normal": list(clip.normal),
                                "center": list(clip.center),
                                "radius": float(clip.radius),
                                "offset": float(clip.offset),
                                "mode": int(clip.mode),
                            },
                        ))

        # Check colormap
        cmap = None
        if hasattr(obj, 'gpu_objects'):
            cmap = obj.gpu_objects.colormap
        if isinstance(cmap, Colormap) and getattr(cmap, 'uniforms', None) is not None:
            buf = getattr(cmap.uniforms, '_buffer', None)
            if buf is not None:
                key = id(buf)
                if key in registry._buffers:
                    buf_id = registry._buffers[key][0]
                    if buf_id not in seen_buf_ids:
                        seen_buf_ids.add(buf_id)
                        interactions.append(ExportInteraction(
                            type="colormap_range", buffer_id=buf_id,
                            config={"min": float(cmap.minval), "max": float(cmap.maxval)},
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

    return interactions
