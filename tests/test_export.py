"""Tests for the export pipeline: capture, serialize, deserialize."""

import numpy as np


def _make_triangle(color=(0.0, 1.0, 0.0, 1.0), label="tri"):
    from webgpu.triangles import TriangulationRenderer

    points = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
        dtype=np.float32,
    )
    return TriangulationRenderer(points, color=color, label=label)


def test_export_basic(webgpu_env):
    """Export a single-renderer scene, verify blob structure after roundtrip."""
    from webgpu.export.serialize import deserialize_scene

    webgpu_env.ensure_canvas(400, 400)
    renderer = _make_triangle()
    scene = webgpu_env.wj.Draw([renderer], width=400, height=400)

    blob = scene.export()

    assert isinstance(blob, bytes)
    assert len(blob) > 0

    exported = deserialize_scene(blob)
    assert len(exported.buffers) > 0
    assert len(exported.render_passes) > 0
    assert exported.camera is not None
    assert "matrix" in exported.camera


def test_export_roundtrip(webgpu_env):
    """Export, deserialize, verify buffer data matches GPU readback."""
    from webgpu.export.capture import capture_scene
    from webgpu.export.serialize import serialize_scene, deserialize_scene

    webgpu_env.ensure_canvas(400, 400)
    renderer = _make_triangle()
    scene = webgpu_env.wj.Draw([renderer], width=400, height=400)

    captured = capture_scene(scene)
    blob = serialize_scene(captured)
    restored = deserialize_scene(blob)

    # Every buffer from capture should appear in restored with identical data
    for buf_id, orig_buf in captured.buffers.items():
        assert buf_id in restored.buffers, f"Missing buffer {buf_id}"
        assert restored.buffers[buf_id].data == orig_buf.data
        assert restored.buffers[buf_id].size == orig_buf.size

    # Render pass count must match
    assert len(restored.render_passes) == len(captured.render_passes)


def test_export_multiple_renderers(webgpu_env):
    """Scene with two renderers produces two render passes."""
    from webgpu.export.serialize import deserialize_scene

    webgpu_env.ensure_canvas(400, 400)
    r1 = _make_triangle(color=(1.0, 0.0, 0.0, 1.0), label="red")
    r2 = _make_triangle(color=(0.0, 0.0, 1.0, 1.0), label="blue")
    scene = webgpu_env.wj.Draw([r1, r2], width=400, height=400)

    blob = scene.export()
    exported = deserialize_scene(blob)

    assert len(exported.render_passes) >= 2
