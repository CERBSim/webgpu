"""Tests for the webgpu Python framework.

All tests use the webgpu_env fixture which provides a live WebGPU device
connected to headless Chrome via the websocket bridge.
"""

import numpy as np


class TestConnection:
    """Python <-> browser websocket bridge."""

    def test_platform_connected(self, webgpu_env):
        assert webgpu_env.platform.js is not None

    def test_device_exists(self, webgpu_env):
        from webgpu.utils import get_device

        dev = get_device()
        assert dev is not None

    def test_device_has_limits(self, webgpu_env):
        from webgpu.utils import get_device

        dev = get_device()
        assert dev.limits.maxBufferSize > 0
        assert dev.limits.maxStorageBufferBindingSize > 0


class TestBuffers:
    """GPU buffer creation and data transfer."""

    def test_create_buffer(self, webgpu_env):
        from webgpu.utils import create_buffer

        buf = create_buffer(1024, label="test_buf")
        assert buf is not None
        assert buf.size >= 1024

    def test_buffer_from_array(self, webgpu_env):
        from webgpu.utils import buffer_from_array
        from webgpu.webgpu_api import BufferUsage

        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf = buffer_from_array(
            data,
            usage=BufferUsage.STORAGE | BufferUsage.COPY_DST,
            label="test_array_buf",
        )
        assert buf is not None
        assert buf.size >= data.nbytes

    def test_buffer_reuse(self, webgpu_env):
        from webgpu.utils import create_buffer

        buf1 = create_buffer(1024, label="reuse_test")
        buf2 = create_buffer(512, label="reuse_test", reuse=buf1)
        # Should reuse since buf1.size >= 512
        assert buf2.size == buf1.size


class TestShaders:
    """Shader file loading."""

    def test_read_shader_file(self, webgpu_env):
        from webgpu.utils import read_shader_file

        src = read_shader_file("camera.wgsl")
        assert len(src) > 0

    def test_shader_dir_registered(self, webgpu_env):
        from webgpu.utils import _shader_directories

        assert "" in _shader_directories


class TestRendering:
    """Scene and renderer pipeline using TriangulationRenderer."""

    def _make_triangle_renderer(self):
        from webgpu.triangles import TriangulationRenderer

        # A single triangle in the XY plane
        points = np.array(
            [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float32,
        )
        return TriangulationRenderer(
            points, color=(0.0, 1.0, 0.0, 1.0), label="test_tri"
        )

    def test_draw_triangulation(self, webgpu_env):
        webgpu_env.ensure_canvas(400, 400)

        renderer = self._make_triangle_renderer()
        scene = webgpu_env.wj.Draw([renderer], width=400, height=400)

        assert scene is not None
        assert scene.bounding_box is not None

    def test_draw_triangulation_readback(self, webgpu_env):
        webgpu_env.ensure_canvas(400, 400)

        renderer = self._make_triangle_renderer()
        scene = webgpu_env.wj.Draw([renderer], width=400, height=400)
        webgpu_env.page.wait_for_timeout(500)

        webgpu_env.output_dir.mkdir(parents=True, exist_ok=True)
        path = webgpu_env.output_dir / "triangle.png"
        webgpu_env.readback_texture(scene, path)

        assert path.exists()
        webgpu_env.assert_matches_baseline(path, "triangle.png")