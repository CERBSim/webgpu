"""WebGPU infrastructure and connection tests."""

from pathlib import Path

TESTS_DIR = Path(__file__).parent


class TestInfrastructure:
    """Pure-JS checks — proves Docker + Chromium + WebGPU work."""

    def _goto_file(self, page, filename):
        path = TESTS_DIR / filename
        page.goto(f"file://{path}")

    def test_webgpu_available(self, page):
        self._goto_file(page, "gpu_test.html")
        page.wait_for_timeout(2000)
        text = page.inner_text("#r")
        assert "navigator.gpu: object" in text, f"No navigator.gpu: {text}"

    def test_webgpu_adapter(self, page):
        self._goto_file(page, "gpu_test.html")
        page.wait_for_timeout(2000)
        text = page.inner_text("#r")
        assert "adapter: ok" in text, f"No WebGPU adapter: {text}"

    def test_webgpu_clear_red(self, page):
        """Clear canvas to red via JS-only WebGPU, verify pixel."""
        self._goto_file(page, "gpu_clear_red.html")
        page.wait_for_timeout(2000)
        text = page.inner_text("#r")
        assert text.startswith("OK"), f"Expected OK, got: {text}"


class TestConnection:
    """Python <-> browser websocket bridge."""

    def test_platform_connected(self, webgpu_env):
        assert webgpu_env.platform.js is not None

    def test_device_exists(self, webgpu_env):
        from webgpu.utils import get_device

        assert get_device() is not None