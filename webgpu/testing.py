"""Reusable pytest infrastructure for WebGPU visual regression tests.

Downstream packages use this by adding to their conftest.py::

    pytest_plugins = ["webgpu.testing"]

Then configure output/baseline dirs on the webgpu_env fixture::

    @pytest.fixture(scope="session", autouse=True)
    def _configure_dirs(webgpu_env):
        webgpu_env.output_dir = Path(__file__).parent / "output"
        webgpu_env.baseline_dir = Path(__file__).parent / "baselines"

Provided fixtures:
    _playwright   – session-scoped Playwright instance
    browser       – session-scoped headless Chromium with WebGPU
    page          – per-test fresh browser page
    webgpu_env    – session-scoped full WebGPU environment (WS bridge + browser)
"""

import base64
import importlib
import os
import shutil
import tempfile
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

CHROMIUM_WEBGPU_ARGS = [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan,UnsafeWebGPU",
    "--use-vulkan=native",
    "--ignore-gpu-blocklist",
    "--disable-dev-shm-usage",
    "--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist",
]

UPDATE_BASELINES = os.environ.get("UPDATE_BASELINES", "") == "1"

_READBACK_JS = """
window._doReadback = async (device, texture, width, height) => {
    const bytesPerPixel = 4;
    const bytesPerRow = Math.ceil(width * bytesPerPixel / 256) * 256;
    const size = bytesPerRow * height;

    const buffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'readback',
    });

    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow },
        [width, height, 1],
    );
    device.queue.submit([encoder.finish()]);

    await buffer.mapAsync(GPUMapMode.READ, 0, size);
    const mapped = new Uint8Array(buffer.getMappedRange(0, size));

    let binary = '';
    const chunkSize = 8192;
    for (let i = 0; i < mapped.length; i += chunkSize) {
        const chunk = mapped.subarray(i, Math.min(i + chunkSize, mapped.length));
        binary += String.fromCharCode.apply(null, chunk);
    }

    buffer.unmap();
    buffer.destroy();

    return btoa(binary);
};
"""

_test_scene_counter = [0]


def _find_link_js():
    """Locate link.js from the installed webgpu package."""
    spec = importlib.util.find_spec("webgpu")
    if spec is None:
        raise RuntimeError("webgpu package not installed")
    if spec.submodule_search_locations:
        pkg_dir = Path(spec.submodule_search_locations[0])
    elif spec.origin:
        pkg_dir = Path(spec.origin).parent
    else:
        raise RuntimeError("Cannot locate webgpu package directory")
    link_js = pkg_dir / "link" / "link.js"
    if not link_js.exists():
        raise FileNotFoundError(f"link.js not found at {link_js}")
    return link_js


def _make_test_html(ws_port, link_js_content):
    """Generate HTML with canvas and websocket connection to Python."""
    inline_js = link_js_content.replace("export ", "")
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>webgpu test</title></head>
<body style="margin:0;background:#222;">
<div id="__webgpu_container"></div>
<script>
{inline_js}
WebsocketLink('ws://127.0.0.1:{ws_port}');
{_READBACK_JS}
// Disable patchedRequestAnimationFrame: its rAF+queue.submit breaks
// subsequent bridge mapAsync calls in headless Chrome. Tests read
// target_texture directly via _doReadback, so canvas display is unnecessary.
window.patchedRequestAnimationFrame = (device, context, target) => {{}};
</script>
</body>
</html>"""


def _patched_init_html(scene, width, height, flex=None):
    from webgpu.renderer import Renderer
    from webgpu.scene import Scene as WScene

    if isinstance(scene, Renderer):
        scene = [scene]
    if isinstance(scene, list):
        scene = WScene(scene)
    id_ = f"__webgpu_{_test_scene_counter[0]}_"
    _test_scene_counter[0] += 1
    return scene, id_


class _QuietHTTPHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# WebGPUTestEnv – helpers available via the webgpu_env fixture
# ---------------------------------------------------------------------------

class WebGPUTestEnv:
    """Helpers for tests that use the webgpu connection.

    Set ``output_dir`` and ``baseline_dir`` before calling methods that
    need them (``screenshot``, ``assert_matches_baseline``).
    """

    def __init__(self, page, wj, platform):
        self.page = page
        self.wj = wj
        self.platform = platform
        self.output_dir = None
        self.baseline_dir = None

    def ensure_canvas(self, width=600, height=600):
        """Inject a <canvas> matching the next wj.Draw() counter.  Returns canvas_id."""
        counter = _test_scene_counter[0]
        cid = f"__webgpu_{counter}_canvas"
        gid = f"__webgpu_{counter}_lilgui"
        self.page.evaluate(f"""(() => {{
            const root = document.createElement('div');
            root.id = '__webgpu_{counter}_root';
            root.style.cssText = 'display:flex;';
            root.innerHTML = `
                <canvas id="{cid}" width="{width}" height="{height}"
                    style="background:#d0d0d0;width:{width}px;height:{height}px;">
                </canvas>
                <div id="{gid}" style="flex:1;"></div>`;
            document.getElementById('__webgpu_container').appendChild(root);
        }})()""")
        return cid

    def screenshot(self, name, canvas_id=None):
        """Screenshot a canvas element.  Returns Path to the saved PNG."""
        assert self.output_dir, "output_dir not configured on WebGPUTestEnv"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        selector = f"#{canvas_id}" if canvas_id else "canvas"
        el = self.page.query_selector(selector)
        assert el, f"No element found for selector: {selector}"
        path = self.output_dir / f"{name}.png"
        el.screenshot(path=str(path))
        return path

    def assert_matches_baseline(self, output_path, baseline_name, *, threshold=0.01):
        """Compare an output image against its baseline.

        If ``UPDATE_BASELINES=1`` env-var is set, copies output -> baseline instead.
        *threshold* is the max fraction of pixels allowed to differ.
        """
        import numpy as np
        from PIL import Image

        assert self.baseline_dir, "baseline_dir not configured on WebGPUTestEnv"
        baseline_path = self.baseline_dir / baseline_name

        if UPDATE_BASELINES:
            self.baseline_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, baseline_path)
            print(f"  Updated baseline: {baseline_name}")
            return

        if not baseline_path.exists():
            pytest.skip(
                f"No baseline {baseline_name} — run with UPDATE_BASELINES=1 to create"
            )

        out = np.array(Image.open(output_path))
        ref = np.array(Image.open(baseline_path))

        if out.shape != ref.shape:
            pytest.fail(f"Shape mismatch: output {out.shape} vs baseline {ref.shape}")

        diff = np.abs(out.astype(int) - ref.astype(int))
        bad_pixels = (diff.max(axis=-1) > 2).sum()  # per-channel tolerance of 2
        total = out.shape[0] * out.shape[1]
        ratio = bad_pixels / total

        if ratio > threshold:
            assert self.output_dir, "output_dir not configured on WebGPUTestEnv"
            diff_path = self.output_dir / f"diff_{baseline_name}"
            diff_img = np.clip(diff * 10, 0, 255).astype(np.uint8)
            Image.fromarray(diff_img).save(str(diff_path))
            pytest.fail(
                f"{ratio:.1%} pixels differ (threshold {threshold:.1%}). "
                f"Diff saved to {diff_path}"
            )

    def readback_texture(self, scene, path):
        """Read back the scene's rendered texture via JS-side buffer readback.

        Uses ``window._doReadback()`` which performs the entire readback in a
        single bridge call, avoiding multi-message interleaving issues.
        """
        import numpy as np
        from PIL import Image
        from webgpu import platform

        texture = scene.canvas.target_texture
        w, h = texture.width, texture.height
        fmt = texture.format
        bytes_per_row = (w * 4 + 255) // 256 * 256

        b64_data = platform.js._doReadback(
            scene.device.handle,
            scene.canvas.target_texture,
            w, h,
        )

        raw = base64.b64decode(b64_data)
        data = np.frombuffer(raw, dtype=np.uint8).reshape(
            (h, bytes_per_row // 4, 4)
        )
        data = data[:, :w, :]

        if fmt == "bgra8unorm":
            data = data[:, :, [2, 1, 0, 3]]

        img = Image.fromarray(data[:, :, :3])
        img.save(str(path))
        return path


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _playwright():
    pw = sync_playwright().start()
    yield pw
    pw.stop()


@pytest.fixture(scope="session")
def browser(_playwright):
    b = _playwright.chromium.launch(
        channel="chrome",
        headless=False,
        args=["--headless=new"] + CHROMIUM_WEBGPU_ARGS,
    )
    yield b
    b.close()


@pytest.fixture
def page(browser):
    """Fresh browser page — no webgpu connection, for pure-JS tests."""
    p = browser.new_page()
    yield p
    p.close()


@pytest.fixture(scope="session")
def webgpu_env(browser):
    """Full webgpu environment: platform + device + browser page with canvas.

    Threading choreography::

        main thread                         background thread
        -----------                         -----------------
                                            platform.init() starts WS server
                                            on_server_ready() -> signals port
        receives port, writes HTML
        page.goto(test page)
          -> browser JS connects to WS ->  platform.init() unblocks
        init_device_sync()
        yield env
    """
    import webgpu.platform as platform

    # Prevent webgpu.jupyter auto-init on import
    _orig_init = platform.init
    platform.init = lambda *a, **kw: None
    import webgpu.jupyter as wj

    platform.init = _orig_init

    # Patch _init_html — we create canvas elements ourselves
    wj._init_html = _patched_init_html

    # HTTP server for the test page
    link_js_content = _find_link_js().read_text()
    tmpdir = Path(tempfile.mkdtemp(prefix="webgpu_test_"))
    handler = partial(_QuietHTTPHandler, directory=str(tmpdir))
    http_server = HTTPServer(("127.0.0.1", 0), handler)
    http_port = http_server.server_address[1]
    threading.Thread(target=http_server.serve_forever, daemon=True).start()

    # Run platform.init in background (blocks until browser connects)
    port_ready = threading.Event()
    init_done = threading.Event()
    ws_port = [None]
    init_error = [None]

    def on_server_ready(server):
        ws_port[0] = server.port
        port_ready.set()

    def run_init():
        try:
            platform.init(before_wait_for_connection=on_server_ready)
        except Exception as e:
            init_error[0] = e
        init_done.set()

    threading.Thread(target=run_init, daemon=True).start()
    assert port_ready.wait(timeout=10), "WS server did not start"

    # Write HTML now that we know the WS port
    (tmpdir / "index.html").write_text(
        _make_test_html(ws_port[0], link_js_content)
    )

    # Navigate browser -> triggers WS connection -> unblocks platform.init
    test_page = browser.new_page()
    test_page.goto(f"http://127.0.0.1:{http_port}/index.html")
    assert init_done.wait(timeout=30), "Platform init timed out (WS handshake)"
    if init_error[0]:
        raise init_error[0]

    # Initialize WebGPU device over the bridge
    from webgpu.utils import init_device_sync

    init_device_sync()

    yield WebGPUTestEnv(page=test_page, wj=wj, platform=platform)

    test_page.close()
    http_server.shutdown()
    shutil.rmtree(tmpdir, ignore_errors=True)