import base64
import itertools
import os
import time

from . import platform, utils
from .canvas import Canvas
from .lilgui import LilGUI
from .link import js_code as _link_js_code
from .renderer import *
from .scene import Scene
from .triangles import *
from .utils import init_device_sync
from .webgpu_api import *


_id_counter = itertools.count()


def _init_html(scene, width, height, flex=None):
    from IPython.display import HTML, display

    if isinstance(scene, Renderer):
        scene = [scene]
    if isinstance(scene, list):
        scene = Scene(scene)

    id_ = f"__webgpu_{next(_id_counter)}_"

    style = f"background-color: white; width: {width}px; height: {height}px;"
    if flex is not None:
        style += f" flex: {flex};"

    display(
        HTML(
            f"""
            <div id='{id_}root'
            style="position: relative; width: {width}px; max-width: 100%; overflow: hidden;"
            >
                <canvas 
                    id='{id_}canvas'
                    style='{style} max-width: 100%; display: block;'
                >
                </canvas>
                <div id='{id_}lilgui'
                    style='position: absolute; top: 0; right: 0; z-index: 10;'
                ></div>
            </div>
            """
        )
    )

    return scene, id_


def _draw_scene(scene: Scene, width, height, id_):
    html_canvas = platform.js.document.getElementById(f"{id_}canvas")

    while html_canvas is None:
        html_canvas = platform.js.document.getElementById(f"{id_}canvas")
    gui_element = platform.js.document.getElementById(f"{id_}lilgui")

    # Lazily initialize the WebGPU device the first time we draw.
    canvas = Canvas(init_device_sync(), html_canvas)
    scene.gui = LilGUI(gui_element, scene)
    scene.init(canvas)
    scene.render()


def _DrawPyodide(b64_data: str):
    import pickle

    data = base64.b64decode(b64_data.encode("utf-8"))
    id_, scene, width, height = pickle.loads(data)

    _draw_scene(scene, width, height, id_)
    return scene


_engine_emitted = False
_export_keep_alive = None  # holds Playwright objects to prevent GC


class _NullGUI:
    """Swallow all GUI calls during export.

    User notebooks routinely call ``add_options_to_gui(scene.gui)`` to wire
    sliders/checkboxes. In export mode the GUI is rebuilt client-side from
    ExportInteraction entries, so any Python-side calls during export are
    irrelevant — they just need to not crash.
    """

    def _noop(self, *a, **kw):
        return self

    def __getattr__(self, name):
        return self._noop

    def __call__(self, *a, **kw):
        return self


def _DrawHTML(scene, width=640, height=600):
    global _engine_emitted
    from IPython.display import Javascript, display

    from .engine import engine_js

    scene, id_ = _init_html(scene, width, height)

    # Create a canvas in the headless Chrome DOM for GPU initialization.
    # (_init_html only emits HTML to notebook output, not to Chrome's DOM.)
    canvas_id = f"{id_}canvas"
    js = platform.js
    html_canvas = js.document.createElement("canvas")
    html_canvas.width = width
    html_canvas.height = height
    html_canvas.id = canvas_id
    js.document.body.appendChild(html_canvas)

    canvas = Canvas(init_device_sync(), html_canvas)
    scene.init(canvas)
    # Disable render() — patchedRequestAnimationFrame hangs in headless Chrome.
    # scene.init() already filled all GPU buffers via update().
    scene.render = lambda *a, **kw: None
    scene._on_camera_changed = lambda *a, **kw: None

    # Provide a no-op GUI stub so user code that calls add_options_to_gui()
    # doesn't crash during export. The real GUI is rebuilt client-side by
    # the JS Interactions class from ExportInteraction entries.
    scene.gui = _NullGUI()

    # Export scene to blob (scene.init already filled all GPU buffers)
    blob = scene.export()
    blob_b64 = base64.b64encode(blob).decode()

    if not _engine_emitted:
        display(Javascript(engine_js))
        _engine_emitted = True

    display(Javascript(f'RenderEngine.create("{canvas_id}", "{blob_b64}");'))
    return scene


def _init_export_gpu():
    """Start headless Chrome with WebGPU for buffer export during nbconvert.

    Both Playwright and platform.init run in background threads to avoid
    conflicts with the asyncio event loop used by nbconvert.
    """
    import threading
    import tempfile
    from pathlib import Path
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    from functools import partial

    CHROMIUM_WEBGPU_ARGS = [
        "--no-sandbox",
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,UnsafeWebGPU",
        "--use-vulkan=native",
        "--ignore-gpu-blocklist",
        "--disable-dev-shm-usage",
        "--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist",
    ]

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "playwright is required for WEBGPU_EXPORTING. "
            "Install with: pip install playwright && playwright install chrome"
        )

    port_ready = threading.Event()
    port_container = [None]
    errors = []
    server_done = threading.Event()

    def _start_server():
        try:
            def _on_server(server):
                port_container[0] = server.port
                port_ready.set()
            platform.init(
                before_wait_for_connection=_on_server,
                block_on_connection=True,
            )
            server_done.set()
        except Exception as e:
            errors.append(e)
            port_ready.set()
            server_done.set()

    def _start_browser():
        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(
                channel="chrome",
                headless=False,
                args=["--headless=new"] + CHROMIUM_WEBGPU_ARGS,
            )
            page = browser.new_page()
            port_ready.wait(timeout=30)
            if port_container[0] is None:
                errors.append(RuntimeError("Server failed to start"))
                return

            ws_port = port_container[0]

            # Serve HTML via a separate HTTP server (WebGPU needs secure context)
            tmpdir = Path(tempfile.mkdtemp(prefix="webgpu_export_"))
            html = (
                "<html><body><script>\n"
                # Disable patchedRequestAnimationFrame — it blocks mapAsync in headless
                + "window.patchedRequestAnimationFrame = () => {};\n"
                + _link_js_code + "\n"
                + f"WebsocketLink('ws://127.0.0.1:{ws_port}');\n"
                + "</script></body></html>"
            )
            (tmpdir / "index.html").write_text(html)

            class _Quiet(SimpleHTTPRequestHandler):
                def __init__(self, *a, **kw):
                    super().__init__(*a, directory=str(tmpdir), **kw)
                def log_message(self, *a):
                    pass

            http = HTTPServer(("127.0.0.1", 0), _Quiet)
            http_port = http.server_address[1]
            threading.Thread(target=http.serve_forever, daemon=True).start()

            page.goto(f"http://127.0.0.1:{http_port}/index.html")
            global _export_keep_alive
            _export_keep_alive = (pw, browser, page, http)
        except Exception as e:
            errors.append(e)

    t_server = threading.Thread(target=_start_server, daemon=True)
    t_browser = threading.Thread(target=_start_browser, daemon=True)
    t_server.start()
    t_browser.start()

    # Wait for browser to connect and server to finish init
    t_browser.join(timeout=30)
    if errors:
        raise RuntimeError(f"WEBGPU_EXPORTING init failed: {errors[0]}")

    server_done.wait(timeout=30)
    if errors:
        raise RuntimeError(f"WEBGPU_EXPORTING init failed: {errors[0]}")

    init_device_sync()


def Draw(
    scene: Scene | list[Renderer] | Renderer,
    width: int | None = None,
    height: int | None = None,
):
    flex = 3 if width is None else None

    width = width if width is not None else 640
    height = height if height is not None else 600

    scene, id_ = _init_html(scene, width, height, flex)

    _draw_scene(scene, width, height, id_)
    return scene


if not platform.is_pyodide:
    try:
        from IPython.display import Javascript, display
    except ImportError:
        pass  # Not in a notebook (e.g. test framework) — skip auto-init.

if not platform.is_pyodide and "Javascript" in dir():
    is_exporting = "WEBGPU_EXPORTING" in os.environ

    if is_exporting:
        # Launch headless Chrome for GPU access during export
        _init_export_gpu()
        Draw = _DrawHTML
    else:
        # Not exporting and not running in pyodide -> Start a websocket server
        # and wait for the client to connect.

        def _webgpu_js(server):
            js = _link_js_code + """
const __is_vscode = (typeof location !== 'undefined' && location.protocol === 'vscode-webview:');
const __webgpu_host = __is_vscode ? '127.0.0.1' : ((typeof location !== 'undefined' && location.hostname) || '127.0.0.1');
WebsocketLink('ws://' + __webgpu_host + ':{port}');
""".format(port=server.port)
            display(Javascript(js))

        is_vscode = "VSCODE_PID" in os.environ
        platform.init(
            before_wait_for_connection=_webgpu_js,
            block_on_connection=True,
        )
