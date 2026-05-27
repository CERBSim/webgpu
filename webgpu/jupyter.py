import base64
import itertools
import os
import time

from . import platform, utils
from .canvas import Canvas
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

    style = (
        f"background-color: var(--webgpu-canvas-bg, #ffffff); width: min({width}px, 100%); max-width: 100%; "
        f"height: auto; aspect-ratio: {width} / {height}; touch-action: none;"
    )
    if flex is not None:
        style += f" flex: {flex};"

    display(
        HTML(
            f"""
            <style>
              #{id_}root {{ --webgpu-canvas-bg: #ffffff; }}
              @media (prefers-color-scheme: dark) {{
                #{id_}root {{ --webgpu-canvas-bg: #adadad; }}
              }}
            </style>
            <div id='{id_}root'
            style="position: relative; width: min({width}px, 100%); max-width: 100%; overflow: hidden;"
            >
                <canvas
                    id='{id_}canvas'
                    style='{style} display: block;'
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
    global _engine_emitted
    html_canvas = platform.js.document.getElementById(f"{id_}canvas")

    while html_canvas is None:
        html_canvas = platform.js.document.getElementById(f"{id_}canvas")
    # Lazily inject the JS render engine into the browser scope so that
    # Scene.init() can call RenderEngine.createLive(...).
    if not _engine_emitted:
        try:
            from IPython.display import Javascript, display
            from .engine import engine_js
            display(Javascript(engine_js))
            _engine_emitted = True
        except Exception as e:
            print(f'warning: could not inject engine_js: {e}')

    # Lazily initialize the WebGPU device the first time we draw.
    canvas = Canvas(init_device_sync(), html_canvas)
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

    # Export scene to blob (scene.init already filled all GPU buffers)
    blob = scene.export()
    blob_b64 = base64.b64encode(blob).decode()

    if not _engine_emitted:
        display(Javascript(engine_js))
        _engine_emitted = True

    display(Javascript(f'RenderEngine.create("{canvas_id}", "{blob_b64}");'))
    return scene


def _DrawHTMLLazy(scene, width=640, height=600):
    """Export scene as a lazy-loading HTML snippet with a pre-baked screenshot.

    The screenshot is rendered in a separate subprocess (own Chrome + GPU device)
    so it doesn't interfere with the main build process's mapAsync calls.
    The scene blob and engine JS are saved as separate static files and only
    fetched when the user clicks to interact.
    """
    global _engine_emitted
    from IPython.display import HTML, Javascript, display

    from .engine import engine_js

    if isinstance(scene, Renderer):
        scene = [scene]
    if isinstance(scene, list):
        scene = Scene(scene)

    id_ = f"__webgpu_{next(_id_counter)}_"
    canvas_id = f"{id_}canvas"

    # Create a canvas in the headless Chrome DOM for GPU initialization.
    js = platform.js
    html_canvas = js.document.createElement("canvas")
    html_canvas.width = width
    html_canvas.height = height
    html_canvas.id = canvas_id
    js.document.body.appendChild(html_canvas)

    canvas = Canvas(init_device_sync(), html_canvas)
    scene.init(canvas)

    # Disable render() — patchedRequestAnimationFrame hangs in headless Chrome.
    scene.render = lambda *a, **kw: None
    scene._on_camera_changed = lambda *a, **kw: None

    # Export scene to blob (scene.init already filled all GPU buffers)
    blob = scene.export()
    blob_b64 = base64.b64encode(blob).decode()

    # Save blob and engine JS as static files for deferred loading.
    # Use a unique hash to avoid collisions across notebooks.
    import hashlib
    blob_hash = hashlib.md5(blob).hexdigest()[:10]
    static_dir = _get_static_dir()

    # Compute relative prefix from the notebook page to _static/.
    # static_dir is <docs_root>/_static/webgpu_scenes, so its grandparent
    # is the docs source root.  The kernel cwd is the notebook's directory.
    # We need the relative path from cwd to that docs root.
    from pathlib import Path
    docs_root = static_dir.parent.parent  # _static/../
    rel_prefix = os.path.relpath(docs_root, Path.cwd()) + "/"
    if rel_prefix == "./":
        rel_prefix = ""

    # Save blob as a JS file (script src works on file:// unlike fetch)
    scene_filename = f"scene_{blob_hash}.js"
    scene_var = f"__webgpu_blob_{blob_hash}"
    (static_dir / scene_filename).write_text(
        f"window.{scene_var} = \"{blob_b64}\";"
    )
    scene_url = f"{rel_prefix}_static/webgpu_scenes/{scene_filename}"

    if not _engine_emitted:
        (static_dir / "engine.js").write_text(engine_js)
        _engine_emitted = True
    engine_url = f"{rel_prefix}_static/webgpu_scenes/engine.js"

    # Capture screenshots (light + dark) in a separate subprocess.
    screenshot_light_b64 = _capture_screenshot_subprocess(blob_b64, width, height, color_scheme="light")
    screenshot_dark_b64 = _capture_screenshot_subprocess(blob_b64, width, height, color_scheme="dark")

    screenshot_light_filename = f"screenshot_{blob_hash}_light.png"
    screenshot_dark_filename = f"screenshot_{blob_hash}_dark.png"

    if screenshot_light_b64:
        (static_dir / screenshot_light_filename).write_bytes(base64.b64decode(screenshot_light_b64))
    if screenshot_dark_b64:
        (static_dir / screenshot_dark_filename).write_bytes(base64.b64decode(screenshot_dark_b64))

    screenshot_light_url = f"{rel_prefix}_static/webgpu_scenes/{screenshot_light_filename}"
    # Fallback to light if dark capture failed
    screenshot_dark_url = (
        f"{rel_prefix}_static/webgpu_scenes/{screenshot_dark_filename}"
        if screenshot_dark_b64 else screenshot_light_url
    )

    # Emit the lazy-load HTML: only screenshot + overlay, everything else loaded on click
    lazy_html = f"""
    <style>
      #{id_}root {{ --webgpu-canvas-bg: #ffffff; }}
      #{id_}img_light {{ display: block; }}
      #{id_}img_dark {{ display: none; }}
      @media (prefers-color-scheme: dark) {{
        #{id_}root {{ --webgpu-canvas-bg: #adadad; }}
        #{id_}img_light {{ display: none; }}
        #{id_}img_dark {{ display: block; }}
      }}
    </style>
    <div id='{id_}root'
         style="position: relative; width: min({width}px, 100%); max-width: 100%; overflow: hidden;"
    >
        <img id='{id_}img_light'
             src='{screenshot_light_url}'
             style='width: 100%; max-width: 100%; height: auto; aspect-ratio: {width} / {height}; display: block;'
        />
        <img id='{id_}img_dark'
             src='{screenshot_dark_url}'
             style='width: 100%; max-width: 100%; height: auto; aspect-ratio: {width} / {height}; display: none;'
        />
        <div id='{id_}overlay'
             style='position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                    display: flex; align-items: center; justify-content: center;
                    background: rgba(0,0,0,0); cursor: pointer; transition: background 0.2s;'
             onmouseover="this.style.background='rgba(0,0,0,0.18)'; this.querySelector('span').style.opacity='1'"
             onmouseout="this.style.background='rgba(0,0,0,0)'; this.querySelector('span').style.opacity='0'"
             onclick="(function() {{ var r = document.getElementById('{id_}root'); if (r.__activated) return; r.__activated = true; function activate() {{ var il = document.getElementById('{id_}img_light'); if (il) il.style.display = 'none'; var id = document.getElementById('{id_}img_dark'); if (id) id.style.display = 'none'; document.getElementById('{id_}overlay').style.display = 'none'; document.getElementById('{canvas_id}').style.display = 'block'; RenderEngine.create('{canvas_id}', window.{scene_var}); }} function loadBlob() {{ var s = document.createElement('script'); s.src = '{scene_url}'; s.onload = activate; document.head.appendChild(s); }} if (typeof RenderEngine === 'undefined') {{ var s = document.createElement('script'); s.src = '{engine_url}'; s.onload = loadBlob; document.head.appendChild(s); }} else {{ loadBlob(); }} }})()"
        >
            <span style='color: white; font-size: 1.3em; font-weight: bold;
                         text-shadow: 0 1px 4px rgba(0,0,0,0.7); pointer-events: none;
                         opacity: 0; transition: opacity 0.2s;'
            >&#9654; Click to interact</span>
        </div>
        <canvas
            id='{canvas_id}'
            style='background-color: var(--webgpu-canvas-bg, #ffffff); width: 100%; max-width: 100%; height: auto;
                   aspect-ratio: {width} / {height}; touch-action: none; display: none;'
        ></canvas>
        <div id='{id_}lilgui'
             style='position: absolute; top: 0; right: 0; z-index: 10;'
        ></div>
    </div>
    """

    display(HTML(lazy_html))
    return scene


def _get_static_dir():
    """Get or create the static directory for webgpu scene assets."""
    from pathlib import Path
    # Walk up from cwd to find the _static directory that Sphinx will copy
    # into the build output.  The kernel cwd may be a subdirectory of the
    # docs source tree (e.g. docs/notebooks/) so a simple relative check
    # is not sufficient.
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        candidate = parent / "_static" / "webgpu_scenes"
        if (parent / "_static").is_dir():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        # Also check docs/_static in case we're at the repo root
        docs_candidate = parent / "docs" / "_static" / "webgpu_scenes"
        if (parent / "docs" / "_static").is_dir():
            docs_candidate.mkdir(parents=True, exist_ok=True)
            return docs_candidate
    # Fallback: create in current directory
    fallback = Path("_static/webgpu_scenes")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# Persistent screenshot worker process (launched once, handles all screenshots)
_screenshot_worker = None


def _capture_screenshot_subprocess(blob_b64, width, height, color_scheme):
    """Send a screenshot request to the persistent worker process."""
    import subprocess
    import sys

    global _screenshot_worker
    if _screenshot_worker is None or _screenshot_worker.poll() is not None:
        _screenshot_worker = subprocess.Popen(
            [sys.executable, "-m", "webgpu.export.screenshot"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for READY signal
        ready = _screenshot_worker.stdout.readline().strip()
        if ready != "READY":
            print(f"Warning: screenshot worker failed to start: {ready}")
            _screenshot_worker = None
            return ""

    try:
        _screenshot_worker.stdin.write(f"{width} {height} {color_scheme} {blob_b64}\n")
        _screenshot_worker.stdin.flush()
        result = _screenshot_worker.stdout.readline().strip()
        return result
    except Exception as e:
        print(f"Warning: screenshot capture failed: {e}")
        return ""


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
        is_lazy_load = "WEBGPU_LAZY_LOAD" in os.environ
        if is_lazy_load:
            Draw = _DrawHTMLLazy
        else:
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
