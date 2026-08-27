"""Persistent screenshot worker for exported scene blobs.

Launches one Chrome instance on a virtual display (Xvfb) and handles multiple
screenshot requests over stdin/stdout.

Protocol (line-based):
    Request:  <width> <height> <color_scheme> <blob_b64>\n
              where <color_scheme> is "light" or "dark".

    Response: <png_b64>\n
    Shutdown: (close stdin)

Requires: Xvfb, playwright, chrome
"""

import sys
import os
import base64
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler


# Start well above any display a real session would use, so a collision with
# the user's desktop is not even a possibility.
_FIRST_DISPLAY = 90
_LAST_DISPLAY = 130


def _display_is_free(disp):
    return not os.path.exists(f"/tmp/.X{disp}-lock") and not os.path.exists(
        f"/tmp/.X11-unix/X{disp}"
    )


def _start_xvfb():
    """Start Xvfb on an explicitly picked display and return (process, display).

    Deliberately not ``-displayfd``: that makes Xvfb scan for a free display
    starting at :0, and probing a display that is already taken can unlink the
    socket of the running server there, killing the user's real session.
    """
    for disp in range(_FIRST_DISPLAY, _LAST_DISPLAY):
        if not _display_is_free(disp):
            continue
        proc = subprocess.Popen(
            ['Xvfb', f':{disp}', '-screen', '0', '1280x1024x24', '-ac'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        socket_path = f"/tmp/.X11-unix/X{disp}"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break  # died (lost a race for this display) — try the next one
            if os.path.exists(socket_path):
                return proc, disp
            time.sleep(0.05)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    raise RuntimeError(
        f"could not start Xvfb on any display in "
        f":{_FIRST_DISPLAY}..:{_LAST_DISPLAY - 1}"
    )


def main():
    xvfb_proc, disp_num = _start_xvfb()
    os.environ['DISPLAY'] = f':{disp_num}'
    os.environ.pop('WAYLAND_DISPLAY', None)

    tmpdir = Path(tempfile.mkdtemp(prefix="webgpu_ss_"))
    try:
        _run_worker(tmpdir)
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        xvfb_proc.terminate()
        xvfb_proc.wait()


def _run_worker(tmpdir):
    from playwright.sync_api import sync_playwright

    ARGS = [
        "--no-sandbox",
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,UnsafeWebGPU",
        "--use-vulkan=native",
        "--ignore-gpu-blocklist",
        "--disable-dev-shm-usage",
        "--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist",
        "--ozone-platform=x11",
    ]

    # Load engine JS once
    engine_js_path = Path(__file__).parent.parent / "engine"
    js_files = ["format.js", "compute.js", "camera.js", "input.js", "interactions.js", "engine.js"]
    engine_js = "\n".join(
        (engine_js_path / f).read_text().replace("export ", "") for f in js_files
    )
    engine_js += "\nif (typeof window !== 'undefined') { window.RenderEngine = RenderEngine; }\n"

    # Start HTTP server for serving pages to Chrome
    class Quiet(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmpdir), **kw)
        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Quiet)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # Launch browser once (not headless — uses the Xvfb virtual display)
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        channel="chrome", headless=False,
        args=ARGS,
    )
    page = browser.new_page()

    # Signal ready
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # Process requests from stdin
    request_id = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 3)
        if len(parts) != 4:
            sys.stdout.write("\n")
            sys.stdout.flush()
            continue

        width, height = int(parts[0]), int(parts[1])
        color_scheme, blob_b64 = parts[2], parts[3]
        if color_scheme not in ("light", "dark"):
            color_scheme = "light"

        html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;overflow:hidden;">
<canvas id="c" width="{width}" height="{height}" style="display:block;"></canvas>
<script>
{engine_js}
(async () => {{
    try {{
        const engine = await RenderEngine.create('c', `{blob_b64}`);
        document.title = 'READY';
    }} catch(e) {{
        document.title = 'ERROR:' + (e.stack || e.message || e);
    }}
}})();
</script></body></html>"""

        (tmpdir / "index.html").write_text(html)
        page.emulate_media(color_scheme=color_scheme)
        request_id += 1
        page.goto(f"http://127.0.0.1:{port}/index.html?rid={request_id}")

        try:
            page.wait_for_function(
                "document.title === 'READY' || document.title.startsWith('ERROR:')",
                timeout=30000,
            )
            title = page.title()
            if title.startswith('ERROR:'):
                print(f"[screenshot] JS error: {title}", file=sys.stderr)
                sys.stdout.write("\n")
                sys.stdout.flush()
                continue

            # Wait for rAF compositing then screenshot the canvas element
            page.wait_for_timeout(100)
            el = page.query_selector('#c')
            png_bytes = el.screenshot() if el else b''
            sys.stdout.write(base64.b64encode(png_bytes).decode() + "\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"[screenshot] error: {e}", file=sys.stderr)
            sys.stdout.write("\n")
            sys.stdout.flush()

    browser.close()
    pw.stop()
    server.shutdown()


if __name__ == "__main__":
    main()
