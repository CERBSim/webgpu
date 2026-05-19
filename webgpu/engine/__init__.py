from pathlib import Path

_engine_dir = Path(__file__).parent
_js_files = ["format.js", "compute.js", "camera.js", "input.js", "interactions.js", "engine.js"]

# Remove export statements for inlining
def _load_js():
    parts = []
    for f in _js_files:
        p = _engine_dir / f
        parts.append(f"// --- {f} ---")
        parts.append(p.read_text().replace("export ", ""))
    # Top-level `class`/`function` declarations in a <script> are scoped to
    # the script, not the global object. Explicitly publish RenderEngine so
    # the websocket bridge / Pyodide can resolve `platform.js.RenderEngine`.
    parts.append(
        "if (typeof window !== 'undefined') { window.RenderEngine = RenderEngine; }"
    )
    return "\n".join(parts)

engine_js = _load_js()
