import os
from pathlib import Path

_engine_dir = Path(__file__).parent
_js_files = ["format.js", "compute.js", "camera.js", "input.js", "interactions.js", "engine.js"]


def _power_preference():
    pref = os.environ.get("WEBGPU_POWER_PREFERENCE", "high-performance")
    return pref if pref in ("high-performance", "low-power") else "high-performance"


# Remove export statements for inlining
def _load_js():
    parts = []
    pref = _power_preference()
    parts.append(
        "if (typeof window !== 'undefined') { window.__webgpuPowerPreference = "
        f"'{pref}'; }}"
    )
    parts.append("if (typeof window === 'undefined' || !window.RenderEngine) {")
    for f in _js_files:
        p = _engine_dir / f
        parts.append(f"// --- {f} ---")
        parts.append(p.read_text().replace("export ", ""))
    parts.append(
        "if (typeof window !== 'undefined') { window.RenderEngine = RenderEngine; }"
    )
    parts.append("}")
    return "\n".join(parts)

engine_js = _load_js()
