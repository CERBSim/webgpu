import re
from pathlib import Path

_link_js_path = Path(__file__).parent / "link.js"
if not _link_js_path.exists():
    raise FileNotFoundError(
        f"webgpu link.js not found at {_link_js_path}. "
        "The webgpu package may be installed incorrectly."
    )

js_code = re.sub(r"^export ", "", _link_js_path.read_text(), flags=re.MULTILINE)
