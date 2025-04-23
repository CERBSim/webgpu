from pathlib import Path

js_code = (Path(__file__).parent / "link.js").read_text().replace("export ", "")
