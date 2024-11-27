from pathlib import Path
import glob


def get_shader_code(files=glob.glob(str(Path(__file__).parent / "*.wgsl"))):
    code = ""
    for file in files:
        file = Path(file)
        if not file.is_absolute():
            file = Path(__file__).parent / file
        code += file.read_text()
    return code
