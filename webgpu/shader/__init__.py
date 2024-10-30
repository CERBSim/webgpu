from pathlib import Path


def get_shader_code(names):
    code = ""
    for name in names:
        code += (Path(__file__).parent / name).read_text()
    return code
