import json
import tempfile
from typing import Any, Dict

import requests
import yaml

# with tempfile.NamedTemporaryFile('w', delete=False) as f:
#     data = requests.get('https://raw.githubusercontent.com/webgpu-native/webgpu-headers/refs/heads/main/webgpu.yml').text
#     f.write(data)
#     f.flush()
#     f.close()
#     with open(f.name, 'r') as f:
#         data = yaml.safe_load(f)
#
# json.dump(data, open('webgpu.json', 'w'), indent=4)
# json.dump(data, open('webgpu.min.json', 'w'))

data = json.load(open("webgpu.json", "r"))


def fix_enum_name(name):
    if name[0].isdigit():
        name = f"_{name}"
    return name


def generate_enum(enum: Dict[str, Any]) -> str:
    """Generate Python enum class."""
    enum_name = parse_type(enum["name"])
    doc = enum["doc"]
    entries = enum.get("entries", [])

    lines = [f"class {enum_name}(str,Enum):"]
    # lines.append(f'    """{doc}"""')

    for entry in entries:
        if entry is None:
            continue
        entry_name = fix_enum_name(entry["name"])
        # entry_doc = entry['doc']
        entry_value = entry.get("value", entry_name)
        value_line = (
            f'    {entry_name} = "{entry_value}"'
            if entry_value
            else f"    {entry_name}"
        )
        # lines.append(f"{value_line}  # {entry_doc}")
        lines.append(f"{value_line}")

    _enums.add(enum_name)
    return "\n".join(lines)


_bitflags = set()
_enums = set()


def generate_bitflags(bitflag: Dict[str, Any]) -> str:
    """Generate Python class for bit flags."""
    bitflag_name = parse_type(bitflag["name"])
    _bitflags.add(bitflag_name)
    doc = bitflag["doc"]
    entries = bitflag.get("entries", [])

    lines = [f"class {bitflag_name}(IntFlag):"]
    # lines.append(f'    """{doc}"""')

    for i, entry in enumerate(entries):
        entry_name = fix_enum_name(entry["name"]).upper()
        # entry_doc = entry['doc']
        # entry_value = entry.get('value', None)
        entry_value = entry.get("value", 1 << i)
        value_line = (
            f"    {entry_name} = {entry_value}" if entry_value else f"    {entry_name}"
        )
        lines.append(f"{value_line}")

    return "\n".join(lines)


def to_camel_case(snake_str):
    return "".join(x.capitalize() if i else x for i,x in enumerate(snake_str.lower().split("_")))

def to_pascal_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


_types = {
    "str": "str",
    "int": "int",
    "string_with_default_empty": "str",
    "out_string": "str",
    "c_void": "int",
    "usize": "int",
    "uint32": "int",
    "int32": "int",
    "float64": "float",
    "float32": "float",
    "float": "float",
    "uint64": "int",
    "int64": "int",
    "nullable_string": "str",
}

_defaults = {
    "float": 0.0,
    "int": 0,
    "str": '""',
}


def parse_type(name):

    if name in _types:
        return _types[name]

    if name.startswith("callback."):
        return "Callable | None"

    name = name.replace("array", "list")
    name = name.replace("<", '["')
    name = name.replace(">", '"]')
    name = name.replace("struct.", "")
    name = name.replace("object.", "")
    name = name.replace("bitflag.", "")
    name = name.replace("enum.", "")

    if "[" in name:
        tokens = name.split("[")
        name = tokens[0] + "[" + parse_type(tokens[1])
    elif not name in ["int", "str", "bool", "float"]:
        name = to_pascal_case(name)

    return name


def parse_type_and_default(typ, factory=True):
    name = typ["type"]
    default = typ.get("default", None)

    if isinstance(default, str) and default.endswith("_not_used"):
        default = None

    if default in ["undefined", "none"]:
        default = None

    if type(default) == str:
        default = f'"{default}"'

    if name in _types:
        name = _types[name]
        return _types[name], _defaults[name]

    if name.startswith("callback."):
        return "Callable | None", None

    if name.startswith("uint") or name.startswith("int") or name == "usize":
        name = "int"
        if type(default) == str or default is None:
            name = "int | None"
            default = None
        return name, default

    # if name.startswith("struct.") and not default:
    #     default = "{}"
    #
    # if name.startswith("object."):
    #     default = "{}"

    name = name.replace("array", "list")
    name = name.replace("<", "[")
    name = name.replace(">", "]")
    name = name.replace("struct.", "")
    name = name.replace("object.", "")
    name = name.replace("bitflag.", "")
    name = name.replace("enum.", "")

    if name.startswith("list") and not default:
        default = "field(default_factory=list)" if factory else "[]"

    if "[" in name:
        tokens = name.split("[")
        print("parse ", tokens)
        name = tokens[0] + "[" + parse_type(tokens[1][:-1]) + "]"
        print("\tgot ", name)
    elif not name in ["int", "str", "bool", "float"]:
        name = to_pascal_case(name)

    if default is None:
        name = name + " | None"


    name = name.replace("[", '["')
    name = name.replace("]", '"]')

    if name in _bitflags and default:
        default = name + "." + default.replace('"', "").upper()

    if name in _enums and default:
        default = name + "." + default.replace('"', "")

    return name, default


def generate_struct(struct: Dict[str, Any]) -> str:
    """Generate Python class for a struct."""
    struct_name = parse_type(struct["name"])
    members = struct.get("members", [])

    lines = [f"@dataclass"]
    lines.append(f"class {struct_name}(BaseWebGPUObject):")
    # lines.append("  model_config = ConfigDict(arbitrary_types_allowed=True)")
    for member in members:
        name = member["name"]
        typ, default_value = parse_type_and_default(member)
        if not typ.startswith("list") or typ in ["int", "str"]:
            typ = f'"{typ}"'
        lines.append(f"  {name}: {typ} = {default_value}")

    return "\n".join(lines)


def generate_property(prop: Dict[str, Any]) -> str:
    name = to_camel_case(prop["name"][4:])
    print(prop)
    typ = parse_type(prop["returns"]["type"])
    s =  f"  @property\n"
    s += f"  def {name}(self) -> \"{typ}\":\n"
    s += f"    return self.handle.{name}()\n"
    return s

def generate_object(obj: Dict[str, Any]) -> str:
    """Generate Python class for a struct."""
    struct_name = parse_type(obj["name"])
    methods = obj.get("methods", [])

    lines = [f"class {struct_name}:"]
    lines.append(f"  handle: pyodide.ffi.JsProxy")
    lines.append(f"  def __init__(self, handle):")
    lines.append(f"    self.handle = handle")
    # lines.append(f"  handle: object")
    for method in methods:
        if method["name"].startswith("get_") and not "args" in method and "returns" in method:
            lines.append(generate_property(method))
            continue
        ret = parse_type(method["returns"]["type"]) if "returns" in method else None
        name = to_camel_case(method["name"])
        signature = ["self"]
        args = []
        for arg in method.get("args", []):
            typ, default = parse_type_and_default(arg, factory=False)
            if not typ.startswith("list") or typ in ["int", "str"]:
                typ = f'"{typ}"'
            signature.append(f"{arg['name']}: {typ} = {default}")
            args.append(arg["name"])
        signature = ", ".join(signature)
        args = ", ".join([arg["name"] for arg in method.get("args", [])])
        lines.append(f'  def {name}( {signature} ) -> "{ret}":')
        lines.append(f"    return self.handle.{name}({args})")

    return "\n".join(lines)


_header = """
from enum import Enum, IntFlag
from dataclasses import dataclass, field
from typing import Callable
import pyodide.ffi
import js
def _to_js(value):
  return pyodide.ffi.to_js(value, dict_converter=js.Object.fromEntries)

class BaseWebGPUObject():
    def _to_js(self):
        return _to_js(self.__dict__)
"""


def generate_api(schema: Dict[str, Any]) -> str:
    """Generate Python API from schema."""
    lines = []

    for enum in schema["enums"]:
        lines.append(generate_enum(enum))

    for bitflag in schema["bitflags"]:
        lines.append(generate_bitflags(bitflag))

    for struct in schema["structs"]:
        lines.append(generate_struct(struct))

    for obj in schema["objects"]:
        lines.append(generate_object(obj))

    return _header + "\n\n".join(lines)


# Example Usage
# Load the schema from JSON
with open("webgpu.json", "r") as file:
    webgpu_schema = json.load(file)

# Generate the API
python_api_code = generate_api(webgpu_schema)

# Write the generated API to a file
with open("webgpu/webgpu_api.py", "w") as file:
    file.write(python_api_code)

print("WebGPU API has been generated successfully!")
