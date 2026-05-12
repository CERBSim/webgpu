"""Helpers to build generic 'gui' ExportInteraction entries from Python.

The exported HTML's `gui` JS handler interprets the schema produced here:
controls bind to a small `vars` dict; `writes` describe how those vars (and a
`t` parameter, in seconds) map to GPU buffer writes, either per-frame
(animation) or one-shot on a var change (`trigger`).
"""

from dataclasses import dataclass, field
from typing import Any

from .format import ExportInteraction


@dataclass
class Target:
    buffer_id: str
    offset: int = 0
    dtype: str = "f32"  # "f32" | "u32" | "i32"

    def to_dict(self):
        return {"buffer_id": self.buffer_id, "offset": self.offset, "dtype": self.dtype}


@dataclass
class Checkbox:
    var: str
    name: str | None = None
    default: bool = False

    def to_dict(self):
        return {"kind": "checkbox", "var": self.var, "name": self.name or self.var}


@dataclass
class Slider:
    var: str
    name: str | None = None
    default: float = 0.0
    min: float = 0.0
    max: float = 1.0
    step: float = 0.01

    def to_dict(self):
        return {
            "kind": "slider", "var": self.var, "name": self.name or self.var,
            "min": self.min, "max": self.max, "step": self.step,
        }


@dataclass
class Dropdown:
    var: str
    options: dict
    name: str | None = None
    default: Any = None

    def to_dict(self):
        return {
            "kind": "dropdown", "var": self.var, "name": self.name or self.var,
            "options": self.options,
        }


@dataclass
class Write:
    targets: list  # list[Target]
    expr: str | None = None       # JS expression over vars + t
    value: Any = None             # constant (used if expr is None)
    when: str | None = None       # JS predicate; if falsy, skip write
    trigger: str | None = None    # var name; one-shot on change instead of per-frame

    def to_dict(self):
        d = {"targets": [t.to_dict() for t in self.targets]}
        if self.expr is not None: d["expr"] = self.expr
        if self.value is not None: d["value"] = self.value
        if self.when is not None: d["when"] = self.when
        if self.trigger is not None: d["trigger"] = self.trigger
        return d


def gui_interaction(label, controls, writes, vars=None) -> ExportInteraction:
    """Build a generic ``gui`` ``ExportInteraction``.

    ``vars`` defaults to ``{c.var: c.default}`` over all controls.
    """
    if vars is None:
        vars = {c.var: c.default for c in controls if hasattr(c, "default") and c.default is not None}
        # Always include checkbox vars even when default is False
        for c in controls:
            if isinstance(c, Checkbox) and c.var not in vars:
                vars[c.var] = c.default
    return ExportInteraction(
        type="gui",
        buffer_id="",  # unused by gui handler
        config={
            "label": label,
            "vars": vars,
            "controls": [c.to_dict() for c in controls],
            "writes": [w.to_dict() for w in writes],
        },
    )
