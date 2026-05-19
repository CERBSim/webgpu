"""Declarative GUI parameters that work in both live and export paths.

A GuiParam describes a single interactive control (dropdown, slider, checkbox)
that writes to one or more GPU uniform buffer fields. Multiple renderers can
share the same GuiParam instance — it appears once in the GUI.
"""
import ctypes as ct
import json


def _field_info(uniform, field_name):
    """Get (byte_offset, dtype_str) for a named field on a ctypes.Structure uniform."""
    for name, ftype in uniform._fields_:
        if name == field_name:
            descriptor = getattr(type(uniform), field_name)
            offset = descriptor.offset
            if ftype == ct.c_float:
                return offset, "f32"
            elif ftype == ct.c_int32:
                return offset, "i32"
            else:
                return offset, "u32"
    raise ValueError(f"Field '{field_name}' not found on {type(uniform).__name__}")


def _resolve(ref):
    """Resolve a uniform reference to the actual uniform object.
    
    Accepts:
      - A UniformBase instance directly
      - A callable that returns a UniformBase
      - An object with a .uniform attribute (e.g. FunctionSettings)
      - An object with a .uniforms attribute (e.g. Colormap)
    """
    if callable(ref) and not isinstance(ref, type):
        return ref()
    if hasattr(ref, '_fields_'):
        # It's already a ctypes Structure (UniformBase)
        return ref
    if hasattr(ref, 'uniform'):
        return ref.uniform
    if hasattr(ref, 'uniforms'):
        return ref.uniforms
    return ref


class GuiParam:
    """Declarative GUI parameter. Works in both live and export paths.
    
    Examples:
        component = GuiParam("dropdown", "Component",
                             options={"Norm": -1, "x": 0, "y": 1}, default=-1)
        component.bind(settings, "component")       # settings.uniform.component
        component.affects(colormap, "min", values={-1: 0.1, 0: -1.0, 1: 0.5})
        
        shrink = GuiParam("slider", "Shrink", default=1.0, min=0.0, max=1.0)
        shrink.bind(mesh_uniform, "shrink")
    """

    def __init__(self, kind, label, *, default, options=None, min=None, max=None, step=None):
        self.kind = kind          # "dropdown" | "slider" | "checkbox"
        self.label = label
        self.default = default
        self.options = options    # dropdown: {"Norm": -1, "x": 0, ...}
        self.min = min
        self.max = max
        self.step = step
        self._bindings = []       # [(ref, field_name)]
        self._effects = []        # [(ref, field_name, {option_val: write_val})]

    def bind(self, ref, field_name):
        """Write selected value directly to this uniform field on change.
        
        ref: a UniformBase, or object with .uniform/.uniforms attr, or callable.
        """
        self._bindings.append((ref, field_name))
        return self

    def affects(self, ref, field_name, *, values):
        """On change, look up selected value in dict and write result to field.
        
        ref: a UniformBase, or object with .uniform/.uniforms attr, or callable.
        values: dict mapping option values to write values, or a callable
            returning such a dict (resolved at export time).
            E.g. {-1: 0.1, 0: -1.0, 1: 0.5}
        """
        self._effects.append((ref, field_name, values))
        return self

    def export(self, registry):
        """Produce an Interaction for the JS engine."""
        from .export.gui import gui_interaction, Dropdown, Slider, Checkbox, Write, Target

        var = self.label.lower().replace(" ", "_")

        # Build control
        if self.kind == "dropdown":
            control = Dropdown(var=var, name=self.label, options=self.options, default=self.default)
        elif self.kind == "slider":
            control = Slider(var=var, name=self.label, default=self.default,
                           min=self.min, max=self.max, step=self.step or 0.01)
        elif self.kind == "checkbox":
            control = Checkbox(var=var, name=self.label, default=self.default)
        else:
            raise ValueError(f"Unknown GuiParam kind: {self.kind}")

        writes = []

        # Direct bindings: write param value to uniform fields
        targets = []
        seen_targets = set()
        for ref, field_name in self._bindings:
            uniform = _resolve(ref)
            if uniform is None:
                continue
            buf = getattr(uniform, "_buffer", None)
            if buf is None:
                continue
            key = id(buf)
            if key not in registry._buffers:
                continue
            buf_id = registry._buffers[key][0]
            offset, dtype = _field_info(uniform, field_name)
            tk = (buf_id, offset, dtype)
            if tk not in seen_targets:
                targets.append(Target(buf_id, offset=offset, dtype=dtype))
                seen_targets.add(tk)
        if targets:
            writes.append(Write(targets=targets, expr=var, trigger=var))

        # Side effects: write looked-up values
        seen_effect_keys = set()
        for ref, field_name, values_or_fn in self._effects:
            uniform = _resolve(ref)
            if uniform is None:
                continue
            buf = getattr(uniform, "_buffer", None)
            if buf is None:
                continue
            key = id(buf)
            if key not in registry._buffers:
                continue
            values_dict = values_or_fn() if callable(values_or_fn) else values_or_fn
            buf_id = registry._buffers[key][0]
            offset, dtype = _field_info(uniform, field_name)
            effect_key = (buf_id, offset, dtype)
            if effect_key in seen_effect_keys:
                continue
            seen_effect_keys.add(effect_key)
            expr = _values_to_js_expr(values_dict, var)
            writes.append(Write(
                targets=[Target(buf_id, offset=offset, dtype=dtype)],
                expr=expr,
                trigger=var,
            ))

        if not writes:
            return None
        return gui_interaction(self.label, [control], writes)


def _values_to_js_expr(values_dict, var_name):
    """Build a JS expression that looks up var_name in a values dict."""
    obj = json.dumps({str(k): v for k, v in values_dict.items()})
    return f'{obj}[String({var_name})]'
