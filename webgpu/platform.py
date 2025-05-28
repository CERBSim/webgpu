"""Platform specific code, currenty there are two possibilities:

1. Running in a browser with Pyodide
   Webgpu calls are done directly in the browser using the Pyodide interface provided by the "js" module.

2. Running in a Python environment with a websocket connection to a browser
   Webgpu calls are transferred via a websocket connection to the browser environment (using the webgpu.link.websocket module)
"""

from collections.abc import Mapping

is_pyodide = False
create_proxy = None
destroy_proxy = None
js = None
websocket_server = None
link = None

try:
    import js
    import pyodide.ffi
    from pyodide.ffi import JsPromise, JsProxy, create_proxy

    def destroy_proxy(proxy):
        proxy.destroy()

    is_pyodide = True

    def _default_converter(value, a, b):
        from .webgpu_api import BaseWebGPUHandle, BaseWebGPUObject

        if isinstance(value, BaseWebGPUHandle):
            return pyodide.ffi.to_js(value.handle)
        if isinstance(value, BaseWebGPUObject):
            return value.__dict__

    def _convert(d):
        from .webgpu_api import BaseWebGPUHandle, BaseWebGPUObject

        if d is None:
            return None
        if isinstance(d, BaseWebGPUHandle):
            return d.handle
        if isinstance(d, BaseWebGPUObject):
            return _convert(d.__dict__) if d.__dict__ else None
        if isinstance(d, Mapping):
            if not d:
                return None
            ret = {}
            for key in d:
                value = _convert(d[key])
                if value is not None:
                    ret[key] = value
            return ret

        if isinstance(d, list):
            return [_convert(value) for value in d]

        return d

    def toJS(value):
        value = _convert(value)
        ret = pyodide.ffi.to_js(
            value,
            dict_converter=js.Object.fromEntries,
            default_converter=_default_converter,
            create_pyproxies=False,
        )
        return ret

except ImportError:
    pass

if not is_pyodide:
    from .link.proxy import Proxy as JsProxy
    from .link.websocket import WebsocketLinkServer

    toJS = lambda x: x

    class JsPromise:
        pass


if is_pyodide:
    import json
    import js as pyodide_js

    from .link.base import LinkBase

    LinkBase.register_serializer(JsProxy, lambda v: json.loads(pyodide_js.JSON.stringify(v)))

_funcs_after_init = []


def execute_when_init(func):
    if js is not None:
        func(js)
    else:
        _funcs_after_init.append(func)


def init(before_wait_for_connection=None):
    global js, create_proxy, destroy_proxy, websocket_server, link
    if is_pyodide or js is not None:
        return
    websocket_server = WebsocketLinkServer()
    create_proxy = websocket_server.create_proxy
    destroy_proxy = websocket_server.destroy_proxy
    link = websocket_server

    websocket_server.wait_for_server_running()

    if before_wait_for_connection:
        before_wait_for_connection(websocket_server)

    websocket_server.wait_for_connection()
    js = websocket_server.get(None, None)

    from .link.base import LinkBase
    from .webgpu_api import BaseWebGPUHandle, BaseWebGPUObject

    LinkBase.register_serializer(BaseWebGPUHandle, lambda v: v.handle)
    LinkBase.register_serializer(BaseWebGPUObject, lambda v: v.__dict__ or None)

    websocket_server._start_handling_messages.set()
    for func in _funcs_after_init:
        func(js)
    _funcs_after_init.clear()


def init_pyodide(link_):
    global link
    link = link_
    global js
    js = link.get(None, None)

    from .link.base import LinkBase
    from .webgpu_api import BaseWebGPUHandle, BaseWebGPUObject

    LinkBase.register_serializer(BaseWebGPUHandle, lambda v: v.handle)
    LinkBase.register_serializer(BaseWebGPUObject, lambda v: v.__dict__ or None)
