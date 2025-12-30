"""Platform specific code, currenty there are two possibilities:

1. Running in a browser with Pyodide
   Webgpu calls are done directly in the browser using the Pyodide interface provided by the "js" module.

2. Running in a Python environment with a websocket connection to a browser
   Webgpu calls are transferred via a websocket connection to the browser environment (using the webgpu.link.websocket module)
"""

from collections.abc import Mapping
import threading

is_pyodide = False
is_pyodide_main_thread = False
create_proxy = None
destroy_proxy = None
js = None
websocket_server = None
link = None


def create_event_handler(
    func,
    prevent_default=True,
    stop_propagation=False,
    stop_immediate_propagation=False,
    return_value=None,
):
    options = {
        "preventDefault": prevent_default,
        "stopPropagation": stop_propagation,
        "stopImmediatePropagation": stop_immediate_propagation,
        "returnValue": return_value,
    }
    return js.createEventHandler(func, options)


try:
    import js as pyodide_js
    import pyodide.ffi
    from pyodide.ffi import JsPromise, JsProxy
    from pyodide.ffi import create_proxy as _create_proxy

    def create_proxy(func, ignore_return_value=False):
        return _create_proxy(func)

    def destroy_proxy(proxy):
        proxy.destroy()

    is_pyodide = True
    try:
        is_pyodide_main_thread = bool(pyodide_js.window.document)
    except:
        is_pyodide_main_thread = False

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
            dict_converter=pyodide_js.Object.fromEntries,
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

    def _serialize_jsproxy(link, value):
        if hasattr(value, "unwrap"):
            u = value.unwrap()
            return link._dump_data(u)

        return json.loads(pyodide_js.JSON.stringify(value))

    LinkBase.register_serializer(JsProxy, _serialize_jsproxy)

_funcs_after_init = []


def execute_when_init(func):
    """Register a callback to run once the JS side is ready.

    If the platform has already been initialized, the callback is executed
    immediately. Otherwise it is queued and executed from ``init`` once the
    websocket connection has been established and ``js`` is set.
    """

    if js is not None:
        func(js)
    else:
        _funcs_after_init.append(func)


def init(before_wait_for_connection=None, block_on_connection: bool = True):
    """Initialize the websocket link to the browser.

    In the default (classic Jupyter) mode, this blocks until the browser has
    connected via websocket so that ``js`` is ready to use.

    In environments like VS Code notebooks, outputs are typically only
    processed once the cell has finished executing. In that situation calling
    ``init`` with ``block_on_connection=False`` avoids a deadlock by moving the
    blocking ``wait_for_connection`` part to a background thread. Code that
    depends on ``js`` should use :func:`execute_when_init` so it runs once the
    connection is ready.
    """

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

    from .link.base import LinkBase
    from .webgpu_api import BaseWebGPUHandle, BaseWebGPUObject

    def _finish_init():
        websocket_server.wait_for_connection()
        js_local = websocket_server.get(None, None)

        LinkBase.register_serializer(BaseWebGPUHandle, lambda _, v: v.handle)
        LinkBase.register_serializer(
            BaseWebGPUObject, lambda _, v: v.__dict__ or None
        )

        # Publish js and run any deferred callbacks.
        globals()["js"] = js_local
        websocket_server._start_handling_messages.set()
        for func in _funcs_after_init:
            func(js_local)
        _funcs_after_init.clear()

    if block_on_connection:
        _finish_init()
    else:
        thread = threading.Thread(target=_finish_init, daemon=True)
        thread.start()


def init_pyodide(link_):
    global link
    link = link_
    global js
    js = link.get(None, None)

    from .link.base import LinkBase
    from .webgpu_api import BaseWebGPUHandle, BaseWebGPUObject

    LinkBase.register_serializer(BaseWebGPUHandle, lambda _, v: v.handle)
    LinkBase.register_serializer(BaseWebGPUObject, lambda _, v: v.__dict__ or None)
