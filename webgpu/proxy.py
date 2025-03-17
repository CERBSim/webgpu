import asyncio
import base64
import itertools
import json
import threading
import os


class JsRemote:
    _request_id: itertools.count
    _requests: dict
    _callback_loop: asyncio.AbstractEventLoop
    _websocket_loop: asyncio.AbstractEventLoop
    _websocket_thread: threading.Thread
    _websocket_port: int = -1
    _websocket_server_started: threading.Event
    _callback_thread: threading.Thread
    _callback_queue: asyncio.Queue
    _connected_clients: set

    _object_id: itertools.count
    _objects: dict

    def __init__(self):
        self._request_id = itertools.count()
        self._requests = {}
        self._objects = {}
        self._object_id = itertools.count()
        self._connected_clients = set()

        self._callback_loop = asyncio.new_event_loop()
        self._callback_queue = asyncio.Queue()
        self._callback_thread = threading.Thread(
            target=self._start_callback_server, daemon=True
        )

        self._websocket_loop = asyncio.new_event_loop()
        self._websocket_server_started = threading.Event()
        self._websocket_thread = threading.Thread(
            target=self._start_websocket_server, daemon=True
        )

        self._websocket_thread.start()
        self._callback_thread.start()

        self._websocket_server_started.wait()

    async def _send_async(self, message):
        if self._connected_clients:
            await asyncio.gather(*(ws.send(message) for ws in self._connected_clients))
        else:
            raise Exception("No connected clients")

    def call_function(self, parent_id, id: int, args=[]):
        return self._send(
            {
                "request_id": next(self._request_id),
                "parent_id": parent_id,
                "type": "call_function",
                "id": id,
                "args": args,
            }
        )

    def set_prop(self, id, prop, value):
        return self._send(
            {
                "request_id": next(self._request_id),
                "type": "set_prop",
                "id": id,
                "prop": prop,
                "value": value,
            }
        )

    def get_prop(self, id, prop=None):
        return self._send(
            {
                "request_id": next(self._request_id),
                "type": "get_prop",
                "id": id,
                "prop": prop,
            }
        )

    def get_keys(self, id):
        return json.loads(
            self._send(
                {
                    "request_id": next(self._request_id),
                    "type": "get_keys",
                    "id": id,
                }
            )
        )

    def on_canvas_resize(self, canvas):
        return self._send(
            {
                "type": "on_canvas_resize",
                "canvas": canvas._to_js(),
            }
        )

    def _send(self, data):
        """Sende a message to the JS environment,
        if request_id is set, (blocking-)wait for the response and return it"""

        if not self._connected_clients:
            print("no clients connected")
            return
        request_id = data.get("request_id", None)
        message = json.dumps(data)
        if request_id is not None:
            event = threading.Event()
            self._requests[request_id] = event
        asyncio.run_coroutine_threadsafe(
            self._send_async(message), self._websocket_loop
        )
        if request_id is not None:
            event.wait()
            return self._requests.pop(request_id)

    def _parse_result(self, data):
        """Parse the result of a message from the JS environment"""
        if data["type"] == "proxy":
            return JsProxy(data["id"], data.get("parent_id", None))

        if data["type"] == "value":
            return data.get("value", None)

        if data["type"] == "binary_value":
            return base64.b64decode(data["value"])

        raise Exception(f"Unknown result type: {data}")

    async def _websocket_handler(self, websocket, path=""):
        try:
            self._connected_clients.add(websocket)
            async for message in websocket:
                self._on_message(json.loads(message))
        finally:
            self._connected_clients.remove(websocket)

    def _on_message(self, data):
        request_id = data.get("request_id", None)
        if request_id is not None:
            event = self._requests[request_id]
            self._requests[request_id] = self._parse_result(data)
            event.set()
            return

        type = data.get("type", None)
        if type == "call_function":
            args = data["args"]
            self._objects[data["id"]](args)

    def _start_callback_server(self):
        async def handle_callbacks():
            while True:
                try:
                    func, args = await self._callback_queue.get()
                    func(*args)
                except asyncio.QueueEmpty:
                    pass
                except Exception as e:
                    print("error in callback", type(e), str(e))
                # await asyncio.sleep(0.01)

        try:
            self._callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._callback_loop)
            self._callback_loop.create_task(handle_callbacks())
            self._callback_loop.run_forever()
        except Exception as e:
            print("exception in _start_callback_server", e)

    def _start_websocket_server(self):
        import websockets

        async def start_websocket():
            port = 8700
            while True:
                try:
                    async with websockets.serve(
                        self._websocket_handler, "", port
                    ) as server:
                        self._websocket_port = port
                        self._websocket_server_started.set()
                        await server.serve_forever()
                except OSError as e:
                    port += 1
                except Exception as e:
                    print("error in websocket server", e)

        try:
            asyncio.set_event_loop(self._websocket_loop)
            self._websocket_loop.create_task(start_websocket())
            self._websocket_loop.run_forever()
        except Exception as e:
            print("exception in _start_websocket_server", e)


remote: JsRemote = None
convert = None


class JsProxyIterator:
    def __init__(self, proxy):
        self._proxy = proxy
        self._keys = proxy._get_keys()
        self._index = 0

    def __next__(self):
        if self._index < len(self._keys):
            key = self._keys[self._index]
            self._index += 1
            return key
        else:
            raise StopIteration

    def __iter__(self):
        return self


class JsProxy:
    _id: int
    _parent_id: int

    def __init__(self, id=None, parent_id=None):
        self._id = id
        self._parent_id = parent_id

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __getattr__(self, key):
        if (
            isinstance(key, str)
            and key.startswith("__")
            and not key.startswith("__vue")
        ):
            return super().__getattr__(key)
        return remote.get_prop(self._id, key)

    def __setattr__(self, key, value):
        if key in ["_id", "_parent_id"]:
            return super().__setattr__(key, value)

        return remote.set_prop(self._id, key, convert(value))

    def __call__(self, *args):
        return remote.call_function(self._parent_id, self._id, convert(args))

    def _to_js(self):
        return {
            "__python_proxy_type__": "proxy",
            "id": self._id,
            "parent_id": self._parent_id,
        }

    def _get_keys(self):
        return remote.get_keys(self._id)

    def __iter__(self):
        return JsProxyIterator(self)


def create_render_proxy(func):
    id = next(remote._object_id)

    def wrapper(args):
        asyncio.run_coroutine_threadsafe(
            remote._callback_queue.put((func, args)), remote._callback_loop
        )

    remote._objects[id] = wrapper
    return {"__python_proxy_type__": "render", "id": id}


_is_exporting = "WEBGPU_EXPORT" in os.environ

remote: JsRemote = None

try:
    import pyodide
    import pyodide.ffi
    import js

    _is_pyodide = True
    create_proxy = pyodide.ffi.create_proxy

except ImportError:
    _is_pyodide = False
    js = None

    def create_proxy(func):
        id = next(remote._object_id)

        def wrapper(args):
            asyncio.run_coroutine_threadsafe(
                remote._callback_queue.put((func, args)), remote._callback_loop
            )

        remote._objects[id] = wrapper
        return {"__python_proxy_type__": "function", "id": id}


def init():
    global remote, js
    remote = JsRemote()
    js = JsProxy()
