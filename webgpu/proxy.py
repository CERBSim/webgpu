import asyncio
import itertools
import json
import threading

import websockets

WS_PORT = 8765


class JsRemote:
    _request_id: itertools.count
    _requests: dict
    _loop: asyncio.AbstractEventLoop
    _thread: threading.Thread
    _connected_clients: set

    _object_id: itertools.count
    _objects: dict

    def __init__(self):
        self._request_id = itertools.count()
        self._requests = {}
        self._objects = {}
        self._object_id = itertools.count()
        self._connected_clients = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_server, daemon=True)
        self._thread.start()

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
        asyncio.run_coroutine_threadsafe(self._send_async(message), self._loop)
        if request_id is not None:
            event.wait()
            return self._requests.pop(request_id)

    def _parse_result(self, data):
        """Parse the result of a message from the JS environment"""
        if data["type"] == "proxy":
            return JsProxy(data["id"], data.get("parent_id", None))

        if data["type"] == "value":
            return data.get("value", None)

        raise Exception(f"Unknown result type: {data}")

    async def _websocket_handler(self, websocket, path=""):
        try:
            print("========================\nnew websocket connection")
            self._connected_clients.add(websocket)
            async for message in websocket:
                self._on_message(json.loads(message))
        finally:
            print("========================\nclose websocket connection")
            self._connected_clients.remove(websocket)

    def _on_message(self, data):
        request_id = data.get("request_id", None)
        if request_id is not None:
            event = self._requests[request_id]
            self._requests[request_id] = self._parse_result(data)
            event.set()
            return

        type = data.get("type", None)
        print("on message", data)
        if type == "call_function":
            print("got call_function message", data)
            args = json.loads(data["args"])
            self._objects[data["id"]](*args)

    def _start_server(self):
        async def start_websocket():
            async with websockets.serve(self._websocket_handler, "", WS_PORT) as server:
                await server.serve_forever()

        try:
            asyncio.set_event_loop(self._loop)

            self._loop.create_task(start_websocket())
            self._loop.run_forever()
        except Exception as e:
            print("exception in _start_servers", e)


remote = None
convert = None
js = None


class JsProxyIterator:
    def __init__(self, proxy):
        self._proxy = proxy
        self._keys = proxy._get_keys()
        print("keys", self._keys)
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

    def __getattr__(self, key):
        if key.startswith("__"):
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


def create_proxy(func, **kwargs):
    id = next(remote._object_id)
    def wrapper(*args):
        import threading
        threading.Thread(target=func, args=args).start()
    remote._objects[id] = wrapper
    return { "__python_proxy_type__": "function", "id": id } | kwargs

if __name__ == "__main__":
    remote = JsRemote()
    root = JsProxy()
    while True:
        import time

        time.sleep(2)
        if remote._connected_clients:
            html = root.document.body.innerHTML[-100:]
            print("html", html)
