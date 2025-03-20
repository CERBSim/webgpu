import asyncio
import base64
import itertools
import json
import threading
from collections.abc import Mapping
from typing import Callable


class LinkBase:
    _request_id: itertools.count
    _requests: dict

    _objects: dict

    _send_loop: asyncio.AbstractEventLoop
    _callback_loop: asyncio.AbstractEventLoop
    _callback_queue: asyncio.Queue
    _callback_thread: threading.Thread

    _serializers: dict[type, Callable] = {}

    @staticmethod
    def register_serializer(type_, serializer):
        LinkBase._serializers[type_] = serializer

    def __init__(self):
        self._request_id = itertools.count()
        self._requests = {}
        self._objects = {}

        self._send_loop = asyncio.new_event_loop()
        self._callback_loop = asyncio.new_event_loop()
        self._callback_queue = asyncio.Queue()

        self._callback_thread = threading.Thread(
            target=self._start_callback_thread, daemon=True
        )
        self._callback_thread.start()

    def wait_for_connection(self):
        raise NotImplementedError

    def expose(self, name: str, obj):
        self._objects[str(name)] = obj

    def call(self, id, args=[], parent_id=None):
        return self._send_data(
            {
                "request_id": next(self._request_id),
                "id": id,
                "type": "call",
                "parent_id": parent_id,
                "args": self._dump_data(args),
            }
        )

    def set_item(self, id, key, value):
        return self._send_data(
            {
                "type": "set",
                "id": id,
                "key": key,
                "value": self._dump_data(value),
            }
        )

    def set(self, id, prop, value):
        return self._send_data(
            {
                "type": "set",
                "id": id,
                "prop": prop,
                "value": self._dump_data(value),
            }
        )

    def get_keys(self, id):
        return self._send_data(
            {
                "request_id": next(self._request_id),
                "type": "get_keys",
                "id": id,
            }
        )

    def get_item(self, id, key):
        return self._send_data(
            {
                "request_id": next(self._request_id),
                "type": "get",
                "id": id,
                "key": key,
            }
        )

    def get(self, id, prop: str | None = None):
        return self._send_data(
            {
                "request_id": next(self._request_id),
                "type": "get",
                "id": id,
                "prop": prop,
            }
        )

    def create_handle(self, obj):
        id_ = id(obj)
        self._objects[id_] = obj
        return {"__is_crosslink_type__": True, "type": "proxy", "id": id_}

    def create_proxy(self, func):
        def wrapper(*args):
            asyncio.run_coroutine_threadsafe(
                self._callback_queue.put((func, args)), self._callback_loop
            )

        id_ = id(wrapper)
        self._objects[id_] = wrapper
        return {"__is_crosslink_type__": True, "type": "proxy", "id": id_}

    def _send_response(self, request_id, data):
        return self._send_data(
            {
                "request_id": request_id,
                "type": "response",
                "value": self._dump_data(data),
            }
        )

    def _get_obj(self, data):
        obj = self._objects
        id_ = data.get("id", None)
        prop = data.get("prop", None)
        key = data.get("key", None)

        if id_ is not None:
            obj = obj[data["id"]]
        if prop is not None:
            obj = obj.__getattribute__(data["prop"])
        if key is not None:
            obj = obj[data["key"]]
        return obj

    def _on_message(self, data):
        try:
            msg_type = data.get("type", None)
            request_id = data.get("request_id", None)

            response = None

            match msg_type:
                case "response":
                    event = self._requests[request_id]
                    self._requests[request_id] = self._load_data(
                        data.get("value", None)
                    )
                    event.set()
                    return

                case "call":
                    args = data["args"]
                    func = self._get_obj(data)
                    response = func(*args)

                case "get":
                    response = self._get_obj(data)

                case "get_keys":
                    response = []

                case "set":
                    prop = data.pop("prop", None)
                    key = data.pop("key", None)
                    obj = self._get_obj(data)
                    if prop is not None:
                        obj.__setattr__(prop, data["value"])
                    elif key is not None:
                        obj[key] = self._load_data(data["value"])

                case _:
                    print("unknown message type", msg_type)

            if request_id is not None:
                self._send_response(request_id, response)
        except Exception as e:
            print("error in on_message", data, type(e), str(e))

    def _dump_data(self, data):
        from .proxy import Proxy

        type_ = type(data)
        for ser_type in self._serializers:
            if issubclass(type_, ser_type):
                data = self._serializers[ser_type](data)
                break

        if isinstance(data, (int, float, str, bool, type(None))):
            return data

        if isinstance(data, (bytes, memoryview)):
            return {
                "__is_crosslink_type__": True,
                "type": "bytes",
                "value": base64.b64encode(data).decode(),
            }

        if isinstance(data, dict):
            return {k: self._dump_data(v) for k, v in data.items()}

        if isinstance(data, Mapping):
            return {k: self._dump_data(v) for k, v in data.items()}

        if isinstance(data, (list, tuple)):
            return [self._dump_data(v) for v in data]

        if isinstance(data, Proxy):
            return {
                "__is_crosslink_type__": True,
                "type": "object",
                "id": data._id,
                "parent_id": data._parent_id,
            }

        # complex type - store it in objects and only send its id
        id_ = id(data)
        self._objects[id_] = data
        return {"__is_crosslink_type__": True, "type": "proxy", "id": id_}

    def _load_data(self, data):
        """Parse the result of a message from the remote environment"""
        from .proxy import Proxy

        if not isinstance(data, dict):
            return data

        if not data.get("__is_crosslink_type__", False):
            return {k: self._load_data(v) for k, v in data.items()}

        if data["type"] == "object":
            return self._objects[data["id"]]

        if data["type"] == "proxy":

            return Proxy(self, data.get("parent_id", None), data.get("id", None))

        if data["type"] == "bytes":
            return base64.b64decode(data["value"])

        raise Exception(f"Unknown result type: {data}")

    def _send_data(self, data):
        """Send data to the remote environment,
        if request_id is set, (blocking-)wait for the response and return it"""

        request_id = data.get("request_id", None)
        type = data.get("type", None)
        message = json.dumps(data)
        event = None
        if type != "response" and request_id is not None:
            event = threading.Event()
            self._requests[request_id] = event

        asyncio.run_coroutine_threadsafe(self._send_async(message), self._send_loop)
        if event:
            event.wait()
            return self._requests.pop(request_id)

    async def _send_async(self, message):
        raise NotImplementedError

    def _start_callback_thread(self):
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
            print("exception in _start_callback_thread", e)
