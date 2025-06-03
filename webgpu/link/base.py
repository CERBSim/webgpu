import asyncio
import base64
import itertools
import json
import threading
from collections.abc import Mapping
from typing import Callable


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class LinkBase:
    _request_id: itertools.count
    _requests: dict
    _objects: dict

    _serializers: dict[type, Callable] = {}

    def _send_data(self, data):
        raise NotImplementedError

    @staticmethod
    def register_serializer(type_, serializer):
        LinkBase._serializers[type_] = serializer

    def __init__(self):
        self._request_id = itertools.count()
        self._requests = {}
        self._objects = {}

    def _call_data(self, id, prop, args, ignore_result=False):
        return {
            "request_id": next(self._request_id) if not ignore_result else None,
            "prop": prop,
            "type": "call",
            "id": id,
            "args": self._dump_data(args),
        }

    def call_new(self, id=None, prop=None, args=[], ignore_result=False):
        return self._send_data(self._call_data(id, prop, args, ignore_result) | {"type": "new"})

    def call_method(self, id=None, prop=None, args=[], ignore_result=False):
        return self._send_data(self._call_data(id, prop, args, ignore_result))

    def call_method_ignore_return(self, id=None, prop=None, args=[]):
        return self.call(id, prop, args, ignore_result=True)

    def call(self, id, args=[], parent_id=None, ignore_result=False):
        return self._send_data(
            self._call_data(id, None, args, ignore_result) | {"parent_id": parent_id}
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
            return {k: self._dump_data(data[k]) for k in list(data.keys())}

        if isinstance(data, Mapping):
            return {k: self._dump_data(data[k]) for k in list(data.keys())}

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
        # print("complex type", data)
        id_ = id(data)
        self._objects[id_] = data
        return {"__is_crosslink_type__": True, "type": "proxy", "id": id_}

    def _load_data(self, data):
        """Parse the result of a message from the remote environment"""
        from .proxy import Proxy

        # print("load data", data, type(data))

        if isinstance(data, list):
            return [self._load_data(v) for v in data]

        if not isinstance(data, dict):
            return data

        if not data.get("__is_crosslink_type__", False):
            return AttrDict({k: self._load_data(v) for k, v in data.items()})

        if data["type"] == "object":
            return self._objects[data["id"]]

        if data["type"] == "proxy":

            return Proxy(self, data.get("parent_id", None), data.get("id", None))

        if data["type"] == "bytes":
            return base64.b64decode(data["value"])

        raise Exception(f"Unknown result type: {data}")

    def expose(self, name: str, obj):
        self._objects[str(name)] = obj

    def create_proxy(self, func, ignore_return_value=False):
        raise NotImplementedError

    def destroy_proxy(self, proxy):
        del self._objects[proxy["id"]]

    def _send_response(self, request_id, data):
        if type(data) is bytes:
            data = request_id.to_bytes(4, "big") + data
        else:
            data = {
                "request_id": request_id,
                "type": "response",
                "value": self._dump_data(data),
            }

        self._send_data(data)

    def _get_obj(self, data):
        obj = self._objects
        id_ = data.get("id", None)
        prop = data.get("prop", None)
        key = data.get("key", None)

        if id_ is not None:
            obj = obj[data["id"]]
        if prop is not None:
            obj = obj.__getattribute__(prop)
        if key is not None:
            obj = obj[data["key"]]
        return obj

    async def _on_message_async(self, message: str):
        data = json.loads(message)
        try:
            msg_type = data.get("type", None)
            request_id = data.get("request_id", None)

            response = None

            match msg_type:
                case "response":
                    event = self._requests[request_id]
                    self._requests[request_id] = self._load_data(data.get("value", None))
                    event.set()
                    return

                case "call":
                    func = self._get_obj(data)
                    args = self._load_data(data["args"])
                    response = func(*args)
                    try:
                        response = await response
                    except TypeError:
                        pass
                    except Exception as e:
                        print("error in call", type(e), str(e))

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
            import traceback
            import sys

            print("error in on_message", data, type(e), str(e), file=sys.stderr)
            if not isinstance(e, str):
                traceback.print_exception(*sys.exc_info(), file=sys.stderr)

    def _on_message(self, message: str):
        data = json.loads(message)
        try:
            msg_type = data.get("type", None)
            request_id = data.get("request_id", None)

            response = None

            match msg_type:
                case "response":
                    event = self._requests[request_id]
                    self._requests[request_id] = self._load_data(data.get("value", None))
                    event.set()
                    return

                case "call":
                    func = self._get_obj(data)
                    args = self._load_data(data["args"])
                    # print("call", func, args)
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
            from ngapp.utils import print_exception

            print("error in on_message", data, type(e), str(e))
            print_exception(e)


class PyodideLink(LinkBase):
    def __init__(self, send_message, size_buffer, result_buffer):
        super().__init__()
        self._send_message = send_message
        self._size_buffer = size_buffer
        self._result_buffer = result_buffer

    def create_proxy(self, func, ignore_return_value=False):
        id_ = id(func)
        self._objects[id_] = func
        return {
            "__is_crosslink_type__": True,
            "type": "proxy",
            "id": id_,
            "ignore_return_value": ignore_return_value,
        }

    def _send_data(self, data):
        if type(data) is bytes:
            self._send_message(data)
        else:
            if (
                data.get("request_id", None) is not None
                and data["type"] != "response"
                and not data.get("ignore_return_value", False)
            ):
                import js

                js.Atomics.store(self._size_buffer, 0, 0)
                self._send_message(json.dumps(data))
                js.Atomics.wait(self._size_buffer, 0, 0, 10000)
                n = self._size_buffer[0]
                res = bytes(self._result_buffer.slice(0, n))
                s = res.decode("utf-8")
                data = json.loads(s)
                return self._load_data(data.get("value", None))
            else:
                self._send_message(json.dumps(data))


class LinkBaseAsync(LinkBase):
    _send_loop: asyncio.AbstractEventLoop
    _callback_loop: asyncio.AbstractEventLoop
    _callback_queue: asyncio.Queue
    _callback_thread: threading.Thread

    def __init__(self):
        super().__init__()
        self._send_loop = asyncio.new_event_loop()
        self._callback_loop = asyncio.new_event_loop()
        self._callback_queue = asyncio.Queue()

        self._callback_thread = threading.Thread(target=self._start_callback_thread, daemon=True)
        self._callback_thread.start()

    def wait_for_connection(self):
        raise NotImplementedError

    def create_proxy(self, func, ignore_return_value=False):
        def wrapper(*args):
            asyncio.run_coroutine_threadsafe(
                self._callback_queue.put((func, args)), self._callback_loop
            )

        id_ = id(wrapper)
        self._objects[id_] = wrapper
        return {
            "__is_crosslink_type__": True,
            "type": "proxy",
            "id": id_,
            "ignore_return_value": ignore_return_value,
        }

    def _send_data(self, data):
        """Send data to the remote environment,
        if request_id is set, (blocking-)wait for the response and return it"""
        # print("send data", data)

        request_id = data.get("request_id", None)
        type = data.get("type", None)
        # print("send response", data)
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
