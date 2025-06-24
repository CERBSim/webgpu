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


def _pack_message(data: dict) -> tuple[dict, bytes | str]:
    buffer = data.pop("buffer", None)
    if buffer is None:
        return data, json.dumps(data)

    offset = 0
    offsets = [0]
    for b in buffer:
        offset += len(b)
        offsets.append(offset)
    data["buffer_offsets"] = offsets

    bdata = json.dumps(data).encode("utf-8")
    bdata = len(bdata).to_bytes(4, "little") + bdata + b"".join(buffer)

    return data, bdata


class LinkBase:
    _request_id: itertools.count
    _requests: dict
    _objects: dict
    _cache: dict

    _serializers: dict[type, Callable] = {}
    _max_message_size = 100 * 1024 * 1024  # 100 MB

    def _send_data(self, metadata, data, key=None):
        raise NotImplementedError

    def _split_and_send(self, data, key=None):
        request_id = data.get("request_id", None)
        metadata, data = _pack_message(data)

        size = len(data)

        if size <= self._max_message_size:
            return self._send_data(metadata, data, key)

        if isinstance(data, str):
            data = data.encode("utf-8")
            data = len(data).to_bytes(4, "little") + data

        # Split the data into chunks
        chunks = []
        while len(data) > self._max_message_size:
            chunk = data[: self._max_message_size]
            data = data[self._max_message_size :]
            chunks.append(chunk)

        if data:
            chunks.append(data)

        for i, chunk in enumerate(chunks):
            mchunk, dchunk = _pack_message(
                {
                    "parent_request_id": request_id,
                    "type": "chunk",
                    "buffer": [chunk],
                    "chunk_id": i,
                    "offset": i * self._max_message_size,
                    "size": len(chunk),
                    "total_size": size,
                    "n_chunks": len(chunks),
                }
            )
            if i == len(chunks) - 1:
                mchunk["request_id"] = request_id
                mchunk["key"] = key
                return self._send_data(mchunk, dchunk)
            else:
                self._send_data(mchunk, dchunk)

    @staticmethod
    def register_serializer(type_, serializer):
        LinkBase._serializers[type_] = serializer

    def __init__(self):
        self._request_id = itertools.count()
        self._requests = {}
        self._objects = {}
        self._cache = {}

    def _call_data(self, id, prop, args, ignore_result=False):
        buffer = []
        args = self._dump_data(args, buffer)
        return {
            "request_id": next(self._request_id) if not ignore_result else None,
            "prop": prop,
            "type": "call",
            "id": id,
            "args": args,
            "buffer": buffer if buffer else None,
        }

    def call_new(self, id=None, prop=None, args=[], ignore_result=False):
        return self._split_and_send(
            self._call_data(id, prop, args, ignore_result) | {"type": "new"}
        )

    def call_method(self, id=None, prop=None, args=[], ignore_result=False):
        return self._split_and_send(self._call_data(id, prop, args, ignore_result))

    def call_method_ignore_return(self, id=None, prop=None, args=[]):
        return self.call(id, prop=prop, args=args, ignore_result=True)

    def call(self, id, args=[], parent_id=None, ignore_result=False, prop=None):
        return self._split_and_send(
            self._call_data(id, prop, args, ignore_result) | {"parent_id": parent_id}
        )

    def set_item(self, id, key, value):
        buffer = []
        value = self._dump_data(value, buffer)
        return self._split_and_send(
            {
                "type": "set",
                "id": id,
                "key": key,
                "value": value,
                "buffer": buffer if buffer else None,
            }
        )

    def set(self, id, prop, value):
        value = self._dump_data(value)
        return self._split_and_send(
            {
                "type": "set",
                "id": id,
                "prop": prop,
                "value": value,
            }
        )

    def get_keys(self, id):
        return self._split_and_send(
            {
                "request_id": next(self._request_id),
                "type": "get_keys",
                "id": id,
            }
        )

    def get_item(self, id, key):
        return self._split_and_send(
            {
                "request_id": next(self._request_id),
                "type": "get",
                "id": id,
                "key": key,
            }
        )

    def get(self, id, prop: str | None = None):
        if (id, prop) in self._cache:
            return self._cache[(id, prop)]

        return self._split_and_send(
            {
                "request_id": next(self._request_id),
                "type": "get",
                "id": id,
                "prop": prop,
            },
            key=(id, prop),
        )

    def create_handle(self, obj):
        id_ = id(obj)
        self._objects[id_] = obj
        return {"__is_crosslink_type__": True, "type": "proxy", "id": id_}

    def _dump_data(self, data, buffer=None):
        from .proxy import Proxy

        type_ = type(data)
        for ser_type in self._serializers:
            if issubclass(type_, ser_type):
                data = self._serializers[ser_type](self, data)
                break

        if isinstance(data, (int, float, str, bool, type(None))):
            return data

        if isinstance(data, (bytes, memoryview)):
            if buffer is None:
                return {
                    "__is_crosslink_type__": True,
                    "type": "bytes",
                    "value": base64.b64encode(data).decode(),
                }
            index = len(buffer)
            buffer.append(bytes(data))
            return {
                "__is_crosslink_type__": True,
                "type": "buffer",
                "index": index,
            }

        if isinstance(data, (list, tuple)):
            return [self._dump_data(v, buffer) for v in data]

        if isinstance(data, (dict, Mapping)):
            return {k: self._dump_data(data[k], buffer) for k in list(data.keys())}

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
            buffer = []
            value = self._dump_data(data, buffer)
            data = {
                "request_id": request_id,
                "type": "response",
                "value": value,
                "buffer": buffer if buffer else None,
            }

            self._split_and_send(data)

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
                    event, key = self._requests[request_id]
                    self._requests[request_id] = self._load_data(data.get("value", None))
                    if key and data.get("cache", False):
                        self._cache[key] = self._requests[request_id]
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
            import sys
            import traceback

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
                    event, key = self._requests[request_id]
                    response = self._load_data(data.get("value", None))
                    self._requests[request_id] = response
                    if data.get("cache", False):
                        self._cache[key] = response
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

    def _send_data(self, metadata, data, key=None):
        if type(data) is bytes:
            self._send_message(data)
        else:
            if (
                metadata.get("request_id", None) is not None
                and metadata["type"] != "response"
                and not metadata.get("ignore_return_value", False)
            ):
                import js

                js.Atomics.store(self._size_buffer, 0, 0)
                self._send_message(data)
                js.Atomics.wait(self._size_buffer, 0, 0, 10000)
                n = self._size_buffer[0]
                res = bytes(self._result_buffer.slice(0, n))
                s = res.decode("utf-8")
                data = json.loads(s)
                return self._load_data(data.get("value", None))
            else:
                self._send_message(data)


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

    def _send_data(self, metadata, data, key=None):
        """Send data to the remote environment,
        if request_id is set, (blocking-)wait for the response and return it"""
        # print("send data", data)

        request_id = metadata.get("request_id", None)
        type = metadata.get("type", None)
        # print("send response", data)
        event = None
        if type != "response" and request_id is not None:
            event = threading.Event()
            self._requests[request_id] = event, key

        asyncio.run_coroutine_threadsafe(self._send_async(data), self._send_loop)
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
