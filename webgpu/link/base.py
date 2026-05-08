import asyncio
import collections
import sys
import base64
import itertools
import json
import time
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


def _unpack_message(msg: str | memoryview | bytes) -> tuple[dict, list]:
    if isinstance(msg, (memoryview, bytes)):
        prefix_size = 4 + int.from_bytes(msg[:4], byteorder="little")
        data = json.loads(bytes(msg[4:prefix_size]).decode("utf-8"))
        buffer_offsets = data["buffer_offsets"]
        buffers = []

        for i in range(len(buffer_offsets) - 1):
            offset = buffer_offsets[i]
            size = buffer_offsets[i + 1] - offset
            buffers.append(bytes(msg[prefix_size + offset : prefix_size + offset + size]))
        return data, buffers
    return json.loads(msg), []


class LinkBase:
    _request_id: itertools.count
    _requests: dict
    _objects: dict
    _cache: dict

    _serializers: dict[type, Callable] = {}
    _max_message_size = 100 * 1024 * 1024  # 100 MB

    def _send_data(self, metadata, data, key=None):
        raise NotImplementedError

    def _cache_add(self, key, raw_value):
        """Add a raw value to the cache."""
        self._cache[key] = raw_value

    def _cache_evict(self, key):
        """Remove a cache entry, queueing release of IDs whose refcount is now 0."""
        raw_value = self._cache.pop(key, None)
        if raw_value is None:
            return
        if isinstance(raw_value, dict) and raw_value.get("__is_crosslink_type__"):
            vid = raw_value.get("id")
            vpid = raw_value.get("parent_id")
            if vid is not None and self._js_id_refcount.get(vid, 0) <= 0:
                self._js_id_refcount.pop(vid, None)
                self._release_queue.append(vid)
            if vpid is not None and self._js_id_refcount.get(vpid, 0) <= 0:
                self._js_id_refcount.pop(vpid, None)
                self._release_queue.append(vpid)

    def _flush_release_queue(self):
        if not self._release_queue:
            return []
        ids = []
        while self._release_queue:
            ids.append(self._release_queue.popleft())
        # Filter out IDs that were re-created by another thread since being queued
        with self._refcount_lock:
            ids = [i for i in ids if self._js_id_refcount.get(i, 0) <= 0]
        if not ids:
            return []
        # evict cache entries keyed by released ids OR whose value references a released id
        ids_set = set(ids)
        to_remove = []
        for k, v in self._cache.items():
            if k[0] in ids_set:
                to_remove.append(k)
            elif isinstance(v, dict) and v.get("id") in ids_set:
                to_remove.append(k)
        for k in to_remove:
            self._cache_evict(k)
        # _cache_evict may have added more ids to the queue — drain those too
        while self._release_queue:
            extra = self._release_queue.popleft()
            if self._js_id_refcount.get(extra, 0) <= 0:
                ids.append(extra)
        return list(set(ids))

    def _split_and_send(self, data, key=None):
        if data.get("type") != "delete_batch":
            deletes = self._flush_release_queue()
            if deletes:
                data["_deletes"] = deletes
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
        self._release_queue = collections.deque()
        self._js_id_refcount = {}  # js_id → number of live Proxies referencing it
        self._refcount_lock = threading.Lock()

        next(self._request_id)  # make sure first id is 1, in case 0 is interpreted as None

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
            return self._load_data(self._cache[(id, prop)])

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
        # sys.meta_path None can happen if called while Python shutdown. In that case, we don't care.
        if sys is None or sys.meta_path is None:
            return data
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

    def _load_data(self, data, buffers=None):
        """Parse the result of a message from the remote environment"""
        from .proxy import Proxy

        if isinstance(data, list):
            return [self._load_data(v) for v in data]

        if not isinstance(data, dict):
            return data

        if not data.get("__is_crosslink_type__", False):
            return AttrDict({k: self._load_data(v) for k, v in data.items()})

        if data["type"] == "object":
            return self._objects[data["id"]]

        if data["type"] == "proxy":
            id_ = data.get("id", None)
            parent_id = data.get("parent_id", None)
            with self._refcount_lock:
                if id_ is not None:
                    self._js_id_refcount[id_] = self._js_id_refcount.get(id_, 0) + 1
                if parent_id is not None:
                    self._js_id_refcount[parent_id] = self._js_id_refcount.get(parent_id, 0) + 1
            return Proxy(self, parent_id, id_)

        if data["type"] == "bytes":
            return base64.b64decode(data["value"])

        if data["type"] == "buffer":
            return buffers[data["index"]]

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
            obj = self._objects.get(id_)
            if obj is None:
                raise KeyError(f"Object with id {id_} not found (already released?)")
        if prop is not None:
            obj = obj.__getattribute__(prop)
        if key is not None:
            obj = obj[data["key"]]
        return obj

    async def _on_message_async(self, message: str | memoryview | bytes):
        data, buffers = _unpack_message(message)
        obj = None
        try:
            msg_type = data.get("type", None)
            request_id = data.get("request_id", None)

            response = None

            match msg_type:
                case "response":
                    event, key = self._requests[request_id]
                    self._requests[request_id] = self._load_data(data.get("value", None), buffers)
                    if key and data.get("cache", False):
                        self._cache_add(key, data.get("value", None))

                    if isinstance(event, asyncio.Future):
                        event.set_result(self._requests[request_id])
                    else:
                        event.set()
                    return

                case "call":
                    func = obj = self._get_obj(data)
                    args = self._load_data(data["args"], buffers)
                    response = func(*args)
                    try:
                        response = await response
                    except TypeError:
                        pass
                    except Exception as e:
                        print("error in call", type(e), str(e))

                case "get":
                    response = obj = self._get_obj(data)

                case "get_keys":
                    response = []

                case "set":
                    prop = data.pop("prop", None)
                    key = data.pop("key", None)
                    obj = self._get_obj(data)
                    if prop is not None:
                        obj.__setattr__(prop, data["value"])
                    elif key is not None:
                        obj[key] = self._load_data(data["value"], buffers)

                case "release_batch":
                    for id_ in data["ids"]:
                        self._objects.pop(id_, None)

                case _:
                    print("unknown message type", msg_type)

            if request_id is not None:
                self._send_response(request_id, response)
        except Exception as e:
            import sys
            import traceback

            print("error in on_message", data, obj, type(e), str(e), file=sys.stderr)
            if not isinstance(e, str):
                traceback.print_exception(*sys.exc_info(), file=sys.stderr)

    def _on_message(self, message: str):
        data, buffers = _unpack_message(message)
        try:
            msg_type = data.get("type", None)
            request_id = data.get("request_id", None)

            response = None

            match msg_type:
                case "response":
                    event, key = self._requests[request_id]
                    response = self._load_data(data.get("value", None), buffers)
                    self._requests[request_id] = response
                    if key and data.get("cache", False):
                        self._cache_add(key, data.get("value", None))
                    event.set()
                    return

                case "call":
                    func = self._get_obj(data)
                    args = self._load_data(data["args"], buffers)
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
                        obj[key] = self._load_data(data["value"], buffers)

                case "release_batch":
                    for id_ in data["ids"]:
                        self._objects.pop(id_, None)

                case _:
                    print("unknown message type", msg_type, data, type(message))

            if request_id is not None:
                self._send_response(request_id, response)
        except Exception as e:
            print("error in on_message", data, type(e), str(e))
            import traceback
            traceback.print_exception(*sys.exc_info())


class PyodideLink(LinkBase):
    def __init__(self, send_message):
        super().__init__()
        self._send_message = send_message
        self._requests = {}

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
        """Send data to the remote environment,
        if request_id is set, (blocking-)wait for the response and return it"""
        request_id = metadata.get("request_id", None)
        type = metadata.get("type", None)
        event = None
        self._send_message(data)
        if type != "response" and request_id is not None:
            # from pyodide.ffi import run_sync
            import asyncio

            event = asyncio.Future()
            self._requests[request_id] = event, key
            # todo: this shouldn't be necessary
            # but run_sync(event) gives an error
            while not event.done():
                time.sleep(0.001)

            return self._requests.pop(request_id)


class LinkBaseAsync(LinkBase):
    _send_loop: asyncio.AbstractEventLoop
    _callback_loop: asyncio.AbstractEventLoop
    _callback_queue: asyncio.Queue
    _callback_thread: threading.Thread
    _callback_task: asyncio.Task | None

    def __init__(self):
        super().__init__()
        self._send_loop = asyncio.new_event_loop()
        self._callback_loop = asyncio.new_event_loop()
        self._callback_task = None
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

        coro = self._send_async(data)
        try:
            asyncio.run_coroutine_threadsafe(coro, self._send_loop)
        except RuntimeError:
            coro.close()
            # Event loop is closed — connection is dead, clean up and bail.
            if event:
                self._requests.pop(request_id, None)
            return None
        if event:
            if not event.wait(timeout=120):
                self._requests.pop(request_id, None)
                raise TimeoutError(f"Request {request_id} timed out after 120s")
            return self._requests.pop(request_id)

    async def _send_async(self, message):
        raise NotImplementedError

    def _schedule_flush(self):
        """Schedule a fire-and-forget flush of the release queue via the send loop.
        Uses a 50ms debounce to let concurrent threads finish their operations."""
        coro = self._debounced_flush()
        try:
            asyncio.run_coroutine_threadsafe(coro, self._send_loop)
        except RuntimeError:
            coro.close()

    async def _debounced_flush(self):
        await asyncio.sleep(0.05)  # 50ms coalescing window
        await self._async_flush()

    async def _async_flush(self):
        if not self._release_queue:
            return
        ids = []
        while self._release_queue:
            ids.append(self._release_queue.popleft())
        if not ids:
            return
        # Only flush IDs that are safe: not in cache (as key or value) and refcount=0
        with self._refcount_lock:
            cache_key_ids = set(k[0] for k in self._cache)
            cached_value_ids = set()
            for v in self._cache.values():
                if isinstance(v, dict):
                    vid = v.get("id")
                    if vid is not None:
                        cached_value_ids.add(vid)
            safe_ids = [i for i in ids if i not in cache_key_ids
                        and i not in cached_value_ids
                        and self._js_id_refcount.get(i, 0) <= 0]
            # Put back unsafe IDs for later sync flush
            for i in ids:
                if i not in safe_ids:
                    self._release_queue.append(i)
        if not safe_ids:
            return
        data = {"type": "delete_batch", "ids": safe_ids}
        _, msg = _pack_message(data)
        await self._send_async(msg)

    def _start_callback_thread(self):
        async def handle_callbacks():
            while True:
                try:
                    func, args = await self._callback_queue.get()
                    func(*args)
                except asyncio.CancelledError:
                    break
                except RuntimeError:
                    break
                except asyncio.QueueEmpty:
                    pass
                except Exception as e:
                    print("error in callback", type(e), str(e))

        try:
            self._callback_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._callback_loop)
            self._callback_queue = asyncio.Queue()
            self._callback_task = self._callback_loop.create_task(handle_callbacks())
            try:
                self._callback_loop.run_forever()
            finally:
                if self._callback_task is not None and not self._callback_task.done():
                    self._callback_task.cancel()
                    self._callback_loop.run_until_complete(
                        asyncio.gather(self._callback_task, return_exceptions=True)
                    )
                self._callback_loop.close()
        except Exception as e:
            print("exception in _start_callback_thread", e)
