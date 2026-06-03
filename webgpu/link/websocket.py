import asyncio
import json
import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlparse

# Browser WebSocket handshakes automatically attach cookies for the target
# (loopback) host, and the JS WebSocket API exposes no way to omit them. On
# machines with large cookies for localhost/127.0.0.1 the resulting Cookie
# header line can exceed websockets' default 8192-byte limit, making the
# handshake fail with SecurityError("line too long"). Raise the limit before
# importing websockets (the value is read at import time of websockets.http11).
# Note: only honored on websockets versions that support this env var.
os.environ.setdefault("WEBSOCKETS_MAX_LINE_LENGTH", "32768")

import websockets
from websockets.http11 import Response
from websockets.datastructures import Headers

from .base import LinkBaseAsync, _unpack_message


class WebsocketLinkBase(LinkBaseAsync):
    _websocket_thread: threading.Thread
    _connection: object
    _event_is_connected: threading.Event
    _event_is_running: threading.Event
    _start_handling_messages: threading.Event

    def __init__(self):
        super().__init__()
        self._connection = None
        self._event_is_connected = threading.Event()
        self._event_is_running = threading.Event()
        self._start_handling_messages = threading.Event()
        self._send_lock = asyncio.Lock()

        self._websocket_thread = threading.Thread(target=self._connect, daemon=True)
        self._websocket_thread.start()

    def wait_for_server_running(self):
        self._event_is_running.wait()

    def wait_for_connection(self):
        self._event_is_connected.wait()

    async def _send_async(self, message):
        if not self._connection:
            raise Exception("Websocket not connected")
        async with self._send_lock:
            await self._connection.send(message)

    def _connect(self):
        raise NotImplementedError


class WebsocketLinkServer(WebsocketLinkBase):
    _stop: asyncio.Future
    _port: int = None
    _auth_token: str

    def __init__(self):
        self._port = 8700
        self._auth_token = secrets.token_urlsafe(32)
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._chunk_buffers = {}
        self._stop = None
        super().__init__()

    @property
    def auth_token(self):
        return self._auth_token

    @property
    def port(self):
        return self._port

    def _check_auth(self, connection, request):
        """Reject WebSocket connections that don't carry a valid token."""
        params = parse_qs(urlparse(request.path).query)
        tokens = params.get("token", [])
        if not tokens or not secrets.compare_digest(tokens[0], self._auth_token):
            return Response(403, "Forbidden", Headers())
        return None

    @staticmethod
    def _message_type(message):
        """Return the top-level message type, parsing only the JSON header
        (not buffer payloads). Returns None on malformed input."""
        try:
            if isinstance(message, (memoryview, bytes)):
                prefix_size = 4 + int.from_bytes(message[:4], byteorder="little")
                header = json.loads(bytes(message[4:prefix_size]).decode("utf-8"))
            else:
                header = json.loads(message)
            return header.get("type") if isinstance(header, dict) else None
        except Exception:
            return None

    def _is_response(self, message):
        return self._message_type(message) == "response"

    def _is_chunk(self, message):
        return isinstance(message, (memoryview, bytes)) and self._message_type(message) == "chunk"

    def _reassemble_chunk(self, message):
        data, buffers = _unpack_message(message)
        pid = data["parent_request_id"]
        buf = self._chunk_buffers.get(pid)
        if buf is None:
            buf = bytearray(data["total_size"])
            self._chunk_buffers[pid] = buf
        chunk = buffers[0]
        offset = data["offset"]
        buf[offset : offset + len(chunk)] = chunk
        if data["chunk_id"] + 1 == data["n_chunks"]:
            del self._chunk_buffers[pid]
            return bytes(buf)
        return None

    def _dispatch(self, message):
        if self._is_response(message):
            self._on_message(message)
        else:
            self._executor.submit(self._on_message, message)

    async def _websocket_handler(self, websocket, path=""):
        if self._connection is not None:
            await websocket.close(4000, "Another session is already active")
            return
        try:
            self._connection = websocket
            self._event_is_connected.set()
            async for message in websocket:
                # Handle responses inline to avoid deadlock: if all executor
                # threads are blocked waiting for JS responses, queued response
                # messages would never be processed. Chunks are reassembled
                # inline (single-threaded, ordered) then dispatched.
                if self._is_chunk(message):
                    full = self._reassemble_chunk(message)
                    if full is not None:
                        self._dispatch(full)
                else:
                    self._dispatch(message)
        finally:
            self._connection = None
            self._chunk_buffers.clear()

    def _connect(self):
        async def start_websocket():
            self._stop = asyncio.get_event_loop().create_future()
            while True:
                try:
                    async with websockets.serve(
                        self._websocket_handler,
                        "127.0.0.1",
                        self._port,
                        max_size=110 * 1024**2,  # slightly above 100 MB chunk size
                        compression=None,
                        process_request=self._check_auth,
                    ):
                        self._event_is_running.set()
                        await self._stop
                        break
                except OSError as e:
                    self._port += 1
                except Exception as e:
                    print("error in websocket server", e)

        try:
            asyncio.set_event_loop(self._send_loop)
            self._send_loop.run_until_complete(start_websocket())
        except Exception as e:
            print("exception in _start_websocket_server", e)
        finally:
            pending = [
                task for task in asyncio.all_tasks(self._send_loop)
                if not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                self._send_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._send_loop.run_until_complete(self._send_loop.shutdown_asyncgens())
            self._send_loop.run_until_complete(self._send_loop.shutdown_default_executor())
            self._send_loop.close()

    def stop(self):
        self._executor.shutdown(wait=False)
        try:
            if self._stop is not None and not self._stop.done():
                self._send_loop.call_soon_threadsafe(self._stop.set_result, None)
        except RuntimeError:
            pass  # Event loop already closed
        if threading.current_thread() is not self._websocket_thread:
            self._websocket_thread.join(timeout=2)

        # Stop the callback event loop so the _callback_thread exits.
        try:
            self._callback_loop.call_soon_threadsafe(self._callback_loop.stop)
        except RuntimeError:
            pass  # Event loop already closed
        if threading.current_thread() is not self._callback_thread:
            self._callback_thread.join(timeout=1)

        # Unblock any threads stuck waiting for websocket RPC responses.
        for rid, val in list(self._requests.items()):
            if isinstance(val, tuple):
                event, key = val
                if isinstance(event, threading.Event):
                    event.set()
