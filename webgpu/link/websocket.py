import asyncio
import json
import threading

import websockets
import websockets.asyncio.client

from .base import LinkBaseAsync


class WebsocketLinkBase(LinkBaseAsync):
    _websocket_thread: threading.Thread
    _connection: websockets.asyncio.client.ClientConnection
    _event_is_connected: threading.Event
    _start_handling_messages: threading.Event

    def __init__(self):
        super().__init__()
        self._event_is_connected = threading.Event()
        self._start_handling_messages = threading.Event()
        self._send_loop = asyncio.new_event_loop()

        self._websocket_thread = threading.Thread(target=self._connect)
        self._websocket_thread.start()

    def wait_for_connection(self):
        self._event_is_connected.wait()

    async def _send_async(self, message):
        if self._connection:
            await self._connection.send(message)
        else:
            raise Exception("Websocket not connected")

    def _connect(self):
        raise NotImplementedError


class WebsocketLinkClient(WebsocketLinkBase):
    _url: str

    def __init__(self, url):
        super().__init__()
        self._url = url

    def _connect(self):
        async def start_websocket():
            async for websocket in websockets.connect(self._url):
                try:
                    # print("client connected")
                    self._connection = websocket
                    self._event_is_connected.set()
                    self._start_handling_messages.wait()
                    async for message in websocket:
                        self._on_message(message)
                except websockets.exceptions.ConnectionClosed:
                    continue
                except Exception:
                    break

        try:
            asyncio.set_event_loop(self._send_loop)
            self._send_loop.run_until_complete(start_websocket())
        except Exception:
            print("closing connection")


class WebsocketLinkServer(WebsocketLinkBase):
    _stop: asyncio.Future
    _port: int = 0

    def __init__(self):
        super().__init__()
        self._stop = self._send_loop.create_future()
        self._port = 8700

    @property
    def port(self):
        return self._port

    async def _websocket_handler(self, websocket, path=""):
        try:
            # print("client connected")
            self._connection = websocket
            self._event_is_connected.set()
            async for message in websocket:
                thread = threading.Thread(target=self._on_message, args=(message,))
                thread.start()
        finally:
            self._connection = None

    def _connect(self):
        async def start_websocket():
            while True:
                try:
                    async with websockets.serve(
                        self._websocket_handler, "", self._port, max_queue=128
                    ):
                        # print("server running on port", self._port)
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
        print("stopped websocket")

    def stop(self):
        self._send_loop.call_soon_threadsafe(self._stop.set_result, None)
