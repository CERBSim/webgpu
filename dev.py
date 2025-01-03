import asyncio
import http.server
import socketserver
from threading import Timer

try:
    import websockets
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    _have_dev_dependencies = True
except:
    print("watchdog and/or websockets are not installed, no hot-reloading support")
    _have_dev_dependencies = False


# Disable caching in HTTP server
class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )
        super().end_headers()


def run_http_server():
    running = False
    PORT = 8000
    while not running:
        try:
            Handler = NoCacheHTTPRequestHandler
            httpd = socketserver.TCPServer(("", PORT), Handler)
            print(f"Serving HTTP on port {PORT}")
            httpd.serve_forever()
            running = True
        except OSError as e:
            if e.errno == 98:
                PORT += 1
            else:
                raise


clients = set()


async def websocket_handler(websocket, path=None):
    clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        clients.remove(websocket)


async def notify_clients(message):
    print(f"notify {len(clients)} clients", message)
    if clients:  # Send the message to all connected clients
        for client in clients:
            await client.send(message)


class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
        self.debounce_timer = None
        self.last_event = None

    def _debounced_notify(self):
        if self.last_event:
            asyncio.run_coroutine_threadsafe(notify_clients("update"), self.loop)

    def on_any_event(self, event):
        if event.event_type != "closed":
            return
        self.last_event = event
        if self.debounce_timer:
            self.debounce_timer.cancel()

        self.debounce_timer = Timer(0.1, self._debounced_notify)
        self.debounce_timer.start()


async def main():
    loop = asyncio.get_running_loop()
    from threading import Thread

    if _have_dev_dependencies:
        await websockets.serve(websocket_handler, "localhost", 6789)

    http_thread = Thread(target=run_http_server)
    http_thread.start()

    if _have_dev_dependencies:
        event_handler = FileChangeHandler(loop)
        observer = Observer()
        observer.schedule(event_handler, path="webgpu", recursive=True)
        observer.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        if _have_dev_dependencies:
            observer.stop()
    if _have_dev_dependencies:
        observer.join()
    http_thread.join()


if __name__ == "__main__":
    asyncio.run(main())
