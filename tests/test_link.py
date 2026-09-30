"""Delivery order of calls over the websocket link.

``TestServer`` drives a bare WebsocketLinkServer with a Python websocket
client, the other tests go through link.js in the browser.
"""

import json
import threading
import time

import pytest


def _wait(cond, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Recorder:
    def __init__(self, work=None):
        self.started = []
        self.done = []
        self.work = work
        self.running = 0
        self.overlap = False
        self._lock = threading.Lock()

    def ev(self, i, *rest):
        with self._lock:
            self.running += 1
            self.overlap |= self.running > 1
            self.started.append(i)
        try:
            if self.work:
                self.work(i)
        finally:
            with self._lock:
                self.running -= 1
                self.done.append(i)

    def answer(self):
        return 42


@pytest.fixture
def server():
    from websockets.sync.client import connect

    from webgpu.link.websocket import WebsocketLinkServer

    srv = WebsocketLinkServer()
    srv.wait_for_server_running()
    ws = connect(f"ws://127.0.0.1:{srv.port}?token={srv.auth_token}")
    srv.wait_for_connection()
    srv.client = ws
    yield srv
    ws.close()
    srv.stop()


def _send_calls(ws, id_, n, prop=None, **flags):
    for i in range(n):
        msg = {"type": "call", "id": id_, "args": [i]} | flags
        if prop:
            msg["prop"] = prop
        ws.send(json.dumps(msg))


class TestServer:
    def test_proxy_calls_keep_wire_order(self, server):
        rec = Recorder()
        handle = server.create_proxy(rec.ev, True)
        _send_calls(server.client, handle["id"], 500)
        assert _wait(lambda: len(rec.done) == 500)
        assert rec.done == list(range(500))

    def test_ordered_calls_run_serially_in_order(self, server):
        rec = Recorder(lambda i: time.sleep(0.001))
        server.expose("rec", rec)
        _send_calls(server.client, "rec", 300, prop="ev", ordered=True)
        assert _wait(lambda: len(rec.done) == 300)
        assert rec.started == list(range(300))
        assert not rec.overlap

    def test_ordered_methods(self, server):
        rec = Recorder(lambda i: time.sleep(0.001))
        server.expose("rec", rec)
        server.ordered_methods.add("ev")
        _send_calls(server.client, "rec", 200, prop="ev")
        assert _wait(lambda: len(rec.done) == 200)
        assert rec.started == list(range(200))
        assert not rec.overlap

    def test_unordered_calls_still_concurrent(self, server):
        rec = Recorder(lambda i: time.sleep(0.2))
        server.expose("rec", rec)
        t0 = time.time()
        _send_calls(server.client, "rec", 4, prop="ev")
        assert _wait(lambda: len(rec.done) == 4)
        assert time.time() - t0 < 0.6
        assert rec.overlap

    def test_slow_ordered_handler_releases_lane(self, server):
        server.ordered_timeout = 0.2
        rec = Recorder(lambda i: time.sleep(1.5) if i == 0 else None)
        server.expose("rec", rec)
        _send_calls(server.client, "rec", 5, prop="ev", ordered=True)
        assert _wait(lambda: len(rec.done) == 4, timeout=1.0)
        assert rec.started == list(range(5))
        assert rec.done == [1, 2, 3, 4]
        assert _wait(lambda: len(rec.done) == 5)

    def test_coalesce_keeps_newest(self, server):
        rec = Recorder(lambda i: time.sleep(0.05))
        server.expose("rec", rec)
        _send_calls(server.client, "rec", 40, prop="ev", ordered=True, coalesce="move")
        assert _wait(lambda: rec.done and rec.done[-1] == 39)
        time.sleep(0.1)
        assert len(rec.done) < 40
        assert rec.done == sorted(rec.done)

    def test_disconnect_and_stop(self, server):
        rec = Recorder(lambda i: time.sleep(0.05))
        server.expose("rec", rec)
        _send_calls(server.client, "rec", 5, prop="ev", ordered=True)
        server.client.close()
        # already received events still run, in order
        assert _wait(lambda: len(rec.done) == 5)
        assert rec.done == list(range(5))
        assert _wait(lambda: server._connection is None)
        lane = server._ordered._thread
        server.stop()
        lane.join(timeout=2)
        assert not lane.is_alive()
        server._ordered.push("{}")
        assert not server._ordered._queue


@pytest.fixture
def js_rec(webgpu_env):
    pl = webgpu_env.platform
    rec = Recorder()
    pl.js.window.linkTestRec = rec
    pl.js.window.linkTestSink = pl.create_proxy(rec.ev, True)
    pl.js.eval("0")  # sets are fire-and-forget, a round trip flushes them
    yield rec
    webgpu_env.page.evaluate("delete window.linkTestRec; delete window.linkTestSink")


class TestBrowser:
    def test_proxy_calls_keep_js_call_order(self, webgpu_env, js_rec):
        # args of different depth take different numbers of microtasks to serialize
        webgpu_env.page.evaluate("""() => {
            for (let i = 0; i < 400; i++)
                if (i % 2) window.linkTestSink(i, [[[[i]]]], {a: [[i]]});
                else window.linkTestSink(i);
        }""")
        assert _wait(lambda: len(js_rec.done) == 400)
        assert js_rec.done == list(range(400))

    def test_ordered_method_calls(self, webgpu_env, js_rec):
        js_rec.work = lambda i: time.sleep(0.001)
        webgpu_env.page.evaluate("""() => {
            for (let i = 0; i < 300; i++)
                window.linkTestRec.callMethodIgnoreResult('ev', [i], {ordered: true});
        }""")
        assert _wait(lambda: len(js_rec.done) == 300)
        assert js_rec.started == list(range(300))
        assert not js_rec.overlap

    def test_nested_round_trips_in_ordered_handler(self, webgpu_env, js_rec):
        pl = webgpu_env.platform
        pl.js.eval("window.linkTestReenter = async () => await window.linkTestRec.callMethod('answer', [])")
        results = []

        def work(i):
            results.append((pl.js.eval(f"{i} + 1"), pl.js.linkTestReenter()))

        js_rec.work = work
        webgpu_env.page.evaluate("""() => {
            for (let i = 0; i < 20; i++)
                window.linkTestRec.callMethodIgnoreResult('ev', [i], {ordered: true});
        }""")
        assert _wait(lambda: len(js_rec.done) == 20)
        assert results == [(i + 1, 42) for i in range(20)]
        assert js_rec.done == list(range(20))

    def test_serialized_event_has_timestamp(self, webgpu_env, js_rec):
        events = []
        js_rec.work = None
        webgpu_env.platform.js.window.linkTestEvent = webgpu_env.platform.create_proxy(
            lambda ev: events.append(ev), True
        )
        webgpu_env.platform.js.eval("0")
        webgpu_env.page.evaluate("() => window.linkTestEvent(new KeyboardEvent('keydown', {key: 'a'}))")
        assert _wait(lambda: events)
        assert events[0]["key"] == "a" and events[0]["timeStamp"] > 0
