"""Tests for object lifecycle and memory management across the Python<->JS bridge.

These tests verify that:
- JS objects are released when Python drops Proxy references
- Python objects are released when JS drops proxy wrappers
- Repeated bridge operations don't leak unboundedly
- Concurrent proxy creation/destruction is safe

Many of these tests WILL FAIL until the memory management implementation is done.
That's intentional — this is TDD.
"""

import gc
import threading
import time

import numpy as np
import pytest


def js_object_count(page):
    """Count numeric-keyed entries in the JS CrossLink.objects registry."""
    return page.evaluate("""(() => {
        if (!window.__crosslink) return -1;
        let count = 0;
        for (const key of Object.keys(window.__crosslink.objects)) {
            if (!isNaN(key)) count++;
        }
        return count;
    })()""")


def py_object_count(link):
    """Count non-string-keyed entries in the Python _objects registry."""
    return sum(1 for k in link._objects if not isinstance(k, str))


class TestJSObjectRelease:
    """When Python drops a Proxy that references a JS object, JS should free it."""

    def test_js_objects_freed_after_proxy_gc(self, webgpu_env):
        """200 JS objects created via bridge must all be freed when Python drops them."""
        page = webgpu_env.page
        platform = webgpu_env.platform

        # Warm up: cache the document/createElement proxies on the Proxy-level cache
        _ = platform.js.document.createElement("div")
        del _
        gc.collect()
        _ = platform.js.document.title
        time.sleep(0.2)

        before = js_object_count(page)

        # Create 200 JS DOM elements via bridge
        proxies = []
        for _ in range(200):
            proxies.append(platform.js.document.createElement("div"))

        during = js_object_count(page)
        # Each createElement stores the result + intermediates (document, createElement ref).
        # At minimum 200 new entries for the div elements themselves.
        assert during >= before + 200

        # Drop all Python references and force GC
        del proxies
        gc.collect()
        gc.collect()

        # Trigger flush — a bridge call causes the release queue to be sent
        _ = platform.js.document.title
        time.sleep(0.3)

        after = js_object_count(page)
        assert after <= before, (
            f"Expected all JS objects freed (back to {before}), got {after} "
            f"(leaked {after - before})"
        )

    def test_function_call_results_freed(self, webgpu_env):
        """50 JS objects created and immediately dropped must all be freed."""
        page = webgpu_env.page
        platform = webgpu_env.platform

        before = js_object_count(page)

        for _ in range(50):
            el = platform.js.document.createElement("span")
            del el
        gc.collect()
        _ = platform.js.document.title  # flush
        time.sleep(0.3)

        after = js_object_count(page)
        assert after <= before, (
            f"50 create+delete cycles leaked {after - before} JS objects"
        )

    def test_primitive_return_does_not_store_js_objects(self, webgpu_env):
        """Accessing a property that returns a primitive should not grow JS object storage."""
        page = webgpu_env.page
        platform = webgpu_env.platform

        # Prime the navigator proxy so it's cached
        nav = platform.js.navigator
        gc.collect()
        _ = platform.js.document.title
        time.sleep(0.1)

        before = js_object_count(page)

        # userAgent is a string (primitive) — JS should not store it in objects
        for _ in range(100):
            _ = nav.userAgent

        gc.collect()
        _ = platform.js.document.title  # flush
        time.sleep(0.2)

        after = js_object_count(page)
        # Growth of 0 is ideal. The flush call itself may create 1 intermediate.
        assert after - before <= 1, (
            f"100 primitive property reads grew JS objects by {after - before}"
        )
        del nav


class TestPythonObjectRelease:
    """When JS drops a proxy to a Python object, Python should free it."""

    def test_python_callbacks_freed_on_explicit_destroy(self, webgpu_env):
        """50 proxies created via create_proxy must all be freed by destroy_proxy."""
        platform = webgpu_env.platform
        link = platform.link

        before = py_object_count(link)

        proxies = []
        for i in range(50):
            p = platform.create_proxy(lambda: i)
            proxies.append(p)

        during = py_object_count(link)
        assert during == before + 50

        for p in proxies:
            platform.destroy_proxy(p)

        after = py_object_count(link)
        assert after == before, (
            f"After destroying 50 proxies, expected {before}, got {after}"
        )

    def test_release_batch_frees_python_objects(self, webgpu_env):
        """A release_batch message must remove all specified objects from _objects."""
        platform = webgpu_env.platform
        link = platform.link

        # Insert 30 dummy objects with known IDs
        test_ids = []
        for i in range(30):
            obj = f"test_object_{i}"
            oid = id(obj)
            link._objects[oid] = obj
            test_ids.append(oid)

        # Simulate JS sending a release_batch message
        import json
        msg = json.dumps({"type": "release_batch", "ids": test_ids})
        link._on_message(msg)

        remaining = [oid for oid in test_ids if oid in link._objects]
        assert len(remaining) == 0, (
            f"{len(remaining)}/30 objects still present after release_batch"
        )


class TestNoLeakInRenderLoop:
    """Rendering repeatedly must not grow object registries."""

    def test_render_loop_no_growth(self, webgpu_env):
        """50 render cycles must not grow Python or JS object registries."""
        from webgpu.triangles import TriangulationRenderer

        page = webgpu_env.page
        platform = webgpu_env.platform
        link = platform.link

        webgpu_env.ensure_canvas(200, 200)
        pts = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
        scene = webgpu_env.wj.Draw(
            [TriangulationRenderer(pts, color=(1, 0, 0, 1))], width=200, height=200
        )
        # Let initialization settle
        time.sleep(0.5)
        gc.collect()
        _ = platform.js.document.title
        time.sleep(0.2)

        py_before = py_object_count(link)
        js_before = js_object_count(page)

        for _ in range(50):
            scene._render_objects(to_canvas=False)
        gc.collect()
        _ = platform.js.document.title  # flush
        time.sleep(0.5)

        py_after = py_object_count(link)
        js_after = js_object_count(page)

        assert py_after == py_before, (
            f"Python objects grew by {py_after - py_before} over 50 renders"
        )
        assert js_after == js_before, (
            f"JS objects grew by {js_after - js_before} over 50 renders"
        )

    def test_buffer_write_loop_no_growth(self, webgpu_env):
        """100 buffer writes must not grow object registries."""
        from webgpu.utils import get_device
        from webgpu.webgpu_api import BufferUsage

        page = webgpu_env.page
        platform = webgpu_env.platform
        link = platform.link

        device = get_device()
        buf = device.createBuffer(
            size=1024,
            usage=BufferUsage.COPY_DST | BufferUsage.STORAGE,
            label="leak_test_buf",
        )

        gc.collect()
        _ = platform.js.document.title
        time.sleep(0.2)

        py_before = py_object_count(link)
        js_before = js_object_count(page)

        data = np.zeros(256, dtype=np.float32)
        for i in range(100):
            data[0] = float(i)
            device.queue.writeBuffer(buf, 0, data.tobytes())
        gc.collect()
        _ = platform.js.document.title
        time.sleep(0.3)

        py_after = py_object_count(link)
        js_after = js_object_count(page)

        assert py_after == py_before, (
            f"Python objects grew by {py_after - py_before} over 100 buffer writes"
        )
        assert js_after == js_before, (
            f"JS objects grew by {js_after - js_before} over 100 buffer writes"
        )


class TestConcurrentProxyLifecycle:
    """Thread safety: concurrent creation and destruction of proxies must not crash."""

    def test_concurrent_proxy_create_destroy(self, webgpu_env):
        """4 threads each creating+destroying 100 proxies must not raise."""
        platform = webgpu_env.platform
        errors = []

        def worker(thread_id):
            try:
                for i in range(100):
                    el = platform.js.document.createElement("div")
                    _ = el.tagName
                    del el
                gc.collect()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Threads raised exceptions: {errors}"

    def test_gc_concurrent_with_bridge_calls(self, webgpu_env):
        """GC running on one thread while another does bridge calls must not crash."""
        platform = webgpu_env.platform
        errors = []
        stop = threading.Event()

        def gc_thread():
            try:
                while not stop.is_set():
                    gc.collect()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("gc", e))

        def work_thread():
            try:
                for _ in range(50):
                    _ = platform.js.document.title
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("work", e))

        gc_t = threading.Thread(target=gc_thread)
        work_t = threading.Thread(target=work_thread)

        gc_t.start()
        work_t.start()
        work_t.join(timeout=30)
        stop.set()
        gc_t.join(timeout=5)

        assert not errors, f"Threads raised exceptions: {errors}"


class TestCacheEviction:
    """Cache entries for dead objects must be evicted."""

    def test_cache_entries_evicted_for_dead_objects(self, webgpu_env):
        """Accessing properties on objects that are then GC'd must not grow cache."""
        platform = webgpu_env.platform
        link = platform.link

        cache_before = len(link._cache)

        # 5 rounds: create 20 objects, access 2 props each (= 200 cache entries), delete all
        for _ in range(5):
            elements = []
            for _ in range(20):
                el = platform.js.document.createElement("div")
                elements.append(el)
            for el in elements:
                _ = el.tagName
                _ = el.nodeName
            del elements
            gc.collect()
            _ = platform.js.document.title  # flush
            time.sleep(0.2)

        cache_after = len(link._cache)
        assert cache_after == cache_before, (
            f"Cache grew by {cache_after - cache_before} entries for dead objects"
        )


class TestExplicitCleanup:
    """Explicit cleanup APIs must release resources."""

    def test_scene_cleanup_frees_render_proxy(self, webgpu_env):
        """scene.cleanup() must reduce Python object count (frees _js_render proxy)."""
        from webgpu.triangles import TriangulationRenderer

        platform = webgpu_env.platform
        link = platform.link

        webgpu_env.ensure_canvas(200, 200)
        pts = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
        scene = webgpu_env.wj.Draw(
            [TriangulationRenderer(pts, color=(0, 1, 0, 1))], width=200, height=200
        )
        time.sleep(0.3)

        py_before = py_object_count(link)
        scene.cleanup()

        py_after = py_object_count(link)
        # cleanup destroys _js_render proxy, so count must decrease
        assert py_after < py_before, (
            f"Expected object count to decrease after cleanup, "
            f"went from {py_before} to {py_after}"
        )

    def test_destroy_proxy_removes_from_python_objects(self, webgpu_env):
        """destroy_proxy must remove the exact object from _objects."""
        platform = webgpu_env.platform
        link = platform.link

        proxy = platform.create_proxy(lambda x: x * 2)
        proxy_id = proxy["id"]
        assert proxy_id in link._objects

        platform.destroy_proxy(proxy)
        assert proxy_id not in link._objects


class TestSynchronization:
    """Races and synchronization issues in the bridge."""

    def test_callback_during_destroy_does_not_crash(self, webgpu_env):
        """If JS invokes a Python callback while another thread destroys it, must not crash."""
        platform = webgpu_env.platform
        link = platform.link
        errors = []
        call_count = [0]

        def callback():
            call_count[0] += 1

        proxy = platform.create_proxy(callback)
        proxy_id = proxy["id"]

        # Have JS fire-and-forget call this callback rapidly
        webgpu_env.page.evaluate(f"""
            window.__race_stop = false;
            window.__race_id = {proxy_id};
            void (async () => {{
                while (!window.__race_stop) {{
                    window.__crosslink.callIgnoreResult(window.__race_id, []);
                    await new Promise(r => setTimeout(r, 5));
                }}
            }})();
        """)
        time.sleep(0.2)  # let calls accumulate

        # Destroy the callback from Python while JS is still calling it
        platform.destroy_proxy(proxy)
        time.sleep(0.2)
        webgpu_env.page.evaluate("window.__race_stop = true;")
        time.sleep(0.1)

        # Some calls succeeded before destroy, and the bridge didn't crash
        assert call_count[0] > 0, "Callback was never invoked"

    def test_concurrent_get_set_same_property(self, webgpu_env):
        """Concurrent get/set on the same JS property must not crash or hang."""
        platform = webgpu_env.platform
        errors = []
        stop = threading.Event()

        # Create a JS object to use as shared state
        obj = platform.js.document.createElement("div")

        def writer():
            try:
                for i in range(50):
                    obj.title = f"value_{i}"
                    if stop.is_set():
                        break
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(50):
                    _ = obj.title
                    if stop.is_set():
                        break
            except Exception as e:
                errors.append(("reader", e))

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        stop.set()

        assert not errors, f"Concurrent get/set raised: {errors}"
        del obj

    def test_request_timeout_does_not_hang_forever(self, webgpu_env):
        """A bridge call to a non-existent object must not hang indefinitely."""
        platform = webgpu_env.platform
        link = platform.link

        # Call a non-existent function ID — JS will error and may not respond.
        # Python should not hang forever.
        import signal

        hung = [False]

        def handler(signum, frame):
            hung[0] = True
            raise TimeoutError("Bridge call hung")

        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)  # 5 second deadline

        try:
            # Call an object ID that doesn't exist on JS side
            try:
                link.call(999888777, args=[], parent_id=None, ignore_result=False)
            except Exception:
                pass  # Any exception is fine — just must not hang
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        assert not hung[0], "Bridge call hung forever — no timeout on event.wait()"


class TestDeleteBatchProtocol:
    """The delete_batch wire protocol must work end-to-end."""

    def test_js_processes_delete_batch(self, webgpu_env):
        """Sending delete_batch from Python must remove those entries from JS objects."""
        page = webgpu_env.page
        platform = webgpu_env.platform

        # Insert 3 objects with known IDs directly into JS
        page.evaluate("""(() => {
            window.__crosslink.objects[99901] = {test: 1};
            window.__crosslink.objects[99902] = {test: 2};
            window.__crosslink.objects[99903] = {test: 3};
        })()""")
        assert page.evaluate("99901 in window.__crosslink.objects") is True

        # Send delete_batch from Python
        from webgpu.link.base import _pack_message
        platform.link._send_data(
            *_pack_message({"type": "delete_batch", "ids": [99901, 99902, 99903]})
        )
        time.sleep(0.2)

        # All 3 must be gone
        assert page.evaluate("99901 in window.__crosslink.objects") is False
        assert page.evaluate("99902 in window.__crosslink.objects") is False
        assert page.evaluate("99903 in window.__crosslink.objects") is False

    def test_delete_batch_uses_delete_operator(self, webgpu_env):
        """delete_batch must use JS `delete` (remove key), not assign undefined."""
        page = webgpu_env.page
        platform = webgpu_env.platform

        page.evaluate("window.__crosslink.objects[99999] = {x: 1};")

        from webgpu.link.base import _pack_message
        platform.link._send_data(
            *_pack_message({"type": "delete_batch", "ids": [99999]})
        )
        time.sleep(0.2)

        has_key = page.evaluate(
            "window.__crosslink.objects.hasOwnProperty('99999')"
        )
        assert has_key is False, (
            "Key still exists — delete_batch must use `delete`, not `= undefined`"
        )
