// Input handling for WebGPU canvas
// Port of camera mouse handling from webgpu/camera.py

class InputHandler {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {import('./camera.js').Camera} camera
   * @param {() => void} onRender — called after every camera change
   * @param {((x: number, y: number) => number[]|null)|null} getPositionFn — for dblclick center
   */
  constructor(canvas, camera, onRender, getPositionFn = null) {
    this.canvas = canvas;
    this.camera = camera;
    this.onRender = onRender;
    this.getPositionFn = getPositionFn;

    // Live mode: sink for classified non-camera events forwarded to the host.
    this._eventSink = null;
    // High-frequency move events (drag/hover) use true backpressure: at most one
    // is in flight to the host (Python) at a time. While one is being processed,
    // further moves are coalesced into a single pending event (deltas summed so
    // the rotation/pan total is preserved); when the host acks, the latest
    // pending move is sent. This caps the event rate to the host's actual
    // processing rate, so a fast drag can't pile up a backlog that replays slowly.
    this._pendingMove = null;
    this._moveInFlight = false;
    this._moveWatchdog = null;
    this.onGestureEnd = null;
    this.rotateSensitivity = 0.5;

    this._downButton = 0;
    this._downX = 0;
    this._downY = 0;
    this._moved = false;

    this._isRotating = false;
    this._isPanning = false;

    // Multi-touch gesture state
    this._activeTouches = new Map(); // id -> {x, y}
    this._prevTouchDist = null;
    this._prevTouchAngle = null;
    this._prevTouchCentroid = null;

    // Prevent browser scroll/zoom on the canvas
    canvas.style.touchAction = 'none';

    // Bind handlers for clean removal
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerLeave = this._onPointerLeave.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onDblClick = this._onDblClick.bind(this);
    this._onContextMenu = (ev) => ev.preventDefault();
    this._onTouchStart = this._onTouchStart.bind(this);
    this._onTouchMove = this._onTouchMove.bind(this);
    this._onTouchEnd = this._onTouchEnd.bind(this);

    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointerup', this._onPointerUp);
    canvas.addEventListener('pointermove', this._onPointerMove);
    canvas.addEventListener('pointerleave', this._onPointerLeave);
    canvas.addEventListener('wheel', this._onWheel, { passive: false });
    canvas.addEventListener('dblclick', this._onDblClick);
    canvas.addEventListener('contextmenu', this._onContextMenu);
    canvas.addEventListener('touchstart', this._onTouchStart, { passive: false });
    canvas.addEventListener('touchmove', this._onTouchMove, { passive: false });
    canvas.addEventListener('touchend', this._onTouchEnd);
    canvas.addEventListener('touchcancel', this._onTouchEnd);
  }

  setEventSink(fn) {
    this._eventSink = fn;
  }

  _buildPayload(type, ev) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    return {
      type,
      button: ev.button == null ? 0 : ev.button,
      buttons: ev.buttons == null ? 0 : ev.buttons,
      x: ev.clientX,
      y: ev.clientY,
      canvasX: Math.round((ev.clientX - rect.left) * dpr),
      canvasY: Math.round((ev.clientY - rect.top) * dpr),
      movementX: ev.movementX || 0,
      movementY: ev.movementY || 0,
      deltaX: ev.deltaX || 0,
      deltaY: ev.deltaY || 0,
      ctrlKey: !!ev.ctrlKey,
      shiftKey: !!ev.shiftKey,
      altKey: !!ev.altKey,
    };
  }

  _send(payload) {
    try {
      this._eventSink(payload);
    } catch (e) {
      console.warn('[input] event sink failed:', e && (e.message || e));
    }
  }

  // Discrete events (click, dblclick, wheel, mouseout): flush any pending
  // coalesced move first so ordering is preserved, then send immediately.
  _forward(type, ev) {
    if (!this._eventSink) return;
    this._flushMove();
    this._send(this._buildPayload(type, ev));
  }

  // High-frequency move events (drag/hover): backpressured. Accumulate deltas
  // into a single pending event while one is in flight; send it once the host
  // acks the previous one.
  _forwardMove(type, ev) {
    if (!this._eventSink) return;
    const payload = this._buildPayload(type, ev);
    const prev = this._pendingMove;
    if (prev && prev.type === type) {
      payload.movementX += prev.movementX;
      payload.movementY += prev.movementY;
    }
    this._pendingMove = payload;
    this._flushMove();
  }

  _flushMove() {
    if (this._moveInFlight) return;          // wait for the host's ack
    const p = this._pendingMove;
    if (!p) return;
    this._pendingMove = null;
    this._moveInFlight = true;
    this._send(p);
    // Watchdog: the host acks via ackInput() once it has processed the event.
    // If that ack is ever lost, don't stall input forever.
    if (typeof setTimeout !== 'undefined') {
      this._moveWatchdog = setTimeout(() => this._releaseMove(), 1000);
    }
  }

  // Called (via engine.ackInput) once the host finished the in-flight move.
  _releaseMove() {
    if (!this._moveInFlight) return;
    this._moveInFlight = false;
    if (this._moveWatchdog != null) {
      clearTimeout(this._moveWatchdog);
      this._moveWatchdog = null;
    }
    if (!this._pendingMove) return;
    // Defer so we never recurse synchronously when the host (Pyodide) acks
    // inline within the same call stack.
    if (typeof Promise !== 'undefined') {
      Promise.resolve().then(() => this._flushMove());
    } else {
      this._flushMove();
    }
  }

  // --- Pointer events (mouse & single touch fallback) ---

  _onPointerDown(ev) {
    ev.preventDefault();
    this._downButton = ev.button;
    this._downX = ev.clientX;
    this._downY = ev.clientY;
    this._moved = false;
    // ctrl/alt are reserved for the host and never start a camera gesture.
    const hostModified = ev.ctrlKey || ev.altKey;
    if (!hostModified && ev.button === 0 && !ev.shiftKey) {
      this._isRotating = true;
    } else if (!hostModified && (ev.button === 1 || (ev.button === 0 && ev.shiftKey))) {
      this._isPanning = true;
    }
    this.canvas.setPointerCapture(ev.pointerId);
  }

  _onPointerUp(ev) {
    const wasGesture = this._isRotating || this._isPanning;
    this._isRotating = false;
    this._isPanning = false;
    // Apply any pending coalesced drag before the gesture ends, so the final
    // pointer position isn't dropped.
    this._flushMove();
    // A press with no movement is a click; camera gestures set _moved.
    if (this._eventSink && !this._moved && ev.button === this._downButton) {
      this._forward('click', ev);
    }
    if (wasGesture && this._moved && this.onGestureEnd) {
      try { this.onGestureEnd(); } catch (e) { /* ignore */ }
    }
  }

  _onPointerMove(ev) {
    // Suppress pointer-based rotation/pan while a multi-touch gesture is active
    if (this._activeTouches.size >= 2) return;

    // On high-DPI screens (e.g. macOS Retina) the OS reports pointer movement
    // in device pixels, which are then divided by devicePixelRatio to produce
    // CSS-pixel deltas.  Multiply back by DPR so that the same physical finger
    // movement always produces the same camera rotation/pan, regardless of
    // the screen's pixel density.
    const dpr = window.devicePixelRatio || 1;
    const t = this.camera.transform;
    if (this._isRotating) {
      this._moved = true;
      const s = this.rotateSensitivity * dpr;
      t.rotate(s * ev.movementY, s * ev.movementX);
      this.camera._notify();
      this.onRender();
      return;
    }
    if (this._isPanning) {
      this._moved = true;
      t.translate(0.01 * dpr * ev.movementX, -0.01 * dpr * ev.movementY);
      this.camera._notify();
      this.onRender();
      return;
    }
    // Not a camera gesture — forward to the host (hover, or a modified drag).
    // Coalesced to one event per frame so a fast drag doesn't pile up a backlog.
    if (!this._eventSink) return;
    if (ev.buttons !== 0) {
      this._moved = true;
      this._forwardMove('drag', ev);
    } else {
      this._forwardMove('mousemove', ev);
    }
  }

  _onPointerLeave(ev) {
    if (this._eventSink) this._forward('mouseout', ev);
  }

  _onWheel(ev) {
    ev.preventDefault();
    if (this._eventSink && (ev.ctrlKey || ev.altKey)) {
      this._forward('wheel', ev);
      return;
    }
    const t = this.camera.transform;
    t.scale(1 - ev.deltaY / 1000, t._center);
    this.camera._notify();
    this.onRender();
  }

  _onDblClick(ev) {
    if (this._eventSink) this._forward('dblclick', ev);
    if (!this.getPositionFn) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const p = this.getPositionFn(x, y);
    if (p != null) {
      this.camera.transform.setCenter(p);
      this.camera._notify();
      this.onRender();
    }
  }

  // --- Multi-touch gesture handling ---

  _onTouchStart(ev) {
    ev.preventDefault();
    for (const touch of ev.changedTouches) {
      this._activeTouches.set(touch.identifier, { x: touch.clientX, y: touch.clientY });
    }
    if (this._activeTouches.size >= 2) {
      this._initGestureState();
    }
  }

  _onTouchMove(ev) {
    ev.preventDefault();
    for (const touch of ev.changedTouches) {
      if (this._activeTouches.has(touch.identifier)) {
        this._activeTouches.set(touch.identifier, { x: touch.clientX, y: touch.clientY });
      }
    }
    if (this._activeTouches.size >= 2) {
      this._handleMultiTouchGesture();
    }
  }

  _onTouchEnd(ev) {
    for (const touch of ev.changedTouches) {
      this._activeTouches.delete(touch.identifier);
    }
    // Reset gesture state when fewer than 2 touches remain
    if (this._activeTouches.size < 2) {
      this._prevTouchDist = null;
      this._prevTouchAngle = null;
      this._prevTouchCentroid = null;
    }
  }

  /** Compute initial distance, angle, and centroid for a two-finger gesture. */
  _initGestureState() {
    const [a, b] = this._getTwoTouches();
    this._prevTouchDist = this._distance(a, b);
    this._prevTouchAngle = this._angle(a, b);
    this._prevTouchCentroid = this._centroid(a, b);
  }

  /** Process ongoing two-finger gesture: pinch-zoom, rotation, and pan. */
  _handleMultiTouchGesture() {
    const [a, b] = this._getTwoTouches();
    const dist = this._distance(a, b);
    const angle = this._angle(a, b);
    const centroid = this._centroid(a, b);
    const t = this.camera.transform;

    // Pinch-to-zoom
    if (this._prevTouchDist != null && this._prevTouchDist > 0) {
      const scaleFactor = dist / this._prevTouchDist;
      t.scale(scaleFactor, t._center);
    }

    // Two-finger rotation
    if (this._prevTouchAngle != null) {
      let angleDelta = angle - this._prevTouchAngle;
      // Normalize to [-PI, PI]
      if (angleDelta > Math.PI) angleDelta -= 2 * Math.PI;
      if (angleDelta < -Math.PI) angleDelta += 2 * Math.PI;
      const degrees = angleDelta * (180 / Math.PI);
      t.rotate(0, degrees);
    }

    // Two-finger pan (centroid movement)
    if (this._prevTouchCentroid != null) {
      const dx = centroid.x - this._prevTouchCentroid.x;
      const dy = centroid.y - this._prevTouchCentroid.y;
      t.translate(0.01 * dx, -0.01 * dy);
    }

    this._prevTouchDist = dist;
    this._prevTouchAngle = angle;
    this._prevTouchCentroid = centroid;

    this.camera._notify();
    this.onRender();
  }

  /** Return the first two active touch positions. */
  _getTwoTouches() {
    const iter = this._activeTouches.values();
    return [iter.next().value, iter.next().value];
  }

  _distance(a, b) {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  _angle(a, b) {
    return Math.atan2(b.y - a.y, b.x - a.x);
  }

  _centroid(a, b) {
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  }

  // --- Cleanup ---

  dispose() {
    this._pendingMove = null;
    this._moveInFlight = false;
    if (this._moveWatchdog != null) { clearTimeout(this._moveWatchdog); this._moveWatchdog = null; }
    const c = this.canvas;
    c.removeEventListener('pointerdown', this._onPointerDown);
    c.removeEventListener('pointerup', this._onPointerUp);
    c.removeEventListener('pointermove', this._onPointerMove);
    c.removeEventListener('pointerleave', this._onPointerLeave);
    c.removeEventListener('wheel', this._onWheel);
    c.removeEventListener('dblclick', this._onDblClick);
    c.removeEventListener('contextmenu', this._onContextMenu);
    c.removeEventListener('touchstart', this._onTouchStart);
    c.removeEventListener('touchmove', this._onTouchMove);
    c.removeEventListener('touchend', this._onTouchEnd);
    c.removeEventListener('touchcancel', this._onTouchEnd);
  }
}

export { InputHandler };
