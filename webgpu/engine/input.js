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
    this._onWheel = this._onWheel.bind(this);
    this._onDblClick = this._onDblClick.bind(this);
    this._onContextMenu = (ev) => ev.preventDefault();
    this._onTouchStart = this._onTouchStart.bind(this);
    this._onTouchMove = this._onTouchMove.bind(this);
    this._onTouchEnd = this._onTouchEnd.bind(this);

    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointerup', this._onPointerUp);
    canvas.addEventListener('pointermove', this._onPointerMove);
    canvas.addEventListener('wheel', this._onWheel, { passive: false });
    canvas.addEventListener('dblclick', this._onDblClick);
    canvas.addEventListener('contextmenu', this._onContextMenu);
    canvas.addEventListener('touchstart', this._onTouchStart, { passive: false });
    canvas.addEventListener('touchmove', this._onTouchMove, { passive: false });
    canvas.addEventListener('touchend', this._onTouchEnd);
    canvas.addEventListener('touchcancel', this._onTouchEnd);
  }

  // --- Pointer events (mouse & single touch fallback) ---

  _onPointerDown(ev) {
    ev.preventDefault();
    if (ev.button === 0 && !ev.shiftKey && !ev.ctrlKey && !ev.altKey) {
      this._isRotating = true;
    } else if (ev.button === 1 || (ev.button === 0 && ev.shiftKey)) {
      this._isPanning = true;
    }
    this.canvas.setPointerCapture(ev.pointerId);
  }

  _onPointerUp(ev) {
    this._isRotating = false;
    this._isPanning = false;
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
      t.rotate(0.3 * dpr * ev.movementY, 0.3 * dpr * ev.movementX);
      this.camera._notify();
      this.onRender();
    } else if (this._isPanning) {
      t.translate(0.01 * dpr * ev.movementX, -0.01 * dpr * ev.movementY);
      this.camera._notify();
      this.onRender();
    }
  }

  _onWheel(ev) {
    ev.preventDefault();
    const t = this.camera.transform;
    t.scale(1 - ev.deltaY / 1000, t._center);
    this.camera._notify();
    this.onRender();
  }

  _onDblClick(ev) {
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
    const c = this.canvas;
    c.removeEventListener('pointerdown', this._onPointerDown);
    c.removeEventListener('pointerup', this._onPointerUp);
    c.removeEventListener('pointermove', this._onPointerMove);
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
