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

    // Bind handlers for clean removal
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onDblClick = this._onDblClick.bind(this);
    this._onContextMenu = (ev) => ev.preventDefault();

    canvas.addEventListener('pointerdown', this._onPointerDown);
    canvas.addEventListener('pointerup', this._onPointerUp);
    canvas.addEventListener('pointermove', this._onPointerMove);
    canvas.addEventListener('wheel', this._onWheel, { passive: false });
    canvas.addEventListener('dblclick', this._onDblClick);
    canvas.addEventListener('contextmenu', this._onContextMenu);
  }

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
    const t = this.camera.transform;
    if (this._isRotating) {
      t.rotate(0.3 * ev.movementY, 0.3 * ev.movementX);
      this.camera._notify();
      this.onRender();
    } else if (this._isPanning) {
      t.translate(0.01 * ev.movementX, -0.01 * ev.movementY);
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

  dispose() {
    const c = this.canvas;
    c.removeEventListener('pointerdown', this._onPointerDown);
    c.removeEventListener('pointerup', this._onPointerUp);
    c.removeEventListener('pointermove', this._onPointerMove);
    c.removeEventListener('wheel', this._onWheel);
    c.removeEventListener('dblclick', this._onDblClick);
    c.removeEventListener('contextmenu', this._onContextMenu);
  }
}

export { InputHandler };
