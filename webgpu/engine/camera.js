// Camera and Transform for WebGPU render engine
// Port of webgpu/camera.py — produces byte-identical uniform buffers

const DEG2RAD = Math.PI / 180;
const UNIFORM_BUFFER_SIZE = 336;

// ---------------------------------------------------------------------------
// 4x4 matrix helpers — row-major Float64Array[16], index = row*4 + col
// ---------------------------------------------------------------------------

function mat4Identity() {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Mul(a, b) {
  const o = new Float64Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[i * 4 + k] * b[k * 4 + j];
      o[i * 4 + j] = s;
    }
  return o;
}

function mat4Inv(m) {
  // Gauss-Jordan elimination with partial pivoting
  const a = new Float64Array(m);
  const inv = mat4Identity();
  for (let col = 0; col < 4; col++) {
    let maxVal = Math.abs(a[col * 4 + col]);
    let maxRow = col;
    for (let row = col + 1; row < 4; row++) {
      const v = Math.abs(a[row * 4 + col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxRow !== col) {
      for (let j = 0; j < 4; j++) {
        let tmp;
        tmp = a[col * 4 + j]; a[col * 4 + j] = a[maxRow * 4 + j]; a[maxRow * 4 + j] = tmp;
        tmp = inv[col * 4 + j]; inv[col * 4 + j] = inv[maxRow * 4 + j]; inv[maxRow * 4 + j] = tmp;
      }
    }
    const diag = a[col * 4 + col];
    for (let j = 0; j < 4; j++) {
      a[col * 4 + j] /= diag;
      inv[col * 4 + j] /= diag;
    }
    for (let row = 0; row < 4; row++) {
      if (row === col) continue;
      const f = a[row * 4 + col];
      for (let j = 0; j < 4; j++) {
        a[row * 4 + j] -= f * a[col * 4 + j];
        inv[row * 4 + j] -= f * inv[col * 4 + j];
      }
    }
  }
  return inv;
}

// ---------------------------------------------------------------------------
// Transform — mirrors Python Transform exactly
// ---------------------------------------------------------------------------

class Transform {
  constructor() {
    this._mat = mat4Identity();
    this._center = [0, 0, 0];
  }

  copy() {
    const t = new Transform();
    t._mat = new Float64Array(this._mat);
    t._center = [...this._center];
    return t;
  }

  init(pmin, pmax) {
    const center = [
      0.5 * (pmin[0] + pmax[0]),
      0.5 * (pmin[1] + pmax[1]),
      0.5 * (pmin[2] + pmax[2]),
    ];
    this._center = center;
    const dx = pmax[0] - pmin[0];
    const dy = pmax[1] - pmin[1];
    const dz = pmax[2] - pmin[2];
    const scale = 2 / Math.sqrt(dx * dx + dy * dy + dz * dz);
    this._mat = mat4Identity();
    this.translate(-center[0], -center[1], -center[2]);
    this.scale(scale);
    if (!(Math.abs(pmin[2]) < 1e-12 && Math.abs(pmax[2]) < 1e-12)) {
      this.rotate(270, 0);
      this.rotate(0, -20);
      this.rotate(20, 0);
    }
  }

  translate(dx = 0, dy = 0, dz = 0) {
    if (Array.isArray(dx)) { [dx, dy, dz] = dx; }
    const t = mat4Identity();
    t[3] = dx; t[7] = dy; t[11] = dz;
    this._mat = mat4Mul(t, this._mat);
  }

  scale(s, center = null) {
    this._withCentering(center, () => {
      const sc = new Float64Array(16);
      sc[0] = s; sc[5] = s; sc[10] = s; sc[15] = 1;
      this._mat = mat4Mul(sc, this._mat);
    });
  }

  rotate(angX, angY = 0, center = null) {
    const rx = angX * DEG2RAD;
    const cx = Math.cos(rx), sx = Math.sin(rx);
    const rotX = mat4Identity();
    rotX[5] = cx; rotX[6] = -sx;
    rotX[9] = sx; rotX[10] = cx;

    const ry = angY * DEG2RAD;
    const cy = Math.cos(ry), sy = Math.sin(ry);
    const rotY = mat4Identity();
    rotY[0] = cy; rotY[2] = sy;
    rotY[8] = -sy; rotY[10] = cy;

    this._withCentering(center, () => {
      this._mat = mat4Mul(rotX, mat4Mul(rotY, this._mat));
    });
  }

  setCenter(center) {
    const mapped = this.mapPoint(center);
    this.translate(-mapped[0], -mapped[1], -mapped[2]);
    this._center = [...center];
  }

  resetXY(flip = false) {
    const s = this._currentScale();
    this._mat = mat4Identity();
    this.translate(-this._center[0], -this._center[1], -this._center[2]);
    this.scale(s);
    if (flip) this.rotate(0, 180);
  }

  resetXZ(flip = false) {
    this.resetXY();
    this.rotate(-90, 0);
    if (flip) this.rotate(0, 180);
  }

  resetYZ(flip = false) {
    this.resetXY();
    this.rotate(-90, 0);
    this.rotate(0, -90);
    if (flip) this.rotate(0, 180);
  }

  mapPoint(point) {
    const m = this._mat;
    const x = point[0], y = point[1], z = point[2];
    const w = m[12] * x + m[13] * y + m[14] * z + m[15];
    return [
      (m[0] * x + m[1] * y + m[2] * z + m[3]) / w,
      (m[4] * x + m[5] * y + m[6] * z + m[7]) / w,
      (m[8] * x + m[9] * y + m[10] * z + m[11]) / w,
    ];
  }

  get mat() { return this._mat; }

  _currentScale() {
    const m = this._mat;
    // Norm of column 0 of upper-left 3x3 (row-major: col 0 = m[0], m[4], m[8])
    return Math.sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
  }

  _withCentering(center, fn) {
    const c = center != null ? center : this._center;
    const mapped = this.mapPoint(c);
    this.translate(-mapped[0], -mapped[1], -mapped[2]);
    fn();
    this.translate(mapped[0], mapped[1], mapped[2]);
  }
}

// ---------------------------------------------------------------------------
// Camera — wraps Transform, provides uniform buffer and observer pattern
// ---------------------------------------------------------------------------

class Camera {
  constructor() {
    this.transform = new Transform();
    this._observers = [];
  }

  reset(pmin, pmax) {
    this.transform.init(pmin, pmax);
    this._notify();
  }

  resetXY(flip = false) { this.transform.resetXY(flip); this._notify(); }
  resetXZ(flip = false) { this.transform.resetXZ(flip); this._notify(); }
  resetYZ(flip = false) { this.transform.resetYZ(flip); this._notify(); }

  registerObserver(cb) {
    if (!this._observers.includes(cb)) this._observers.push(cb);
  }

  unregisterObserver(cb) {
    this._observers = this._observers.filter(c => c !== cb);
  }

  _notify() {
    for (const cb of this._observers) cb();
  }

  /**
   * Compute camera uniform buffer matching CameraUniforms layout (336 bytes).
   *
   * Struct layout (all mat4 are 16 floats = 64 bytes):
   *   view                  [0..63]    column-major
   *   model_view            [64..127]  column-major
   *   model_view_projection [128..191] column-major
   *   rot_mat               [192..255] column-major
   *   normal_mat            [256..319] row-major (shader reads as inverse-transpose)
   *   aspect  (f32)         [320..323]
   *   width   (u32)         [324..327]
   *   height  (u32)         [328..331]
   *   padding (u32)         [332..335]
   */
  updateUniforms(width, height) {
    if (height === 0) return null;

    // --- Projection (matches Python exactly) ---
    const near = 0.1, far = 10, fov = 45;
    const aspect = width / height;
    const zoom = 1.0;
    const top = near * Math.tan(fov * DEG2RAD / 2) * zoom;
    const h = 2 * top;
    const w = aspect * h;
    const left = -0.5 * w;
    const right = left + w;
    const bottom = top - h;

    const px = 2 * near / (right - left);
    const py = 2 * near / (top - bottom);
    const a = (right + left) / (right - left);
    const b = (top + bottom) / (top - bottom);
    const c = -far / (far - near);
    const d = (-far * near) / (far - near);

    const proj = new Float64Array(16);
    proj[0] = px; proj[2] = a;
    proj[5] = py; proj[6] = b;
    proj[10] = c; proj[11] = d;
    proj[14] = -1;

    // --- View matrix: identity with z-translation = -3 ---
    const view = mat4Identity();
    view[11] = -3;

    const modelView = mat4Mul(view, this.transform.mat);
    const modelViewProj = mat4Mul(proj, modelView);
    const normalMat = mat4Inv(modelView);

    // --- Rotation matrix: normalized columns of upper-left 3x3 of modelView ---
    const rotMat = mat4Identity();
    for (let j = 0; j < 3; j++) {
      let norm = 0;
      for (let i = 0; i < 3; i++) norm += modelView[i * 4 + j] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < 3; i++) rotMat[i * 4 + j] = modelView[i * 4 + j] / norm;
    }

    // --- Write to buffer ---
    const buf = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);

    // Column-major: transpose row-major → f32[offset + col*4 + row]
    const writeColMajor = (offset, m) => {
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++)
          f32[offset + j * 4 + i] = m[i * 4 + j];
    };

    // Row-major: direct copy
    const writeRowMajor = (offset, m) => {
      for (let k = 0; k < 16; k++) f32[offset + k] = m[k];
    };

    writeColMajor(0, view);            // view
    writeColMajor(16, modelView);       // model_view
    writeColMajor(32, modelViewProj);   // model_view_projection
    writeColMajor(48, rotMat);          // rot_mat
    writeRowMajor(64, normalMat);       // normal_mat (row-major = col-major of transpose)

    f32[80] = aspect;
    u32[81] = width;
    u32[82] = height;
    u32[83] = 0;

    return buf;
  }
}

export { Camera, Transform, UNIFORM_BUFFER_SIZE };
