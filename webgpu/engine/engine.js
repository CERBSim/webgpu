// RenderEngine — core WebGPU renderer for exported scene blobs.
// Depends on: parseSceneBlob (format.js), Camera (camera.js),
//             ComputeDAG (compute.js), InputHandler (input.js)
// All are concatenated into the same scope by the Python loader — no imports.

const SAMPLE_COUNT = 4;
const LIGHT_CLEAR_COLOR = Object.freeze({ r: 1.0, g: 1.0, b: 1.0, a: 1.0 });
const DARK_CLEAR_COLOR  = Object.freeze({ r: 0.8078, g: 0.8392, b: 0.8667, a: 1.0 });
const LIGHT_CANVAS_BG = '#ffffff';
const DARK_CANVAS_BG  = '#ced6dd';
const DEPTH_FORMAT = 'depth24plus';
// Object-id / pick texture format. Matches Python canvas.select_format. The
// select pass renders into a host-owned texture of this format (non-MSAA).
const SELECT_FORMAT = 'rgba32uint';

const TRANSPARENT_BLEND = {
  color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

const OPAQUE_BLEND = {
  color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
};

// ---------------------------------------------------------------------------
// Helper: buffer usage flags from usage string
// ---------------------------------------------------------------------------

function bufferUsageFlags(usage) {
  const U = GPUBufferUsage;
  switch (usage) {
    case 'uniform':       return U.UNIFORM | U.COPY_DST | U.COPY_SRC;
    case 'storage':       return U.STORAGE | U.COPY_DST | U.COPY_SRC;
    case 'storage-write': return U.STORAGE | U.COPY_DST | U.COPY_SRC;
    case 'vertex':        return U.VERTEX | U.COPY_DST;
    case 'index':         return U.INDEX | U.COPY_DST;
    case 'indirect':      return U.INDIRECT | U.STORAGE | U.COPY_DST | U.COPY_SRC;
    default:              return U.STORAGE | U.COPY_DST | U.COPY_SRC;
  }
}

// ---------------------------------------------------------------------------
// Helper: find the sampler resource ID that is co-located with a texture
//
// Convention: for a texture at binding N, the sampler is at binding N+1
// and its ID starts with "sampler_". We look it up in the bindings dict.
// ---------------------------------------------------------------------------

function findSamplerIdForTexture(bindings, textureBindingNum) {
  const nextBinding = String(textureBindingNum + 1);
  const id = bindings[nextBinding];
  if (id && id.startsWith('sampler_')) return { binding: textureBindingNum + 1, id };
  return null;
}

// ---------------------------------------------------------------------------
// RenderEngine
// ---------------------------------------------------------------------------

class RenderEngine {

  /**
   * Create and initialize from a canvas element and a base64-encoded blob.
   */
  static async create(canvasId, base64Blob) {
    const binary = atob(base64Blob);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    const engine = new RenderEngine();
    await engine.init(document.getElementById(canvasId), bytes.buffer);
    return engine;
  }

  /**
   * Live entry point: bind the engine to GPU resources owned by an external
   * runtime (e.g. Pyodide). The descriptor provides already-allocated
   * GPUBuffer / GPUTexture / GPUSampler proxies.
   *
   * descriptor = {
   *   device: GPUDevice,
   *   context: GPUCanvasContext,           // already configure()d
   *   canvasFormat?: string,
   *   buffers:  Map<string, GPUBuffer>  | object,
   *   textures: Map<string, GPUTexture> | object,
   *   samplers: Map<string, GPUSampler> | object,
   *   render_passes:  ExportRenderPass[],
   *   compute_passes: ExportComputePass[],
   *   interactions?:  ExportInteraction[],
   *   camera?: { matrix?, center?, buffer_id? },
   *   light?:  { buffer_id?, data? },
   * }
   */
  static async createLive(canvas, descriptor) {
    try {
      const engine = new RenderEngine();
      await engine.initLive(canvas, descriptor);
      canvas.__engine = engine;
      return engine;
    } catch (err) {
      console.error('[engine] createLive failed:', err && (err.stack || err.message || err));
      throw err;
    }
  }

  async init(canvas, arrayBuffer) {
    this.canvas = canvas;
    this.mode = 'blob';

    // --- Parse blob ---
    this.scene = parseSceneBlob(arrayBuffer);

    // --- WebGPU device ---
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const powerPreference = (typeof window !== 'undefined' && window.__webgpuPowerPreference) || 'high-performance';
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) throw new Error('No WebGPU adapter found');
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });

    // --- Canvas context ---
    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context = canvas.getContext('webgpu');
    this.context.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    // --- Create GPU resources ---
    this.buffers = new Map();   // id → GPUBuffer
    this.textures = new Map();  // id → GPUTexture
    this.samplers = new Map();  // id → GPUSampler (keyed by the sampler_ id)
    this.texViews = new Map();  // id → GPUTextureView
    this.frameBuffers = new Map(); // id → ArrayBuffer (CPU-only frame snapshots)

    this._createBuffers();
    this._createTextures();
    this._createDepthAndMSAA();

    // --- Camera ---
    this.camera = new Camera();
    if (this.scene.camera) {
      const c = this.scene.camera;
      if (c.matrix) this.camera.transform._mat = Float64Array.from(c.matrix);
      if (c.center) this.camera.transform._center = [...c.center];
    }
    this._updateCameraBuffer();

    // --- Light ---
    this._applyLight();

    await this._finishInit();
  }

  async initLive(canvas, descriptor) {
    this.canvas = canvas;
    this.mode = 'live';

    if (!descriptor.device)  throw new Error('createLive: descriptor.device required');
    if (!descriptor.context) throw new Error('createLive: descriptor.context required');

    this.device = descriptor.device;
    this.context = descriptor.context;
    this.canvasFormat = descriptor.canvasFormat
      || (navigator.gpu && navigator.gpu.getPreferredCanvasFormat())
      || 'bgra8unorm';

    // Synthesize a scene-shaped object so the rest of the engine stays generic.
    // No raw bytes — every GPU resource is supplied by the host.
    this.scene = {
      buffers: {},
      textures: {},
      compute_passes: descriptor.compute_passes || [],
      render_passes:  descriptor.render_passes  || [],
      interactions:   descriptor.interactions   || [],
      camera: descriptor.camera || {},
      light:  descriptor.light  || {},
      theme: descriptor.theme || {},
    };

    // Live-mode host callbacks (optional).
    this._onEvent = descriptor.on_event || null;
    this._onCameraChanged = descriptor.on_camera_changed || null;
    // Host clear color [r,g,b,a] overriding the built-in light/dark color.
    this._clearColorOverride = descriptor.clear_color || null;

    this.buffers  = _toMap(descriptor.buffers);
    this.textures = _toMap(descriptor.textures);
    this.samplers = new Map();
    this.texViews = new Map();
    this.frameBuffers = new Map();

    // Frame buffers (CPU-only blobs for animation snapshots)
    if (descriptor.frame_buffers) {
      for (const [id, data] of Object.entries(descriptor.frame_buffers)) {
        this.frameBuffers.set(id, data);
      }
    }

    for (const [id, buf] of this.buffers) {
      this.scene.buffers[id] = { usage: _usageFromId(id), size: buf.size };
    }
    for (const [id, tex] of this.textures) {
      this.scene.textures[id] = { format: tex.format, width: tex.width, height: tex.height };
      this.texViews.set(id, tex.createView());
    }

    // Sampler lookup is supplied directly by the host.
    this._samplerLookup = _toMap(descriptor.samplers || {});

    this._createDepthAndMSAA();

    // --- Camera ---
    this.camera = new Camera();
    if (this.scene.camera) {
      const c = this.scene.camera;
      if (c.matrix) this.camera.transform._mat = Float64Array.from(c.matrix);
      if (c.center) this.camera.transform._center = [...c.center];
    }
    this._updateCameraBuffer();

    // In live mode the host typically writes the light buffer directly.
    if (this.scene.light && this.scene.light.data) this._applyLight();

    await this._finishInit();
  }

  /** Translate a serialized compute pass (snake_case) into ComputeDAG.addPass args. */
  _translateComputePass(cp) {
    const indirectSetup = cp.indirect_setup ? {
      counterId: cp.indirect_setup.counter_id,
      indirectId: cp.indirect_setup.indirect_id,
      vertexCount: cp.indirect_setup.vertex_count,
    } : null;
    const countThenFill = cp.count_then_fill ? {
      counterId: cp.count_then_fill.counter_id,
      outputId: cp.count_then_fill.output_id,
      elementSize: cp.count_then_fill.element_size,
      indirectId: cp.count_then_fill.indirect_id || null,
      vertexCount: cp.count_then_fill.vertex_count || 0,
      // indexed: indirect buffer is the 5-u32 drawIndexedIndirect layout
      // (indexCount, instanceCount, firstIndex, baseVertex, firstInstance)
      // instead of the 4-u32 drawIndirect layout.
      indexed: !!cp.count_then_fill.indexed,
      initialCount: cp.count_then_fill.initial_count || 0,
      // siblings: extra output buffers resized in lockstep with the primary
      // from the same counter (e.g. a vector's positions/directions/values
      // arrays). Each: { id, elementSize }. The primary (outputId) still
      // drives the instanceCount cap.
      siblings: (cp.count_then_fill.siblings || []).map(s => ({
        id: s.id,
        elementSize: s.element_size,
      })),
    } : null;
    return {
      shader: cp.shader,
      bindings: this._intKeyBindings(cp.bindings),
      workgroups: cp.workgroups,
      triggers: cp.triggers,
      resetBuffers: cp.reset_buffers,
      entryPoint: cp.entry_point || 'main',
      indirectSetup,
      countThenFill,
    };
  }

  /** Build the compute DAG from scratch (first install). */
  async _buildComputeDAG() {
    this.computeDAG = new ComputeDAG();
    for (const cp of this.scene.compute_passes) {
      this.computeDAG.addPass(cp.id, this._translateComputePass(cp));
    }
    await this.computeDAG.initPipelines(this.device, this.buffers);
    for (const cp of this.scene.compute_passes) {
      for (const t of cp.triggers) this.computeDAG.markDirty(t);
    }
  }

  /**
   * Incrementally reconcile the compute DAG with this.scene.compute_passes
   * (called from update() when a renderer with a compute pass is toggled).
   * Adds newly-present passes and removes vanished ones, leaving UNCHANGED
   * passes — and their already-converged JS-owned buffers and in-flight
   * readbacks — completely untouched. A wholesale rebuild here instead would
   * discard those buffers and invalidate references the host/bridge still
   * holds (observed as "mapAsync: external Instance reference no longer exists").
   */
  async _syncComputeDAG() {
    if (!this.computeDAG) { await this._buildComputeDAG(); return; }
    const want = new Map(this.scene.compute_passes.map(cp => [cp.id, cp]));
    // Remove passes that are gone.
    for (const id of [...this.computeDAG.passes.keys()]) {
      if (!want.has(id)) this.computeDAG.removePass(id);
    }
    // Add passes that are new; init + trigger only those.
    for (const [id, cp] of want) {
      if (this.computeDAG.passes.has(id)) continue;
      this.computeDAG.addPass(id, this._translateComputePass(cp));
      await this.computeDAG._initPass(this.device, this.buffers, id);
      for (const t of cp.triggers) this.computeDAG.markDirty(t);
    }
  }

  async _finishInit() {
    const canvas = this.canvas;

    // --- Compute ---
    await this._buildComputeDAG();

    // --- Render pipelines ---
    this.renderPassObjects = [];
    await this._createRenderPipelines();

    // --- Input + camera ---
    // The JS engine owns the camera and input loop in both modes. In live mode
    // it also forwards non-camera events to the host and syncs settled camera
    // transforms back.
    this.camera.registerObserver(() => {
      this._updateCameraBuffer();
      if (this._cameraBufferId) {
        this.computeDAG.markDirty(this._cameraBufferId);
      }
      this.render();
      this._notifyCameraChanged();
    });
    this.input = new InputHandler(canvas, this.camera, () => this.render());
    if (this._onEvent) {
      this.input.setEventSink((ev) => {
        try { this._onEvent(ev); }
        catch (e) { console.warn('[engine] on_event failed:', e && (e.message || e)); }
      });
    }
    if (this._onCameraChanged) {
      this.input.onGestureEnd = () => this._notifyCameraChanged(true);
    }

    // --- Interactions (lil-gui) ---
    const guiContainerId = canvas.id.replace('canvas', 'lilgui');
    const guiContainer = document.getElementById(guiContainerId);
    this.interactions = new Interactions(this, guiContainer);
    if (this.scene.interactions && this.scene.interactions.length > 0) {
      try {
        await Promise.race([
          this.interactions.setup(this.scene.interactions),
          new Promise((_, reject) => setTimeout(() => reject(new Error('interactions setup timed out')), 5000)),
        ]);
      } catch (e) {
        console.warn('[engine] interactions setup failed:', e.message || e);
      }
    }

    // --- Resize handling ---
    // Live mode: the host owns the canvas and drives resize via its own
    // observer; it tells us when to react via engine.handleResize().
    if (this.mode !== 'live') {
      this._resizeObserver = new ResizeObserver(() => this._onResize());
      this._resizeObserver.observe(canvas);
    }

    // --- Theme handling ---
    this._applyTheme();
    this._setupThemeObserver();

    // --- First frame ---
    // In live mode, the host decides when to render. Don't auto-render.
    if (this.mode !== 'live') this.render();
  }

  /**
   * Capture the next rendered frame as a CPU-readable Uint8Array.
   * Resolves with { data, width, height, format } after the next render.
   * Used by screenshots and headless tests.
   */
  captureNextFrame() {
    return new Promise((resolve) => {
      (this._frameCaptureRequests = this._frameCaptureRequests || []).push(resolve);
      this._renderNow();
    });
  }

  /**
   * Capture the next rendered frame and return just the raw ArrayBuffer.
   * Unlike captureNextFrame(), the ArrayBuffer is returned at the top level
   * (not nested in an object) so it transfers correctly through the bridge.
   * Width/height/format should be read from the canvas separately.
   */
  /**
   * Drive count-then-fill passes to convergence. Each pass starts with a tiny
   * output buffer and grows it from a GPU-read atomic counter over several
   * frames; in the live GUI that happens across animation frames, but a one-shot
   * capture (screenshot, headless test) has no such loop — so we render and
   * await processReadbacks() in a loop here until no pass resizes any more. This
   * is what makes count-then-fill fields (e.g. clipping/surface vectors) appear
   * in screenshots and headless renders instead of coming out blank.
   */
  async settle(maxIterations = 32) {
    if (!this.computeDAG) return 0;
    // Force the count-then-fill passes to (re)run so they can converge.
    this.notifyDirty(null);
    let i = 0;
    for (; i < maxIterations; i++) {
      this._renderNow(null, /*autoReadback=*/false);
      const resized = await this.computeDAG.processReadbacks(
        this.device, this.buffers, () => this._rebuildRenderBindGroups()
      );
      if (!resized) break;
    }
    return i;
  }

  async captureFrameBuffer() {
    await this.settle();
    return new Promise((resolve) => {
      (this._frameCaptureRequests = this._frameCaptureRequests || []).push(
        ({ data }) => resolve(data)
      );
      this._renderNow();
    });
  }

  _ensureCaptureTexture() {
    const w = this.width, h = this.height;
    if (!this._captureTex || this._captureTex.width !== w || this._captureTex.height !== h) {
      if (this._captureTex) this._captureTex.destroy();
      this._captureTex = this.device.createTexture({
        size: [w, h],
        format: this.canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        label: 'capture-target',
      });
    }
    return this._captureTex;
  }

  enableHeadlessCapture() {
    this._headless = true;
    this._ensureCaptureTexture();
    return this._captureTex;
  }

  captureTexture() {
    return this._ensureCaptureTexture();
  }

  /**
   * Push updated render/compute pass descriptors and rebuild pipelines.
   * Used by the host (Python) when the renderer set or its options change.
   */
  async update({ render_passes, compute_passes, interactions, buffers, textures, samplers, frame_buffers, camera } = {}) {
    if (render_passes)  this.scene.render_passes  = render_passes;
    if (compute_passes) this.scene.compute_passes = compute_passes;
    if (interactions)   this.scene.interactions   = interactions;
    if (camera) {
      // The buffer registry can reassign the camera's string id when the
      // render-object set changes, so a cached _cameraBufferId would point at
      // whatever buffer now holds that id (e.g. the 4-byte "subdivision"
      // uniform). Refresh the camera and force _updateCameraBuffer to
      // re-resolve the id from the new descriptor.
      this.scene.camera = camera;
      this._cameraBufferId = (camera && camera.buffer_id) || null;
    }
    // Update live buffer/texture/sampler maps when the host recreates GPU resources.
    // Preserve buffers that the JS engine has resized via countThenFill —
    // Python doesn't know about JS-side resizes, so its references are stale
    // for those buffers. The JS engine is the authority for output buffer sizing.
    if (buffers) {
      const incoming = _toMap(buffers);
      const resized = this.computeDAG ? this.computeDAG._resizedBufferIds : new Set();
      // Rebuild this.buffers to exactly (engine-resized ∪ incoming host set).
      // Any id previously held that is neither is dropped, so the engine stops
      // referencing host buffers whose Python wrappers were freed — the host
      // then destroys them at its render-safe flush. Without this prune the map
      // grows forever and freed buffers would still be bound in a frame. We do
      // NOT destroy host buffers here (the host owns them); engine-owned
      // (resized) buffers are kept and freed in dispose().
      const next = new Map();
      for (const [id, buf] of this.buffers) {
        if (resized.has(id)) next.set(id, buf);
      }
      for (const [id, buf] of incoming) {
        if (!resized.has(id)) next.set(id, buf);
      }
      this.buffers = next;
    }
    if (textures) {
      this.textures = _toMap(textures);
      this.texViews = new Map();
      for (const [id, tex] of this.textures) {
        this.texViews.set(id, tex.createView());
      }
    }
    if (samplers) {
      this._samplerLookup = _toMap(samplers);
    }
    if (frame_buffers) {
      for (const [id, data] of Object.entries(frame_buffers)) {
        this.frameBuffers.set(id, data);
      }
    }
    // Re-setup interactions if they changed (e.g. new animation frames)
    if (interactions && this.interactions) {
      if (this.interactions.gui) {
        this.interactions.gui.destroy();
        this.interactions.gui = null;
      }
      try {
        await this.interactions.setup(this.scene.interactions);
      } catch (e) {
        console.warn('[engine] interactions setup in update() failed:', e.message || e);
      }
    }
    // Reconcile the compute DAG with the new pass set (a renderer with a
    // compute pass was toggled on/off, e.g. clip vectors). Without this, a
    // newly-added pass lives in scene.compute_passes but is never built into
    // the DAG and never runs. Done incrementally so unchanged passes keep their
    // converged buffers and in-flight readbacks.
    if (compute_passes) {
      const have = this.computeDAG ? this.computeDAG.passes : new Map();
      const want = this.scene.compute_passes.map(p => p.id);
      const changed = have.size !== want.length || want.some(id => !have.has(id));
      if (changed) {
        this._updating = true;
        try {
          await this._syncComputeDAG();
        } catch (e) {
          console.error('[engine] _syncComputeDAG failed in update():', e.message || e);
        }
        this._updating = false;
      }
    }

    // Rebuild render pipelines from scratch — cheap relative to a full reload.
    this._updating = true;
    this.renderPassObjects = [];
    try {
      await this._createRenderPipelines();
    } catch (e) {
      console.error('[engine] _createRenderPipelines failed in update():', e.message || e);
    }
    this._updating = false;
    // The host dispatches this async update() WITHOUT awaiting it and then
    // immediately calls notifyDirty()+render(); those ran while _updating was
    // still true (pipelines mid-rebuild) and were dropped by _renderNow's guard.
    // Now that pipelines are ready, trigger the post-update frame ourselves.
    // notifyDirty() first so compute passes re-run — their output feeds the
    // render pass within the same command encoder, so the frame isn't stale.
    // (Previously this was masked whenever another renderer's compute kept a
    // continuous render loop alive; with a compute-only renderer like clip
    // vectors as the sole content, nothing else re-triggered a frame.)
    this.notifyDirty();
    this.render();
  }

  /**
   * Live-mode hook: the host has resized the canvas. Recreate depth + MSAA
   * targets and re-render. The host is also responsible for updating the
   * camera uniform (aspect ratio change).
   */
  handleResize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width > 0 ? Math.round(rect.width * dpr) : this.canvas.width;
    const h = rect.height > 0 ? Math.round(rect.height * dpr) : this.canvas.height;
    if (w === this.width && h === this.height) return;
    if (w === 0 || h === 0) return;
    if (this.depthTexture) this.depthTexture.destroy();
    if (this.msaaTexture)  this.msaaTexture.destroy();
    this._createDepthAndMSAA();
    // Refresh the camera uniform (aspect/width/height) for the new size.
    this._updateCameraBuffer();
    this.render();
  }

  /** Live-mode hook: host sets the camera transform (reset/view/bookmark). */
  setCameraTransform(matrix, center) {
    if (!this.camera) return;
    this._suppressCameraNotify = true;
    try {
      if (matrix) this.camera.transform._mat = Float64Array.from(matrix);
      if (center) this.camera.transform._center = Array.from(center);
      this._updateCameraBuffer();
      if (this._cameraBufferId) this.computeDAG.markDirty(this._cameraBufferId);
      this.render();
    } finally {
      this._suppressCameraNotify = false;
    }
  }

  /**
   * Live-mode hook: recenter the camera on a world-space point (double-click).
   *
   * Applied to the engine's OWN (authoritative) camera using its current
   * matrix, so the view-translate and the rotation pivot (_center) update
   * together and can't desync. Doing this host-side instead — pushing a
   * Python-computed matrix+center via setCameraTransform — risks the pivot and
   * the matrix disagreeing, which makes the next rotate orbit the old center
   * and visibly snap the view back. The move is notified to the host so its
   * camera mirror (used for picking/bookmarks) stays in sync.
   */
  setCenter(point) {
    if (!this.camera || !point) return;
    this.camera.transform.setCenter(Array.from(point));
    this._updateCameraBuffer();
    if (this._cameraBufferId) this.computeDAG.markDirty(this._cameraBufferId);
    this.render();
    this._notifyCameraChanged(true);
  }

  /** Live-mode hook: host sets the background clear color [r,g,b,a] (0..1). */
  setClearColor(rgba) {
    this._clearColorOverride = rgba || null;
    this._applyTheme();
    this.render();
  }

  /**
   * Live-mode hook: the host acks that it finished processing the last input
   * event, releasing the input handler's move backpressure so the next
   * coalesced move can be sent.
   */
  ackInput() {
    if (this.input && this.input._releaseMove) this.input._releaseMove();
  }

  /** Report the current camera transform to the host (debounced). */
  _notifyCameraChanged(immediate = false) {
    if (!this._onCameraChanged || this._suppressCameraNotify) return;
    const send = () => {
      this._camNotifyTimer = null;
      try {
        this._onCameraChanged({
          matrix: Array.from(this.camera.transform._mat),
          center: Array.from(this.camera.transform._center),
        });
      } catch (e) {
        console.warn('[engine] on_camera_changed failed:', e && (e.message || e));
      }
    };
    if (immediate) {
      if (this._camNotifyTimer) { clearTimeout(this._camNotifyTimer); }
      send();
      return;
    }
    if (this._camNotifyTimer) return;  // trailing-edge debounce already scheduled
    this._camNotifyTimer = setTimeout(send, 100);
  }

  /**
   * Live-mode hook: mark a buffer (or a list of them) as dirty so any
   * compute pass triggered by it re-runs on the next render. Pass no args
   * to mark every compute trigger dirty (used after pipeline rebuilds).
   */
  notifyDirty(bufferIds) {
    if (!bufferIds) {
      for (const cp of this.scene.compute_passes) {
        for (const t of cp.triggers) this.computeDAG.markDirty(t);
      }
      return;
    }
    const ids = Array.isArray(bufferIds) ? bufferIds : [bufferIds];
    for (const id of ids) this.computeDAG.markDirty(id);
  }

  // =========================================================================
  // GPU resource creation
  // =========================================================================

  _createBuffers() {
    const device = this.device;
    // Create GPU buffers (skip CPU-only frame snapshots)
    for (const [id, info] of Object.entries(this.scene.buffers)) {
      if (info.usage === 'frame') {
        // CPU-only blob (e.g. animation frame snapshot). Keep raw bytes
        // available via this.frameBuffers for interactions to upload later.
        this.frameBuffers.set(id, info.data);
        continue;
      }
      const buf = device.createBuffer({
        size: info.size,
        usage: bufferUsageFlags(info.usage),
        label: id,
      });
      if (info.data && info.data.byteLength > 0) {
        device.queue.writeBuffer(buf, 0, info.data);
      }
      this.buffers.set(id, buf);
    }
  }

  _createTextures() {
    const device = this.device;
    // Create textures
    for (const [id, info] of Object.entries(this.scene.textures)) {
      const tex = device.createTexture({
        size: [info.width, info.height],
        format: info.format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        label: id,
      });
      if (info.data && info.data.byteLength > 0) {
        const bytesPerRow = _bytesPerPixel(info.format) * info.width;
        device.queue.writeTexture(
          { texture: tex },
          info.data,
          { bytesPerRow },
          [info.width, info.height],
        );
      }
      this.textures.set(id, tex);
      this.texViews.set(id, tex.createView());

      // Create sampler if descriptor present
      if (info.sampler) {
        // The sampler is stored on the texture info; we need to associate it
        // with the sampler_* id used in bindings. Find it by scanning render passes.
        const sampler = device.createSampler(info.sampler);
        // Store under tex id — we'll resolve sampler_ ids below
        this.samplers.set(id, sampler);
      }
    }

    // Build sampler_id → GPUSampler mapping by finding which texture each sampler references.
    // Convention: sampler and texture are co-located (sampler at binding N+1 of texture at N).
    this._samplerLookup = new Map(); // sampler_id → GPUSampler
    const allPasses = [...this.scene.render_passes, ...this.scene.compute_passes];
    for (const pass of allPasses) {
      for (const [bStr, resId] of Object.entries(pass.bindings)) {
        if (resId.startsWith('sampler_')) {
          // Find the texture at binding - 1
          const bNum = parseInt(bStr);
          const texResId = pass.bindings[String(bNum - 1)];
          if (texResId && this.samplers.has(texResId)) {
            this._samplerLookup.set(resId, this.samplers.get(texResId));
          } else {
            console.warn(`[engine] sampler lookup failed: ${resId}`);
          }
        }
      }
    }
  }

  _createDepthAndMSAA() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = rect.width > 0 ? Math.round(rect.width * dpr) : this.canvas.width;
    const h = rect.height > 0 ? Math.round(rect.height * dpr) : this.canvas.height;
    // Set canvas backing resolution
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;

    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: SAMPLE_COUNT,
      label: 'depth',
    });

    this.msaaTexture = this.device.createTexture({
      size: [w, h],
      format: this.canvasFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: SAMPLE_COUNT,
      label: 'msaa',
    });
  }

  // =========================================================================
  // Render pipeline creation
  // =========================================================================

  async _createRenderPipelines() {
    for (const rp of this.scene.render_passes) {
      const obj = await this._buildRenderPassObject(rp);
      this.renderPassObjects.push(obj);
    }
  }

  async _buildRenderPassObject(rp) {
    const device = this.device;
    const bindings = this._intKeyBindings(rp.bindings);
    const isTransparent = !rp.depth_write;

    const module = device.createShaderModule({ code: rp.shader, label: rp.id });

    const blend = isTransparent ? TRANSPARENT_BLEND : OPAQUE_BLEND;

    // Build explicit layout — avoids layout:"auto" which strips unreachable bindings
    const vis = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const layoutEntries = buildLayoutEntries(bindings, vis, null);
    const bindGroupLayout = device.createBindGroupLayout({ entries: layoutEntries });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    // Build vertex buffer layouts from export data
    const vertexBufferLayouts = (rp.vertex_buffers || []).map(vb => ({
      arrayStride: vb.arrayStride,
      stepMode: vb.stepMode || 'vertex',
      attributes: vb.attributes.map(a => ({
        format: a.format,
        offset: a.offset,
        shaderLocation: a.shaderLocation,
      })),
    }));

    const pipeline = await device.createRenderPipelineAsync({
      layout: pipelineLayout,
      vertex: { module, entryPoint: rp.vertex_entry_point || 'vertex_main', buffers: vertexBufferLayouts },
      fragment: {
        module,
        entryPoint: rp.fragment_entry_point || 'fragment_main',
        targets: [{ format: this.canvasFormat, blend }],
      },
      primitive: { topology: rp.topology || 'triangle-list' },
      depthStencil: {
        format: DEPTH_FORMAT,
        depthWriteEnabled: rp.depth_write,
        depthCompare: 'less',
        depthBias: rp.depth_bias || 0,
      },
      multisample: { count: SAMPLE_COUNT },
      label: rp.id,
    });

    // Build bind group entries
    const entries = this._buildBindGroupEntries(bindings);
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries,
      label: rp.id,
    });

    // Indirect draw buffer reference
    let indirectBuffer = null;
    if (rp.draw_indirect) {
      indirectBuffer = this.buffers.get(rp.draw_indirect);
    }

    const vertexBufferRefs = (rp.vertex_buffers || []).map(vb => this.buffers.get(vb.buffer_id));

    let indexBuffer = null;
    let indexFormat = null;
    if (rp.index_buffer_id) {
      indexBuffer = this.buffers.get(rp.index_buffer_id);
      indexFormat = rp.index_format || 'uint32';
    }

    return {
      id: rp.id,
      enabled: true,
      pipeline,
      bindGroup,
      // Kept so the select pipeline can reuse the exact same explicit layout
      // (→ the same bind group is valid for both render and select pipelines)
      // and so _rebuildRenderBindGroups rebuilds against a stable layout.
      bindGroupLayout,
      vertexBufferLayouts,
      vertexCount: rp.vertex_count,
      instanceCount: rp.instance_count,
      drawIndirect: !!rp.draw_indirect,
      indirectBuffer,
      vertexBufferRefs,
      indexBuffer,
      indexFormat,
      // Select (object-id/pick) pass metadata; pipeline built lazily in
      // _ensureSelectPipelines(). null shader → this pass is not pickable.
      selectShader: rp.select_shader || null,
      selectEntryPoint: rp.select_entry_point || null,
      vertexEntryPoint: rp.vertex_entry_point || 'vertex_main',
      topology: rp.topology || 'triangle-list',
      depthBias: rp.depth_bias || 0,
      selectPipeline: null,
    };
  }

  _buildBindGroupEntries(bindings) {
    const entries = [];
    for (const [bindingNum, resId] of Object.entries(bindings)) {
      const b = parseInt(bindingNum);
      const type = resourceType(resId);

      if (type === 'buffer') {
        const buf = this.buffers.get(resId);
        if (!buf) console.error(`[engine]   MISSING buffer: ${resId} at binding ${b}`);
        entries.push({ binding: b, resource: { buffer: buf } });
      } else if (type === 'texture') {
        const view = this.texViews.get(resId);
        if (!view) console.error(`[engine]   MISSING texture view: ${resId} at binding ${b}`);
        entries.push({ binding: b, resource: view });
      } else if (type === 'sampler') {
        const sampler = this._samplerLookup.get(resId);
        if (!sampler) {
          console.error(`[engine]   MISSING sampler: ${resId} at binding ${b} (samplerLookup keys: ${[...this._samplerLookup.keys()]})`);
        } else {
          entries.push({ binding: b, resource: sampler });
        }
      } else if (type === 'storage-texture') {
        const view = this.texViews.get(resId);
        if (!view) console.error(`[engine]   MISSING storage texture view: ${resId} at binding ${b}`);
        entries.push({ binding: b, resource: view });
      }
    }
    return entries;
  }

  // =========================================================================
  // Camera
  // =========================================================================

  _updateCameraBuffer() {
    const buf = this.camera.updateUniforms(
      this.canvas.width || 1, this.canvas.height || 1
    );
    if (!buf) return;

    // Use buffer_id from scene metadata if available
    if (!this._cameraBufferId) {
      this._cameraBufferId = (this.scene.camera && this.scene.camera.buffer_id) || null;
      // Fallback: first uniform buffer
      if (!this._cameraBufferId) {
        for (const [id, info] of Object.entries(this.scene.buffers)) {
          if (info.usage === 'uniform') {
            this._cameraBufferId = id;
            break;
          }
        }
      }
    }
    if (this._cameraBufferId) {
      this.device.queue.writeBuffer(this.buffers.get(this._cameraBufferId), 0, buf);
    }
  }

  // =========================================================================
  // Light
  // =========================================================================

  _applyLight() {
    if (!this.scene.light || !this.scene.light.data) return;
    // Light data is base64-encoded bytes
    const b64 = this.scene.light.data;
    const raw = atob(b64);
    const data = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) data[i] = raw.charCodeAt(i);

    // Use buffer_id from metadata, fallback to second uniform buffer
    let lightBufId = this.scene.light.buffer_id || null;
    if (!lightBufId) {
      for (const [id, info] of Object.entries(this.scene.buffers)) {
        if (info.usage === 'uniform' && id !== this._cameraBufferId) {
          lightBufId = id;
          break;
        }
      }
    }
    if (lightBufId) {
      this.device.queue.writeBuffer(this.buffers.get(lightBufId), 0, data);
    }
  }

  _rebuildRenderBindGroups() {
    for (let i = 0; i < this.renderPassObjects.length; i++) {
      const rp = this.scene.render_passes[i];
      const bindings = this._intKeyBindings(rp.bindings);
      const entries = this._buildBindGroupEntries(bindings);
      // Use the explicit layout the render pipeline was built with — the select
      // pipeline shares it, so the rebuilt bind group stays valid for both.
      const layout = this.renderPassObjects[i].bindGroupLayout
        || this.renderPassObjects[i].pipeline.getBindGroupLayout(0);
      this.renderPassObjects[i].bindGroup = this.device.createBindGroup({
        layout, entries, label: rp.id,
      });
      // Also update direct buffer references in case a buffer was recreated
      if (rp.vertex_buffers) {
        this.renderPassObjects[i].vertexBufferRefs = rp.vertex_buffers.map(vb => this.buffers.get(vb.buffer_id));
      }
      if (rp.draw_indirect) {
        this.renderPassObjects[i].indirectBuffer = this.buffers.get(rp.draw_indirect);
      }
      if (rp.index_buffer_id) {
        this.renderPassObjects[i].indexBuffer = this.buffers.get(rp.index_buffer_id);
      }
    }
  }

  // =========================================================================
  // Select (object-id / pick) pass
  // =========================================================================

  /**
   * Lazily build a "select" render pipeline for every pass that declared a
   * select shader. It reuses the pass's explicit bind-group layout (so the
   * pass's existing bind group is valid as-is) and renders the SELECT_PIPELINE
   * shader variant into an rgba32uint object-id target with a non-MSAA depth
   * buffer. Rebuilt automatically after update()/_createRenderPipelines, which
   * replace renderPassObjects (selectPipeline resets to null).
   */
  _ensureSelectPipelines() {
    if (!this.renderPassObjects) return;
    const device = this.device;
    for (const po of this.renderPassObjects) {
      if (po.selectPipeline === 'failed') continue;  // don't retry a bad pass every select
      if (po.selectPipeline || !po.selectShader || !po.selectEntryPoint) continue;
      try {
        const module = device.createShaderModule({ code: po.selectShader, label: po.id + '-select' });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [po.bindGroupLayout] });
        po.selectPipeline = device.createRenderPipeline({
          layout: pipelineLayout,
          vertex: { module, entryPoint: po.vertexEntryPoint, buffers: po.vertexBufferLayouts },
          // Integer (uint) targets cannot blend.
          fragment: { module, entryPoint: po.selectEntryPoint, targets: [{ format: SELECT_FORMAT }] },
          primitive: { topology: po.topology },
          // Always write/test depth so the front-most object wins per pixel,
          // even for passes that are transparent in the visible render.
          depthStencil: {
            format: DEPTH_FORMAT,
            depthWriteEnabled: true,
            depthCompare: 'less',
            depthBias: po.depthBias || 0,
          },
          multisample: { count: 1 },
          label: po.id + '-select',
        });
      } catch (e) {
        console.warn('[engine] select pipeline build failed for', po.id, e && (e.message || e));
        po.selectPipeline = 'failed';
      }
    }
  }

  /**
   * Live-mode hook: render the object-id / pick buffer into host-owned textures.
   *
   * The host (Python Scene._do_select / get_position) owns the select color +
   * depth textures and reads a pixel back itself; we just populate them. Unlike
   * the host's own Python select path, this reuses every pass's real GPU
   * buffers — including the ones the compute DAG filled (clip cross-section
   * cut_trigs + its indirect draw count, surface/clip vectors, …) — so the pick
   * buffer matches exactly what is drawn on screen. colorTexture must be
   * SELECT_FORMAT; depthTexture must be DEPTH_FORMAT; both non-MSAA and the same
   * size as the canvas.
   */
  renderSelectInto(colorTexture, depthTexture) {
    if (!this.device || !this.computeDAG) return;
    if (!colorTexture || !depthTexture) return;
    this._ensureSelectPipelines();

    const device = this.device;
    const encoder = device.createCommandEncoder();

    // Do NOT run the compute DAG here. The on-screen render loop owns the
    // execute()→processReadbacks() convergence cycle for count-then-fill passes
    // (clip cross-section, surface/clip vectors); running execute() out-of-band
    // from this hover-triggered path consumes the dirty set, bumps generation
    // counters and schedules readbacks at the wrong time, breaking that
    // convergence (vectors stop drawing, stale geometry appears mid-gesture).
    // The select pass only reads the already-filled buffers, so reuse them
    // as-is — the latest on-screen frame has populated them.

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: colorTexture.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1.0,
      },
    });

    for (const po of this.renderPassObjects) {
      if (po.enabled === false || !po.selectPipeline || po.selectPipeline === 'failed') continue;
      pass.setPipeline(po.selectPipeline);
      pass.setBindGroup(0, po.bindGroup);
      if (po.vertexBufferRefs) {
        for (let i = 0; i < po.vertexBufferRefs.length; i++) {
          pass.setVertexBuffer(i, po.vertexBufferRefs[i]);
        }
      }
      if (po.indexBuffer) {
        pass.setIndexBuffer(po.indexBuffer, po.indexFormat);
      }
      if (po.drawIndirect && po.indexBuffer) {
        pass.drawIndexedIndirect(po.indirectBuffer, 0);
      } else if (po.drawIndirect) {
        pass.drawIndirect(po.indirectBuffer, 0);
      } else if (po.indexBuffer) {
        pass.drawIndexed(po.vertexCount, po.instanceCount);
      } else {
        pass.draw(po.vertexCount, po.instanceCount);
      }
    }

    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // =========================================================================
  // Render
  // =========================================================================

  /**
   * Coalesced render entry point. Many render requests within a single frame
   * (e.g. dragging the clipping plane fires a mousemove + scene.render() per
   * event, each marking the clipping compute pass dirty) collapse into one
   * requestAnimationFrame, so the compute DAG + render pass run at most once
   * per frame instead of once per event.
   *
   * Callers that need the actual frame to run (capture, readback) still work:
   * a frame is guaranteed to be scheduled, and capture requests / readbacks are
   * processed when it runs.
   */
  render() {
    if (this._headless) { this._renderNow(); return; }
    if (this._renderScheduled) return;
    if (typeof requestAnimationFrame === 'undefined') { this._renderNow(); return; }
    this._renderScheduled = true;
    requestAnimationFrame(() => {
      this._renderScheduled = false;
      this._renderNow();
    });
  }

  _renderNow(captureTarget = null, autoReadback = true) {
    if (!this.device || !this.context) return;
    if (this.width === 0 || this.height === 0) return;
    if (this._updating) return;

    const device = this.device;
    const encoder = device.createCommandEncoder();

    // Run compute passes
    this.computeDAG.execute(device, encoder, this.buffers);

    const target = captureTarget || (this._headless ? this._ensureCaptureTexture() : null);
    const canvasTexture = target || this.context.getCurrentTexture();

    // Reconcile our MSAA/depth attachments to the ACTUAL swapchain size. The
    // canvas backing store (which getCurrentTexture() follows) is also resized by
    // the host (Python Canvas.resize) on layout changes, independently of our
    // handleResize(). Between those two events a frame would mix a fresh
    // swapchain with a stale-size MSAA target — WebGPU rejects the whole command
    // buffer ("resolve target size does not match the other attachments" + a
    // flood of "[Invalid CommandBuffer]"). Recreating on mismatch here makes the
    // render pass self-consistent every frame, whoever triggered the resize.
    if (!this.msaaTexture
        || this.msaaTexture.width !== canvasTexture.width
        || this.msaaTexture.height !== canvasTexture.height) {
      const w = canvasTexture.width, h = canvasTexture.height;
      if (this.depthTexture) this.depthTexture.destroy();
      if (this.msaaTexture) this.msaaTexture.destroy();
      this.depthTexture = device.createTexture({
        size: [w, h], format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: SAMPLE_COUNT,
        label: 'depth',
      });
      this.msaaTexture = device.createTexture({
        size: [w, h], format: this.canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: SAMPLE_COUNT,
        label: 'msaa',
      });
      this.width = w; this.height = h;
      this._updateCameraBuffer();   // refresh aspect/width/height for the new size
    }

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.msaaTexture.createView(),
        resolveTarget: (target || canvasTexture).createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: this.clearColor || LIGHT_CLEAR_COLOR,
      }],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1.0,
      },
    });

    for (const pass of this.renderPassObjects) {
      if (pass.enabled === false) continue;
      renderPass.setPipeline(pass.pipeline);
      renderPass.setBindGroup(0, pass.bindGroup);
      if (pass.vertexBufferRefs) {
        for (let i = 0; i < pass.vertexBufferRefs.length; i++) {
          renderPass.setVertexBuffer(i, pass.vertexBufferRefs[i]);
        }
      }
      if (pass.indexBuffer) {
        renderPass.setIndexBuffer(pass.indexBuffer, pass.indexFormat);
      }
      if (pass.drawIndirect && pass.indexBuffer) {
        // Indexed geometry (e.g. arrow glyphs) with a GPU-computed instance
        // count: 5-u32 indirect buffer filled by the countThenFill pass.
        renderPass.drawIndexedIndirect(pass.indirectBuffer, 0);
      } else if (pass.drawIndirect) {
        renderPass.drawIndirect(pass.indirectBuffer, 0);
      } else if (pass.indexBuffer) {
        renderPass.drawIndexed(pass.vertexCount, pass.instanceCount);
      } else {
        renderPass.draw(pass.vertexCount, pass.instanceCount);
      }
    }

    renderPass.end();

    // If a captureNextFrame() is pending, copy the resolved canvas texture
    // into a staging buffer before submission.
    let pendingCaptures = null;
    let captureBuffer = null;
    let captureMeta = null;
    if (this._frameCaptureRequests && this._frameCaptureRequests.length) {
      pendingCaptures = this._frameCaptureRequests;
      this._frameCaptureRequests = [];
      const w = canvasTexture.width;
      const h = canvasTexture.height;
      const bpp = _bytesPerPixel(this.canvasFormat) || 4;
      const bytesPerRow = Math.ceil((w * bpp) / 256) * 256;
      captureBuffer = device.createBuffer({
        size: bytesPerRow * h,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'frame-capture',
      });
      encoder.copyTextureToBuffer(
        { texture: canvasTexture },
        { buffer: captureBuffer, bytesPerRow, rowsPerImage: h },
        [w, h, 1],
      );
      captureMeta = { width: w, height: h, bytesPerRow, format: this.canvasFormat };
    }

    device.queue.submit([encoder.finish()]);

    if (pendingCaptures) {
      captureBuffer.mapAsync(GPUMapMode.READ).then(() => {
        const src = new Uint8Array(captureBuffer.getMappedRange()).slice();
        captureBuffer.unmap();
        captureBuffer.destroy();
        // Strip row padding so callers get tight width*bpp*height bytes.
        const { width, height, bytesPerRow, format } = captureMeta;
        const bpp = _bytesPerPixel(format) || 4;
        const tight = new Uint8Array(width * height * bpp);
        for (let y = 0; y < height; y++) {
          tight.set(src.subarray(y * bytesPerRow, y * bytesPerRow + width * bpp), y * width * bpp);
        }
        // Resolve with the raw ArrayBuffer so the websocket bridge transfers
        // the bytes via its dedicated buffer channel rather than serializing
        // a typed-array proxy.
        for (const resolve of pendingCaptures) resolve({ data: tight.buffer, width, height, format });
      });
    }

    // Async readback for count-then-fill passes. Skipped when a caller drives
    // the readbacks itself (settle()) to converge synchronously.
    if (autoReadback) {
      this.computeDAG.processReadbacks(this.device, this.buffers, () => {
        this._rebuildRenderBindGroups();
        this.render();
      });
    }
  }

  // =========================================================================
  // Resize
  // =========================================================================

  _onResize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (w === this.width && h === this.height) return;
    if (w === 0 || h === 0) return;

    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;

    // Recreate depth and MSAA textures
    this.depthTexture.destroy();
    this.msaaTexture.destroy();
    this._createDepthAndMSAA();

    this._updateCameraBuffer();
    this.render();
  }

  // =========================================================================
  // Helpers
  // =========================================================================

  /** Convert string-keyed bindings dict to int-keyed object */
  _intKeyBindings(bindings) {
    const out = {};
    for (const [k, v] of Object.entries(bindings)) out[parseInt(k)] = v;
    return out;
  }

  _isDarkMode() {
    try {
      // Prefer explicit page-level theme (e.g. pydata-sphinx-theme toggle)
      const htmlTheme = document.documentElement && document.documentElement.dataset && document.documentElement.dataset.theme;
      if (htmlTheme === 'dark') return true;
      if (htmlTheme === 'light') return false;
      // Fall back to OS-level preference
      return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    } catch {
      return false;
    }
  }

  _applyTheme() {
    const dark = this._isDarkMode();
    const ov = this._clearColorOverride;
    this.clearColor = ov
      ? { r: ov[0], g: ov[1], b: ov[2], a: ov.length > 3 ? ov[3] : 1.0 }
      : (dark ? DARK_CLEAR_COLOR : LIGHT_CLEAR_COLOR);
    if (this.canvas && this.canvas.style) {
      this.canvas.style.backgroundColor = ov
        ? `rgb(${Math.round(ov[0] * 255)},${Math.round(ov[1] * 255)},${Math.round(ov[2] * 255)})`
        : (dark ? DARK_CANVAS_BG : LIGHT_CANVAS_BG);
    }
    // Update colorbar background color to match theme
    if (this.scene && this.scene.theme && this.scene.theme.buffer_id) {
      const buf = this.buffers && this.buffers.get(this.scene.theme.buffer_id);
      if (buf) {
        const c = this.clearColor;
        this.device.queue.writeBuffer(buf, 16, new Float32Array([c.r, c.g, c.b]));
      }
    }
  }

  _setupThemeObserver() {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    this._themeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    this._onThemeChange = () => {
      this._applyTheme();
      this.render();
    };
    if (this._themeMediaQuery.addEventListener) {
      this._themeMediaQuery.addEventListener('change', this._onThemeChange);
    } else if (this._themeMediaQuery.addListener) {
      this._themeMediaQuery.addListener(this._onThemeChange);
    }
    // Also observe data-theme attribute changes (e.g. pydata-sphinx-theme)
    if (document.documentElement && typeof MutationObserver !== 'undefined') {
      this._themeAttrObserver = new MutationObserver(() => {
        this._applyTheme();
        this.render();
      });
      this._themeAttrObserver.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['data-theme'],
      });
    }
  }

  _teardownThemeObserver() {
    if (!this._themeMediaQuery || !this._onThemeChange) return;
    if (this._themeMediaQuery.removeEventListener) {
      this._themeMediaQuery.removeEventListener('change', this._onThemeChange);
    } else if (this._themeMediaQuery.removeListener) {
      this._themeMediaQuery.removeListener(this._onThemeChange);
    }
    this._themeMediaQuery = null;
    this._onThemeChange = null;
    if (this._themeAttrObserver) {
      this._themeAttrObserver.disconnect();
      this._themeAttrObserver = null;
    }
  }

  dispose() {
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this._camNotifyTimer) { clearTimeout(this._camNotifyTimer); this._camNotifyTimer = null; }
    this._teardownThemeObserver();
    if (this.input) this.input.dispose();
    if (this.interactions) this.interactions.dispose();
    if (this.depthTexture) this.depthTexture.destroy();
    if (this.msaaTexture) this.msaaTexture.destroy();
    // In live mode the host owns buffers/textures/samplers and the device.
    if (this.mode !== 'live') {
      for (const buf of this.buffers.values()) buf.destroy();
      for (const tex of this.textures.values()) tex.destroy();
      if (this.device) this.device.destroy();
    } else if (this.computeDAG) {
      // Live mode: the host owns the buffers/textures it supplied and frees
      // them itself (Python GC -> deferred flush_pending_destroys). But buffers
      // the ENGINE created — countThenFill compute outputs it resized, plus
      // indirect/staging/cap scratch — have no Python wrapper, so free them
      // here or they leak each time a live scene is discarded.
      this.computeDAG.dispose(this.buffers);
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers shared by createLive
// ---------------------------------------------------------------------------

/** Accept either a Map or a plain object and return a Map. */
function _toMap(x) {
  if (!x) return new Map();
  if (x instanceof Map) return x;
  return new Map(Object.entries(x));
}

/** Recover buffer usage category from the id prefix used by BufferRegistry. */
function _usageFromId(id) {
  if (id.startsWith('uniform_'))  return 'uniform';
  if (id.startsWith('storage_'))  return 'storage';
  if (id.startsWith('vertex_'))   return 'vertex';
  if (id.startsWith('index_'))    return 'index';
  if (id.startsWith('indirect_')) return 'indirect';
  return 'storage';
}

// ---------------------------------------------------------------------------
// Texture format → bytes per pixel
// ---------------------------------------------------------------------------

function _bytesPerPixel(format) {
  const map = {
    'rgba8unorm': 4, 'rgba8snorm': 4, 'rgba8uint': 4, 'rgba8sint': 4,
    'bgra8unorm': 4, 'rg32float': 8, 'rgba16float': 8, 'rgba32float': 16,
    'r8unorm': 1, 'r16float': 2, 'r32float': 4, 'rg8unorm': 2, 'rg16float': 4,
  };
  return map[format] || 4;
}

export { RenderEngine };
