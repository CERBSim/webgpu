// RenderEngine — core WebGPU renderer for exported scene blobs.
// Depends on: parseSceneBlob (format.js), Camera (camera.js),
//             ComputeDAG (compute.js), InputHandler (input.js)
// All are concatenated into the same scope by the Python loader — no imports.

const SAMPLE_COUNT = 4;
const CLEAR_COLOR = { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
const DEPTH_FORMAT = 'depth24plus';

const TOPOLOGY_MAP = {
  'triangle-list': 'triangle-list',
  'triangle-strip': 'triangle-strip',
  'line-list': 'line-list',
  'line-strip': 'line-strip',
  'point-list': 'point-list',
};

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

  async init(canvas, arrayBuffer) {
    this.canvas = canvas;

    // --- Parse blob ---
    this.scene = parseSceneBlob(arrayBuffer);

    // --- WebGPU device ---
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
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

    // --- Compute ---
    this.computeDAG = new ComputeDAG();
    for (const cp of this.scene.compute_passes) {
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
      } : null;
      this.computeDAG.addPass(cp.id, {
        shader: cp.shader,
        bindings: this._intKeyBindings(cp.bindings),
        workgroups: cp.workgroups,
        triggers: cp.triggers,
        resetBuffers: cp.reset_buffers,
        indirectSetup,
        countThenFill,
      });
    }
    await this.computeDAG.initPipelines(this.device, this.buffers);
    // Mark all compute triggers dirty for first frame
    for (const cp of this.scene.compute_passes) {
      for (const t of cp.triggers) this.computeDAG.markDirty(t);
    }

    // --- Render pipelines ---
    this.renderPassObjects = [];
    await this._createRenderPipelines();

    // --- Input ---
    this.camera.registerObserver(() => {
      this._updateCameraBuffer();
      // Mark camera-dependent compute passes dirty
      for (const cp of this.scene.compute_passes) {
        for (const t of cp.triggers) this.computeDAG.markDirty(t);
      }
      this.render();
    });
    this.input = new InputHandler(canvas, this.camera, () => this.render());

    // --- Interactions (lil-gui) ---
    const guiContainerId = canvas.id.replace('canvas', 'lilgui');
    const guiContainer = document.getElementById(guiContainerId);
    this.interactions = new Interactions(this, guiContainer);
    if (this.scene.interactions && this.scene.interactions.length > 0) {
      await this.interactions.setup(this.scene.interactions);
    }

    // --- Resize handling ---
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(canvas);

    // --- First frame ---
    this.render();
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
    const w = this.canvas.width || this.canvas.clientWidth;
    const h = this.canvas.height || this.canvas.clientHeight;
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
      primitive: { topology: TOPOLOGY_MAP[rp.topology] || 'triangle-list' },
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
      vertexCount: rp.vertex_count,
      instanceCount: rp.instance_count,
      drawIndirect: !!rp.draw_indirect,
      indirectBuffer,
      vertexBufferRefs,
      indexBuffer,
      indexFormat,
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
      const layout = this.renderPassObjects[i].pipeline.getBindGroupLayout(0);
      this.renderPassObjects[i].bindGroup = this.device.createBindGroup({
        layout, entries, label: rp.id,
      });
    }
  }

  // =========================================================================
  // Render
  // =========================================================================

  render() {
    if (!this.device || !this.context) return;
    if (this.width === 0 || this.height === 0) return;

    const device = this.device;
    const encoder = device.createCommandEncoder();

    // Run compute passes
    this.computeDAG.execute(device, encoder, this.buffers);

    const canvasTexture = this.context.getCurrentTexture();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.msaaTexture.createView(),
        resolveTarget: canvasTexture.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: CLEAR_COLOR,
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
      if (pass.drawIndirect) {
        renderPass.drawIndirect(pass.indirectBuffer, 0);
      } else if (pass.indexBuffer) {
        renderPass.drawIndexed(pass.vertexCount, pass.instanceCount);
      } else {
        renderPass.draw(pass.vertexCount, pass.instanceCount);
      }
    }

    renderPass.end();
    device.queue.submit([encoder.finish()]);

    // Async readback for count-then-fill passes
    this.computeDAG.processReadbacks(this.device, this.buffers, () => {
      this._rebuildRenderBindGroups();
      this.render();
    });
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

  dispose() {
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this.input) this.input.dispose();
    if (this.interactions) this.interactions.dispose();
    for (const buf of this.buffers.values()) buf.destroy();
    for (const tex of this.textures.values()) tex.destroy();
    if (this.depthTexture) this.depthTexture.destroy();
    if (this.msaaTexture) this.msaaTexture.destroy();
    if (this.device) this.device.destroy();
  }
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
