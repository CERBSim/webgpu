// ---------------------------------------------------------------------------
// Helper: determine GPU resource type from id prefix
// ---------------------------------------------------------------------------

function resourceType(id) {
  if (id.startsWith('uniform_')) return 'buffer';
  if (id.startsWith('storage_')) return 'buffer';
  if (id.startsWith('tex_'))     return 'texture';
  if (id.startsWith('sampler_')) return 'sampler';
  if (id.startsWith('stex_'))    return 'storage-texture';
  return 'buffer'; // fallback
}

// ---------------------------------------------------------------------------
// Helper: build explicit GPUBindGroupLayout entries.
// For render passes, storage buffers are always read-only.
// For compute passes, parse shader to find read_write bindings.
// ---------------------------------------------------------------------------

function buildLayoutEntries(bindings, visibility, shader) {
  // For compute: find which bindings are read_write from shader declarations
  const rwBindings = new Set();
  if (shader) {
    const re = /@binding\((\d+)\)\s*var<storage\s*,\s*read_write>/g;
    let m;
    while ((m = re.exec(shader)) !== null) rwBindings.add(parseInt(m[1]));
  }

  const entries = [];
  for (const [bindingNum, resId] of Object.entries(bindings)) {
    const b = parseInt(bindingNum);
    const type = resourceType(resId);
    const entry = { binding: b, visibility };
    if (type === 'buffer') {
      if (resId.startsWith('uniform_')) {
        entry.buffer = { type: 'uniform' };
      } else if (rwBindings.has(b)) {
        entry.buffer = { type: 'storage' };
      } else {
        entry.buffer = { type: 'read-only-storage' };
      }
    } else if (type === 'texture') {
      entry.texture = { sampleType: 'float' };
    } else if (type === 'sampler') {
      entry.sampler = { type: 'filtering' };
    } else if (type === 'storage-texture') {
      entry.storageTexture = { access: 'write-only', format: 'rgba8unorm' };
    }
    entries.push(entry);
  }
  return entries;
}

// Built-in shader: cap the indirect instanceCount at the output buffer capacity.
const CAP_INDIRECT_SHADER = `
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> indirect: array<u32, 4>;
@group(0) @binding(2) var<uniform> max_instances: u32;

@compute @workgroup_size(1)
fn main() {
  let count = atomicLoad(&counter);
  indirect[1] = min(count, max_instances);
}
`;

class ComputeDAG {
  constructor() {
    this.passes = new Map(); // id → pass descriptor
    this.pipelines = new Map(); // id → GPUComputePipeline
    this.bindGroups = new Map(); // id → GPUBindGroup
    this.dirty = new Set(); // currently dirty trigger ids
    this.stagingBuffers = new Map(); // id → GPUBuffer for counter readback
    this._pendingReadbacks = [];
    this._readbackInFlight = false;
    this._capPipelines = new Map(); // id → { pipeline, bindGroup, maxBuf }
  }

  addPass(id, { shader, bindings, workgroups, triggers, resetBuffers, indirectSetup, countThenFill }) {
    this.passes.set(id, { shader, bindings, workgroups, triggers: triggers || [], resetBuffers: resetBuffers || [], indirectSetup: indirectSetup || null, countThenFill: countThenFill || null });
    this._checkCycles();
  }

  markDirty(triggerId) {
    this.dirty.add(triggerId);
  }

  execute(device, commandEncoder, buffers) {
    if (this.dirty.size === 0) return;

    // Collect triggered passes, cascading through the graph
    const triggered = new Set();
    const queue = [...this.dirty];
    while (queue.length > 0) {
      const tid = queue.shift();
      for (const [id, pass] of this.passes) {
        if (!triggered.has(id) && pass.triggers.includes(tid)) {
          triggered.add(id);
          queue.push(id); // this pass's id can trigger downstream
        }
      }
    }

    if (triggered.size === 0) {
      this.dirty.clear();
      return;
    }

    // Topo-sort triggered passes
    const sorted = this._topoSort(triggered);

    for (const id of sorted) {
      const pass = this.passes.get(id);
      const pipeline = this.pipelines.get(id);
      const bindGroup = this.bindGroups.get(id);

      // Clear reset buffers
      for (const bufId of pass.resetBuffers) {
        const buf = buffers.get(bufId);
        commandEncoder.clearBuffer(buf, 0, buf.size);
      }

      // Dispatch compute pass
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      const [x, y, z] = pass.workgroups;
      computePass.dispatchWorkgroups(x, y, z);
      computePass.end();

      // After dispatch: build drawIndirect args from atomic counter
      if (pass.indirectSetup) {
        const { counterId, indirectId, vertexCount } = pass.indirectSetup;
        const counterBuf = buffers.get(counterId);
        const indirectBuf = buffers.get(indirectId);
        device.queue.writeBuffer(indirectBuf, 0, new Uint32Array([vertexCount]));
        // Baseline: copy raw counter (fallback if cap shader unavailable)
        commandEncoder.copyBufferToBuffer(counterBuf, 0, indirectBuf, 4, 4);
        // Cap shader overwrites indirect[1] with min(counter, capacity)
        const cap = this._capPipelines.get(id + '_indirect');
        if (cap) {
          const capPass = commandEncoder.beginComputePass();
          capPass.setPipeline(cap.pipeline);
          capPass.setBindGroup(0, cap.bindGroup);
          capPass.dispatchWorkgroups(1);
          capPass.end();
        }
      }

      // countThenFill: set up indirect for current frame + copy counter to staging
      if (pass.countThenFill) {
        const { counterId, indirectId, vertexCount } = pass.countThenFill;
        if (indirectId) {
          const counterBuf = buffers.get(counterId);
          const indirectBuf = buffers.get(indirectId);
          device.queue.writeBuffer(indirectBuf, 0, new Uint32Array([vertexCount]));
          // Baseline: copy raw counter (fallback if cap shader unavailable)
          commandEncoder.copyBufferToBuffer(counterBuf, 0, indirectBuf, 4, 4);
          // Cap shader overwrites indirect[1] with min(counter, capacity)
          const cap = this._capPipelines.get(id + '_ctf');
          if (cap) {
            const capPass = commandEncoder.beginComputePass();
            capPass.setPipeline(cap.pipeline);
            capPass.setBindGroup(0, cap.bindGroup);
            capPass.dispatchWorkgroups(1);
            capPass.end();
          }
        }
        // Only copy to staging for resize readback if no readback is in flight,
        // otherwise the staging buffer is mapped/pending-map and the submit
        // would fail with "used in submit while mapped".
        if (!this._readbackInFlight) {
          const counterBuf = buffers.get(counterId);
          const staging = this.stagingBuffers.get(id);
          commandEncoder.copyBufferToBuffer(counterBuf, 0, staging, 0, 4);
          this._pendingReadbacks.push(id);
        }
      }

      this.dirty.add(id);
    }

    this.dirty.clear();
  }

  async initPipelines(device, buffers) {
    // Create the shared cap-indirect pipeline (one shader for all passes)
    if (!this._capModule) {
      try {
        this._capModule = device.createShaderModule({ code: CAP_INDIRECT_SHADER });
        const capLayout = device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          ],
        });
        this._capLayout = capLayout;
        this._capPipelineGPU = await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({ bindGroupLayouts: [capLayout] }),
          compute: { module: this._capModule, entryPoint: 'main' },
        });
      } catch (e) {
        console.warn('[ComputeDAG] Cap-indirect pipeline failed, using fallback:', e);
        this._capModule = null;
      }
    }

    for (const [id, pass] of this.passes) {
      const module = device.createShaderModule({ code: pass.shader });

      // Build explicit layout — parse shader for read_write bindings
      const vis = GPUShaderStage.COMPUTE;
      const layoutEntries = buildLayoutEntries(pass.bindings, vis, pass.shader);
      const bindGroupLayout = device.createBindGroupLayout({ entries: layoutEntries });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

      const pipeline = await device.createComputePipelineAsync({
        layout: pipelineLayout,
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(id, pipeline);

      const entries = Object.entries(pass.bindings).map(([binding, bufId]) => ({
        binding: parseInt(binding),
        resource: { buffer: buffers.get(bufId) },
      }));
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries,
      });
      this.bindGroups.set(id, bindGroup);

      if (pass.countThenFill) {
        const staging = device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          label: `${id}_staging`,
        });
        this.stagingBuffers.set(id, staging);
        this._buildCapBindGroup(device, buffers, id, '_ctf', pass.countThenFill);
      }
      if (pass.indirectSetup) {
        this._buildCapBindGroup(device, buffers, id, '_indirect', pass.indirectSetup);
      }
    }
  }

  _buildCapBindGroup(device, buffers, passId, suffix, setup) {
    const { counterId, indirectId, outputId, elementSize } = setup;
    const counterBuf = buffers.get(counterId);
    const indirectBuf = buffers.get(indirectId);
    // Compute max instances from output buffer size
    const outputBuf = outputId ? buffers.get(outputId) : null;
    const elemSize = elementSize || 64;
    const maxInstances = outputBuf ? Math.floor(outputBuf.size / elemSize) : 0xFFFFFFFF;

    // Destroy previous maxBuf if rebuilding
    const key = passId + suffix;
    const prev = this._capPipelines.get(key);
    if (prev && prev.maxBuf) prev.maxBuf.destroy();

    // Uniform buffer holding max_instances
    const maxBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: `${passId}${suffix}_cap_max`,
    });
    device.queue.writeBuffer(maxBuf, 0, new Uint32Array([maxInstances]));

    const bindGroup = device.createBindGroup({
      layout: this._capLayout,
      entries: [
        { binding: 0, resource: { buffer: counterBuf } },
        { binding: 1, resource: { buffer: indirectBuf } },
        { binding: 2, resource: { buffer: maxBuf } },
      ],
    });
    this._capPipelines.set(key, { pipeline: this._capPipelineGPU, bindGroup, maxBuf });
  }

  async processReadbacks(device, buffers, onRerender) {
    if (this._pendingReadbacks.length === 0) return;
    if (this._readbackInFlight) return;

    const passIds = [...this._pendingReadbacks];
    this._pendingReadbacks = [];
    this._readbackInFlight = true;

    let needsRerender = false;

    for (const id of passIds) {
      const pass = this.passes.get(id);
      const ctf = pass.countThenFill;
      const staging = this.stagingBuffers.get(id);

      await staging.mapAsync(GPUMapMode.READ);
      const count = new Uint32Array(staging.getMappedRange())[0];
      staging.unmap();

      const neededSize = Math.max(64, count * ctf.elementSize);
      const currentBuf = buffers.get(ctf.outputId);

      if (neededSize > currentBuf.size || neededSize < currentBuf.size / 4) {
        // Overallocate 2x when growing to provide headroom for rapid changes
        // (e.g. dragging clipping slider). Shrink uses exact size.
        const allocSize = neededSize > currentBuf.size ? neededSize * 2 : neededSize;
        const newBuf = device.createBuffer({
          size: allocSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
          label: ctf.outputId,
        });
        buffers.set(ctf.outputId, newBuf);
        this._rebuildBindGroups(device, buffers, ctf.outputId);
        // Rebuild cap bind group with updated max_instances
        this._buildCapBindGroup(device, buffers, id, '_ctf', ctf);
        needsRerender = true;
      }

      if (ctf.indirectId) {
        const indirectBuf = buffers.get(ctf.indirectId);
        device.queue.writeBuffer(indirectBuf, 0, new Uint32Array([ctf.vertexCount, count, 0, 0]));
      }
    }

    this._readbackInFlight = false;

    if (needsRerender) {
      for (const id of passIds) {
        const pass = this.passes.get(id);
        for (const t of pass.triggers) this.markDirty(t);
      }
      onRerender();
    }
  }

  _rebuildBindGroups(device, buffers, changedBufferId) {
    for (const [id, pass] of this.passes) {
      const usesBuffer = Object.values(pass.bindings).includes(changedBufferId);
      if (!usesBuffer) continue;

      const pipeline = this.pipelines.get(id);
      const layout = pipeline.getBindGroupLayout(0);
      const entries = Object.entries(pass.bindings).map(([binding, bufId]) => ({
        binding: parseInt(binding),
        resource: { buffer: buffers.get(bufId) },
      }));
      this.bindGroups.set(id, device.createBindGroup({ layout, entries }));
    }
  }

  // Build adjacency: if pass B has trigger "A" and A is a pass, then A → B
  _buildGraph() {
    const adj = new Map();
    const passIds = new Set(this.passes.keys());
    for (const id of passIds) adj.set(id, []);
    for (const [id, pass] of this.passes) {
      for (const t of pass.triggers) {
        if (passIds.has(t)) {
          adj.get(t).push(id);
        }
      }
    }
    return adj;
  }

  _checkCycles() {
    const adj = this._buildGraph();
    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = new Map();
    for (const id of adj.keys()) color.set(id, WHITE);

    const dfs = (u) => {
      color.set(u, GRAY);
      for (const v of adj.get(u)) {
        if (color.get(v) === GRAY) throw new Error(`ComputeDAG: cycle detected involving "${u}" and "${v}"`);
        if (color.get(v) === WHITE) dfs(v);
      }
      color.set(u, BLACK);
    };

    for (const id of adj.keys()) {
      if (color.get(id) === WHITE) dfs(id);
    }
  }

  _topoSort(triggered) {
    const adj = this._buildGraph();
    // Restrict to triggered passes
    const inDeg = new Map();
    for (const id of triggered) inDeg.set(id, 0);
    for (const id of triggered) {
      for (const v of adj.get(id)) {
        if (triggered.has(v)) {
          inDeg.set(v, inDeg.get(v) + 1);
        }
      }
    }

    const queue = [];
    for (const [id, deg] of inDeg) {
      if (deg === 0) queue.push(id);
    }

    const sorted = [];
    while (queue.length > 0) {
      const u = queue.shift();
      sorted.push(u);
      for (const v of adj.get(u)) {
        if (!triggered.has(v)) continue;
        const d = inDeg.get(v) - 1;
        inDeg.set(v, d);
        if (d === 0) queue.push(v);
      }
    }
    return sorted;
  }
}

export { ComputeDAG };
