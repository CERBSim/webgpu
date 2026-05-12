// Interactions — GUI controls that update uniform buffers and trigger compute/render

async function loadLilGui() {
  if (window.lil && window.lil.GUI) return window.lil.GUI;
  if (window.createLilGUI) {
    // link.js provides createLilGUI — call it to ensure lil-gui is loaded
    const gui = await window.createLilGUI({ autoPlace: false });
    gui.destroy();
    return window.lil.GUI;
  }
  const url = 'https://cdn.jsdelivr.net/npm/lil-gui@0.20';
  if (window.define === undefined) {
    await import(url);
  } else {
    await new Promise((resolve) => {
      require([url], (module) => {
        window.lil = module;
        resolve();
      });
    });
  }
  return window.lil.GUI;
}

// ---------------------------------------------------------------------------
// Interaction handlers
// ---------------------------------------------------------------------------

const interactionHandlers = {};

// --- Clipping Plane ---
// ClippingUniforms (48 bytes):
//   plane:   vec4<f32> [0..15]  — (nx, ny, nz, d)
//   sphere:  vec4<f32> [16..31] — (cx, cy, cz, radius)
//   mode:    u32       [32..35]
//   padding: u32*3     [36..47]

interactionHandlers.clipping_plane = function (engine, interaction, gui) {
  const cfg = interaction.config;
  const bufferId = interaction.buffer_id;
  const buffer = engine.buffers.get(bufferId);

  const state = {
    nx: cfg.normal[0],
    ny: cfg.normal[1],
    nz: cfg.normal[2],
    offset: cfg.offset || 0,
    mode: cfg.mode || 1,  // Default to PLANE mode
    enabled: cfg.mode !== 0 && cfg.mode !== undefined,
  };
  const center = [...cfg.center];
  const radius = cfg.radius || 1.0;

  function writeBuffer() {
    let nx = state.nx, ny = state.ny, nz = state.nz;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (len > 1e-12) { nx /= len; ny /= len; nz /= len; }
    else { nx = 0; ny = 0; nz = -1; }

    const cx = center[0] + nx * state.offset;
    const cy = center[1] + ny * state.offset;
    const cz = center[2] + nz * state.offset;
    const d = -(cx * nx + cy * ny + cz * nz);

    const data = new ArrayBuffer(48);
    const f32 = new Float32Array(data);
    const u32 = new Uint32Array(data);

    f32[0] = nx; f32[1] = ny; f32[2] = nz; f32[3] = d;
    f32[4] = cx; f32[5] = cy; f32[6] = cz; f32[7] = radius;
    u32[8] = state.enabled ? state.mode : 0;
    u32[9] = 0; u32[10] = 0; u32[11] = 0;

    engine.device.queue.writeBuffer(buffer, 0, data);
    if (engine.computeDAG) engine.computeDAG.markDirty(bufferId);
    engine.render();
  }

  const folder = gui.addFolder('Clipping');
  folder.add(state, 'enabled').name('Enabled').onChange(writeBuffer);
  folder.add(state, 'nx', -1, 1, 0.01).name('Normal X').onChange(writeBuffer);
  folder.add(state, 'ny', -1, 1, 0.01).name('Normal Y').onChange(writeBuffer);
  folder.add(state, 'nz', -1, 1, 0.01).name('Normal Z').onChange(writeBuffer);
  folder.add(state, 'offset', -2, 2, 0.01).name('Offset').onChange(writeBuffer);
};

// --- Colormap Range ---
// ColormapUniforms: min (f32), max (f32), discrete (u32), n_colors (u32)
// config provides offset_min / offset_max (byte offsets into the buffer)

interactionHandlers.colormap_range = function (engine, interaction, gui) {
  const cfg = interaction.config;
  const bufferId = interaction.buffer_id;
  const buffer = engine.buffers.get(bufferId);

  const offsetMin = cfg.offset_min !== undefined ? cfg.offset_min : 0;
  const offsetMax = cfg.offset_max !== undefined ? cfg.offset_max : 4;

  const state = {
    min: cfg.min !== undefined ? cfg.min : 0,
    max: cfg.max !== undefined ? cfg.max : 1,
  };

  function writeBuffer() {
    const tmp = new Float32Array(1);
    tmp[0] = state.min;
    engine.device.queue.writeBuffer(buffer, offsetMin, tmp);
    tmp[0] = state.max;
    engine.device.queue.writeBuffer(buffer, offsetMax, tmp);
    engine.render();
  }

  const folder = gui.addFolder('Colormap');
  folder.add(state, 'min', -10, 10, 0.01).name('Min').onChange(writeBuffer);
  folder.add(state, 'max', -10, 10, 0.01).name('Max').onChange(writeBuffer);
};

// --- Time Animation ---
// config: { frames: [frame_buf_id_for_target_0, frame_buf_id_for_target_1, ...] }
// buffer_id: the live GPU buffer to overwrite when the slider moves.
// On change, copies the selected frame's bytes into the GPU buffer.

interactionHandlers.time_animation = function (engine, interaction, gui) {
  const cfg = interaction.config;
  const targets = cfg.targets || [
    { buffer_id: interaction.buffer_id, frames: cfg.frames || [] },
  ];
  const nFrames = targets.length > 0 ? targets[0].frames.length : 0;
  if (nFrames === 0) return;

  const state = { time: 0 };

  function applyFrame() {
    const i = Math.max(0, Math.min(nFrames - 1, Math.round(state.time)));
    for (const t of targets) {
      const buf = engine.buffers.get(t.buffer_id);
      const data = engine.frameBuffers.get(t.frames[i]);
      if (buf && data) {
        engine.device.queue.writeBuffer(buf, 0, data);
      }
    }
    engine.render();
  }

  const folder = gui.addFolder(cfg.label || 'Animation');
  folder.add(state, 'time', 0, nFrames - 1, 1).name('Frame').onChange(applyFrame);

  // Apply initial frame so the canvas matches frame 0 on load.
  applyFrame();
};

// --- Generic GUI ---
// config: {
//   label: string,
//   vars: { name: defaultValue, ... },
//   controls: [{kind: 'checkbox'|'slider'|'dropdown', var, name, ...kindOpts}],
//   writes: [{
//     targets: [{buffer_id, offset, dtype: 'f32'|'u32'|'i32'}],
//     expr?: string,        // JS expr over vars + t (seconds since loop start)
//     value?: any,          // constant (used if expr absent)
//     when?: string,        // JS expr; if falsy, skip
//     trigger?: string,     // var name; one-shot on change instead of per-frame
//   }, ...]
// }
//
// Per-frame writes (no `trigger`) run inside a requestAnimationFrame loop
// while at least one of them has a truthy `when`. Trigger writes fire once
// whenever the named var changes via the GUI.

interactionHandlers.gui = function (engine, interaction, gui) {
  const cfg = interaction.config;
  const vars = { ...(cfg.vars || {}) };
  const controls = cfg.controls || [];
  const writes = cfg.writes || [];

  function compile(expr) {
    if (expr == null) return null;
    const keys = Object.keys(vars);
    const decl = keys.length
      ? `const {${keys.join(',')}} = __v;`
      : '';
    return new Function('__v', 't', `${decl} return (${expr});`);
  }

  const compiled = writes.map(w => ({
    targets: w.targets || [],
    exprFn: compile(w.expr),
    whenFn: compile(w.when),
    value: w.value,
    trigger: w.trigger,
  }));

  function dtypeArray(dtype, v) {
    if (dtype === 'u32') return new Uint32Array([v]);
    if (dtype === 'i32') return new Int32Array([v]);
    return new Float32Array([v]);
  }

  function executeWrite(w, t) {
    if (w.whenFn && !w.whenFn(vars, t)) return;
    const v = w.exprFn ? w.exprFn(vars, t) : w.value;
    if (v == null) return;
    for (const tgt of w.targets) {
      const buf = engine.buffers.get(tgt.buffer_id);
      if (!buf) continue;
      engine.device.queue.writeBuffer(
        buf, tgt.offset || 0, dtypeArray(tgt.dtype || 'f32', v),
      );
    }
  }

  const perFrame = compiled.filter(w => !w.trigger);
  const triggers = compiled.filter(w => w.trigger);

  let raf = null;
  let t0 = 0;

  function needsLoop() {
    for (const w of perFrame) {
      if (!w.whenFn || w.whenFn(vars, 0)) return true;
    }
    return false;
  }

  function loop(now) {
    const t = (now - t0) / 1000;
    for (const w of perFrame) executeWrite(w, t);
    engine.render();
    raf = needsLoop() ? requestAnimationFrame(loop) : null;
  }

  function ensureLoop() {
    if (raf == null && needsLoop()) {
      t0 = performance.now();
      raf = requestAnimationFrame(loop);
    }
  }

  function onVarChange(varName) {
    for (const w of triggers) {
      if (w.trigger === varName) executeWrite(w, 0);
    }
    if (triggers.length) engine.render();
    ensureLoop();
  }

  const folder = gui.addFolder(cfg.label || 'Controls');
  for (const c of controls) {
    const name = c.name || c.var;
    if (c.kind === 'checkbox') {
      folder.add(vars, c.var).name(name).onChange(() => onVarChange(c.var));
    } else if (c.kind === 'slider') {
      folder.add(vars, c.var, c.min, c.max, c.step || 0.01)
            .name(name).onChange(() => onVarChange(c.var));
    } else if (c.kind === 'dropdown') {
      folder.add(vars, c.var, c.options).name(name)
            .onChange(() => onVarChange(c.var));
    }
  }
};

// ---------------------------------------------------------------------------

class Interactions {
  constructor(engine, guiContainer) {
    this.engine = engine;
    this.guiContainer = guiContainer;
    this.gui = null;
  }

  async setup(interactions) {
    if (!interactions || interactions.length === 0) return;

    const GUI = await loadLilGui();
    this.gui = new GUI({ container: this.guiContainer, autoPlace: !this.guiContainer });
    this.gui.close();

    for (const interaction of interactions) {
      const handler = interactionHandlers[interaction.type];
      if (handler) {
        handler(this.engine, interaction, this.gui);
      } else {
        console.warn('Unknown interaction type:', interaction.type);
      }
    }
  }

  dispose() {
    if (this.gui) {
      this.gui.destroy();
      this.gui = null;
    }
  }
}

export { Interactions };
