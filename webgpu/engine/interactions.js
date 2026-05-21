// Interactions — GUI controls that update uniform buffers and trigger compute/render

/**
 * Get an existing child folder by title, or create a new one.
 * Prevents duplicate folders when multiple renderers export the same interaction type.
 */
function getOrCreateFolder(gui, title, { open = false } = {}) {
  for (const f of gui.folders) {
    if (f._title === title) return f;
  }
  const folder = gui.addFolder(title);
  if (!open) folder.close();

  // Remove any 'Empty' placeholder leaf nodes that lil-gui inserts for empty folders.
  try {
    const el = folder.domElement;
    if (el) {
      // Remove any leaf element whose trimmed text content is exactly 'Empty'
      const leaves = el.querySelectorAll('*');
      for (const node of leaves) {
        if (node.childElementCount === 0 && node.textContent && node.textContent.trim() === 'Empty') {
          node.remove();
        }
      }
    }
  } catch (e) {
    // ignore
  }

  return folder;
}

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

  const folder = getOrCreateFolder(gui, cfg.label || 'Animation', { open: !!cfg.open });
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
//                           // May return a TypedArray/ArrayBuffer for bulk writes.
//     value?: any,          // constant (used if expr absent)
//     when?: string,        // JS expr; if falsy, skip
//     trigger?: string,     // var name or '*' (any); one-shot on change
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
      if (tgt.dtype === 'pass_enable') {
        const pass = engine.renderPassObjects.find(p => p.id === tgt.buffer_id);
        if (pass) pass.enabled = !!v;
        continue;
      }
      const buf = engine.buffers.get(tgt.buffer_id);
      if (!buf) continue;
      // If the expr returned a typed array or ArrayBuffer, write it directly.
      const data = (v.buffer instanceof ArrayBuffer || v instanceof ArrayBuffer)
        ? v : dtypeArray(tgt.dtype || 'f32', v);
      engine.device.queue.writeBuffer(buf, tgt.offset || 0, data);
      // Mark the buffer dirty so compute passes that depend on it re-run.
      if (engine.computeDAG) engine.computeDAG.markDirty(tgt.buffer_id);
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
      if (w.trigger === '*' || w.trigger === varName) executeWrite(w, 0);
    }
    if (triggers.length) engine.render();
    ensureLoop();
  }

  const container = getOrCreateFolder(gui, cfg.label || 'Controls', { open: !!cfg.open });
  for (const c of controls) {
    const name = c.name || c.var;
    if (c.kind === 'checkbox') {
      container.add(vars, c.var).name(name).onChange(() => onVarChange(c.var));
    } else if (c.kind === 'slider') {
      container.add(vars, c.var, c.min, c.max, c.step || 0.01)
            .name(name).onChange(() => onVarChange(c.var));
    } else if (c.kind === 'dropdown') {
      container.add(vars, c.var, c.options).name(name)
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
    this.gui = new GUI({ container: this.guiContainer, autoPlace: !this.guiContainer, title: 'Controls' });
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
