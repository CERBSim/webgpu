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
