console.log("loaded webgui_scene.js");
export default {
  template: `
    <div style="position:relative;" data-initializing>
      <canvas id="canvas" style=""></canvas>
      <div style="position:absolute;pointer-events:none;top:0"></div>
      <div style="position:absolute;pointer-events:none;top:0"></div>
    </div>`,

  mounted() {
    const canvas = this.$el.querySelector("canvas");
    canvas.width = this.width;
    canvas.height = this.height;
    console.log("mounted");
    main();
    this.is_initialized = false;
  },

  beforeDestroy() {
    //window.removeEventListener("resize", this.resize);
    //window.removeEventListener("DOMContentLoaded", this.resize);
  },

  methods: {
    create(type, id, parent_id, ...args) {
      if (!this.is_initialized) return;
    },
    name(object_id, name) {
      return "";
    },
    move(object_id, x, y, z) {},
    scale(object_id, sx, sy, sz) {},
    rotate(object_id, R) {},
    visible(object_id, value) {},
    delete(object_id) {},
    resize() {
      const { clientWidth, clientHeight } = this.$el;
      //this.renderer.setSize(clientWidth, clientHeight);
    },
    init_objects(data) {
      this.resize();
      this.$el.removeAttribute("data-initializing");
      this.is_initialized = true;
    },

    async run_user_function(data) {
      const user_function = pyodide.runPython(
        "import webgpu.main; webgpu.main.user_function",
      );
      await user_function(data);
    },
  },

  props: {
    width: Number,
    height: Number,
    click_events: Array,
    background_color: String,
  },
};

let pyodide = null;
console.log("load init_webgpu.js");

const files = [
  "__init__.py",
  "colormap.py",
  "font.py",
  "fonts.json",
  "gpu.py",
  "input_handler.py",
  "main.py",
  "mesh.py",
  "uniforms.py",
  "utils.py",
  "shader/__init__.py",
  "shader/eval.wgsl",
  "shader/shader.wgsl",
  "shader/compute.wgsl",
  "shader/mesh.wgsl",
  "shader/uniforms.wgsl",
];

async function reload() {
  try {
    pyodide.FS.mkdir("webgpu");
    pyodide.FS.mkdir("webgpu/shader");
  } catch {}
  for (var file of files) {
    const data = await (
      await fetch(`./webgpu/${file}`, { method: "GET", cache: "no-cache" })
    ).text();

    pyodide.FS.writeFile("webgpu/" + file, data);
  }
  await pyodide.runPythonAsync(
    "import webgpu.main; await webgpu.main.reload();",
  );
}

async function main() {
  console.log("loading pyodide", performance.now());
  // const blob = await (await fetch("./pyodide/snapshot.bin")).blob();
  // const decompressor = new DecompressionStream('gzip');
  // const stream = blob.stream().pipeThrough(decompressor);
  // const response = new Response(stream);
  // return await response.arrayBuffer();

  pyodide = await loadPyodide({
    // _loadSnapshot: blob.arrayBuffer(),
  });

  pyodide.setDebug(true);
  console.log("loaded pyodide", performance.now());
  console.log(pyodide);
  await pyodide.loadPackage([
    "netgen",
    "ngsolve",
    "packaging",
    "numpy",
    "micropip",
  ]);
  await pyodide.runPythonAsync(
    "import micropip; await micropip.install('dill');",
  );
  console.log("loaded netgen", performance.now());

  //try {
  //  const socket = new WebSocket("ws://localhost:6789");
  //  socket.addEventListener("open", function (event) {
  //    console.log("WebSocket connection opened");
  //  });
  //  socket.addEventListener("message", function (event) {
  //    console.log("Message from server ", event.data);
  //    reload();
  //  });
  //} catch {
  //  console.log("WebSocket connection failed");
  //}
  reload();
}
