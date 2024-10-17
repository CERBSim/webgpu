let pyodide = null;

const files = [
  "__init__.py",
  "colormap.py",
  "gpu.py",
  "input_handler.py",
  "main.py",
  "mesh.py",
  "shader.wgsl",
  "uniforms.py",
  "utils.py",
];

async function reload() {
  try {
    pyodide.FS.mkdir("webgpu");
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
  await pyodide.loadPackage(["netgen"]);
  console.log("loaded netgen", performance.now());

  try {
    const socket = new WebSocket("ws://localhost:6789");
    socket.addEventListener("open", function (event) {
      console.log("WebSocket connection opened");
    });
    socket.addEventListener("message", function (event) {
      console.log("Message from server ", event.data);
      reload();
    });
  } catch {
    console.log("WebSocket connection failed");
  }
  reload();
}
main();
