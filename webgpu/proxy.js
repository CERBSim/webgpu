function serializeEvent(event) {
  event.preventDefault();
  const keys = [
    "button",
    "altKey",
    "metaKey",
    "ctrlKey",
    "shiftKey",
    "x",
    "y",
    "deltaX",
    "deltaY",
    "deltaMode",
    "movementX",
    "movementY",
  ];
  return Object.fromEntries(keys.map((k) => [k, event[k]]));
}

window.lil_guis = {};

function initLilGUI() {
    // In generated html files, requirejs is imported before lil-gui is loaded.
    // Thus, we must load lil-gui using require, use import otherwise.
    const lil_url = "https://cdn.jsdelivr.net/npm/lil-gui@0.20";
    if(window.define === undefined){
        import(lil_url);
    } else {
        require([lil_url], (module) => {
            window.lil = module;
        });
    }
}
initLilGUI();

function isPrimitive(value) {
  return (
    value === null || (typeof value !== "object" && typeof value !== "function")
  );
}

class Remote {
  constructor({ port, host }) {
    this.counter = 1;
    this.objects = {};

    this.socket = new WebSocket(`${host}:${port}`);

    this.socket.onopen = () => {
      console.log("Connection established");
    };
    this.socket.onclose = () => {
      console.log("Connection closed");
    };
    this.socket.onmessage = (data) => this.onMessage(data);

    this.resize_observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const canvas = entry.target;
        const width = entry.contentBoxSize[0].inlineSize;
        const height = entry.contentBoxSize[0].blockSize;
        canvas.width = Math.max(1, width);
        canvas.height = Math.max(1, height);
      }
    });

    window.patchedRequestAnimationFrame = (
      device_id,
      context_id,
      target_texture_id,
    ) => {
      requestAnimationFrame((t) => {
        const device = this.objects[device_id];
        const context = this.objects[context_id];
        const target = this.objects[target_texture_id];

        const current = context.getCurrentTexture();
        const encoder = device.createCommandEncoder();

        //console.log("sizes ", target.width, target.height, current.width, current.height);

        encoder.copyTextureToTexture(
          { texture: target },
          { texture: current },
          { width: current.width, height: current.height },
        );
        device.queue.submit([encoder.finish()]);
      });
    };
  }

  convertValue(value) {
    if (value === null) return undefined;

    if (value instanceof MouseEvent) return serializeEvent(value);
    if (value instanceof Event) return serializeEvent(value);
    if (value instanceof InputEvent) return serializeEvent(value);

    if (value === undefined || value === null || typeof value !== "object")
      return value;

    if (value.__python_proxy_type__ == "bytes")
      return Uint8Array.from(atob(value.data), (c) => c.charCodeAt(0)).buffer;

    if (value.__python_proxy_type__ == "function") {
      return (...args) => {
        this.socket.send(
          JSON.stringify({
            type: "call_function",
            args: this.convertObject(args),
            id: value.id,
          }),
        );
      };
    }

    if (value.__python_proxy_type__ == "render") {
      return (...args) => {
        const texture_view = this.objects[value.context_id]
          .getCurrentTexture()
          .createView();
        const view_id = this.counter++;
        this.objects[view_id] = texture_view;
        this.socket.send(
          JSON.stringify({
            type: "call_function",
            args: [view_id, ...args],
            id: value.id,
          }),
        );
      };
    }

    if (value.__python_proxy_type__ == "proxy") {
      return this.objects[value.id];
    }

    return this.convertObject(value);
  }

  convertObject(data) {
    if (
      data === undefined ||
      data === null ||
      typeof data !== "object" ||
      data.__python_proxy_type__ == "proxy" ||
      data.__python_proxy_type__ == "bytes"
    )
      return this.convertValue(data);

    Object.keys(data).map((key) => {
      data[key] = this.convertValue(data[key]);
    });
    return data;
  }

  async sendResult(ret, request_id, parent_id) {
    if (request_id === undefined) {
      return;
    }

    const result = await Promise.resolve(ret);
    // console.log("send result", typeof result, isPrimitive(result), result);

    if(result instanceof ArrayBuffer) {
      this.socket.send(
        JSON.stringify({
          type: "binary_value",
          request_id,
          value: btoa(String.fromCharCode.apply(null, new Uint8Array(result))),
        }),
      );
      return;
    }

    if (isPrimitive(result)) {
      this.socket.send(
        JSON.stringify({
          type: "value",
          request_id,
          value: result,
        }),
      );
      return;
    }

    const id = this.counter++;
    this.objects[id] = result;
    this.socket.send(
      JSON.stringify({
        type: "proxy",
        request_id,
        parent_id,
        id,
      }),
    );
  }

  onMessage(event) {
    const data = JSON.parse(event.data);
    const request_id = data.request_id;

    const obj = data.id ? this.objects[data.id] : self;

    if (data.type === "call_function") {
      const args = this.convertObject(data.args);
      return this.sendResult(
        obj.apply(this.objects[data.parent_id], args),
        request_id,
      );
    }

    if (data.type === "get_keys")
      return this.sendResult(JSON.stringify(Object.keys(obj)), request_id);

    if (data.type === "get_prop")
      return this.sendResult(obj[data.prop], request_id, data.id);

    if (data.type === "set_prop") {
      obj[data.prop] = this.convertObject(data.value);
      return this.sendResult(undefined, request_id);
    }

    if (data.type === "on_canvas_resize") {
      const canvas = this.convertObject(data.canvas);
      this.resize_observer.observe(canvas);
      return;
    }

    if (data.type === "delete_object") {
      this.objects[data.id] = undefined;
      return;
    }

    console.error("Unknown message type:", data, data.type);
  }
}

const remote = new Remote({
  port: WEBSOCKET_PORT,
  host: "ws://localhost",
});
