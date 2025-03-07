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
    this.socket.onmessage = (data) => this.onMessage(data);

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
    let result = ret;
    if (ret instanceof Promise) {
      result = await ret;
      //console.log("awaited promise", ret, result);
    }

    //console.log("send result", isPrimitive(result), result);

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

    if (data.type === "delete_object") {
      this.objects[data.id] = undefined;
      return;
    }

    console.error("Unknown message type:", data);
  }
}

const remote = new Remote({
  port: 8765,
  host: "ws://localhost",
});
