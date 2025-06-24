/* eslint-disable */

function serializeEvent(event) {
  try {
    event.preventDefault();
  } catch (e) {
    // ignore, some events do not support preventDefault
  }
  const keys = [
    'button',
    'code',
    'key',
    'altKey',
    'metaKey',
    'ctrlKey',
    'shiftKey',
    'repeat',
    'x',
    'y',
    'deltaX',
    'deltaY',
    'deltaMode',
    'movementX',
    'movementY',
  ];
  let obj = Object.fromEntries(keys.map((k) => [k, event[k]]));
  if (event.x !== undefined) {
    let rect = event.target.getBoundingClientRect();
    obj.canvasX = event.x - Math.floor(rect.x);
    obj.canvasY = event.y - Math.floor(rect.y);
  }
  return obj;
}

function decodeB64(data) {
  return Uint8Array.from(atob(data), (c) => c.charCodeAt(0)).buffer;
}

function encodeB64(buffer) {
  const u8 = new Uint8Array(buffer);
  let str = '';
  for (let i = 0; i < u8.length; i++) {
    str += String.fromCharCode(u8[i]);
  }
  return btoa(str);
}

function isPrimitive(value) {
  return (
    value === null || (typeof value !== 'object' && typeof value !== 'function')
  );
}

function isPrimitiveDict(obj) {
  if (obj === window) return false;
  if (obj === navigator) return false;

  if (typeof obj !== 'object' || obj === null) return false;
  if (Array.isArray(obj)) {
    return obj.every(isPrimitiveDict);
  }

  if (obj.$children) return false;
  if (obj.constructor !== Object) return false;

  for (let key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const value = obj[key];

      if (!isPrimitive(value) && !isPrimitiveDict(value)) {
        return false;
      }
    }
  }

  return true;
}

function createProxy(link, id, parent_id, ignore_return = false) {
  const target = ignore_return
    ? function (...args) {
        link.callIgnoreResult(id, args, parent_id);
      }
    : function (...args) {
        return link.call(id, args, parent_id);
      };
  target.handleEvent = async (eventType, event) => {
    return link.callIgnoreResult(id, [eventType, event], parent_id);
  };
  target.callMethodIgnoreResult = (prop, args) => {
    link.callMethodIgnoreResult(id, prop, args);
  };
  if (ignore_return) {
    target.callFunction = (self, arg) => {
      link.callIgnoreResult(id, [arg], parent_id);
    };
    target.callMethod = async (prop, args) => {
      link.callMethodIgnoreResult(id, prop, args);
    };
  } else {
    target.callFunction = (self, arg) => {
      return link.call(id, [arg], parent_id);
    };
    target.callMethod = async (prop, args) => {
      return await link.callMethod(id, prop, args);
    };
  }

  const handler = {
    get: function (obj, prop) {
      if (prop === '_parent_id') return parent_id;
      if (prop === '_id') return id;
      if (prop === '_link') return link;
      if (prop === 'handleEvent') return target.handleEvent;
      if (prop === 'call') {
        return target.callFunction;
      }
      if (prop === 'then') return Reflect.get(...arguments);
      if (prop.startsWith('__')) return Reflect.get(...arguments);
      if (prop === 'callMethodIgnoreResult') return Reflect.get(...arguments);
      if (prop === 'callMethod') return Reflect.get(...arguments);
      if (prop === '_is_webapp_proxy') return true;

      return link.getProp(id, prop);
    },
    set: function (obj, prop, value) {
      link.setProp(id, prop, value);
      return true;
    },

    apply: ignore_return
      ? function (obj, thisArg, args) {
          link.callIgnoreResult(id, args, parent_id);
        }
      : function (obj, thisArg, args) {
          return link.call(id, args, parent_id);
        },
  };
  return new Proxy(target, handler);
}

class CrossLink {
  constructor(connection) {
    this.requestCounter = 1;
    this.requests = {};

    this.counter = 1;
    this.objects = {};

    this.connection = connection;
    this.connection.onMessage((data) => this.onMessage(data));
    this.connected = new Promise((resolve) => {
      this.connection.onOpen(() => {
        this.expose('importPackage', window.importPackage);
        this.expose('addStyleFile', window.addStyleFile);
        console.log('connection open');
        resolve();
      });
    });
  }

  async _sendRequestAwaitResponse(data) {
    const request_id = this.requestCounter++;
    data.request_id = request_id;
    try {
      const result = await new Promise((resolve, reject) => {
        this.requests[request_id] = resolve;
        this.connection.send(data);
        setTimeout(() => {
          reject(
            new Error(
              `Timeout, request ${request_id}, data: ${JSON.stringify(data)}`
            )
          );
        }, 20000);
      });
      // const t = Date.now() - requestData.sent;
      return result;
    } finally {
      delete this.requests[request_id];
    }
  }

  async getProp(id, prop) {
    return await this._sendRequestAwaitResponse({ type: 'get', id, prop });
  }

  async getItem(id, key) {
    return await this._sendRequestAwaitResponse({ type: 'get', id, key });
  }

  async setProp(id, prop, value) {
    this.connection.send({
      type: 'set',
      id,
      prop,
      value: await this._dumpData(value),
    });
  }

  async setItem(id, key, value) {
    this.connection.send({
      type: 'set',
      id,
      key,
      value: await this._dumpData(value),
    });
  }

  async callIgnoreResult(id, args = [], parent_id = undefined) {
    return await this.connection.send({
      type: 'call',
      id,
      parent_id,
      args: await this._dumpData(args),
    });
  }

  async call(id, args = [], parent_id = undefined, prop = undefined) {
    return await this._sendRequestAwaitResponse({
      type: 'call',
      id,
      parent_id,
      args: await this._dumpData(args),
      prop,
    });
  }

  async callMethod(id, prop, args = []) {
    return await this._sendRequestAwaitResponse({
      type: 'call',
      id,
      args: await this._dumpData(args),
      prop,
    });
  }

  async callMethodIgnoreResult(id, prop, args = []) {
    return await this.connection.send({
      type: 'call',
      id,
      args: await this._dumpData(args),
      prop,
    });
  }

  expose(name, obj) {
    this.objects[name] = obj;
  }

  async _dumpData(data) {
    if (data === null) return undefined;
    if (data === undefined) return undefined;
    if (data._is_webapp_proxy) {
      return {
        type: 'object',
        id: data._id,
      };
    }

    if (data instanceof MouseEvent) return serializeEvent(data);
    if (data instanceof Event) return serializeEvent(data);
    if (data instanceof InputEvent) return serializeEvent(data);

    if (data instanceof ArrayBuffer)
      return {
        __is_crosslink_type__: true,
        type: 'bytes',
        value: encodeB64(data),
      };

    if (isPrimitive(data)) return data;

    if (data.constructor === Array) {
      const result = [];
      for (let item of data) result.push(await this._dumpData(item));
      return result;
    }

    if (data.__is_crosslink_type__) return data;

    if (isPrimitiveDict(data)) {
      return data;
    }
    if (data.constructor === Object) {
      const result = {};
      await Promise.all(
        Object.keys(data).map(async (key) => {
          result[key] = await this._dumpData(data[key]);
        })
      );
      return result;
    }

    // complex type - store it in objects only send its id
    const id = this.counter++;
    this.objects[id] = data;
    return {
      __is_crosslink_type__: true,
      type: 'proxy',
      js_type: typeof data,
      id,
    };
  }

  _loadValue(value, buffer) {
    if (value instanceof ArrayBuffer) return value;
    if (value === null || value === undefined) return undefined;
    if (!value.__is_crosslink_type__) return value;

    if (value.type == 'bytes') return decodeB64(value.value);
    if (value.type == 'object') return this.objects[value.id];
    if (value.type == 'proxy')
      return createProxy(
        this,
        value.id,
        value.parent_id,
        value.ignore_return_value
      );
    if (value.type == 'buffer') {
      const index = value.index;
      const start = buffer.offsets[index];
      const end = buffer.offsets[index + 1];
      return buffer.data.slice(start, end);
    }

    console.error('Cannot load value, unknown value type:', value);
  }

  _loadData(data, buffer) {
    if (
      data === undefined ||
      data === null ||
      typeof data !== 'object' ||
      data.__is_crosslink_type__
    )
      return this._loadValue(data, buffer);

    Object.keys(data).map((key) => {
      data[key] = this._loadData(data[key], buffer);
    });
    return data;
  }

  async sendResponse(data, request_id, parent_id) {
    if (request_id === undefined || request_id === null) {
      return;
    }

    const value = await this._dumpData(await Promise.resolve(data));
    //console.log('encoded data', value);

    this.connection.send({
      type: 'response',
      request_id,
      value,
      cache:
        value && value.__is_crosslink_type__ && value.js_type === 'function',
    });
  }

  _decodeBinaryMessage(bdata) {
    const metadata_size = new Uint32Array(bdata, 0, 1)[0];
    const decoder = new TextDecoder('utf-8');
    const decoded = decoder.decode(new Uint8Array(bdata, 4, metadata_size));
    const data = JSON.parse(decoded);
    const buffer = {
      offsets: data.buffer_offsets,
      data: new Uint8Array(bdata, 4 + metadata_size),
    };

    return {
      data,
      buffer,
    };
  }

  async onMessage(event) {
    let data = event.data;
    let request_id = undefined;
    let result_action = undefined;
    let buffer = undefined;

    if (event.data instanceof ArrayBuffer) {
      const decoded_message = this._decodeBinaryMessage(event.data);
      data = decoded_message.data;
      buffer = decoded_message.buffer;
    } else {
      if (typeof event.data === 'string') {
        data = JSON.parse(event.data);
      }
      request_id = data.request_id;
      result_action = data.result_action || 'send';
    }

    let obj = data.id ? this.objects[data.id] : window;
    // console.log('onMessage', data, obj);

    let response = null;

    switch (data.type) {
      case 'call':
      case 'new':
        const args = this._loadData(data.args, buffer);
        let self = null;
        if (data.prop) {
          self = this.objects[data.id];
          obj = self[data.prop];
        } else {
          self = this.objects[data.parent_id];
        }

        if (data.type === 'call') {
          // console.log("call", data.id, data.prop, obj, args);
          response = obj.apply(self, args);
        } else {
          response = Reflect.construct(obj, args);
        }
        break;
      case 'get_keys':
        response = Object.keys(obj);
        break;
      case 'get':
        if (data.prop) response = obj[data.prop];
        else if (data.key) response = obj[data.key];
        else response = obj;
        if (response && (data.prop || data.key)) {
          response = await this._dumpData(response);
          if (typeof response === 'object') response.parent_id = data.id;
        }
        break;
      case 'set':
        const value = this._loadData(data.value, buffer);
        if (data.prop) obj[data.prop] = value;
        if (data.key) obj[data.key] = value;
        break;
      case 'delete':
        this.objects[data.id] = undefined;
        break;
      case 'response':
        this.requests[request_id](this._loadData(data.value, buffer));
        break;
      case 'chunk':
        const id = 'chunk_' + data.parent_request_id;

        if (this.objects[id] === undefined)
          this.objects[id] = new Uint8Array(data.total_size);
        this.objects[id].set(buffer.data, data.offset);
        if (data.n_chunks === data.chunk_id + 1) {
          const combined_data = this.objects[id].buffer;
          delete this.objects[id];
          this.onMessage({
            data: combined_data,
          });
        }
        break;
      default:
        console.error('Unknown message type:', data, data.type);
    }

    if (request_id !== undefined && data.type !== 'response') {
      response = await response;
      if (data.result_callback) {
        const callback = this._loadData(data.result_callback, buffer);
        await callback(response);
      } else {
        if (result_action === 'send') this.sendResponse(response, request_id);
        else if (result_action === 'store')
          this.objects['result_' + request_id] = response;
        else if (result_action !== 'ignore')
          console.error(
            'Unknown result action:',
            result_action,
            data,
            response
          );
      }
    }
  }
}

export function WebsocketLink(url) {
  const socket = new WebSocket(url);
  socket.binaryType = 'arraybuffer';
  return new CrossLink({
    send: (data) => socket.send(JSON.stringify(data)),
    onMessage: (callback) => (socket.onmessage = callback),
    onOpen: (callback) => (socket.onopen = callback),
  });
}

export function WebworkerLink(worker) {
  // wait for first message, which means the worker has loaded and is ready
  const buffer = new SharedArrayBuffer(1024 * 1024 * 100);
  const size_buffer = new Int32Array(buffer, 0, 1);
  const result_buffer = new Uint8Array(buffer, 4);
  worker.postMessage(buffer);
  worker.postMessage(
    JSON.stringify({
      type: 'settings',
      value: { base_url: window.__webapp_router_base },
    })
  );
  const workerReady = new Promise((resolve) => {
    worker.addEventListener(
      'message',
      (event) => {
        resolve();
      },
      { once: true }
    );
  });
  return new CrossLink({
    send: (data) => {
      if (data.type === 'response') {
        const str = JSON.stringify(data);
        const bytes = new TextEncoder().encode(str);
        result_buffer.set(bytes);
        Atomics.store(size_buffer, 0, bytes.length);
        Atomics.notify(size_buffer, 0, 1);
        return;
      }
      worker.postMessage(JSON.stringify(data));
    },
    onMessage: async (callback) => {
      await workerReady;
      worker.addEventListener('message', callback);
    },
    onOpen: async (callback) => {
      await workerReady;
      callback();
    },
  });
}

window.createLilGUI = async (args) => {
  if (window.lil === undefined) {
    const url = 'https://cdn.jsdelivr.net/npm/lil-gui@0.20';
    if (window.define === undefined) {
      await import(url);
    } else {
      await new Promise(async (resolve) => {
        require([url], (module) => {
          window.lil = module;
          resolve();
        });
      });
    }
  }
  return new window.lil.GUI(args);
};

window.addStyleFile = async (url) => {
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = url;
  document.body.appendChild(link);
};

window.importPackage = async (url) => {
  if (window.define === undefined) {
    const p = await new Promise(async (resolve) => {
      const script = document.createElement('script');
      script.src = url;
      script.onload = () => {
        console.log('script loaded');
        resolve();
      };
      document.body.appendChild(script);
    });
    return p;
  } else {
    const p = await new Promise(async (resolve) => {
      require([url], (module) => {
        resolve(module);
      });
    });
    return p;
  }
};

window.patchedRequestAnimationFrame = (device, context, target) => {
  // context.getCurrentTexture() is only guaranteed to be valid during the requestAnimationFrame callback
  // Thus, in order to render from python asynchroniously, we are always rendering into a separate texture
  // The actual callback here only copies the rendered image from the separate render target texture to the current texture
  requestAnimationFrame((t) => {
    const current = context.getCurrentTexture();
    const encoder = device.createCommandEncoder();

    encoder.copyTextureToTexture(
      { texture: target },
      { texture: current },
      { width: current.width, height: current.height }
    );

    device.queue.submit([encoder.finish()]);
  });
};
