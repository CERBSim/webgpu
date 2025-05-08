/* eslint-disable */

function serializeEvent(event) {
  event.preventDefault();
  const keys = [
    'button',
    'altKey',
    'metaKey',
    'ctrlKey',
    'shiftKey',
    'x',
    'y',
    'deltaX',
    'deltaY',
    'deltaMode',
    'movementX',
    'movementY',
  ];
  return Object.fromEntries(keys.map((k) => [k, event[k]]));
}

function decodeB64(data) {
  return Uint8Array.from(atob(data), (c) => c.charCodeAt(0)).buffer;
}

function encodeB64(buffer) {
  return btoa(String.fromCharCode.apply(null, new Uint8Array(buffer)));
}

function isPrimitive(value) {
  return (
    value === null || (typeof value !== 'object' && typeof value !== 'function')
  );
}

function isPrimitiveDict(obj) {
  if (obj.$children) return false;
  if (obj === window) return false;
  if (obj === navigator) return false;

  if (typeof obj !== 'object' || obj === null) return false;
  if (Array.isArray(obj)) {
    return obj.every(isPrimitiveDict);
  }

  for (let key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
      const value = obj[key];

      if (!isPrimitive(value) && !isPrimitiveDict(value)) {
        return false;
      }
    }
  }

  if (obj.constructor !== Object) return false;

  return true;
}

function createProxy(link, id, parent_id) {
  const target = function (...args) {
    return link.call(id, args, parent_id);
  };
  target.handleEvent = async (eventType, event) => {
    return link.call(id, [eventType, event], parent_id, '_handle');
  };
  target.callFunction = (self, arg) => {
    return link.call(id, [arg], parent_id);
  };
  target.callMethod = async (prop, arg) => {
    return await link.callMethod(id, prop, [arg]);
  };
  target.callMethodIgnoreResult = (prop, arg) => {
    link.callMethodIgnoreResult(id, prop, [arg]);
  };
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

      return link.getProp(id, prop);
    },
    set: function (obj, prop, value) {
      link.setProp(id, prop, value);
    },

    apply: function (obj, thisArg, args) {
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
        console.log('connection open');
        resolve();
      });
    });
  }

  async _sendRequestAwaitResponse(data) {
    //console.log('sending request', data);
    const request_id = this.requestCounter++;
    data.request_id = request_id;
    try {
      const result = await new Promise((resolve, reject) => {
        this.requests[request_id] = resolve;
        this.connection.send(JSON.stringify(data));
        setTimeout(() => {
          reject(
            new Error(
              `Timeout, request ${request_id}, data: ${JSON.stringify(data)}`
            )
          );
        }, 10000);
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
    return await this.connection.send(
      JSON.stringify({
        type: 'call',
        id,
        args: await this._dumpData(args),
        prop,
      })
    );
  }

  expose(name, obj) {
    this.objects[name] = obj;
  }

  async _dumpData(data) {
    //console.log("dumping data", data);
    if (data === null) return undefined;

    if (isPrimitive(data)) return data;

    if (data instanceof MouseEvent) return serializeEvent(data);
    if (data instanceof Event) return serializeEvent(data);
    if (data instanceof InputEvent) return serializeEvent(data);

    if (data instanceof ArrayBuffer)
      return {
        __is_crosslink_type__: true,
        type: 'bytes',
        value: encodeB64(data),
      };

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
      Object.keys(data).map(async (key) => {
        result[key] = await this._dumpData(data[key]);
      });
      return result;
    }

    // complex type - store it in objects only send its id
    const id = this.counter++;
    this.objects[id] = data;
    return {
      __is_crosslink_type__: true,
      type: 'proxy',
      id,
    };
  }

  _loadValue(value) {
    if (value === null || value === undefined) return undefined;
    if (!value.__is_crosslink_type__) return value;

    if (value.type == 'bytes') return decodeB64(value.value);
    if (value.type == 'object') return this.objects[value.id];
    if (value.type == 'proxy')
      return createProxy(this, value.id, value.parent_id);

    console.error('Cannot load value, unknown value type:', value);
  }

  _loadData(data) {
    if (
      data === undefined ||
      data === null ||
      typeof data !== 'object' ||
      data.__is_crosslink_type__
    )
      return this._loadValue(data);

    Object.keys(data).map((key) => {
      data[key] = this._loadData(data[key]);
    });
    return data;
  }

  async sendResponse(data, request_id, parent_id) {
    if (request_id === undefined) {
      return;
    }

    const value = await this._dumpData(await Promise.resolve(data));
    //console.log('encoded data', value);

    this.connection.send(
      JSON.stringify({
        type: 'response',
        request_id,
        value,
      })
    );
  }

  async onMessage(event) {
    const data = JSON.parse(event.data);
    const request_id = data.request_id;

    let obj = data.id ? this.objects[data.id] : window;
    // console.log('onMessage', data, obj);

    let response = null;

    switch (data.type) {
      case 'call':
        const args = this._loadData(data.args);
        let self = null;
        if (data.prop) {
          self = this.objects[data.id];
          obj = self[data.prop];
        } else {
          self = this.objects[data.parent_id];
        }

        // console.log('calling', 'data', data, 'obj', obj, 'self', self, 'args', args);

        response = obj.apply(self, args);
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
        const value = this._loadData(data.value);
        if (data.prop) obj[data.prop] = value;
        if (data.key) obj[data.key] = value;
        break;
      case 'delete':
        this.objects[data.id] = undefined;
        break;
      case 'response':
        this.requests[request_id](this._loadData(data.value));
        return;
      default:
        console.error('Unknown message type:', data, data.type);
    }

    if (request_id !== undefined && data.type !== 'response') {
      response = await response;
      this.sendResponse(response, request_id);
    }
  }
}

export function WebsocketLink(url) {
  const socket = new WebSocket(url);
  return new CrossLink({
    send: (data) => socket.send(data),
    onMessage: (callback) => (socket.onmessage = callback),
    onOpen: (callback) => (socket.onopen = callback),
  });
}

export function WebworkerLink(worker) {
  // wait for first message, which means the worker has loaded and is ready
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
    send: (data) => worker.postMessage(data),
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

window.importPackage=async (url) => {
  if (window.define === undefined) {
    await import(url);
  } else {
    await new Promise(async (resolve) => {
      require([url], (module) => {
        resolve(module);
      });
    });
  }
}

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
