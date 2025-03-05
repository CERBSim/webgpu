
const PORT = {{PORT}};


class Remote {
  constructor() {
    this.counter = 0;
    this.objects = {};

    this.socket = new WebSocket(`ws://localhost:${PORT}`);');

    this.socket.onopen = () => {
      console.log('Connection established');
    };
    this.socket.on('message', (data) => this.onMessage(data));
  }

  sendResult(obj, request_id) {
    if(request_id === undefined) {
      return;
    }

    let result = obj;
    if(obj instanceof Object) {
      const id = this.counter++;
      this.objects[id] = obj;
      result = {
        id,
      };
    }

    this.socket.send(JSON.stringify({
      request_id,
      result
    }));
  }

  onMessage(msg) {
    const data = JSON.parse(msg);
    console.log('Received data:', data);
    const request_id = data.request_id;

    const obj = data.id ? this.objects[data.id] : self;
    const prop = obj[data.prop];

    if(data.type === 'call_function')
      return this.sendResult(prop.apply(obj, data.args), request_id);

    if(data.type === 'get_prop')
      return this.sendResult(prop, request_id);

    if(data.type === 'delete_object') {
      this.objects[data.id] = undefined;
      return;
    }

    console.error('Unknown message type:', data);
  }
};
