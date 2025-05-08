from .base import LinkBase


class _ProxyIterator:
    def __init__(self, proxy):
        self._proxy = proxy
        self._keys = proxy._get_keys()
        self._index = 0

    def __next__(self):
        if self._index < len(self._keys):
            key = self._keys[self._index]
            self._index += 1
            return key
        else:
            raise StopIteration

    def __iter__(self):
        return self


class Proxy:
    _link: LinkBase
    _parent_id: int | None
    _id: int | None

    def __init__(self, link, parent_id=None, id=None):
        self._link = link
        self._id = id
        self._parent_id = parent_id

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __getattr__(self, key):
        if (
            key
            in [
                "_id",
                "_parent_id",
                "_link",
                "_call",
                "_call_method_ignore_return",
                "_call_method",
            ]
            or isinstance(key, str)
            and key.startswith("__")
        ):
            return super().__getattr__(key)
        return self._link.get(self._id, key)

    def __setattr__(self, key, value):
        if key in ["_id", "_parent_id", "_link"]:
            return super().__setattr__(key, value)

        return self._link.set(self._id, key, value)

    def __call__(self, *args, _ignore_result=False):
        return self._link.call(self._id, list(args), self._parent_id, _ignore_result)

    def _call_method(self, prop, args=[], ignore_result=False):
        return self._link.call_method(self._id, prop, args, ignore_result)

    def _call_method_ignore_return(self, prop, args=[]):
        return self._link.call_method_ignore_return(self._id, prop, args)

    def _new(self, *args):
        return self._link.call_new(self._id, args=list(args))

    def _to_js(self):
        return {
            "type": "proxy",
            "id": self._id,
            "parent_id": self._parent_id,
        }

    def _get_keys(self):
        return self._link.get_keys(self._id)

    def __iter__(self):
        return _ProxyIterator(self)
