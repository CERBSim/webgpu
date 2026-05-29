import sys

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
    _noreturn_names: set[str]

    def __init__(self, link, parent_id=None, id=None):
        self._link = link
        self._id = id
        self._parent_id = parent_id
        self._noreturn_names = set()
        self._pcache = {}

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
                "_noreturn_names",
                "_pcache",
            ]
            or isinstance(key, str)
            and key.startswith("__")
        ):
            return super().__getattr__(key)
        cached = self._pcache.get(key)
        if cached is not None:
            return cached
        if key in self._noreturn_names:

            def wrapper(*args):
                return self._call_method_ignore_return(key, list(args))

            return wrapper
        result = self._link.get(self._id, key)
        if isinstance(result, Proxy) and (self._id, key) in self._link._cache:
            self._pcache[key] = result
        return result

    def __setattr__(self, key, value):
        if key in ["_id", "_parent_id", "_link", "_noreturn_names", "_pcache"]:
            return super().__setattr__(key, value)
        self._pcache.pop(key, None)
        return self._link.set(self._id, key, value)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __call__(self, *args, _ignore_result=False):
        return self._link.call(self._id, list(args), self._parent_id, _ignore_result)

    def _call_method(self, prop, args=None, ignore_result=False):
        return self._link.call_method(self._id, prop, args or [], ignore_result)

    def _call_method_ignore_return(self, prop, args=None):
        return self._link.call_method_ignore_return(self._id, prop, args or [])

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

    def __del__(self):
        if sys is None or sys.meta_path is None:
            return
        link = self.__dict__.get('_link')
        id_ = self.__dict__.get('_id')
        parent_id = self.__dict__.get('_parent_id')
        if link is None:
            return
        with link._refcount_lock:
            refcount = link._js_id_refcount
            if id_ is not None:
                rc = refcount.get(id_, 0) - 1
                if rc <= 0:
                    refcount.pop(id_, None)
                    link._release_queue.append(id_)
                else:
                    refcount[id_] = rc
            if parent_id is not None:
                rc = refcount.get(parent_id, 0) - 1
                if rc <= 0:
                    refcount.pop(parent_id, None)
                    link._release_queue.append(parent_id)
                else:
                    refcount[parent_id] = rc
        if hasattr(link, '_schedule_flush'):
            link._schedule_flush()

    def __iter__(self):
        return _ProxyIterator(self)
