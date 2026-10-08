"""One list of class names for the whole program.

YOLO class IDs always come from the loaded model (`model.names`) and are never renumbered.
Every other name (CNN checkpoint `class_names`, classes the user adds) gets an ID after the
last YOLO ID. Lookups are by name, so the CNN and the relay mapping never depend on an ID.
"""
import re
import threading
from typing import Dict, Iterable, List, Optional


def normalize_name(s) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s-]', ' ', str(s).lower())).strip()


class ClassRegistry:
    def __init__(self, yolo_names: Optional[Dict[int, str]] = None, extra_names: Iterable[str] = ()):
        self._lock = threading.RLock()
        self._names: Dict[int, str] = {}
        self._yolo_ids = set()
        self.set_yolo_names(yolo_names or {})
        self.add_names(extra_names)

    def set_yolo_names(self, yolo_names: Dict[int, str]):
        """Use the IDs of a (new) YOLO model. Other names are kept and moved behind the YOLO IDs."""
        with self._lock:
            extras = [name for cid, name in sorted(self._names.items()) if cid not in self._yolo_ids]
            self._names = {int(k): str(v) for k, v in yolo_names.items()}
            self._yolo_ids = set(self._names)
            self.add_names(extras)

    def add_names(self, names: Iterable[str]):
        for name in names:
            self.id_for(name, register=True)

    def name(self, cid, default="UNK") -> str:
        with self._lock:
            return self._names.get(cid, default)

    def id_for(self, name, register=False) -> Optional[int]:
        """ID of a class name (case, spaces and underscores do not matter). None if unknown."""
        key = normalize_name(name)
        if not key:
            return None
        with self._lock:
            for cid, existing in self._names.items():
                if normalize_name(existing) == key:
                    return cid
            if not register:
                return None
            new_id = max(self._names, default=-1) + 1
            self._names[new_id] = str(name)
            return new_id

    def rename(self, old, new) -> bool:
        with self._lock:
            cid = self.id_for(old)
            if cid is None or self.id_for(new) is not None:
                return False
            self._names[cid] = str(new)
            return True

    def remove(self, name) -> bool:
        """Remove a user-added name. A class of the YOLO model cannot be removed (its ID must stay)."""
        with self._lock:
            cid = self.id_for(name)
            if cid is None or cid in self._yolo_ids:
                return False
            del self._names[cid]
            return True

    def is_yolo_class(self, name) -> bool:
        with self._lock:
            return self.id_for(name) in self._yolo_ids

    def snapshot(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._names)

    def names(self) -> List[str]:
        with self._lock:
            return [self._names[k] for k in sorted(self._names)]
