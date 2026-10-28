"""Persistent tag vocabulary: spellings, kinds, counts, learned character -> franchise links and renames.

Keeps tag spelling consistent (re_zero / rezero / Re-Zero -> one tag, yui_hirasawa -> hirasawa_yui) and powers the
tag browser. Only tags the user applied are counted; results of the current session never count as evidence."""
from __future__ import annotations
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from .tags import KINDS, name_key, singular_key, tag_key

VERSION = 2


class TagStore:
    def __init__(self, path: Path | None = None):
        self.path, self.lock = Path(path) if path else None, threading.RLock()
        self.counts: Counter = Counter()
        self.kinds: dict[str, str] = {}
        self.parents: dict[str, Counter] = defaultdict(Counter)    # character -> {franchise: times seen together}
        self.renames: dict[str, str] = {}
        self._spell: dict[str, str] = {}
        self._names: dict[str, str] = {}
        if self.path and self.path.exists():
            try:
                self._load(json.loads(self.path.read_text("utf-8")))
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def _load(self, data: dict):
        if data.get("version") != VERSION:                         # v1: {"tag": count}
            data = {"counts": {k: v for k, v in data.items() if isinstance(v, int)}}
        self.counts.update({k: int(v) for k, v in data.get("counts", {}).items()})
        self.kinds.update({k: v for k, v in data.get("kinds", {}).items() if v in KINDS})
        for ch, fr in data.get("parents", {}).items():
            self.parents[ch].update(fr)
        self.renames.update(data.get("renames", {}))
        for t, _ in self.counts.most_common():
            self._index(t)
        for t in self.kinds:
            self._index(t)

    def _index(self, tag: str):
        self._spell.setdefault(tag_key(tag), tag)
        if self.kinds.get(tag) == "character" and "_" in tag:
            self._names.setdefault(name_key(tag), tag)

    # ---- lookups ----
    def canonical(self, tag: str, kind: str | None = None) -> str:
        with self.lock:
            seen = set()
            while tag in self.renames and tag not in seen:
                seen.add(tag); tag = self.renames[tag]
            key = tag_key(tag)
            if key in self._spell:
                return self._spell[key]
            if (kind in (None, "character")) and "_" in tag and name_key(tag) in self._names:
                return self._names[name_key(tag)]
            if kind in (None, "subject", "theme", "other"):
                s = singular_key(key)
                if s and s in self._spell:
                    return self._spell[s]
            return tag

    def kind_of(self, tag: str) -> str | None:
        with self.lock:
            return self.kinds.get(tag)

    def count(self, tag: str) -> int:
        with self.lock:
            return self.counts.get(tag, 0)

    def franchise_of(self, character: str) -> str | None:
        with self.lock:
            c = self.parents.get(character)
            return c.most_common(1)[0][0] if c else None

    def top(self, n: int = 150, kinds=None) -> list[str]:
        with self.lock:
            tags = [t for t, _ in self.counts.most_common() if kinds is None or self.kinds.get(t) in kinds]
            return tags[:n]

    def all_tags(self) -> list[tuple[str, str, int]]:
        with self.lock:
            names = set(self.counts) | set(self.kinds)
            return sorted(((t, self.kinds.get(t, "other"), self.counts.get(t, 0)) for t in names), key=lambda r: (-r[2], r[0]))

    # ---- learning ----
    def seed(self, entries):
        """[(tag, kind or None)]: fixes spellings and kinds from the start without counting anything."""
        with self.lock:
            for tag, kind in entries:
                if kind in KINDS:
                    self.kinds.setdefault(tag, kind)
                self._index(tag)

    def learn(self, tags):
        """Remember spellings, kinds and character -> franchise links of a result (Tag objects)."""
        with self.lock:
            for t in tags:
                if t.kind in KINDS and t.kind != "other":
                    self.kinds.setdefault(t.name, t.kind)
                self._index(t.name)
            franchises = [t.name for t in tags if t.kind == "franchise"]
            if len(franchises) == 1:
                for t in tags:
                    if t.kind == "character":
                        self.parents[t.name][franchises[0]] += 1

    def commit(self, tag_lists):
        """Called when files are actually renamed: the applied tags become the user's vocabulary."""
        with self.lock:
            for tags in tag_lists:
                for t in tags:
                    self.counts[t] += 1; self._index(t)
            self.save()

    # ---- editing (tag browser) ----
    def rename(self, old: str, new: str):
        with self.lock:
            if old == new:
                return
            self.counts[new] += self.counts.pop(old, 0)
            if old in self.kinds:
                self.kinds.setdefault(new, self.kinds.pop(old))
            if old in self.parents:
                self.parents[new].update(self.parents.pop(old))
            for c in self.parents.values():
                if old in c:
                    c[new] += c.pop(old)
            self.renames = {k: (new if v == old else v) for k, v in self.renames.items()}
            self.renames[old] = new
            self.renames.pop(new, None)
            self._reindex()
            self.save()

    def delete(self, tag: str):
        with self.lock:
            self.counts.pop(tag, None); self.kinds.pop(tag, None); self.parents.pop(tag, None)
            for c in self.parents.values():
                c.pop(tag, None)
            self._reindex()
            self.save()

    def set_kind(self, tag: str, kind: str):
        with self.lock:
            if kind in KINDS:
                self.kinds[tag] = kind; self._reindex(); self.save()

    def _reindex(self):
        self._spell.clear(); self._names.clear()
        for t, _ in self.counts.most_common():
            self._index(t)
        for t in self.kinds:
            self._index(t)

    def save(self):
        if not self.path:
            return
        with self.lock:
            data = {"version": VERSION, "counts": dict(self.counts), "kinds": self.kinds,
                    "parents": {k: dict(v) for k, v in self.parents.items() if v}, "renames": self.renames}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
            tmp.replace(self.path)
