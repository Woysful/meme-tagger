"""Danbooru character database (bundled, ~24k characters): name matching and character -> franchise lookup."""
from __future__ import annotations
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path
from .paths import DATA
from .tags import clean_tag, name_key, tag_key

_SUFFIX = re.compile(r"^(.*?)_\(([^()]+)\)$")


@lru_cache(maxsize=1)
def _short_names() -> dict[str, str]:
    raw = json.loads((DATA / "franchises.json").read_text("utf-8"))
    return {tag_key(k): v for k, v in raw.items() if not k.startswith("_")}


def franchise_tag(copyright: str) -> str:
    """Danbooru copyright -> short franchise tag: 'k-on!' -> 'k-on', 'kirby_(series)' -> 'kirby'."""
    c = re.sub(r"_\(series\)$", "", (copyright or "").strip().lower())
    return _short_names().get(tag_key(c)) or _short_names().get(tag_key(copyright or "")) or clean_tag(c)


def split_danbooru(name: str) -> tuple[str, str | None]:
    """'rem_(re:zero)' -> ('rem', 're:zero'); 'hirasawa_yui' -> ('hirasawa_yui', None)."""
    m = _SUFFIX.match(name)
    return (m.group(1), m.group(2)) if m else (name, None)


class CharacterDB:
    def __init__(self, rows):
        self.by_tag: dict[str, list[tuple[str, int]]] = {}          # base tag -> [(franchise tag, posts)]
        self.by_name: dict[str, list[tuple[str, str, int]]] = {}    # name_key -> [(base tag, franchise, posts)]
        self.franchises: set[str] = set()
        self.by_franchise: dict[str, list[tuple[str, str, int]]] = {}
        for full, cop, posts in rows:
            base, _ = split_danbooru(full)
            base = clean_tag(base)
            fr = franchise_tag(cop) if cop else ""
            if not base:
                continue
            self.by_tag.setdefault(base, []).append((fr, posts))
            self.by_name.setdefault(name_key(base), []).append((base, fr, posts))
            if fr:
                self.franchises.add(fr)
                self.by_franchise.setdefault(fr, []).append((base, fr, posts))
        self._fr_keys = {tag_key(f): f for f in self.franchises} | {tag_key(k): v for k, v in _short_names().items()}

    @classmethod
    def from_file(cls, path: Path):
        rows = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    rows.append((parts[0], parts[1], int(parts[2] or 0)))
        return cls(rows)

    def is_character(self, tag: str) -> bool:
        return tag in self.by_tag

    def known_franchise(self, tag: str) -> str | None:
        """Canonical franchise tag for any spelling of a known franchise ('Re:Zero' -> 're_zero')."""
        return self._fr_keys.get(tag_key(tag))

    def franchise_of(self, tag: str) -> str | None:
        opts = [o for o in self.by_tag.get(tag, []) if o[0]]
        return max(opts, key=lambda o: o[1])[0] if opts else None

    def match(self, name: str, franchise: str | None = None) -> tuple[str, str | None] | None:
        """Free-form character name (+ optional franchise) -> (danbooru tag, franchise tag), or None when the name is
        unknown or ambiguous. A given franchise is never contradicted: 'Asuka' + 'Evangelion' finds souryuu_asuka_langley
        instead of the more literal asuka of another series."""
        raw = (name or "").strip()
        m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", raw)
        if m:
            raw, franchise = m.group(1), franchise or m.group(2)
        base = clean_tag(raw)
        if not base:
            return None
        key = name_key(base)
        cands = self.by_name.get(key, [])
        if franchise:
            fk = tag_key(self.known_franchise(franchise) or clean_tag(franchise))
            same = [c for c in cands if c[1] and _same_franchise(c[1], fk)]
            if not same:
                toks = set(key.split())
                same = [c for f, chars in self.by_franchise.items() if _same_franchise(f, fk)
                        for c in chars if toks <= set(name_key(c[0]).split())]
            if not same:
                return None
            best = max(same, key=lambda c: c[2])
            return best[0], best[1] or None
        if not cands:
            return None
        cands = sorted(cands, key=lambda c: -c[2])
        top = cands[0]
        if " " not in key and top[2] < 300:                 # one-word names are too ambiguous unless very popular
            return None
        if len(cands) == 1 or top[2] >= 5 * cands[1][2]:
            return top[0], top[1] or None
        return None


def _same_franchise(franchise: str, key: str) -> bool:
    fk = tag_key(franchise)
    return fk == key or (len(key) >= 4 and (key in fk or fk in key))


@lru_cache(maxsize=1)
def default_db() -> CharacterDB:
    try:
        return CharacterDB.from_file(DATA / "characters.tsv.gz")
    except OSError:
        return CharacterDB([])
