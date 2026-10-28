"""Память тегов: держит написание тегов единообразным (re_zero / rezero / re-zero -> один тег)
и подсказывает VLM уже существующий словарь пользователя."""
from __future__ import annotations
import json, re, threading
from collections import Counter
from pathlib import Path


def tag_key(tag: str) -> str:
    k = re.sub(r"[\W_]+", "", tag.lower())
    if len(k) > 3 and k.endswith("s") and not k.endswith("ss"):
        k = k[:-1]                       # cats -> cat
    return k or tag


class TagMemory:
    def __init__(self, path: Path | None = None):
        self.path, self.lock = path, threading.Lock()
        self.persisted, self.session, self.spelling, self.seeded = Counter(), Counter(), {}, []
        if path and Path(path).exists():
            try:
                self.persisted.update(json.loads(Path(path).read_text("utf-8")))
            except (OSError, ValueError):
                pass
            for t, _ in self.persisted.most_common():
                self.spelling.setdefault(tag_key(t), t)

    def canonical(self, tag: str) -> str:
        with self.lock:
            return self.spelling.get(tag_key(tag), tag)

    def seed(self, tags):
        """Стартовый словарь: задаёт написание и попадает в подсказку модели, но счётчики не увеличивает."""
        with self.lock:
            for t in tags:
                if self.spelling.setdefault(tag_key(t), t) == t:
                    self.seeded.append(t)

    def count(self, tag: str) -> int:
        with self.lock:
            c = self.spelling.get(tag_key(tag), tag)
            return self.persisted[c] + self.session[c]

    def observe(self, tags):
        """Учитывает теги в рамках сессии (в файл не пишется)."""
        with self.lock:
            for t in tags:
                c = self.spelling.setdefault(tag_key(t), t)
                self.session[c] += 1

    def commit(self, tag_lists):
        """Вызывается при «Apply»: теги реально применены -> сохраняем."""
        with self.lock:
            for tags in tag_lists:
                for t in tags:
                    self.persisted[self.spelling.setdefault(tag_key(t), t)] += 1
            if self.path:
                Path(self.path).write_text(json.dumps(self.persisted, ensure_ascii=False, indent=0), "utf-8")
        self.session.clear()

    def top(self, n=150):
        with self.lock:
            out = [t for t, _ in (self.persisted + self.session).most_common(n)]
            out += [t for t in self.seeded if t not in out]
            return out[:n]
