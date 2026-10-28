"""Парсер vocab.txt (без зависимостей от ML).
Формат:  tag | prompt | min_prob     (prompt и min_prob необязательны)
  [group]   — метки внутри одной группы взаимоисключающие (остаётся только лучшая)
  []        — следующие метки без группы
  ~name     — «отвлекающая» метка: участвует в сравнении, но никогда не становится тегом"""
from __future__ import annotations
import re
from dataclasses import dataclass
from .organizer import clean_tag

VERSION_RE = re.compile(r"#\s*vocab-version:\s*(\d+)")


@dataclass
class Entry:
    label: str
    prompt: str
    group: str | None
    min_p: float
    hidden: bool


def version_of(text: str) -> int:
    m = VERSION_RE.search(text)
    return int(m.group(1)) if m else 0


def parse_vocab(text: str) -> list:
    group, out = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1].strip() or None
            continue
        parts = [p.strip() for p in line.split("|")]
        hidden = parts[0].startswith("~")
        raw = parts[0].lstrip("~")
        label = ("~" if hidden else "") + clean_tag(raw)
        prompt = parts[1] if len(parts) > 1 and parts[1] else f"a picture of {raw}"
        min_p = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        out.append(Entry(label, prompt, None if hidden else group, min_p, hidden))
    return out
