"""Шаблоны имён файлов (без зависимости от Qt).

Переменные:  {original} {meaningful} {text} {description} {tags}
  {meaningful}      исходное имя, но только если оно осмысленное ('bunny karate'; а '00IoBi56ZE8' -> пусто)
  {a|b|c}           первое непустое из переменных
  {text:40}         максимум 40 символов (для {tags:3} — первые 3 тега)
  <  - {text}>      необязательная группа: исчезает целиком, если любая переменная внутри пустая
Расширение файла добавляется автоматически. Пустые [] и () удаляются."""
from __future__ import annotations
import re
from .organizer import clean_name_text, looks_random

VARS = ("original", "meaningful", "text", "description", "tags")
DEFAULT_TEMPLATE = "{original} [{tags}]"
PRESETS = [
    ("Original name + tags", "{original} [{tags}]"),
    ("Image text + tags (else original name)", "{text|original} [{tags}]"),
    ("Keep meaningful names, else image text", "{meaningful|text|original} [{tags}]"),
    ("Original name - text + tags", "{meaningful}< - {text}> [{tags}]"),
    ("Text, else scene description + tags", "{text|description|original} [{tags}]"),
    ("Scene description + tags", "{description|text|original} [{tags}]"),
    ("Tags only", "[{tags}]"),
]
LEGACY_STYLES = {"keep": PRESETS[0][1], "text": PRESETS[2][1], "both": PRESETS[3][1], "tags": PRESETS[6][1]}

_VAR = re.compile(r"\{([^{}]*)\}")
_GROUP = re.compile(r"<([^<>]*)>")
_UNSAFE = re.compile(r"[\\/\x00-\x1f]")


def _squash(s: str) -> str:
    return re.sub(r"[\W_]+", "", (s or "").lower())


def _parse_spec(spec: str):
    """'text|original:40' -> (['text', 'original'], 40, None); при ошибке (_, _, 'сообщение')."""
    names, n = spec, None
    if ":" in spec:
        names, _, num = spec.rpartition(":")
        if not num.strip().isdigit() or int(num) < 1:
            return [], None, f"Bad limit in {{{spec}}} (use a number, e.g. {{text:40}})"
        n = int(num)
    lst = [x.strip().lower() for x in names.split("|")]
    bad = [x for x in lst if x not in VARS]
    if bad or not lst:
        return [], None, f"Unknown variable: {{{bad[0] if bad else spec}}}. Available: " + " ".join("{" + v + "}" for v in VARS)
    return lst, n, None


def validate(template: str):
    """-> (errors, warnings). При ошибках шаблон не применяется; предупреждения — о потере возможностей."""
    errors, warnings = [], []
    t = template or ""
    if not t.strip():
        return ["Template is empty"], []
    if t.count("{") != t.count("}"):
        errors.append("Unbalanced { }")
    if t.count("<") != t.count(">"):
        errors.append("Unbalanced < >")
    if re.search(r"[{}]", _VAR.sub("", t)):
        errors.append("Stray or nested { } (variables cannot be nested)")
    if _GROUP.sub("", t).count("<") or _GROUP.sub("", t).count(">"):
        errors.append("Optional < > groups cannot be nested")
    for m in _VAR.finditer(t):
        _, _, err = _parse_spec(m.group(1))
        if err:
            errors.append(err)
    if errors:
        return errors, warnings
    if "{tags" not in t.lower():
        warnings.append("No {tags} in the template: tags cannot be read back from file names, so already processed files "
                        "will be analyzed again and will not be re-grouped into folders.")
    elif not re.search(r"\[\{tags(?::\d+)?\}\]\s*$", t, re.I):
        warnings.append("End the template with [{tags}] so tags can be read back from file names "
                        "(needed for folder grouping and for skipping already analyzed files).")
    elif re.search(r"\{tags:\d+\}", t, re.I):
        warnings.append("{tags:N} puts only N tags into the name; only those can be read back later.")
    return errors, warnings


def values_for(it) -> dict:
    return {"original": it.base, "text": it.text, "description": getattr(it, "description", ""), "tags": list(it.tags)}


def _light(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", _UNSAFE.sub(" ", s or "")).strip()
    return s[:limit].strip()


def _lim(n, default: int, scale: float) -> int:
    """Лимит символов: явный {x:N} или умолчание, ужатое на scale, но не меньше min(лимит, 8)."""
    base = n or default
    return max(min(base, 8), int(base * scale))


def _render_once(t: str, v: dict, tm: int, dm: int, scale: float, drop_text: bool) -> str:
    def value(name, n):
        if name == "tags":
            return " ".join(v["tags"][:n] if n else v["tags"])
        if name in ("original", "meaningful"):
            if name == "meaningful" and looks_random(v["original"]):
                return ""
            return _light(v["original"], _lim(n, 100, scale))
        if name == "text":
            return "" if drop_text else clean_name_text(v["text"], _lim(n, tm, scale))
        if name == "description":
            return clean_name_text(v["description"], _lim(n, dm, scale))
        return ""

    def evaluate(spec):
        names, n, err = _parse_spec(spec)
        if err:
            return ""
        return next((x for x in (value(nm, n) for nm in names) if x), "")

    def subst(seg, empties):
        def rep(m):
            r = evaluate(m.group(1))
            if not r:
                empties.append(1)
            return r
        return _VAR.sub(rep, seg)

    def group(m):
        empties = []
        out = subst(m.group(1), empties)
        return "" if empties else out

    s = subst(_GROUP.sub(group, t), [])
    s = re.sub(r"\[\s*\]|\(\s*\)", "", s)
    s = _UNSAFE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" -_.,|")


def fit_bytes(s: str, max_bytes: int) -> str:
    """Укорачивает имя, сохраняя хвост [теги]."""
    if len(s.encode()) <= max_bytes:
        return s
    m = re.search(r"\s*\[[^\[\]]*\]\s*$", s)
    suffix, head = (m.group(0), s[:m.start()]) if m else ("", s)
    if len(suffix.encode()) > max_bytes // 2:                      # огромный хвост: режем всё подряд
        suffix, head = "", s
    while head and len((head + suffix).encode()) > max_bytes:
        head = head[:-1]
    return (head.rstrip() + suffix).strip() or s[:max_bytes // 4]


def render(template: str, values: dict, text_max: int = 60, desc_max: int = 80, max_bytes: int = 220) -> str:
    """Имя файла без расширения. Длинное сжимается: сначала текст/описание/исходное имя, в последнюю очередь — теги."""
    errors, _ = validate(template)
    if errors:
        template = DEFAULT_TEMPLATE
    # Дубль «исходное имя + тот же текст» убираем только если {text} и {original} стоят в РАЗНЫХ местах шаблона;
    # {text|original} — это выбор одного из вариантов, там дедупликация сломала бы повторный анализ.
    specs = [_parse_spec(m)[0] for m in _VAR.findall(template)]
    orig = {"original", "meaningful"}
    separate = (any("text" in sp and not set(sp) & orig for sp in specs) and any(set(sp) & orig and "text" not in sp for sp in specs))
    drop_text = separate and bool(_squash(values["text"])) and _squash(values["text"]) in _squash(values["original"])
    s = ""
    for scale in (1.0, 0.75, 0.5, 0.3):
        s = _render_once(template, values, text_max, desc_max, scale, drop_text)
        if len(s.encode()) <= max_bytes:
            break
    else:
        s = fit_bytes(s, max_bytes)
    return s or _light(values["original"], 100) or "image"


def render_filename(template: str, item, ext: str, text_max: int = 60, desc_max: int = 80) -> str:
    budget = 245 - len(ext.encode()) - 6                            # запас под « (2)» при коллизиях
    return render(template, values_for(item), text_max, desc_max, budget) + ext
