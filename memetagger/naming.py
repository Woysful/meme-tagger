"""File names: reading tags back from names, recognizing random names, cleaning image text, and name templates.

Template variables:  {original} {meaningful} {text} {description} {tags}
  {meaningful}      the original name, but only if it means something ('bunny karate'; '00IoBi56ZE8' -> empty)
  {a|b|c}           first non-empty of several variables
  {text:40}         at most 40 characters ({tags:3}: the first 3 tags)
  <  - {text}>      optional group: dropped entirely when any variable inside is empty
The extension is added automatically. Empty [] and () are removed."""
from __future__ import annotations
import re
from .tags import squash as _squash

_SUFFIX = re.compile(r"\s*\[([^\[\]]*)\](?:\s*\(\d+\))?\s*$")   # also accepts the " (2)" added on name collisions


def split_name(stem: str) -> tuple[str, list[str]]:
    """'cat lol [anime k-on]' -> ('cat lol', ['anime', 'k-on'])"""
    m = _SUFFIX.search(stem)
    if not m:
        return stem, []
    return stem[:m.start()], [t for t in m.group(1).split() if t]


def build_filename(base: str, tags, ext: str) -> str:
    """'' base -> the name is just the tags: '[anime k-on].jpg'. Keeps the name within 240 bytes, dropping tags if needed."""
    n = lambda s: len(s.encode())
    base = base.strip()
    while n(base) > 120:
        base = base[:-1]
    tags = list(tags)
    while tags:
        name = f"{base} [{' '.join(tags)}]" if base else f"[{' '.join(tags)}]"
        if n(name) + n(ext) <= 240:
            return name + ext
        tags.pop()
    return (base or "image") + ext


_GENERIC = {"image", "images", "img", "original", "unnamed", "tenor", "giphy", "sticker", "screenshot", "maxresdefault",
            "superthumb", "index", "download", "video", "photo", "file", "untitled", "pic", "picture", "default"}
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _id_like(tok: str) -> bool:
    return (len(tok) in (11, 15, 16) and re.fullmatch(r"[A-Za-z0-9_-]+", tok) is not None and
            (bool(re.search(r"\d", tok) and re.search(r"[A-Za-z]", tok)) or bool(re.search(r"[a-z][A-Z]", tok))))


def looks_random(base: str) -> bool:
    """True for names like 00IoBi56ZE8, F7_c1kIWsAAEucE, photo_2024-01-01_12-00, UUIDs, hashes, 'Image'."""
    b = (base or "").strip()
    if not b:
        return True
    low = b.lower()
    if low in _GENERIC or re.sub(r"[\s_\-().]*\d*$", "", low) in _GENERIC:
        return True
    if re.fullmatch(r"[\W_]*\d[\d\W_]*", b) or _UUID.search(low):
        return True
    if re.search(r"(?=[0-9a-f]*\d)[0-9a-f]{10,}", low) or low.startswith("ezgif"):
        return True
    if re.match(r"(photo|video|doc|img|pxl|screenshot|image)[_\- ]*\d", low):
        return True
    if _id_like(b) or any(_id_like(t) for t in re.split(r"[\s_]+", b)):
        return True
    if re.fullmatch(r"[a-z0-9]{12,14}", b) and sum(c.isdigit() for c in b) >= 3 and sum(c.isalpha() for c in b) >= 3:
        return True                                      # Reddit-style id: 9gwumjr5x5831
    alnum = [c for c in b if c.isalnum()]
    return not alnum or sum(c.isalpha() for c in alnum) / len(alnum) <= 0.5


_URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|ru|net|org|io|ua|me|gg|tv)\b\S*)", re.I)
_DROP_CHARS = re.compile(r'[*?"<>]')                     # dropped: "What?!" -> "What!"
_SPACE_CHARS = re.compile(r'[\\/:|\[\]\x00-\x1f]')         # replaced with a space


def clean_name_text(s: str, max_chars: int = 60) -> str:
    """Image text -> safe part of a file name: no links, @handles, brackets or forbidden characters; long text is cut
    at a word boundary."""
    s = _URL.sub(" ", s or "")
    s = re.sub(r"(^|\s)@\w+", " ", s)
    s = re.sub(r"\s+", " ", _SPACE_CHARS.sub(" ", _DROP_CHARS.sub("", s))).strip(" .-_,")
    if len(s) > max_chars:
        cut = s[:max_chars]; sp = cut.rfind(" ")
        s = (cut[:sp] if sp >= max_chars * 0.5 else cut).strip(" .,-_")
    return s

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
