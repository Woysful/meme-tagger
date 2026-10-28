"""Чистая логика без ML и GUI: имена файлов, иерархия папок, перемещение, откат."""
from __future__ import annotations
import json, re, shutil, time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".jfif"}
_SUFFIX = re.compile(r"\s*\[([^\[\]]*)\](?:\s*\(\d+\))?\s*$")   # допускает хвост " (2)" от разрешения коллизий


def clean_tag(raw: str) -> str:
    t = re.sub(r"[^\w\-]", "", raw.strip().lower().replace(" ", "_"))
    return t.strip("_-")


def split_name(stem: str):
    """'cat lol [anime k-on]' -> ('cat lol', ['anime', 'k-on'])"""
    m = _SUFFIX.search(stem)
    if not m:
        return stem, []
    return stem[:m.start()], [t for t in m.group(1).split() if t]


def build_filename(base: str, tags, ext: str) -> str:
    """base == '' -> имя состоит только из тегов: '[anime k-on].jpg'"""
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
    """True для имён вроде 00IoBi56ZE8, F7_c1kIWsAAEucE, photo_2024-01-01_12-00, UUID, хэшей, 'Image'."""
    b = base.strip()
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
        return True                                      # id в стиле Reddit: 9gwumjr5x5831
    alnum = [c for c in b if c.isalnum()]
    return not alnum or sum(c.isalpha() for c in alnum) / len(alnum) <= 0.5


@dataclass
class Item:
    src: Path
    root: Path
    base: str
    tags: list
    text: str = ""          # текст с картинки (для имени файла); на диске отдельно не хранится
    description: str = ""   # описание сцены (VLM); тоже только в таблице/шаблоне имени


def _item(path: Path, root: Path) -> Item:
    base, tags = split_name(path.stem)
    return Item(path, root, base, tags)


def collect(paths) -> list:
    """Файлы и/или папки (рекурсивно) -> Item'ы. Для папки root = сама папка."""
    items, seen = [], set()
    for p in map(Path, paths):
        if p.is_dir():
            root = p
            files = sorted(f for f in p.rglob("*")
                           if f.is_file() and f.suffix.lower() in IMAGE_EXT
                           and not any(x.startswith(".") for x in f.relative_to(p).parts))
        elif p.is_file() and p.suffix.lower() in IMAGE_EXT:
            root, files = p.parent, [p]
        else:
            continue
        for f in files:
            if f not in seen:
                seen.add(f); items.append(_item(f, root))
    return items


def assign_folders(tag_lists, group_min=12, max_depth=4):
    """Для каждого изображения возвращает путь из тегов-папок, напр. ('anime','k-on').
    Тег становится папкой, когда в текущей папке набралось >= group_min изображений с ним;
    затем то же самое рекурсивно внутри. Самые массовые теги идут первыми
    (поэтому 'anime' оказывается выше 'k-on'); при равенстве — тег, который стоит
    в имени раньше (имена пишутся от общего к частному)."""
    out = [()] * len(tag_lists)

    def rec(idxs, path, skip, depth):
        if depth >= max_depth or len(idxs) < group_min:
            return
        cnt, rank = Counter(), {}
        for i in idxs:
            for pos, t in enumerate(tag_lists[i]):
                if t not in skip:
                    cnt[t] += 1; rank[t] = min(rank.get(t, pos), pos)
        left = set(idxs)
        for t in sorted((t for t, c in cnt.items() if c >= group_min),
                        key=lambda t: (-cnt[t], rank[t], t)):
            members = [i for i in idxs if i in left and t in tag_lists[i]]
            if len(members) < group_min:
                continue
            if depth > 0 and len(members) == len(idxs):
                skip = skip | {t}      # тег есть у всех — новая папка ничего не даст
                continue
            left -= set(members)
            for i in members:
                out[i] = path + (t,)
            rec(members, path + (t,), skip | {t}, depth + 1)

    rec(list(range(len(tag_lists))), (), frozenset(), 0)
    return out


@dataclass
class Move:
    src: Path
    dst: Path
    copy: bool = False


@dataclass
class Plan:
    moves: list
    roots: list


def _unique(dst: Path, src: Path, reserved: set) -> Path:
    cand, n = dst, 1
    while cand in reserved or (cand.exists() and cand != src):
        n += 1; cand = dst.with_name(f"{dst.stem} ({n}){dst.suffix}")
    reserved.add(cand)
    return cand


_URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|ru|net|org|io|ua|me|gg|tv)\b\S*)", re.I)
_DROP_CHARS = re.compile(r'[*?"<>]')                   # просто убираем: «What?!» -> «What!»
_SPACE_CHARS = re.compile(r'[\\/:|\[\]\x00-\x1f]')       # заменяем пробелом
NAME_STYLES = ("keep", "text", "both", "tags")


def clean_name_text(s: str, max_chars: int = 60) -> str:
    """Текст с картинки -> безопасная часть имени файла: без ссылок, @ников, скобок и запрещённых символов;
    длинное обрезается по границе слова."""
    s = _URL.sub(" ", s or "")
    s = re.sub(r"(^|\s)@\w+", " ", s)
    s = re.sub(r"\s+", " ", _SPACE_CHARS.sub(" ", _DROP_CHARS.sub("", s))).strip(" .-_,")
    if len(s) > max_chars:
        cut = s[:max_chars]; sp = cut.rfind(" ")
        s = (cut[:sp] if sp >= max_chars * 0.5 else cut).strip(" .,-_")
    return s


def _squash(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


def compose_base(it, style: str = "keep", keep_meaningful: bool = False, text_max: int = 60) -> str:
    """Что стоит перед [тегами] в имени.
      keep — исходное имя; text — текст с картинки (нет текста -> исходное имя);
      both — «исходное - текст» (если исходное имя случайное, то просто текст); tags — только теги.
    keep_meaningful: осмысленное исходное имя ('bunny karate') не заменяется в режимах text и tags."""
    meaningful = not looks_random(it.base)
    text = clean_name_text(it.text, text_max)
    if style == "tags":
        return it.base if (keep_meaningful and meaningful) or not it.tags else ""
    if style == "text":
        return it.base if (keep_meaningful and meaningful) or not text else text
    if style == "both":
        if not text or _squash(text) in _squash(it.base):          # уже в имени -> не дублируем при повторном анализе
            return it.base
        return text if looks_random(it.base) else f"{it.base} - {text}"
    return it.base


def item_context(it) -> dict:
    """Подсказки из исходного имени файла и из папок, где он лежал."""
    try:
        folders = list(it.src.parent.relative_to(it.root).parts)
    except ValueError:
        folders = []
    return {"name": it.base, "folders": [it.root.name] + folders}


def make_plan(items, out_dir=None, group_min=12, max_depth=4, copy=False, merge_library=True,
              replace_name=False, keep_meaningful=False, name_style=None, text_max=60,
              template=None, desc_max=80) -> Plan:
    groups = defaultdict(list)
    for it in items:
        groups[Path(out_dir) if out_dir else it.root].append((it, copy))
    moves, reserved = [], set()
    for root, entries in groups.items():
        if out_dir and merge_library and root.is_dir():     # «библиотека» растёт: пересобираем её целиком
            known = {it.src for it, _ in entries}
            entries += [(e, False) for e in collect([root]) if e.tags and e.src not in known]
        folders = assign_folders([it.tags for it, _ in entries], group_min, max_depth)
        for (it, cp), ft in zip(entries, folders):
            cp = cp and root not in it.src.parents
            if template:                                              # шаблон пользователя: {original} {text} {description} {tags}
                from .template import render_filename
                fname = render_filename(template, it, it.src.suffix, text_max, desc_max)
            else:                                                     # прежние стили (оставлены для совместимости)
                base = compose_base(it, name_style or ("tags" if replace_name else "keep"), keep_meaningful, text_max)
                fname = build_filename(base, it.tags, it.src.suffix)
            dst = root.joinpath(*ft) / fname
            moves.append(Move(it.src, _unique(dst, it.src, reserved), cp))
    return Plan(moves, list(groups))


def _prune(dirs, roots):
    roots = set(roots)
    for d in dirs:
        while d not in roots and d != d.parent:
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent


def execute(plan: Plan, journal_dir: Path):
    done, errors = [], []
    for m in plan.moves:
        if m.src == m.dst:
            continue
        try:
            m.dst.parent.mkdir(parents=True, exist_ok=True)
            (shutil.copy2 if m.copy else shutil.move)(str(m.src), str(m.dst))
            done.append(m)
        except OSError as e:
            errors.append(f"{m.src.name}: {e}")
    journal_dir.mkdir(parents=True, exist_ok=True)
    (journal_dir / f"journal-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(
        {"roots": [str(r) for r in plan.roots],
         "ops": [{"src": str(m.src), "dst": str(m.dst), "copy": m.copy} for m in done]}))
    _prune([m.src.parent for m in done if not m.copy], plan.roots)
    return len(done), errors


def undo_last(journal_dir: Path):
    files = sorted(journal_dir.glob("journal-*.json"))
    if not files:
        return 0, ["Нечего откатывать"]
    data = json.loads(files[-1].read_text())
    n, errors, dirs = 0, [], []
    for op in reversed(data["ops"]):
        src, dst = Path(op["src"]), Path(op["dst"])
        try:
            if op["copy"]:
                dst.unlink()
            else:
                if src.exists():
                    raise OSError("исходный путь уже занят")
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
            dirs.append(dst.parent); n += 1
        except OSError as e:
            errors.append(f"{dst.name}: {e}")
    _prune(dirs, [Path(r) for r in data["roots"]])
    files[-1].rename(files[-1].with_suffix(".undone"))
    return n, errors
