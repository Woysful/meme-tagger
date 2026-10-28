"""Library organization: collecting files, folder structure by tag kinds, planning moves, executing and undoing.

Folders follow configurable levels of tag kinds, e.g. 'franchise > character' gives k-on/hirasawa_yui/. A folder is
created once at least `min_count` images share the value, and an existing folder is always reused. Files already in
place whose tags did not change are left alone, so the library stays stable as it grows; rebuild=True re-plans it all."""
from __future__ import annotations
import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from . import metadata
from .naming import DEFAULT_TEMPLATE, render_filename, split_name, build_filename
from .tags import KINDS, clean_tag

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".jfif"}
DEFAULT_LEVELS = "franchise > character"
LEVEL_PRESETS = [
    ("Franchise › Character", "franchise > character"),
    ("Franchise or meme › Character", "franchise|meme_format > character"),
    ("Medium › Franchise › Character", "medium > franchise > character"),
    ("Franchise only", "franchise"),
    ("No folders (rename only)", ""),
]


@dataclass
class Item:
    src: Path
    root: Path
    base: str
    tags: list[str]
    text: str = ""                  # text from the image (for the file name)
    description: str = ""           # scene description
    orig_tags: list[str] = field(default_factory=list)      # tags in the file name when it was added
    kinds: dict[str, str] = field(default_factory=dict)     # tag -> kind, from the analysis
    touched: bool = False           # analyzed or edited in this session (metadata is written only for these)

    @property
    def changed(self) -> bool:
        return self.tags != self.orig_tags


def make_item(path: Path, root: Path, read_meta: bool = False) -> Item:
    base, tags = split_name(path.stem)
    it = Item(path, root, base, list(tags), orig_tags=list(tags))
    if read_meta and tags and (m := metadata.read(path)) is not None:     # only files this app already processed
        it.text, it.description = m.title, m.description
        it.kinds = {t: k for t, k in m.kinds.items() if t in tags and k in KINDS}
    return it


def collect(paths, read_meta: bool = False) -> list[Item]:
    """Files and/or folders (recursive) -> Items. For a folder, root = that folder; hidden folders are skipped."""
    items, seen = [], set()
    for p in map(Path, paths):
        if p.is_dir():
            root = p
            files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXT
                           and not any(x.startswith(".") for x in f.relative_to(p).parts))
        elif p.is_file() and p.suffix.lower() in IMAGE_EXT:
            root, files = p.parent, [p]
        else:
            continue
        for f in files:
            if f not in seen:
                seen.add(f); items.append(make_item(f, root, read_meta))
    return items


def item_context(it: Item) -> dict:
    """Hints from the original file name and the folders the file was in."""
    try:
        folders = list(it.src.parent.relative_to(it.root).parts)
    except ValueError:
        folders = []
    return {"name": it.base, "folders": [it.root.name] + folders}


# ---------- folder levels ----------
def parse_levels(spec: str) -> list[tuple[str, ...]]:
    """'franchise|meme_format > character' -> [('franchise', 'meme_format'), ('character',)]; ValueError on unknown kinds."""
    levels = []
    for part in (spec or "").replace("›", ">").split(">"):
        kinds = tuple(k.strip().lower() for k in part.split("|") if k.strip())
        if not kinds:
            continue
        bad = [k for k in kinds if k not in KINDS]
        if bad:
            raise ValueError(f"Unknown tag kind '{bad[0]}'. Use: {', '.join(KINDS)}")
        levels.append(kinds)
    return levels


def assign_folders(entries: list[list[tuple[str, str | None]]], levels, min_count: int = 3,
                   exists: Callable[[tuple], bool] = lambda p: False) -> list[tuple]:
    """entries: per image [(tag, kind)] -> folder path per image, e.g. ('k-on', 'hirasawa_yui').
    At each level an image uses its first tag of the level's kinds; the folder exists when >= min_count images
    share it (or it already exists on disk). Images without a value, or below the threshold, stay at the parent."""
    out = [()] * len(entries)

    def rec(idxs, path, depth):
        if depth >= len(levels):
            return
        groups = defaultdict(list)
        for i in idxs:
            v = next((t for t, k in entries[i] if k in levels[depth] and t not in path), None)
            if v:
                groups[v].append(i)
        for v, members in groups.items():
            sub = path + (v,)
            if len(members) >= min_count or exists(sub):
                for i in members:
                    out[i] = sub
                rec(members, sub, depth + 1)

    rec(list(range(len(entries))), (), 0)
    return out


# ---------- planning ----------
@dataclass
class Move:
    src: Path
    dst: Path
    copy: bool = False
    meta: metadata.Meta | None = None
    meta_mode: str = "off"

    @property
    def moves(self) -> bool:
        return self.src != self.dst


@dataclass
class Plan:
    moves: list[Move]
    roots: list[Path]

    @property
    def operations(self) -> list[Move]:
        return [m for m in self.moves if m.moves or m.meta is not None]


def _unique(dst: Path, src: Path, reserved: set) -> Path:
    cand, n = dst, 1
    while cand in reserved or (cand.exists() and cand != src):
        n += 1; cand = dst.with_name(f"{dst.stem} ({n}){dst.suffix}")
    reserved.add(cand)
    return cand


def _meta_for(it: Item, kind_of) -> metadata.Meta:
    return metadata.Meta(list(it.tags), {t: k for t in it.tags if (k := it.kinds.get(t) or kind_of(t))},
                         it.description, it.text)


def make_plan(items, *, out_dir=None, levels=DEFAULT_LEVELS, min_count: int = 3, copy: bool = False,
              template: str = DEFAULT_TEMPLATE, text_max: int = 60, desc_max: int = 80, library=(),
              rebuild: bool = False, gather: bool = True, kind_of: Callable[[str], str | None] = lambda t: None,
              metadata_mode: str = "off") -> Plan:
    """items: the current batch. library: Items already in out_dir (counted for folder thresholds; moved only on rebuild).
    gather: a settled file still waiting in a parent folder moves (name unchanged) into a subfolder that now reached
    the threshold, so a new k-on/ folder collects the older k-on images instead of leaving them behind."""
    lv = parse_levels(levels) if isinstance(levels, str) else list(levels)
    level_kinds = {k for level in lv for k in level}
    groups: dict[Path, list[tuple[Item, bool]]] = defaultdict(list)
    for it in items:
        groups[Path(out_dir) if out_dir else it.root].append((it, True))
    if out_dir and library:
        known = {it.src for it in items}
        groups[Path(out_dir)] += [(it, False) for it in library if it.src not in known and it.tags]
    moves, reserved = [], set()
    for root, entries in groups.items():
        inside = lambda it: root == it.src.parent or root in it.src.parents
        settled = [not rebuild and inside(it) and not it.changed for it, _ in entries]
        for (it, _), s in zip(entries, settled):
            if s:
                reserved.add(it.src)
        tagged = [[(t, it.kinds.get(t) or kind_of(t)) for t in it.tags] for it, _ in entries]
        folders = assign_folders(tagged, lv, min_count, lambda p: root.joinpath(*p).is_dir())
        for (it, batch), folder, s in zip(entries, folders, settled):
            if s or not batch:
                rel = it.src.parent.relative_to(root).parts if inside(it) else None
                below = rel is not None and len(folder) > len(rel) and folder[:len(rel)] == rel
            if not batch and not rebuild:
                if gather and below:
                    moves.append(Move(it.src, _unique(root.joinpath(*folder) / it.src.name, it.src, reserved)))
                continue
            meta = _meta_for(it, kind_of) if metadata_mode != "off" and batch and it.touched else None
            if s:
                dst = _unique(root.joinpath(*folder) / it.src.name, it.src, reserved) if gather and below else it.src
                moves.append(Move(it.src, dst, False, meta, metadata_mode))
                continue
            fname = render_filename(template, it, it.src.suffix, text_max, desc_max)
            rel = it.src.parent.relative_to(root).parts if inside(it) and not rebuild else None
            # A file already inside the library keeps its folder when that is the assigned one or a subfolder of it,
            # so the user's own folders are respected and nothing is pulled up to the root, unless the subfolder is
            # a tag folder (e.g. k-on/) for a tag the file no longer has.
            keep = rel is not None and rel[:len(folder)] == folder and not any(
                (t := clean_tag(p)) not in it.tags and (it.kinds.get(t) or kind_of(t)) in level_kinds
                for p in rel[len(folder):])
            target = it.src.parent if keep else root.joinpath(*folder)
            dst = _unique(target / fname, it.src, reserved)
            moves.append(Move(it.src, dst, copy and not inside(it), meta, metadata_mode))
    return Plan(moves, list(groups))


def plan_tag_edit(items, mapping: dict[str, str | None], kind_of=lambda t: None, metadata_mode: str = "off") -> Plan:
    """Rename/merge/delete tags in existing files: mapping old -> new tag (None deletes). Files stay in their folder."""
    moves, reserved = [], set()
    for it in items:
        if not any(t in mapping for t in it.tags):
            continue
        new = []
        for t in it.tags:
            t = mapping.get(t, t)
            if t and t not in new:
                new.append(t)
        it2 = Item(it.src, it.root, it.base, new, it.text, it.description,
                   kinds={mapping.get(t) or t: k for t, k in it.kinds.items()}, touched=True)
        dst = _unique(it.src.with_name(build_filename(it.base, new, it.src.suffix)), it.src, reserved)
        meta = _meta_for(it2, kind_of) if metadata_mode != "off" else None
        moves.append(Move(it.src, dst, False, meta, metadata_mode))
    return Plan(moves, sorted({it.root for it in items}))


# ---------- execution ----------
def _prune(dirs, roots):
    roots = set(roots)
    for d in dirs:
        while d not in roots and d != d.parent:
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent


def execute(plan: Plan, journal_dir: Path, progress=None) -> tuple[int, list[str]]:
    done, errors, ops = 0, [], []
    for n, m in enumerate(plan.moves):
        if progress and n % 25 == 0:
            progress(n, len(plan.moves))
        try:
            if m.moves:
                m.dst.parent.mkdir(parents=True, exist_ok=True)
                if m.copy:
                    shutil.copy2(m.src, m.dst)
                    if metadata.sidecar_path(m.src).exists():
                        shutil.copy2(metadata.sidecar_path(m.src), metadata.sidecar_path(m.dst))
                else:
                    shutil.move(str(m.src), str(m.dst))
                    metadata.move_sidecar(m.src, m.dst)
                ops.append({"src": str(m.src), "dst": str(m.dst), "copy": m.copy})
            if m.meta is not None:
                metadata.write(m.dst, m.meta, m.meta_mode)
            if m.moves or m.meta is not None:
                done += 1
        except (OSError, metadata.MetadataError) as e:
            errors.append(f"{m.src.name}: {e}")
    journal_dir.mkdir(parents=True, exist_ok=True)
    (journal_dir / f"journal-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 10**6:06d}.json").write_text(
        json.dumps({"roots": [str(r) for r in plan.roots], "ops": ops}))
    _prune([Path(op["src"]).parent for op in ops if not op["copy"]], plan.roots)
    return done, errors


def undo_last(journal_dir: Path) -> tuple[int, list[str]]:
    """Reverts the last Apply (file moves/copies; metadata written into files stays)."""
    files = sorted(journal_dir.glob("journal-*.json"))
    if not files:
        return 0, ["Nothing to undo"]
    data = json.loads(files[-1].read_text())
    n, errors, dirs = 0, [], []
    for op in reversed(data["ops"]):
        src, dst = Path(op["src"]), Path(op["dst"])
        try:
            if op["copy"]:
                dst.unlink()
                metadata.sidecar_path(dst).unlink(missing_ok=True)
            else:
                if src.exists():
                    raise OSError("the original path is taken")
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dst), str(src))
                metadata.move_sidecar(dst, src)
            dirs.append(dst.parent); n += 1
        except OSError as e:
            errors.append(f"{dst.name}: {e}")
    _prune(dirs, [Path(r) for r in data["roots"]])
    files[-1].rename(files[-1].with_suffix(".undone"))
    return n, errors
