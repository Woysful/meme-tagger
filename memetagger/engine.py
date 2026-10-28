"""Orchestrator. Per image: prepare() (decode once, OCR, WD14) -> analyze() (vision model, verification, fusion).
prepare() is independent of the vision model, so the worker runs it for the next image while the GPU answers."""
from __future__ import annotations
import json
import threading
from dataclasses import dataclass, field
from .characters import CharacterDB, default_db
from .fusion import Evidence, Fusion, Policy
from .imaging import ImageData
from .models import DEFAULT_WD
from .naming import clean_name_text, looks_random
from .paths import config_file
from .tags import KINDS, MEDIUMS, Tag, clean_tag
from .tagstore import TagStore
from .vlm import ALL_WANT, DEFAULT_MODEL, OllamaVLM, VLMError, build_user

INSTALL_HELP = ("Install Ollama with GPU support and pull a vision model, e.g. on Arch:\n"
                "  sudo pacman -S ollama-cuda\n  sudo systemctl enable --now ollama\n  ollama pull qwen3-vl:8b")


@dataclass(frozen=True)
class EngineConfig:
    mode: str = "auto"                 # auto | vlm | classic
    host: str = ""
    model: str = DEFAULT_MODEL
    ocr: bool = True
    wd: bool = True                    # WD14 character recognizer
    wd_model: str = DEFAULT_WD
    wd_device: str = "auto"            # auto | cuda | cpu
    want_tags: bool = True             # when False, no tag models are loaded at all


@dataclass
class TagOptions:
    want: frozenset = ALL_WANT
    max_tags: int = 8
    banned: frozenset = frozenset()
    desc_lang: str = "English"
    limits: dict | None = None
    strictness: str = "balanced"
    wd_trust: float = 0.85
    verify: bool = True
    show_vocab: bool = False
    folder_tags_all: bool = False
    concept_thr: float = 0.30
    general_thr: float = 0.35
    fallback: bool = True


@dataclass
class Prepared:
    image: ImageData
    ocr: str = ""
    wd: object = None
    notes: list = field(default_factory=list)


@dataclass
class Result:
    tags: list[Tag] = field(default_factory=list)
    description: str = ""
    caption: str = ""                  # text for the file name
    ocr_text: str = ""
    source: str = ""                   # vlm | classic | error
    notes: list[str] = field(default_factory=list)
    want: frozenset = ALL_WANT         # what was generated; the UI keeps the other values
    candidates: list[str] = field(default_factory=list)
    suggestions: list[tuple[str, str | None, float]] = field(default_factory=list)   # WD14 characters (tag, franchise, p)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tags]

    @property
    def kinds(self) -> dict[str, str]:
        return {t.name: t.kind for t in self.tags}


_SHARED: dict = {}                     # heavy models survive Engine re-creation (settings changes)
_SHARED_LOCK = threading.Lock()


def shared(key, factory):
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = factory()
        return _SHARED[key]


def reset_shared():
    with _SHARED_LOCK:
        _SHARED.clear()


def load_json(name: str) -> dict:
    try:
        data = json.loads(config_file(name).read_text("utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (OSError, ValueError):
        return {}


def load_seeds() -> list[tuple[str, str | None]]:
    """seed_tags.txt lines: 'tag' or 'tag | kind'."""
    out = []
    try:
        lines = config_file("seed_tags.txt").read_text("utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tag, _, kind = (x.strip() for x in line.partition("|"))
        if (tag := clean_tag(tag)):
            out.append((tag, kind if kind in KINDS else None))
    return out


class Engine:
    def __init__(self, cfg: EngineConfig = EngineConfig(), store: TagStore | None = None, progress=None, *,
                 vlm=None, wd=None, ocr_reader=None, classic=None, chardb: CharacterDB | None = None,
                 folder_hints: dict | None = None, seeds=None, parents: dict | None = None):
        self.cfg, self.store = cfg, store if store is not None else TagStore()
        self.progress = progress or (lambda m: None)
        self.db = chardb or default_db()
        self.parents = parents if parents is not None else load_json("aliases.json")
        self.folder_hints = {k.lower().strip(): v for k, v in (folder_hints if folder_hints is not None else load_json("folder_hints.json")).items()}
        self.store.seed(seeds if seeds is not None else load_seeds())
        self.vlm, self.vlm_note = vlm, ""
        self._wd, self._ocr, self._classic = wd, ocr_reader, classic
        self._wd_failed = self._ocr_failed = False
        if self.vlm is None and cfg.mode != "classic":
            v = OllamaVLM(cfg.host, cfg.model)
            ok, msg = v.status()
            if ok:
                self.vlm = v; self.progress(msg)
            elif cfg.mode == "vlm":
                raise VLMError(msg + "\n\n" + INSTALL_HELP)
            else:
                self.vlm_note = "Ollama unavailable, classic engine used: " + msg.splitlines()[0]
                self.progress(self.vlm_note)

    # ---- lazily loaded models ----
    def ocr(self):
        if self._ocr is None and self.cfg.ocr and not self._ocr_failed:
            try:
                from .ocr import OcrReader
                self._ocr = shared("ocr", lambda: OcrReader(self.progress))
            except Exception as e:                          # OCR is a hint, never a reason to fail
                self._ocr_failed = True; self.progress(f"OCR disabled: {e}")
        return self._ocr

    def wd(self):
        if self._wd is None and self.cfg.wd and not self._wd_failed:
            try:
                from .wd14 import Wd14
                key = ("wd", self.cfg.wd_model, self.cfg.wd_device)
                self._wd = shared(key, lambda: Wd14(self.cfg.wd_model, self.cfg.wd_device, self.progress, self.db))
                self.progress(f"Character recognizer on {self._wd.device}")
            except Exception as e:
                self._wd_failed = True; self.progress(f"WD14 disabled: {e}")
        return self._wd

    def classic(self):
        if self._classic is None:
            from .classic import ClassicTagger
            self._classic = shared("classic", lambda: ClassicTagger(self.progress))
        return self._classic

    def warm_up(self, want=ALL_WANT):
        """Loads what the analysis will need (called once in the worker, so the first image is not slower)."""
        if "tags" in want and self.cfg.want_tags:
            self.wd()
            if self.vlm is None:
                self.classic()
        if want & {"tags", "text"}:
            self.ocr()

    def flush(self):
        if self._ocr is not None:
            self._ocr.flush()

    # ---- stage 1: CPU/GPU-light preparation, safe to run ahead in another thread ----
    def prepare(self, path, opts: TagOptions = TagOptions()) -> Prepared:
        prep = Prepared(ImageData(path))
        _ = prep.image.rgb                                  # decode errors surface here
        want = opts.want
        if want & {"tags", "text"} and (ocr := self.ocr()) is not None:
            try:
                prep.ocr = ocr.read(prep.image)
            except Exception as e:
                prep.notes.append(f"OCR failed: {e}")
        if "tags" in want and (wd := self.wd()) is not None:
            try:
                prep.wd = wd(prep.image.rgb)
            except Exception as e:
                prep.notes.append(f"WD14 failed: {e}")
        if self.vlm is not None:
            prep.image.jpeg_b64()                            # encode ahead too
        return prep

    # ---- stage 2: vision model + fusion ----
    def analyze(self, prep: Prepared, context: dict | None = None, opts: TagOptions = TagOptions()) -> Result:
        want = frozenset(opts.want) & ALL_WANT or ALL_WANT
        do_tags, do_text, do_desc = "tags" in want, "text" in want, "description" in want
        ctx = context or {}
        name = "" if looks_random(ctx.get("name", "")) else ctx.get("name", "")
        notes = list(prep.notes) + ([self.vlm_note] if self.vlm_note else [])
        banned = frozenset(clean_tag(b) for b in opts.banned)
        fusion = Fusion(self.store, self.db, self.parents,
                        Policy(opts.strictness, opts.wd_trust, opts.max_tags, banned, opts.folder_tags_all))
        folder_tags = self.folder_tags(ctx.get("folders", ()), banned) if do_tags else []
        ev = Evidence(f"{name} {prep.ocr}", folder_tags, prep.wd)
        candidates = [c.label() for c in prep.wd.characters] if prep.wd else []
        result = Result(want=want, ocr_text=prep.ocr, candidates=candidates,
                        suggestions=[(c.tag, c.franchise, c.score) for c in prep.wd.characters] if prep.wd else [])

        if self.vlm is not None:
            try:
                user = build_user(want, desc_lang=opts.desc_lang, candidates=candidates, name=name, ocr=prep.ocr,
                                  folder_tags=[t.name for t in folder_tags],
                                  vocabulary=self.store.top(60, {"franchise", "character", "meme_format"}) if opts.show_vocab else ())
                data = self.vlm.analyze(prep.image.jpeg_b64(), user, want, opts.limits)
                result.source = "vlm"
                if do_desc:
                    result.description = str(data.get("description") or "").strip()
                if do_text:
                    result.caption = clean_name_text(str(data.get("name_text") or ""), 120) or clean_name_text(prep.ocr, 120)
                if do_tags:
                    ev.medium = data.get("medium") if isinstance(data.get("medium"), str) else None
                    vlm_claims, wd_claims = fusion.claims(data, prep.wd)
                    tri = fusion.triage(vlm_claims, wd_claims, ev)
                    verdicts = None
                    if tri.to_verify and opts.verify:
                        verdicts = self.vlm.verify(prep.image.jpeg_b64(), [(c.kind, c.tag, c.franchise) for c in tri.to_verify])
                    tri = fusion.resolve(tri, verdicts)
                    result.tags = fusion.assemble(data, tri.accepted, ev)
                    notes += [f"kept {c.describe()}" for c in tri.accepted] + tri.notes
                if result.tags or not do_tags or self.cfg.mode == "vlm":
                    return self._finish(result, notes)
                notes.append("vision model returned no usable tags")
            except VLMError as e:
                if self.cfg.mode == "vlm":
                    raise
                notes.append(f"vision model failed: {e}")

        result.source = "classic"
        result.caption = result.caption or (clean_name_text(prep.ocr, 120) if do_text else "")
        if do_tags and self.cfg.want_tags:
            extra, art = self.classic().tag(prep.image.rgb, prep.wd, opts.concept_thr, opts.general_thr, opts.fallback)
            ev.medium = None if art else "photo"
            _, wd_claims = fusion.claims(None, prep.wd)
            tri = fusion.triage([], wd_claims, ev, have_vlm=False)
            result.tags = fusion.assemble(None, tri.accepted, ev, extra)
            notes += [f"kept {c.describe()}" for c in tri.accepted]
        return self._finish(result, notes)

    def _finish(self, result: Result, notes: list[str]) -> Result:
        result.notes = [n for n in notes if n]
        self.store.learn(result.tags)
        return result

    def tag(self, path, context: dict | None = None, opts: TagOptions = TagOptions()) -> Result:
        prep = self.prepare(path, opts)
        try:
            return self.analyze(prep, context, opts)
        finally:
            prep.image.release()

    def folder_tags(self, folders, banned=frozenset()) -> list[Tag]:
        out, seen = [], set()
        for f in folders:
            for raw in self.folder_hints.get(str(f).lower().strip(), []):
                t = clean_tag(raw)
                kind = self.store.kind_of(t) or ("character" if self.db.is_character(t) else
                                                 "franchise" if self.db.known_franchise(t) else "other")
                t = self.store.canonical(self.db.known_franchise(t) or t if kind == "franchise" else t, kind)
                if t and t not in banned and t not in seen:
                    seen.add(t); out.append(Tag(t, kind, 1.0, "folder"))
        return out


# One-word Danbooru characters that are far more often plain words in a meme library (cat_(kemono_friends)...).
COMMON_WORDS = frozenset({"cat", "dog", "fox", "frog", "bunny", "rabbit", "bear", "wolf", "bird", "ghost", "girl", "boy",
                          "aqua", "teacher", "nurse", "maid", "witch", "angel", "demon", "doge", "pepe", "link"})


def guess_kind(store: TagStore, db: CharacterDB, tag: str) -> str | None:
    """Kind of a tag read from a file name: what the user/analysis taught the store, else the character database.
    One-word names count as characters only when they are popular (megumin, pikachu), not for cat or ghost."""
    if k := store.kind_of(tag):
        return k
    if tag in MEDIUMS:
        return "medium"
    if db.known_franchise(tag) == tag:
        return "franchise"
    if db.is_character(tag) and ("_" in tag or (tag not in COMMON_WORDS and max(p for _f, p in db.by_tag[tag]) >= 500)):
        return "character"
    return None
