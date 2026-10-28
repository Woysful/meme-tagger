"""Оркестратор: подсказки (имя файла, папки, OCR, WD14) -> VLM (Ollama) -> проверка франшиз/персонажей
-> при недоступности VLM классический CLIP+WD14."""
from __future__ import annotations
import json, re, shutil
from dataclasses import dataclass
from pathlib import Path
from .models import user_config_dir
from .organizer import clean_name_text, clean_tag, looks_random
from .tagmemory import TagMemory, tag_key
from .vlm import ALL_WANT, OllamaVLM, VLMError, claims_from_result, tags_from_result

INSTALL_HELP = ("Install Ollama with GPU support and pull a vision model, e.g. on Arch:\n"
                "  sudo pacman -S ollama-cuda\n  sudo systemctl enable --now ollama\n  ollama pull qwen3-vl:8b")


@dataclass
class Result:
    tags: list
    description: str = ""
    source: str = ""
    note: str = ""
    text: str = ""          # сырой текст OCR
    caption: str = ""       # текст/смысл для имени файла
    want: frozenset = ALL_WANT      # что реально генерировалось: tags / text / description (остальное GUI не трогает)


def config_file(name: str) -> Path:
    """Копирует дефолт из пакета в ~/.config/memetagger при первом запуске (дальше файл правит пользователь)."""
    here, dst = Path(__file__).parent, user_config_dir() / name
    if not dst.exists():
        src = here / name.replace(".", "_default.")
        if src.exists():
            shutil.copy(src, dst)
    return dst


_SHARED = {}          # тяжёлые модели (CLIP/WD14) живут между пересозданиями Engine


def reset_shared():
    _SHARED.clear()


def _squash(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.lower())


class Engine:
    def __init__(self, mode="auto", host="", model="qwen3-vl:8b", memory=None, progress=None,
                 classic=None, hints=True, ocr=True, verify=True, ocr_reader=None, folder_hints=None, seeds=None,
                 want_tags=True):
        self.mode, self.memory, self.use_hints, self.verify = mode, memory or TagMemory(), hints, verify
        self.progress = progress or (lambda m: None)
        self._classic, self.vlm, self.vlm_note = classic, None, ""
        self.ocr = ocr_reader
        if ocr and self.ocr is None:
            try:
                from .ocr import OcrReader
                self.ocr = OcrReader(self.progress)
            except Exception as e:                       # OCR — приятное дополнение, не повод падать
                self.progress(f"OCR disabled: {e}")
        self.folder_hints = {k.lower(): v for k, v in (folder_hints if folder_hints is not None else self._load_json("folder_hints.json")).items()
                             if not k.startswith("_")}
        self.memory.seed(seeds if seeds is not None else self._load_seeds())
        if mode != "classic":
            v = OllamaVLM(host, model); ok, msg = v.status()
            if ok:
                self.vlm = v; self.progress(msg)
            elif mode == "vlm":
                raise VLMError(msg + "\n\n" + INSTALL_HELP)
            else:
                self.vlm_note = msg; self.progress("Ollama unavailable, using classic CLIP+WD14: " + msg.splitlines()[0])
        if self.vlm is None and want_tags:          # CLIP/WD14 нужны только для тегов
            self._get_classic()

    # ---- конфиги ----
    @staticmethod
    def _load_json(name):
        try:
            return json.loads(config_file(name).read_text("utf-8"))
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _load_seeds():
        try:
            return [l.strip() for l in config_file("seed_tags.txt").read_text("utf-8").splitlines()
                    if l.strip() and not l.startswith("#")]
        except OSError:
            return []

    def flush(self):
        if self.ocr:
            self.ocr.flush()

    # ---- подсказки ----
    def _get_classic(self):
        if self._classic is None:
            if "classic" not in _SHARED:
                from .taggers import Pipeline
                _SHARED["classic"] = Pipeline(self.progress)
            self._classic = _SHARED["classic"]
        return self._classic

    def _wd14(self):
        if self._classic is not None:
            return self._classic.wd
        if "wd" not in _SHARED:
            from .taggers import Wd14
            _SHARED["wd"] = Wd14(self.progress)
        return _SHARED["wd"]

    def _hints(self, path):
        try:
            from .taggers import load_rgb
            _, chars, series = self._wd14()(load_rgb(path), 1.1, 0.6)
        except Exception as e:
            self.use_hints = False; self.progress(f"WD14 hints disabled: {e}")
            return None
        if not chars and not series:
            return None
        return {"characters": [t.name for t in chars[:3]], "series": [t.name for t in series[:2]]}

    def _folder_tags(self, folders, banned):
        out = []
        for f in folders:
            for t in self.folder_hints.get(f.lower().strip(), []):
                t = self.memory.canonical(clean_tag(t))
                if t and t not in banned and t not in out:
                    out.append(t)
        return out

    def _supported(self, tag, support_keys, text_key):
        k = tag_key(tag)
        if self.memory.count(tag) >= 3:
            return True
        if len(k) >= 3 and k in text_key:
            return True
        return any(len(h) >= 3 and (k in h or h in k) for h in support_keys)

    # ---- основной метод ----
    def tag(self, path, max_tags=6, general_thr=0.35, concept_thr=0.30, banned=frozenset(), fallback=True,
            context=None, desc_lang="English", want=ALL_WANT, limits=None) -> Result:
        """want ⊆ {tags, text, description}. Выключенное не генерируется, а вместе с ним пропускаются связанные этапы:
        tags -> WD14, проверки, словарь в промпте; text -> OCR (если не нужен тегам) и поле name_text; description -> поле description."""
        want = frozenset(want) & ALL_WANT or ALL_WANT
        do_tags, do_text, do_desc = "tags" in want, "text" in want, "description" in want
        banned = frozenset(clean_tag(b) for b in banned)
        ctx, note = context or {}, self.vlm_note
        ocr_text = ""
        if self.ocr and (do_text or do_tags):             # только описание? тогда OCR не нужен
            try:
                ocr_text = self.ocr.read(path)
            except Exception as e:
                note = f"OCR failed: {e}"
        description = caption = ""
        if self.vlm:
            try:
                name = ctx.get("name", "")
                name = "" if looks_random(name) else name
                folder_tags = self._folder_tags(ctx.get("folders", ()), banned) if do_tags else []
                hints = self._hints(path) if (do_tags and self.use_hints) else None
                data = self.vlm.tag(path, self.memory.top(150) if do_tags else (), hints,
                                    {"name": name, "folder_tags": folder_tags, "ocr": ocr_text, "desc_lang": desc_lang, "limits": limits}, want)
                description = str(data.get("description", "")).strip() if do_desc else ""
                caption = (clean_name_text(str(data.get("name_text") or ""), 120) or clean_name_text(ocr_text, 120)) if do_text else ""
                drop, dropped, tags = set(), [], []
                if do_tags:
                    if self.verify:
                        support = [tag_key(t) for t in folder_tags]
                        if hints:
                            support += [tag_key(t) for v in hints.values() for t in v]
                        text_key = _squash(name + " " + ocr_text)
                        for kind, t in claims_from_result(data, self.memory):
                            if t in banned or t in folder_tags or self._supported(t, support, text_key):
                                continue
                            verdict = self.vlm.verify(path, kind, t)
                            if verdict != "yes":
                                drop.add(t); dropped.append(f"{t}({verdict})")
                    tags = tags_from_result(data, self.memory, banned, max_tags, forced=folder_tags, drop=drop)
                if tags or self.mode == "vlm" or not do_tags:
                    self.memory.observe(tags)
                    if dropped:
                        note = (note + "; " if note else "") + "dropped unverified: " + ", ".join(dropped)
                    return Result(tags, description, "vlm", note, ocr_text, caption, want)
                note = "VLM returned no usable tags"
            except VLMError as e:
                if self.mode == "vlm":
                    raise
                note = f"VLM failed: {e}"
        cap = caption or (clean_name_text(ocr_text, 120) if do_text else "")
        if not do_tags:                                   # классическая модель умеет только теги -> её не грузим
            return Result([], description, "classic", note, ocr_text, cap, want)
        tags, seen = [], set()
        for t in self._get_classic().tag(path, max_tags, general_thr, concept_thr, banned, fallback):
            c = self.memory.canonical(t.name)
            if tag_key(c) not in seen and c not in banned:
                tags.append(c); seen.add(tag_key(c))
        self.memory.observe(tags)
        return Result(tags, description, "classic", note, ocr_text, cap, want)
