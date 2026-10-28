"""Локальные модели (onnxruntime, CPU):
 * CLIP zero-shot по словарю vocab.txt — общие понятия (anime, cat, bunny, screenshot, ...)
 * WD14 tagger (danbooru) — персонажи и франшизы; запускается только если CLIP видит арт."""
from __future__ import annotations
import csv, hashlib, json, re, shutil
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from .models import get_model_file, user_config_dir, user_data_dir, WD14, CLIP
from .organizer import clean_tag
from .vocab import parse_vocab, version_of

BLACKLIST = {"solo", "1girl", "1boy", "2girls", "multiple_girls", "multiple_boys", "looking_at_viewer",
             "simple_background", "white_background", "grey_background", "smile", "open_mouth", "blush",
             "long_hair", "short_hair", "closed_mouth", "standing", "upper_body", "english_text", "highres",
             "commentary", "bad_id", "absurdres", "comic", "monochrome", "greyscale", "text_focus"}
ART_LABELS = ("anime", "manga", "cartoon", "comic", "pixel_art")


@dataclass
class Tag:
    name: str
    score: float
    kind: str  # concept | copyright | character | general


def _session(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def load_rgb(path) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return bg.convert("RGB")


class ClipZeroShot:
    def __init__(self, vocab: Path, progress=None):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(get_model_file(CLIP, "tokenizer.json", progress)))
        self.tok.enable_truncation(77)
        self.tok.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")
        self.vis = _session(get_model_file(CLIP, "onnx/vision_model_quantized.onnx", progress))
        self.txt = _session(get_model_file(CLIP, "onnx/text_model_quantized.onnx", progress))
        entries = parse_vocab(vocab.read_text(encoding="utf-8"))
        self.labels = [e.label for e in entries]
        self.group = {e.label: e.group for e in entries}
        self.min_p = {e.label: e.min_p for e in entries}
        prompts = [e.prompt for e in entries]
        key = hashlib.sha1("\n".join(prompts).encode()).hexdigest()[:12]
        cache = user_data_dir() / f"clip_text_{key}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self._text(prompts); np.save(cache, self.emb)

    @staticmethod
    def _out(sess, outs, name):
        names = [o.name for o in sess.get_outputs()]
        v = outs[names.index(name) if name in names else 0]
        v = v.reshape(v.shape[0], -1).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def _text(self, prompts):
        enc = self.tok.encode_batch(prompts)
        feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64)}
        if "attention_mask" in {i.name for i in self.txt.get_inputs()}:
            feeds["attention_mask"] = np.array([e.attention_mask for e in enc], dtype=np.int64)
        return self._out(self.txt, self.txt.run(None, feeds), "text_embeds")

    def probs(self, img: Image.Image) -> dict:
        x = np.asarray(img.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255.0
        x = ((x - [0.4815, 0.4578, 0.4082]) / [0.2686, 0.2613, 0.2758]).transpose(2, 0, 1)[None].astype(np.float32)
        e = self._out(self.vis, self.vis.run(None, {"pixel_values": x}), "image_embeds")
        z = 100.0 * (e @ self.emb.T)[0]
        p = np.exp(z - z.max()); p /= p.sum()
        return dict(zip(self.labels, p.tolist()))

    FALLBACK_MIN = 0.10   # ниже этого «лучшая догадка» уже не берётся

    def classify(self, img, thr, banned=frozenset(), fallback=True):
        """Из каждой группы берётся только лучшая метка; отвлекающие (~) и запрещённые отбрасываются
        (но остаются в сравнении и «забирают» вероятность); метка проходит, если p >= max(thr, её min_p).
        Если ничего не прошло и fallback=True — берём одну лучшую догадку с p >= FALLBACK_MIN."""
        p = self.probs(img); best = {}
        for lab, v in p.items():
            if lab.startswith("~") or lab in banned:
                continue
            g = self.group[lab] or lab            # без группы — сама себе группа
            if g not in best or v > p[best[g]]:
                best[g] = lab
        tags = [Tag(l, p[l], "concept") for l in best.values() if p[l] >= max(thr, self.min_p[l])]
        safe = [l for l in best.values() if self.min_p[l] == 0]      # «капризные» метки (doge...) в догадки не берём
        if not tags and fallback and safe:
            top = max(safe, key=lambda l: p[l])
            if p[top] >= self.FALLBACK_MIN:
                tags = [Tag(top, p[top], "concept")]
        return sorted(tags, key=lambda t: -t.score)[:3], p


class Wd14:
    def __init__(self, progress=None):
        self.sess = _session(get_model_file(WD14, "model.onnx", progress))
        with open(get_model_file(WD14, "selected_tags.csv", progress), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.names, self.cats = [r["name"] for r in rows], [int(r["category"]) for r in rows]
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        self.size = inp.shape[1] if isinstance(inp.shape[1], int) else 448

    def __call__(self, img: Image.Image, general_thr, char_thr=0.85):
        s = max(img.size)
        canvas = Image.new("RGB", (s, s), (255, 255, 255))
        canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
        canvas = canvas.resize((self.size, self.size), Image.BICUBIC)
        arr = np.ascontiguousarray(np.asarray(canvas, dtype=np.float32)[:, :, ::-1])  # RGB -> BGR
        probs = self.sess.run(None, {self.inp: arr[None]})[0][0]
        general, chars, series = [], [], []
        for name, cat, p in zip(self.names, self.cats, probs):
            if cat == 0 and p >= general_thr and name not in BLACKLIST:
                general.append(Tag(clean_tag(name), float(p), "general"))
            elif cat == 4 and p >= char_thr:
                m = re.match(r"^(.*)_\((.+)\)$", name)      # hirasawa_yui_(k-on!)
                if m:
                    series.append(Tag(clean_tag(m.group(2)), float(p), "copyright")); name = m.group(1)
                chars.append(Tag(clean_tag(name), float(p), "character"))
            elif cat == 3 and p >= char_thr:
                series.append(Tag(clean_tag(name), float(p), "copyright"))
        key = lambda t: -t.score
        return sorted(general, key=key), sorted(chars, key=key), sorted(series, key=key)


def _sync_config(cfg: Path):
    """Копирует дефолты в ~/.config; устаревший vocab.txt заменяет (старый -> vocab.txt.bak)."""
    here = Path(__file__).parent
    voc, default = cfg / "vocab.txt", here / "vocab_default.txt"
    if voc.exists() and version_of(voc.read_text("utf-8")) < version_of(default.read_text("utf-8")):
        shutil.move(voc, cfg / "vocab.txt.bak")
    if not voc.exists():
        shutil.copy(default, voc)
    if not (cfg / "aliases.json").exists():
        shutil.copy(here / "aliases_default.json", cfg / "aliases.json")


class Pipeline:
    ART_GATE = 0.10   # суммарная вероятность «арт» по CLIP, выше которой включаем WD14

    def __init__(self, progress=None):
        cfg = user_config_dir(); _sync_config(cfg)
        self.aliases = {k: v for k, v in json.loads((cfg / "aliases.json").read_text("utf-8")).items()
                        if not k.startswith("_")}
        if progress: progress("Loading CLIP…")
        self.clip = ClipZeroShot(cfg / "vocab.txt", progress)
        if progress: progress("Loading WD14…")
        self.wd = Wd14(progress)

    def tag(self, path, max_tags=6, general_thr=0.35, concept_thr=0.30, banned=frozenset(), fallback=True) -> list:
        banned = frozenset(clean_tag(b) for b in banned)
        img = load_rgb(path)
        concepts, probs = self.clip.classify(img, concept_thr, banned, fallback)
        general, chars, series = [], [], []
        if sum(probs.get(k, 0) for k in ART_LABELS) >= self.ART_GATE:
            general, chars, series = self.wd(img, general_thr)
            general, chars, series = ([t for t in ts if t.name not in banned] for ts in (general, chars, series))
            if chars and "anime" not in banned and not any(t.name == "anime" for t in concepts):
                concepts.insert(0, Tag("anime", chars[0].score, "concept"))
        lists = {"concept": concepts, "copyright": series[:2], "character": chars[:3], "general": general}
        names = {t.name for ts in lists.values() for t in ts}
        for n in list(names):                                # алиасы: персонаж -> франшиза и т.п.
            for extra in map(clean_tag, self.aliases.get(n, [])):
                if extra and extra not in names and extra not in banned:
                    lists["copyright"].append(Tag(extra, 1.0, "copyright")); names.add(extra)
        chosen, seen = [], set()
        for kind in ("concept", "copyright", "character", "general"):   # от общего к частному
            for t in lists[kind]:
                if t.name and t.name not in seen and t.name not in banned and len(chosen) < max_tags:
                    chosen.append(t); seen.add(t.name)
        return chosen
