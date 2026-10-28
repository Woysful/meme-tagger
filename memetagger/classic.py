"""Classic engine without Ollama: CLIP zero-shot over vocab.txt for general concepts (+ WD14 general tags for art).
Characters and franchises come from WD14 through the same fusion rules as with the vision model."""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .models import CLIP, get_model_file
from .paths import DATA, user_config_dir, user_data_dir
from .tags import Tag
from .vocab import parse_vocab, version_of
from .wd14 import WdResult

ART_LABELS = ("anime", "manga", "cartoon", "comic", "pixel_art")
GROUP_KINDS = {"medium": "medium", "format": "meme_format", "memes": "meme_format"}


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def sync_vocab() -> Path:
    """Copies the default vocab.txt to the config dir; an outdated copy is replaced (old one -> vocab.txt.bak)."""
    voc, default = user_config_dir() / "vocab.txt", DATA / "vocab_default.txt"
    if voc.exists() and version_of(voc.read_text("utf-8")) < version_of(default.read_text("utf-8")):
        shutil.move(voc, voc.with_suffix(".txt.bak"))
    if not voc.exists():
        shutil.copy(default, voc)
    return voc


class ClipZeroShot:
    FALLBACK_MIN = 0.10

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
        cache = user_data_dir() / f"clip_text_{hashlib.sha1(chr(10).join(prompts).encode()).hexdigest()[:12]}.npy"
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

    def classify(self, img, thr, fallback=True):
        """Best label per group; distractors (~) compete but never become tags; a label passes at p >= max(thr, min_p)."""
        p = self.probs(img); best = {}
        for lab, v in p.items():
            if lab.startswith("~"):
                continue
            g = self.group[lab] or lab
            if g not in best or v > p[best[g]]:
                best[g] = lab
        kind = lambda l: GROUP_KINDS.get(self.group[l] or "", "subject")
        tags = [Tag(l, kind(l), p[l], "classic") for l in best.values() if p[l] >= max(thr, self.min_p[l])]
        safe = [l for l in best.values() if self.min_p[l] == 0]
        if not tags and fallback and safe:
            top = max(safe, key=lambda l: p[l])
            if p[top] >= self.FALLBACK_MIN:
                tags = [Tag(top, kind(top), p[top], "classic")]
        return sorted(tags, key=lambda t: -t.score), p


class ClassicTagger:
    ART_GATE = 0.10

    def __init__(self, progress=None):
        if progress:
            progress("Loading CLIP…")
        self.clip = ClipZeroShot(sync_vocab(), progress)

    def tag(self, img: Image.Image, wd: WdResult | None, concept_thr=0.30, general_thr=0.35, fallback=True) -> tuple[list[Tag], bool]:
        """-> (general tags, looks like art). Characters are handled by the fusion step."""
        tags, p = self.clip.classify(img, concept_thr, fallback)
        art = sum(p.get(k, 0) for k in ART_LABELS) >= self.ART_GATE
        if art and wd is not None:
            tags += [Tag(n, "subject", s, "wd14") for n, s in wd.general if s >= general_thr][:4]
        return tags, art
