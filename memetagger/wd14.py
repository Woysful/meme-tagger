"""WD-format anime taggers via onnxruntime: characters with probabilities, plus general tags for the classic engine.
Runs on the GPU (CUDA) when onnxruntime-gpu is installed, otherwise on the CPU. Built-in models are the WD v3 taggers;
custom ones (see models.resolve_wd) may use another input layout and preprocessing."""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image
from .characters import CharacterDB, default_db, franchise_tag, split_danbooru
from .models import DEFAULT_WD, ModelError, resolve_wd
from .tags import clean_tag

# Danbooru tags that describe composition rather than content: useless as file tags.
GENERAL_BLACKLIST = frozenset({
    "solo", "1girl", "1boy", "2girls", "2boys", "3girls", "multiple_girls", "multiple_boys", "looking_at_viewer",
    "simple_background", "white_background", "grey_background", "smile", "open_mouth", "blush", "long_hair", "short_hair",
    "closed_mouth", "standing", "upper_body", "english_text", "highres", "commentary", "bad_id", "absurdres", "comic",
    "monochrome", "greyscale", "text_focus", "speech_bubble", "translated", "meme", "parody", "sitting", "full_body",
    "cowboy_shot", "portrait", "close-up", "outdoors", "indoors", "day", "night", "teeth", "breasts"})


@dataclass
class WdCandidate:
    tag: str
    franchise: str | None
    score: float

    def label(self) -> str:
        fr = f" ({self.franchise})" if self.franchise else ""
        return f"{self.tag.replace('_', ' ')}{fr} {self.score:.2f}"


@dataclass
class WdResult:
    characters: list[WdCandidate] = field(default_factory=list)
    general: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Preprocess:
    """Square padding, resize and pixel scaling. WD v3: NHWC, BGR, 0..255 on white. NCHW models default to the timm
    convention (RGB, 0..1, mean/std 0.5); preprocess.json next to the model can override size, padding and mean/std:
    {"stages": [{"type": "rescale_pad", "size": 1008, "pad_value": 0}, {"type": "normalize", "mean": [...], "std": [...]}]}"""
    size: int = 448
    nchw: bool = False
    bgr: bool = True
    scale: float = 1.0
    mean: tuple = (0.0, 0.0, 0.0)
    std: tuple = (1.0, 1.0, 1.0)
    pad: int = 255

    @classmethod
    def for_model(cls, shape, config: Path | None = None) -> "Preprocess":
        nchw = len(shape) == 4 and shape[1] == 3 and shape[3] != 3
        dim = shape[2] if nchw else shape[1]
        p = cls(nchw=True, bgr=False, scale=1 / 255, mean=(0.5,) * 3, std=(0.5,) * 3) if nchw else cls()
        if isinstance(dim, int) and dim > 0:
            p.size = dim
        if config is not None:
            try:
                cfg = json.loads(config.read_text("utf-8"))
            except (OSError, ValueError) as e:
                raise ModelError(f"cannot read {config}: {e}") from e
            for st in cfg.get("stages", [cfg]):
                p.size = int(st.get("size", p.size))
                p.pad = int(st.get("pad_value", p.pad))
                if "mean" in st or "std" in st:
                    p.scale = 1 / 255
                    p.mean, p.std = tuple(st.get("mean", p.mean)), tuple(st.get("std", p.std))
        return p

    def __call__(self, img: Image.Image) -> np.ndarray:
        s = max(img.size)
        canvas = Image.new("RGB", (s, s), (self.pad,) * 3)
        canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
        canvas = canvas.resize((self.size, self.size), Image.BICUBIC)
        x = np.asarray(canvas, dtype=np.float32)
        if self.scale != 1.0 or any(self.mean) or any(v != 1.0 for v in self.std):
            x = (x * self.scale - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        if self.bgr:
            x = x[:, :, ::-1]
        if self.nchw:
            x = x.transpose(2, 0, 1)
        return np.ascontiguousarray(x, dtype=np.float32)[None]


def providers_for(device: str) -> list[str]:
    import onnxruntime as ort
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()          # CUDA/cuDNN from the nvidia-* pip wheels (onnxruntime-gpu[cuda,cudnn])
        except Exception:
            pass
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError("CUDA is not available to onnxruntime. Install it with:  pip install 'onnxruntime-gpu[cuda,cudnn]'")
    return ["CPUExecutionProvider"]


def read_tags(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"name", "category"} <= set(rows[0]):
        raise ModelError(f"{path.name} needs 'name' and 'category' columns (WD selected_tags.csv format)")
    return rows


class Wd14:
    def __init__(self, model: str = DEFAULT_WD, device: str = "auto", progress=None, chardb: CharacterDB | None = None):
        import onnxruntime as ort
        files = resolve_wd(model, progress)
        rows = read_tags(files.tags)
        self.name = files.name
        self.sess = ort.InferenceSession(str(files.model), providers=providers_for(device))
        self.device = "GPU (CUDA)" if self.sess.get_providers()[0] == "CUDAExecutionProvider" else "CPU"
        n_out = self.sess.get_outputs()[0].shape[-1]
        if isinstance(n_out, int) and n_out != len(rows):
            raise ModelError(f"{files.model.name} has {n_out} outputs, but {files.tags.name} lists {len(rows)} tags")
        db = chardb or default_db()
        self.char_idx = np.array([i for i, r in enumerate(rows) if r["category"].strip() == "4"], dtype=int)
        if not len(self.char_idx):
            raise ModelError(f"{self.name} has no character tags (category 4 in {files.tags.name})")
        self.chars = []
        for i in self.char_idx:
            base, suffix = split_danbooru(rows[i]["name"])
            tag = clean_tag(base)
            self.chars.append((tag, db.franchise_of(tag) or (franchise_tag(suffix) if suffix else None)))
        self.gen_idx = np.array([i for i, r in enumerate(rows) if r["category"].strip() == "0"
                                 and r["name"] not in GENERAL_BLACKLIST], dtype=int)
        self.gen_names = [clean_tag(rows[i]["name"]) for i in self.gen_idx]
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        self.prep = Preprocess.for_model(list(inp.shape), files.preprocess)

    def probs(self, img: Image.Image) -> np.ndarray:
        p = np.asarray(self.sess.run(None, {self.inp: self.prep(img)})[0][0], dtype=np.float32)
        if p.size and (p.min() < 0 or p.max() > 1):       # a model that outputs logits
            p = 1 / (1 + np.exp(-p))
        return p

    def __call__(self, img: Image.Image, char_thr: float = 0.3, general_thr: float = 0.35) -> WdResult:
        probs = self.probs(img)
        cp = probs[self.char_idx]
        chars = [WdCandidate(*self.chars[j], float(cp[j])) for j in np.argsort(-cp)[:5] if cp[j] >= char_thr]
        gp = probs[self.gen_idx]
        general = [(self.gen_names[j], float(gp[j])) for j in np.argsort(-gp)[:20] if gp[j] >= general_thr]
        return WdResult(chars, general)
