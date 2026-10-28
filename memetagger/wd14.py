"""WD14 v3 anime tagger via onnxruntime: characters with probabilities, plus general tags for the classic engine.
Runs on the GPU (CUDA) when onnxruntime-gpu is installed, otherwise on the CPU."""
from __future__ import annotations
import csv
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from .characters import CharacterDB, default_db, franchise_tag, split_danbooru
from .models import DEFAULT_WD, WD_FILES, WD_MODELS, get_model_file
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


class Wd14:
    def __init__(self, model: str = DEFAULT_WD, device: str = "auto", progress=None, chardb: CharacterDB | None = None):
        import onnxruntime as ort
        repo = WD_MODELS.get(model, WD_MODELS[DEFAULT_WD])[0]
        if progress:
            progress(f"Loading {repo.split('/')[-1]}…")
        path = get_model_file(repo, WD_FILES[0], progress)
        self.sess = ort.InferenceSession(str(path), providers=providers_for(device))
        self.device = "GPU (CUDA)" if self.sess.get_providers()[0] == "CUDAExecutionProvider" else "CPU"
        with open(get_model_file(repo, WD_FILES[1], progress), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        db = chardb or default_db()
        self.char_idx = np.array([i for i, r in enumerate(rows) if r["category"] == "4"])
        self.chars = []
        for i in self.char_idx:
            base, suffix = split_danbooru(rows[i]["name"])
            tag = clean_tag(base)
            self.chars.append((tag, db.franchise_of(tag) or (franchise_tag(suffix) if suffix else None)))
        self.gen_idx = np.array([i for i, r in enumerate(rows) if r["category"] == "0" and r["name"] not in GENERAL_BLACKLIST])
        self.gen_names = [clean_tag(rows[i]["name"]) for i in self.gen_idx]
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        self.size = inp.shape[1] if isinstance(inp.shape[1], int) else 448

    def _input(self, img: Image.Image) -> np.ndarray:
        s = max(img.size)
        canvas = Image.new("RGB", (s, s), (255, 255, 255))
        canvas.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
        canvas = canvas.resize((self.size, self.size), Image.BICUBIC)
        return np.ascontiguousarray(np.asarray(canvas, dtype=np.float32)[:, :, ::-1])[None]   # RGB -> BGR, NHWC

    def __call__(self, img: Image.Image, char_thr: float = 0.3, general_thr: float = 0.35) -> WdResult:
        probs = self.sess.run(None, {self.inp: self._input(img)})[0][0]
        cp = probs[self.char_idx]
        chars = [WdCandidate(*self.chars[j], float(cp[j])) for j in np.argsort(-cp)[:5] if cp[j] >= char_thr]
        gp = probs[self.gen_idx]
        general = [(self.gen_names[j], float(gp[j])) for j in np.argsort(-gp)[:20] if gp[j] >= general_thr]
        return WdResult(chars, general)
