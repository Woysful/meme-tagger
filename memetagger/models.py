"""Model files from Hugging Face. Network is needed only for the first download (or never, if bundled in the AppImage)."""
from __future__ import annotations
import sys
import urllib.request
from pathlib import Path
from .paths import bundled_models_dir, user_data_dir

CLIP = "Xenova/clip-vit-base-patch32"
CLIP_FILES = ("tokenizer.json", "onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx")
WD_MODELS = {
    "eva02-large": ("SmilingWolf/wd-eva02-large-tagger-v3", "WD EVA02-Large v3: best character recognition (~1.2 GB)"),
    "vit-large": ("SmilingWolf/wd-vit-large-tagger-v3", "WD ViT-Large v3 (~1.2 GB)"),
    "swinv2": ("SmilingWolf/wd-swinv2-tagger-v3", "WD SwinV2 v3: smaller and faster (~0.5 GB)"),
}
DEFAULT_WD = "eva02-large"
WD_FILES = ("model.onnx", "selected_tags.csv")


def _local_name(repo: str, filename: str) -> str:
    return f"{repo.replace('/', '__')}__{filename.replace('/', '__')}"


def get_model_file(repo: str, filename: str, progress=None, dest_dir=None) -> Path:
    local = _local_name(repo, filename)
    dirs = [Path(dest_dir)] if dest_dir else [bundled_models_dir(), user_data_dir() / "models"]
    for d in dirs:
        if (d / local).exists():
            return d / local
    dest = dirs[-1] / local
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total, got, shown = int(r.headers.get("Content-Length") or 0), 0, -1
        while chunk := r.read(1 << 20):
            f.write(chunk); got += len(chunk)
            pct = got * 100 // total if total else -1
            if progress and total and pct != shown:
                shown = pct; progress(f"Downloading {repo.split('/')[-1]} {filename}: {pct}% of {total / 1e9:.2f} GB")
    tmp.rename(dest)
    return dest


if __name__ == "__main__":      # python -m memetagger.models models/ [wd-model]  (bundling models into the AppImage)
    out, wd = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_WD)
    for repo, files in ((WD_MODELS[wd][0], WD_FILES), (CLIP, CLIP_FILES)):
        for fn in files:
            print("fetch", repo, fn); get_model_file(repo, fn, dest_dir=out)
