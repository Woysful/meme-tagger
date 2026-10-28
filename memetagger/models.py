"""Поиск/загрузка ONNX-моделей. Сеть нужна только для первой загрузки
(или вообще не нужна, если модели вшиты в AppImage при сборке)."""
import os
import sys
import urllib.request
from pathlib import Path

APP = "memetagger"

# (репозиторий на HuggingFace, файл)
WD14 = "SmilingWolf/wd-swinv2-tagger-v3"
CLIP = "Xenova/clip-vit-base-patch32"
REQUIRED = [
    (WD14, "model.onnx"), (WD14, "selected_tags.csv"),
    (CLIP, "tokenizer.json"),
    (CLIP, "onnx/vision_model_quantized.onnx"),
    (CLIP, "onnx/text_model_quantized.onnx"),
]


def _xdg(var, fallback):
    return Path(os.environ.get(var) or Path.home() / fallback) / APP


def user_data_dir() -> Path:
    p = _xdg("XDG_DATA_HOME", ".local/share"); p.mkdir(parents=True, exist_ok=True); return p


def user_config_dir() -> Path:
    p = _xdg("XDG_CONFIG_HOME", ".config"); p.mkdir(parents=True, exist_ok=True); return p


def bundled_models_dir() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "models"


def get_model_file(repo, filename, progress=None, dest_dir=None) -> Path:
    local = f"{repo.replace('/', '__')}__{filename.replace('/', '__')}"
    dirs = [bundled_models_dir(), user_data_dir() / "models"]
    if dest_dir:
        dirs = [Path(dest_dir)]
    for d in dirs:
        if (d / local).exists():
            return d / local
    dest = dirs[-1] / local
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, got = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk); got += len(chunk)
            if progress and total:
                progress(f"Downloading model {filename}: {got * 100 // total}%")
    tmp.rename(dest)
    return dest


if __name__ == "__main__":  # python -m memetagger.models models/  (для вшивания в AppImage)
    out = sys.argv[1]
    for repo, fn in REQUIRED:
        print("fetch", repo, fn); get_model_file(repo, fn, dest_dir=out)
