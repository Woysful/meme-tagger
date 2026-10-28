"""Model files from Hugging Face. Network is needed only for the first download (or never, if bundled in the AppImage).

Besides the built-in WD taggers, any ONNX tagger in the WD format can be used ("custom" model): a Hugging Face repo
(owner/repo, or owner/repo:file.onnx to pick one of several ONNX files) or a local folder / .onnx file. It needs the
ONNX model and a selected_tags.csv with name and category columns (Danbooru categories: 0 general, 3 copyright,
4 character); an optional preprocess.json describes the input (see wd14.Preprocess)."""
from __future__ import annotations
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from .paths import bundled_models_dir, user_data_dir

CLIP = "Xenova/clip-vit-base-patch32"
CLIP_FILES = ("tokenizer.json", "onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx")
WD_MODELS = {      # key -> (repo, label)
    "eva02-large": ("SmilingWolf/wd-eva02-large-tagger-v3", "EVA02-Large v3 · best · 1.2 GB"),
    "vit-large": ("SmilingWolf/wd-vit-large-tagger-v3", "ViT-Large v3 · 1.2 GB"),
    "swinv2": ("SmilingWolf/wd-swinv2-tagger-v3", "SwinV2 v3 · faster · 0.5 GB"),
}
DEFAULT_WD = "eva02-large"
CUSTOM = "custom"
WD_FILES = ("model.onnx", "selected_tags.csv")
TAGS_FILE, PREPROCESS_FILE = "selected_tags.csv", "preprocess.json"
_REPO = re.compile(r"^[\w.-]+/[\w.-]+(:[\w./-]+\.onnx)?$")


class ModelError(RuntimeError):
    pass


def _local_name(repo: str, filename: str) -> str:
    return f"{repo.replace('/', '__')}__{filename.replace('/', '__')}"


def _request(url: str) -> urllib.request.Request:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})


def _download(repo: str, filename: str, dest: Path, progress=None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        with urllib.request.urlopen(_request(url), timeout=60) as r, open(tmp, "wb") as f:
            total, got, shown = int(r.headers.get("Content-Length") or 0), 0, -1
            while chunk := r.read(1 << 20):
                f.write(chunk); got += len(chunk)
                pct = got * 100 // total if total else -1
                if progress and total and pct != shown:
                    shown = pct; progress(f"Downloading {repo.split('/')[-1]} {filename}: {pct}% of {total / 1e9:.2f} GB")
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        hint = " (private or gated repo: set HF_TOKEN)" if e.code in (401, 403) else ""
        raise ModelError(f"cannot download {repo}/{filename}: HTTP {e.code}{hint}") from e
    tmp.rename(dest)
    return dest


def get_model_file(repo: str, filename: str, progress=None, dest_dir=None) -> Path:
    local = _local_name(repo, filename)
    dirs = [Path(dest_dir)] if dest_dir else [bundled_models_dir(), user_data_dir() / "models"]
    for d in dirs:
        if (d / local).exists():
            return d / local
    return _download(repo, filename, dirs[-1] / local, progress)


@dataclass
class WdFiles:
    name: str
    model: Path
    tags: Path
    preprocess: Path | None = None


def _repo_files(repo: str) -> list[str]:
    try:
        with urllib.request.urlopen(_request(f"https://huggingface.co/api/models/{repo}"), timeout=30) as r:
            return [s["rfilename"] for s in json.load(r).get("siblings", [])]
    except urllib.error.HTTPError as e:
        hint = " (private or gated repo: set HF_TOKEN)" if e.code in (401, 403) else ""
        raise ModelError(f"Hugging Face repo '{repo}' not found: HTTP {e.code}{hint}") from e


def _pick_onnx(names: list[str], where: str) -> str:
    onnx = sorted(n for n in names if n.endswith(".onnx"))
    if not onnx:
        raise ModelError(f"no .onnx model in {where}")
    if "model.onnx" in onnx:
        return "model.onnx"
    if len(onnx) > 1:
        raise ModelError(f"several ONNX files in {where}: {', '.join(onnx)}. Pick one: {where}:{onnx[0]}")
    return onnx[0]


def _custom_repo(spec: str, progress=None) -> WdFiles:
    repo, _, onnx = spec.partition(":")
    cache = user_data_dir() / "models" / "custom" / repo.replace("/", "__")
    try:
        files = _repo_files(repo)
    except urllib.error.URLError as e:          # offline: use what was downloaded before
        if not (cache / TAGS_FILE).exists():
            raise ModelError(f"cannot reach Hugging Face for {repo}: {e.reason}") from e
        files = [p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.is_file() and p.suffix != ".part"]
    onnx = onnx or _pick_onnx([f for f in files if "/" not in f], repo)
    if onnx not in files:
        raise ModelError(f"{repo} has no file {onnx}")
    if TAGS_FILE not in files:
        raise ModelError(f"{repo} has no {TAGS_FILE}: only taggers in the WD format are supported")
    if progress:
        progress(f"Loading {repo}…")
    get = lambda fn: cache / fn if (cache / fn).exists() else _download(repo, fn, cache / fn, progress)
    model, tags = get(onnx), get(TAGS_FILE)
    for extra in files:                          # weights stored next to the graph (models over 2 GB)
        if extra.startswith(onnx) and extra != onnx and extra.endswith("data"):
            get(extra)
    return WdFiles(spec, model, tags, get(PREPROCESS_FILE) if PREPROCESS_FILE in files else None)

def _custom_local(path: Path) -> WdFiles:
    folder = path.parent if path.is_file() else path
    model = path if path.is_file() else folder / _pick_onnx([p.name for p in folder.iterdir()], str(folder))
    tags = folder / TAGS_FILE
    if not tags.exists():
        raise ModelError(f"{TAGS_FILE} is missing next to {model}")
    pre = folder / PREPROCESS_FILE
    return WdFiles(model.name if path.is_file() else folder.name, model, tags, pre if pre.exists() else None)


def resolve_wd(spec: str, progress=None) -> WdFiles:
    """Built-in key, Hugging Face repo spec or local path -> files on disk (downloaded when needed)."""
    spec = (spec or DEFAULT_WD).strip()
    if spec in WD_MODELS:
        repo = WD_MODELS[spec][0]
        if progress:
            progress(f"Loading {repo.split('/')[-1]}…")
        return WdFiles(repo.split("/")[-1], get_model_file(repo, WD_FILES[0], progress),
                       get_model_file(repo, WD_FILES[1], progress))
    path = Path(spec).expanduser()
    if path.exists():
        return _custom_local(path)
    if _REPO.match(spec):
        return _custom_repo(spec, progress)
    raise ModelError(f"'{spec}' is neither a folder/file nor a Hugging Face repo like owner/name")


if __name__ == "__main__":      # python -m memetagger.models models/ [wd-model]  (bundling models into the AppImage)
    out, wd = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_WD)
    for repo, files in ((WD_MODELS[wd][0], WD_FILES), (CLIP, CLIP_FILES)):
        for fn in files:
            print("fetch", repo, fn); get_model_file(repo, fn, dest_dir=out)
