"""Runs images through the engine and writes a Markdown report.
  python -m memetagger.debug ~/test_images --out report.md
  python -m memetagger.debug ~/test_images --models qwen3-vl:8b,gemma3:12b --out compare.md
With several models every image lists the results of all models side by side."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from .engine import Engine, EngineConfig, TagOptions
from .models import DEFAULT_WD, WD_MODELS
from .organizer import collect, item_context
from .tagstore import TagStore
from .vlm import ALL_WANT, DEFAULT_MODEL


def run(paths, models, cfg: EngineConfig = EngineConfig(), opts: TagOptions = TagOptions(), progress=print) -> dict:
    items = collect(paths)
    table = {it.src.name: {} for it in items}
    for model in models:
        progress(f"== {model}")
        eng = Engine(EngineConfig(**{**cfg.__dict__, "model": model}), TagStore(), progress)   # clean memory per model
        for it in items:
            t0 = time.time()
            try:
                r = eng.tag(it.src, item_context(it), opts)
                tags = " ".join(f"{t.name}" for t in r.tags) or "(none)"
                extra = [f"OCR: {r.ocr_text[:120]}"] if r.ocr_text else []
                extra += [f"WD14: {', '.join(r.candidates)}"] if r.candidates else []
                cell = (f"**{tags}** — {r.description} _({time.time() - t0:.1f}s, {r.source})_"
                        + "".join(f"\n    - {x}" for x in extra + r.notes))
            except Exception as e:
                cell = f"ERROR {type(e).__name__}: {e}"
            table[it.src.name][model] = cell
            progress(f"{it.src.name}: {cell}")
        eng.flush()
    return table


def to_markdown(table: dict, models) -> str:
    out = []
    for name, cells in table.items():
        out.append(f"### {name}")
        out += [f"- `{m}`: {cells.get(m, '')}" for m in models]
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--mode", default="auto", choices=("auto", "vlm", "classic"))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="one model")
    ap.add_argument("--models", default="", help="comma-separated: comparison mode")
    ap.add_argument("--host", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--wd-model", default=DEFAULT_WD, choices=tuple(WD_MODELS))
    ap.add_argument("--wd-device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--strictness", default="balanced", choices=("lenient", "balanced", "strict"))
    for flag in ("ocr", "wd", "verify", "tags", "text", "description"):
        ap.add_argument(f"--no-{flag}", action="store_true")
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()] or [a.model]
    want = frozenset(k for k in ("tags", "text", "description") if not getattr(a, f"no_{k}")) or ALL_WANT
    cfg = EngineConfig(a.mode, a.host, models[0], not a.no_ocr, not a.no_wd, a.wd_model, a.wd_device, "tags" in want)
    opts = TagOptions(want=want, verify=not a.no_verify, strictness=a.strictness)
    md = to_markdown(run(a.paths, models, cfg, opts), models)
    if a.out:
        Path(a.out).write_text(md, "utf-8"); print("saved", a.out)


if __name__ == "__main__":
    main()
