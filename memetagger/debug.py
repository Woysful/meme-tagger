"""Прогон изображений через движок и отчёт в Markdown.
  python -m memetagger.debug ~/test_images --out report.md
  python -m memetagger.debug ~/test_images --models qwen3-vl:8b,gemma3:12b,minicpm-v4.5 --out compare.md
При нескольких моделях для каждого изображения выводятся результаты всех моделей рядом."""
import argparse, time
from pathlib import Path
from .engine import Engine
from .organizer import collect, item_context
from .tagmemory import TagMemory


def run(paths, models, mode="auto", host="", progress=print, ocr=True, verify=True, want=None):
    items = collect(paths); table = {it.src.name: {} for it in items}
    for model in models:
        progress(f"== {model}")
        eng = Engine(mode, host, model, TagMemory(), progress, ocr=ocr, verify=verify, want_tags=not want or 'tags' in want)       # у каждой модели своя «чистая» память
        for it in items:
            t0 = time.time()
            try:
                r = eng.tag(it.src, context=item_context(it), **({'want': want} if want else {}))
                ocr = f"[OCR: {r.text[:120]}] " if r.text else ""
                note = f", {r.note}" if r.note else ""
                cell = (f"**{' '.join(r.tags) or '(none)'}** — {r.description} {ocr}"
                        f"_({time.time() - t0:.1f}s, {r.source}{note})_")
            except Exception as e:
                cell = f"ERROR {type(e).__name__}: {e}"
            table[it.src.name][model] = cell; progress(f"{it.src.name}: {cell}")
    return table


def to_markdown(table, models):
    out = []
    for name, cells in table.items():
        out.append(f"### {name}")
        out += [f"- `{m}`: {cells.get(m, '')}" for m in models]
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+"); ap.add_argument("--mode", default="auto")
    ap.add_argument("--model", default="qwen3-vl:8b", help="одна модель")
    ap.add_argument("--models", default="", help="несколько через запятую — режим сравнения")
    ap.add_argument("--host", default=""); ap.add_argument("--out", default="")
    ap.add_argument("--no-ocr", action="store_true"); ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-tags", action="store_true"); ap.add_argument("--no-text", action="store_true"); ap.add_argument("--no-description", action="store_true")
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()] or [a.model]
    md = to_markdown(run(a.paths, models, a.mode, a.host, print, not a.no_ocr, not a.no_verify,
                       frozenset(k for k, off in (('tags', a.no_tags), ('text', a.no_text), ('description', a.no_description)) if not off) or None), models)
    if a.out:
        Path(a.out).write_text(md, "utf-8"); print("saved", a.out)


if __name__ == "__main__":
    main()
