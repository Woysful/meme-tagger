# MemeTagger
Локальный тегер мемов: CLIP (общие понятия) + WD14 (персонажи/франшизы) через onnxruntime.
Файлы получают имя `оригинал [anime k-on hirasawa_yui].jpg`, а когда тегов набирается >= порога — раскладываются
по вложенным папкам (`anime/k-on/hirasawa_yui/…`).

Запуск из исходников: `pip install -r requirements.txt && python run.py [файлы/папки]`
Сборка AppImage: `./build_appimage.sh` (или `BUNDLE_MODELS=1 ./build_appimage.sh`)
Тесты логики: `python tests/test_organizer.py`
Настройки словаря: `~/.config/memetagger/vocab.txt`, алиасы: `~/.config/memetagger/aliases.json`
На целевой системе для Qt 6 может понадобиться `libxcb-cursor0`.

Tuning tags: `python -m memetagger.debug some_meme.jpg` prints CLIP probabilities per label,
then adjust `~/.config/memetagger/vocab.txt` (groups, `~distractors`, per-label min probability).

## Best quality: local vision model via Ollama (GPU)
Memes are text- and franchise-heavy, so the main engine is a multimodal LLM that runs on your GPU:
```
sudo pacman -S ollama-cuda && sudo systemctl enable --now ollama
ollama pull qwen3-vl:8b        # ~6 GB, fits a 12 GB card; check `ollama ps` shows 100% GPU
```
The app (Engine: Auto) uses it when reachable, otherwise falls back to CLIP+WD14. WD14 additionally feeds character
hints into the prompt. `tag_memory.json` keeps spellings consistent (re_zero / rezero / re-zero -> one tag) and is
sent to the model as the "existing vocabulary".

Evaluate on a folder and save a report:
`python -m memetagger.debug ~/test_images --out report.md`
Tests: `python tests/test_organizer.py && python tests/test_vlm.py`

Compare several vision models on your own images (Ollama swaps them in and out of VRAM):
`python -m memetagger.debug ~/test_images --models qwen3-vl:8b,gemma3:12b,minicpm-v4.5 --out compare.md`

## Context, OCR and verification
- **Folder hints** (`~/.config/memetagger/folder_hints.json`): folder name -> tags. Files from `Destiny/`, `Gensh/`, `HTT/`... get
  those tags and the model is told about them. Edit the file to match your own folder names.
- **File-name hint**: meaningful original names ("bunny karate", "persona 4 its peak") are passed to the model; random ones
  (`00IoBi56ZE8`, `F7_c1kIWsAAEucE`, `photo_2024-...`, hashes) are detected and ignored.
- **OCR** (RapidOCR, PP-OCRv5 cyrillic+latin, CPU): text read from the image is given to the model as a hint. Results are cached
  by file content. `pip install rapidocr` (models download on first use; the build script pre-downloads them for the AppImage).
- **Verification**: franchise/character guesses not supported by WD14, folder, file name, OCR or your tag history get one short
  yes/no/unsure query to the same model; only "yes" survives. Dropped guesses are listed in the row tooltip / debug report.
- **Seed vocabulary** (`seed_tags.txt`) keeps spellings consistent from the first image.
- Rename option: "replace original name with tags" + "keep meaningful original names".

## Interface
Header badge shows whether Ollama is reachable; the model box is filled from `/api/tags` (vision-capable models only, via `/api/show`
capabilities) and ⟳ refreshes it; the GPU line shows what `ollama ps` would (warns if the model spilled to CPU).
Toolbar: Pause / Resume / Stop / Re-analyze, filter, ⚙ Config (opens the config folder or a config file, "Reload configuration"),
Undo, Apply. Tests of the controller logic run on a Qt stub: `python tests/test_gui_logic.py`.

## Theme / contrast
`memetagger/theme.py` defines explicit light and dark colour tokens (no system palette roles, which made tabs and borders vanish on
some desktop themes). "Theme: system" only picks light or dark. `python tests/test_theme.py` checks WCAG contrast of all
text/background and border/background pairs, so low-contrast regressions fail the tests.

## File name templates and scene description
Naming tab -> "Template". Variables: `{original}` `{meaningful}` (original name only if it looks meaningful) `{text}` (text read from
the image; short texts verbatim, long ones summarized by the AI) `{description}` (one or two sentences describing the scene, written
by the vision model) `{tags}`. Also: fallback `{text|description|original}` (first non-empty), limits `{text:40}` / `{tags:3}`,
optional parts `< - {text}>` (dropped when a variable inside is empty). The extension is added automatically; empty `[]`/`()` are
removed; long names are shortened (text/description first, tags last). Presets fill the field; edit it freely.
Keep `[{tags}]` at the end: that is how tags are read back from file names (folder grouping, skipping analyzed files) - the UI warns
if the template breaks this. The table has editable "Text on the image" and "Scene description" columns. Description language: English,
Russian or Ukrainian (tags stay English). The description needs the vision model; the classic engine leaves it empty.
Tests: `python tests/test_template.py`.

## Generation switches (Engine tab -> Generate)
Tags / Text from the image / Scene description can be turned off independently. Unneeded work is skipped, not just hidden:
- **Tags off**: no WD14 character hints, no verification queries, no tag vocabulary in the prompt, shorter answer; CLIP/WD14 are not even loaded.
- **Text off**: no `name_text` field; OCR is skipped too unless tags still use it as a hint.
- **Description off**: shorter answer. Description only: OCR is skipped as well.
Values of switched-off items already in the table are kept. "Skip files that already have [tags]" applies while tags are generated.
CLI: `python -m memetagger.debug DIR --no-tags --no-text --no-description`.

## Model management, lengths, settings
- **Model management** (Engine tab): "Pull model" downloads the model named in the Model box (progress, cancel; a cancelled download resumes next time),
  "Unload from memory" is `ollama stop` for the selected model and the models this app used. The option "Unload the model from memory when closing
  the app" (on by default) unloads only models used in this session; errors never block closing.
- **Lengths**: "Ask the model to respect these lengths" turns the character limits into a word target for the model (no character counting, no
  line wrapping) plus a generous hard cap in the answer schema (`maxLength`, silently dropped if the server rejects it). Off: free answer, trimmed only in the file name.
- **Settings** are saved automatically when changed (debounced) and on exit (`~/.config/memetagger/memetagger.conf`), including window geometry, splitter and tab.
