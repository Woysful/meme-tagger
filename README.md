# MemeTagger

Local AI tagger and organizer for a meme library. A vision model in Ollama writes tags, the text on the image and a
scene description. A WD tagger recognizes anime characters, and a Danbooru database (~24k characters) normalizes their
names and adds the franchise. Files are renamed to `name [anime k-on hirasawa_yui].jpg` and sorted into folders like
`k-on/hirasawa_yui/`.

## Install
```
pip install -r requirements.txt     # NVIDIA: use onnxruntime-gpu[cuda,cudnn] instead of onnxruntime
ollama pull qwen3-vl:8b             # ~6 GB, fits in 12 GB VRAM together with the WD tagger
python run.py [files/folders]
```
WD and CLIP models are downloaded from Hugging Face on first use. AppImage: `./build_appimage.sh`
(`BUNDLE_MODELS=1 WD_MODEL=eva02-large` puts the models inside).

## How tagging works
1. **Prepare**: the image is decoded once; OCR reads its text; the WD tagger proposes characters with probabilities.
   This runs ahead for the next image while the vision model works on the current one.
2. **Vision model**: gets the image, WD candidates, OCR text, the file name and folder hints, and returns franchise,
   characters (with confidence), meme format, subjects and themes.
3. **Fusion**: names are normalized via Danbooru (`yui hirasawa` → `hirasawa_yui`). Guesses are confirmed by WD, OCR
   text or folders; unconfirmed ones are kept, double-checked or dropped depending on Strictness.
4. **Vocabulary**: spellings, tag kinds and character → franchise links. Counts grow only on Apply.

"Why these tags?" in the preview explains each decision. Batch report:
`python -m memetagger.debug ~/memes --out report.md` (`--models a,b` compares models).

## Custom models
- **Vision model**: any Ollama vision model, including GGUF from Hugging Face: `hf.co/owner/repo:Q4_K_M`.
- **Character model**: built-in WD v3 (EVA02, ViT, SwinV2) and PixAI v1.0 (8k characters). Or choose
  "Custom model…" and enter a Hugging Face repo (`owner/repo`, or `owner/repo:file.onnx`) or a local folder.

Custom character models must be ONNX taggers in the WD format: `.onnx` + `selected_tags.csv` (Danbooru categories:
4 = character, 3 = copyright). Input layout is detected; an optional `preprocess.json` sets size, padding and
normalization. Gated repos need `HF_TOKEN`. PyTorch-only repos (e.g. `pixai-labs/pixai-tagger-v1.0`) do not run;
the error message lists compatible ONNX versions.

## Folders
Settings → Library → Folders. Levels are tag kinds: `franchise > character` (default), `medium > franchise >
character`, or empty (rename only).
- A folder is created once N images share it (default 3); existing folders are always used.
- Files already in the library are left alone unless their tags change. Your own folders are kept.
- Theme tags (cat, war…) never become folders.

## Metadata
Tags, kinds, text and description are written as XMP into JPEG/PNG/WebP, via exiftool for other formats, or to a
`.xmp` sidecar. Readable by digiKam, Dolphin/Baloo and Lightroom.

## Interface
- Table with per-file status, colored tag chips, filters and search (Ctrl+F). Large folders import in the background.
- Preview with tag chips, autocomplete and WD candidates; tags added with several files selected go to all of them.
- Tag library: rename, merge, change kind or delete tags, optionally in library files too.
- Undo reverts the last Apply.

## Files
- Config: `~/.config/memetagger/` (`folder_hints.json`, `seed_tags.txt`, `aliases.json`, `vocab.txt`).
- Data: `~/.local/share/memetagger/` (models, `tag_memory.json`, undo journal, OCR cache).
- Character database: `python tools/build_chardb.py`.

## Development
```
QT_QPA_PLATFORM=offscreen python -m pytest -q tests
```
