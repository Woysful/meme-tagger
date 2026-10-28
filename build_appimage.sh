#!/usr/bin/env bash
# Сборка MemeTagger-x86_64.AppImage. Нужен интернет (pip + appimagetool).
# Собирайте на не слишком новом дистрибутиве (напр. Ubuntu 22.04) — тогда AppImage запустится почти везде.
#   BUNDLE_MODELS=1 ./build_appimage.sh   — вшить модели (~600 МБ, работает полностью офлайн с 1-го запуска)
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install -r requirements.txt pyinstaller

EXTRA=()
if [ "${BUNDLE_MODELS:-0}" = 1 ]; then
  python -m memetagger.models models
  EXTRA+=(--add-data "models:models")
fi

# OCR-модели рапидокра качаются в каталог пакета; прогреваем здесь, чтобы они попали внутрь образа
python -m memetagger.ocr || echo "WARNING: OCR warm-up failed, OCR will be disabled in the AppImage"

pyinstaller --noconfirm --clean --windowed --name memetagger --collect-all rapidocr \
  --add-data "memetagger/folder_hints_default.json:memetagger" \
  --add-data "memetagger/seed_tags_default.txt:memetagger" \
  --add-data "memetagger/vocab_default.txt:memetagger" \
  --add-data "memetagger/aliases_default.json:memetagger" \
  "${EXTRA[@]}" run.py

rm -rf AppDir && mkdir -p AppDir/usr/bin
cp -r dist/memetagger/* AppDir/usr/bin/
python - <<'PY'
from PIL import Image, ImageDraw
im = Image.new("RGBA", (256, 256), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
d.rounded_rectangle((16, 16, 240, 240), 40, fill=(255, 140, 66, 255))
d.polygon([(70, 128), (128, 70), (200, 70), (200, 142), (142, 200)], fill=(255, 255, 255, 255))
d.ellipse((160, 88, 182, 110), fill=(255, 140, 66, 255))
im.save("AppDir/memetagger.png"); im.save("AppDir/.DirIcon", format="PNG")
PY
cat > AppDir/memetagger.desktop <<'D'
[Desktop Entry]
Type=Application
Name=MemeTagger
Comment=Local AI tagging and sorting of memes
Exec=memetagger %F
Icon=memetagger
Categories=Graphics;Utility;
Terminal=false
D
cat > AppDir/AppRun <<'A'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/memetagger" "$@"
A
chmod +x AppDir/AppRun

[ -f appimagetool ] || { curl -L -o appimagetool https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage; chmod +x appimagetool; }
ARCH=x86_64 ./appimagetool --appimage-extract-and-run AppDir MemeTagger-x86_64.AppImage
echo "Готово: MemeTagger-x86_64.AppImage"
