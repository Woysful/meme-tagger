"""OCR текста на мемах через RapidOCR (onnxruntime, CPU). Кириллица + латиница (модель PP-OCRv5 cyrillic).
Результат кэшируется по содержимому файла."""
from __future__ import annotations
import hashlib, json, threading
from pathlib import Path
from PIL import Image, ImageDraw
from .models import user_data_dir


class OcrUnavailable(RuntimeError):
    pass


def _prep(path, max_side=2000):
    import numpy as np
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    im = bg.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])         # RGB -> BGR (как ждёт OpenCV-пайплайн)


def _warmup_image():
    im = Image.new("RGB", (260, 70), "white"); ImageDraw.Draw(im).text((10, 25), "Test 123", fill="black")
    return im


class OcrReader:
    def __init__(self, progress=None, cache_path=None, engine=None):
        self.lock, self.dirty = threading.Lock(), 0
        self.cache_path = Path(cache_path or user_data_dir() / "ocr_cache.json")
        try:
            self.cache = json.loads(self.cache_path.read_text("utf-8")) if self.cache_path.exists() else {}
        except (OSError, ValueError):
            self.cache = {}
        self.lang = "custom"
        if engine is not None:                      # для тестов
            self.engine = engine; return
        self.engine, self.lang = self._create(progress or (lambda m: None))

    @staticmethod
    def _create(progress):
        try:
            from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
        except ImportError as e:
            raise OcrUnavailable(f"rapidocr is not installed (pip install rapidocr): {e}")
        import numpy as np
        probe = np.asarray(_warmup_image())[:, :, ::-1].copy()
        cyr = getattr(LangRec, "CYRILLIC", None)
        if cyr is None:
            try:
                cyr = LangRec("cyrillic")
            except Exception:
                cyr = None
        attempts = []
        if cyr is not None:
            attempts.append(("cyrillic+latin", {
                "Det.engine_type": EngineType.ONNXRUNTIME, "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE, "Det.ocr_version": OCRVersion.PPOCRV5,
                "Rec.engine_type": EngineType.ONNXRUNTIME, "Rec.lang_type": cyr,
                "Rec.model_type": ModelType.MOBILE, "Rec.ocr_version": OCRVersion.PPOCRV5}))
        attempts.append(("default (Chinese/English only)", None))
        last = None
        for name, params in attempts:
            try:
                progress(f"Loading OCR ({name})…")
                eng = RapidOCR(params=params) if params else RapidOCR()
                eng(probe)                          # в версиях >= 3.10 модели грузятся при первом вызове
                if params is None:
                    progress("Cyrillic OCR model unavailable: Russian/Ukrainian text will be read poorly")
                return eng, name
            except Exception as e:                  # скачивание/несовместимость — пробуем следующий вариант
                last = e
        raise OcrUnavailable(f"OCR failed to start: {last}")

    @staticmethod
    def _key(path) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def read(self, path, min_score=0.5, max_chars=500) -> str:
        key = self._key(path)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        res = self.engine(_prep(path))
        txts, scores = getattr(res, "txts", None) or (), getattr(res, "scores", None) or ()
        text = " ".join(t.strip() for t, s in zip(txts, scores) if s >= min_score and t.strip())[:max_chars]
        with self.lock:
            self.cache[key] = text; self.dirty += 1
            if self.dirty >= 25:
                self._flush_locked()
        return text

    def _flush_locked(self):
        try:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), "utf-8"); self.dirty = 0
        except OSError:
            pass

    def flush(self):
        with self.lock:
            if self.dirty:
                self._flush_locked()


if __name__ == "__main__":      # python -m memetagger.ocr --warmup  (скачивает модели; вызывается при сборке AppImage)
    import sys
    r = OcrReader(print)
    print("OCR ready:", r.lang)
