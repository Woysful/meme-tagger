"""OCR of meme text with RapidOCR (onnxruntime, CPU): PP-OCRv5 Cyrillic+Latin model. Results are cached by file content."""
from __future__ import annotations
import json
import threading
from pathlib import Path
from PIL import Image, ImageDraw
from .paths import user_data_dir


class OcrUnavailable(RuntimeError):
    pass


def _warmup_image():
    im = Image.new("RGB", (260, 70), "white")
    ImageDraw.Draw(im).text((10, 25), "Test 123", fill="black")
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
        if engine is not None:
            self.engine = engine
            return
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
                eng(probe)                          # models are loaded lazily on the first call
                if params is None:
                    progress("Cyrillic OCR model unavailable: Russian/Ukrainian text will be read poorly")
                return eng, name
            except Exception as e:
                last = e
        raise OcrUnavailable(f"OCR failed to start: {last}")

    def read(self, image, min_score=0.5, max_chars=500) -> str:
        """image: ImageData (see imaging.py)."""
        key = image.sha1
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        res = self.engine(image.bgr())
        txts, scores = getattr(res, "txts", None) or (), getattr(res, "scores", None) or ()
        text = " ".join(t.strip() for t, s in zip(txts, scores) if s >= min_score and t.strip())[:max_chars]
        with self.lock:
            self.cache[key] = text; self.dirty += 1
            if self.dirty >= 25:
                self._flush_locked()
        return text

    def _flush_locked(self):
        try:
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False), "utf-8")
            tmp.replace(self.cache_path); self.dirty = 0
        except OSError:
            pass

    def flush(self):
        with self.lock:
            if self.dirty:
                self._flush_locked()


if __name__ == "__main__":      # python -m memetagger.ocr  (downloads the models; used by the AppImage build)
    print("OCR ready:", OcrReader(print).lang)
