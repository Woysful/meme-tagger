"""One decode per image: every stage (OCR, WD14, VLM, thumbnail) reuses the same pixels."""
from __future__ import annotations
import base64
import hashlib
import io
from pathlib import Path
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 300_000_000          # large screenshots are fine; real decompression bombs are still refused


def to_rgb(im: Image.Image) -> Image.Image:
    """Transparent areas become white (memes are usually drawn on white), palette/CMYK/16-bit become RGB."""
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")
    return im.convert("RGB")


class ImageData:
    def __init__(self, path):
        self.path = Path(path)
        self._rgb: Image.Image | None = None
        self._sha1: str | None = None
        self._jpeg: dict[int, str] = {}

    @property
    def rgb(self) -> Image.Image:
        if self._rgb is None:
            with Image.open(self.path) as im:
                im.seek(0)                                  # first frame of animations
                self._rgb = to_rgb(ImageOps.exif_transpose(im))
        return self._rgb

    @property
    def sha1(self) -> str:
        if self._sha1 is None:
            h = hashlib.sha1()
            with open(self.path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._sha1 = h.hexdigest()
        return self._sha1

    def fitted(self, max_side: int) -> Image.Image:
        im = self.rgb
        if max(im.size) <= max_side:
            return im
        im = im.copy()
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        return im

    def jpeg_b64(self, max_side: int = 1344) -> str:
        if max_side not in self._jpeg:
            buf = io.BytesIO()
            self.fitted(max_side).save(buf, "JPEG", quality=90)
            self._jpeg[max_side] = base64.b64encode(buf.getvalue()).decode()
        return self._jpeg[max_side]

    def bgr(self, max_side: int = 2000):
        import numpy as np
        return np.ascontiguousarray(np.asarray(self.fitted(max_side))[:, :, ::-1])

    def release(self):
        self._rgb = None; self._jpeg.clear()


def thumbnail_rgba(path, size: int = 160) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            im.draft("RGB", (size * 2, size * 2))           # fast JPEG downscale while decoding
            im = ImageOps.exif_transpose(im)
            im.thumbnail((size, size))
            return im.convert("RGBA")
    except Exception:
        return None
