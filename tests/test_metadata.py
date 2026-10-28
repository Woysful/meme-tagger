import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memetagger import metadata as md
from memetagger.metadata import Meta, MetadataError

META = Meta(tags=["кот", "shrek", "reaction"], kinds={"кот": "subject", "shrek": "character"},
            description="Кот смотрит в камеру & <удивляется>", title="КОГДА ПЯТНИЦА")

FOREIGN = """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="4"
      xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:rights><rdf:Alt><rdf:li xml:lang="x-default">CC0</rdf:li></rdf:Alt></dc:rights>
   <dc:subject><rdf:Bag><rdf:li>old</rdf:li></rdf:Bag></dc:subject>
   <dc:description><rdf:Alt><rdf:li xml:lang="x-default">old description</rdf:li></rdf:Alt></dc:description>
  </rdf:Description>
  <rdf:Description rdf:about="" xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
   <lr:hierarchicalSubject><rdf:Bag>
    <rdf:li>People|Alice</rdf:li><rdf:li>MemeTagger|other|old</rdf:li>
   </rdf:Bag></lr:hierarchicalSubject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def _image(mode: str = "RGB", size: tuple[int, int] = (64, 48)) -> Image.Image:
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], len(mode)), dtype=np.uint8), mode)


def _pixels(path: Path) -> bytes:
    with Image.open(path) as im:
        im.load()
        return im.tobytes()


def _exif() -> bytes:
    exif = Image.Exif()
    exif[0x010E] = "exif description"
    return exif.tobytes()


def _jpeg_markers(data: bytes) -> list[tuple[int, bytes]]:
    segments, _ = md._jpeg_segments(data)
    return [(s.marker, data[s.payload:s.payload + 6]) for s in segments]


@pytest.fixture
def no_exiftool(monkeypatch):
    monkeypatch.setattr(md.shutil, "which", lambda name: None)


# --- packet -------------------------------------------------------------------

def test_packet_round_trip_cyrillic():
    packet = md.build_packet(META)
    assert packet.startswith('<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>')
    assert packet.endswith('<?xpacket end="w"?>')
    assert "<rdf:Bag>" in packet and 'xml:lang="x-default"' in packet and "ns0" not in packet
    assert "MemeTagger|character|shrek" in packet and "reaction|" not in packet
    assert md.parse_packet(packet) == META


def test_empty_fields_are_removed():
    packet = md.build_packet(Meta(), FOREIGN)
    assert "dc:subject" not in packet and "dc:description" not in packet and "dc:title" not in packet
    assert "People|Alice" in packet and "MemeTagger|" not in packet
    assert md.parse_packet(packet) == Meta()


def test_merge_keeps_foreign_properties():
    packet = md.build_packet(META, FOREIGN)
    assert 'xmp:Rating="4"' in packet and "CC0" in packet and "ns0" not in packet
    assert "old description" not in packet and ">old<" not in packet and "MemeTagger|other|old" not in packet
    assert "People|Alice" in packet
    assert md.parse_packet(packet) == META
    assert packet.count("hierarchicalSubject>") == 2  # a single property element
    assert md.build_packet(META, md.build_packet(META, FOREIGN)) == md.build_packet(META, FOREIGN)


def test_parse_attribute_form_and_multiple_descriptions():
    xml = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
      <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" dc:title="Привет"/>
      <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:subject><rdf:Bag><rdf:li>a</rdf:li><rdf:li>б</rdf:li></rdf:Bag></dc:subject>
        <dc:description><rdf:Alt><rdf:li xml:lang="en">en</rdf:li><rdf:li xml:lang="x-default">def</rdf:li></rdf:Alt></dc:description>
      </rdf:Description></rdf:RDF></x:xmpmeta>"""
    assert md.parse_packet(xml) == Meta(tags=["a", "б"], description="def", title="Привет")
    with pytest.raises(MetadataError):
        md.parse_packet("<x:xmpmeta")


def test_corrupt_existing_packet_is_replaced():
    assert md.parse_packet(md.build_packet(META, "<broken")) == META


# --- JPEG ---------------------------------------------------------------------

@pytest.mark.parametrize("with_exif", [False, True])
def test_jpeg_round_trip(tmp_path, with_exif):
    path = tmp_path / "a.jpg"
    _image().save(path, quality=90, **({"exif": _exif()} if with_exif else {}))
    path.chmod(0o640)
    before = _pixels(path)
    assert md.write(path, META) == "embedded"
    assert md.write(path, META) == "embedded"
    data = path.read_bytes()
    assert data.count(md.JPEG_XMP) == 1
    markers = _jpeg_markers(data)
    xmp_index = markers.index((0xE1, md.JPEG_XMP[:6]))
    assert all(m == 0xE0 or p == md.JPEG_EXIF for m, p in markers[:xmp_index])
    assert md.read(path) == META
    assert _pixels(path) == before
    assert path.stat().st_mode & 0o777 == 0o640
    if with_exif:
        with Image.open(path) as im:
            assert im.getexif()[0x010E] == "exif description"
    assert not md.sidecar_path(path).exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_jpeg_keeps_other_bytes(tmp_path):
    path = tmp_path / "a.jpg"
    _image().save(path, exif=_exif())
    original = path.read_bytes()
    md.write(path, META)
    data = path.read_bytes()
    segments, _ = md._jpeg_segments(data)
    xmp = next(s for s in segments if md._jpeg_is(data, s, md.JPEG_XMP))
    assert data[:xmp.start] + data[xmp.end:] == original


def test_jpeg_oversized_description_is_truncated(tmp_path):
    path = tmp_path / "big.jpg"
    _image().save(path)
    meta = Meta(tags=["a", "b"], description="я" * 70000)
    assert md.write(path, meta) == "embedded"
    got = md.read(path)
    assert got.tags == ["a", "b"]
    assert 0 < len(got.description) < 70000 and meta.description.startswith(got.description)
    segments, _ = md._jpeg_segments(path.read_bytes())
    assert all(s.end - s.payload <= 65533 for s in segments)
    assert meta.description == "я" * 70000  # caller's object is untouched


def test_jpeg_too_many_tags_are_dropped_from_end(tmp_path):
    path = tmp_path / "tags.jpg"
    _image().save(path)
    tags = [f"tag-{i:05d}-xxxxxxxxxx" for i in range(5000)]
    md.write(path, Meta(tags=tags, description="d"))
    got = md.read(path)
    assert got.description == "" and 0 < len(got.tags) < len(tags) and got.tags == tags[:len(got.tags)]


# --- PNG ----------------------------------------------------------------------

def test_png_round_trip(tmp_path):
    path = tmp_path / "a.png"
    _image("RGBA").save(path)
    before = _pixels(path)
    md.write(path, META)
    md.write(path, META)
    data = path.read_bytes()
    assert data.count(md.PNG_XMP_KEY) == 1
    assert data.index(b"IHDR") < data.index(b"iTXt") < data.index(b"IDAT")
    assert md.read(path) == META
    assert _pixels(path) == before
    with Image.open(path) as im:
        assert "MemeTagger|character|shrek" in im.info["XML:com.adobe.xmp"]


# --- WebP ---------------------------------------------------------------------

def test_webp_vp8x_round_trip(tmp_path):
    path = tmp_path / "a.webp"
    _image().save(path, "WEBP", lossless=True, exif=_exif())
    assert path.read_bytes()[12:16] == b"VP8X"
    before = _pixels(path)
    assert md.write(path, META) == "embedded"
    assert md.write(path, META) == "embedded"
    data = path.read_bytes()
    assert data[20] & 0x04 and data.count(b"XMP ") == 1
    assert int.from_bytes(data[4:8], "little") == len(data) - 8
    assert md.read(path) == META
    assert _pixels(path) == before
    with Image.open(path) as im:
        assert im.getexif()[0x010E] == "exif description"


def test_simple_webp_falls_back_to_sidecar(tmp_path, no_exiftool):
    path = tmp_path / "a.webp"
    _image().save(path, "WEBP", lossless=True)
    original = path.read_bytes()
    assert original[12:16] != b"VP8X"
    assert md.write(path, META) == "sidecar"
    assert path.read_bytes() == original
    assert md.sidecar_path(path).exists()
    assert md.read(path) == META


# --- other formats ------------------------------------------------------------

def test_gif_sidecar_fallback_and_merge(tmp_path, no_exiftool):
    path = tmp_path / "a.gif"
    _image().convert("P").save(path)
    md.sidecar_path(path).write_text(FOREIGN, encoding="utf-8")
    assert md.write(path, META) == "sidecar"
    text = md.sidecar_path(path).read_text(encoding="utf-8")
    assert "CC0" in text and "People|Alice" in text
    assert md.read(path) == META
    assert md.write(path, META, "off") == "off"


def test_sidecar_mode_for_jpeg(tmp_path):
    path = tmp_path / "a.jpg"
    _image().save(path)
    original = path.read_bytes()
    assert md.write(path, META, mode="sidecar") == "sidecar"
    assert path.read_bytes() == original and md.read(path) == META
    with pytest.raises(ValueError):
        md.write(path, META, mode="bogus")


def test_exiftool_path(tmp_path, monkeypatch):
    path = tmp_path / "-a.gif"
    _image().convert("P").save(path, "GIF")
    calls, stored = [], {}

    def fake_run(args, **kwargs):
        calls.append(args)
        if "-b" in args:
            return subprocess.CompletedProcess(args, 0, stored.get("xmp", b""), b"")
        source = next(a for a in args if a.startswith("-XMP<="))[len("-XMP<="):]
        stored["xmp"] = Path(source).read_bytes()
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(md.shutil, "which", lambda name: "/usr/bin/exiftool")
    monkeypatch.setattr(md.subprocess, "run", fake_run)
    assert md.write(path, META) == "exiftool"
    write_call = calls[-1]
    assert write_call[0] == "/usr/bin/exiftool" and "-overwrite_original" in write_call
    assert write_call[-1] == str(path.resolve())
    assert md.parse_packet(stored["xmp"].decode()) == META
    assert md.read(path) == META
    assert not md.sidecar_path(path).exists()

    monkeypatch.setattr(md.subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 1, b"", b"boom"))
    with pytest.raises(MetadataError, match="boom"):
        md.write(path, META)


# --- paths and errors ---------------------------------------------------------

def test_sidecar_path_and_move(tmp_path):
    assert md.sidecar_path("/x/a.jpg") == Path("/x/a.jpg.xmp")
    src, dst = tmp_path / "a.jpg", tmp_path / "sub" / "b.png"
    dst.parent.mkdir()
    assert md.move_sidecar(src, dst) is False
    md.sidecar_path(src).write_text("x")
    assert md.move_sidecar(src, dst) is True
    assert md.sidecar_path(dst).read_text() == "x" and not md.sidecar_path(src).exists()


@pytest.mark.parametrize("name,content", [
    ("bad.jpg", b"\xff\xd8\xff\xe1\xff\xffgarbage"),
    ("bad.png", md.PNG_SIG + b"\x00\x00\x10\x00IHDR"),
    ("bad.webp", b"RIFF\xff\xff\xff\xffWEBPVP8X\xff\xff\x00\x00"),
    ("bad.gif", b"GIF89a"),
    ("empty.jpg", b""),
])
def test_read_corrupt_returns_none(tmp_path, no_exiftool, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    assert md.read(path) is None
    assert md.read(tmp_path / "missing.jpg") is None


def test_read_corrupt_xmp_returns_none(tmp_path):
    path = tmp_path / "a.jpg"
    _image().save(path)
    data = path.read_bytes()
    path.write_bytes(md._jpeg_set(data, b"<x:xmpmeta><rdf:RDF"))
    assert md.read(path) is None
    md.sidecar_path(path).write_text(md.build_packet(META), encoding="utf-8")
    assert md.read(path) == META


def test_write_errors_raise_metadata_error(tmp_path):
    with pytest.raises(MetadataError):
        md.write(tmp_path / "missing.jpg", META)
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe1\x00")
    with pytest.raises(MetadataError):
        md.write(path, META)
