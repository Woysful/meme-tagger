"""XMP metadata for meme images.

Tags, tag kinds, a description and the text read from the image are stored as
standard XMP properties so that file managers, digiKam and tracker can search
them:

* ``dc:subject`` (rdf:Bag) - the tags;
* ``lr:hierarchicalSubject`` (rdf:Bag) - ``MemeTagger|<kind>|<tag>`` entries;
* ``dc:description`` / ``dc:title`` (rdf:Alt, ``x-default``).

JPEG, PNG and extended (VP8X) WebP files are rewritten natively; other formats
go through exiftool when it is installed and fall back to a ``<file>.xmp``
sidecar otherwise. Only the standard library is used.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

MODES = ("off", "embed", "sidecar")


class MetadataError(Exception):
    """Raised when metadata cannot be written."""


@dataclass
class Meta:
    tags: list[str] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)
    description: str = ""
    title: str = ""

    def is_empty(self) -> bool:
        return not (self.tags or self.kinds or self.description or self.title)


# --- XMP packet ---------------------------------------------------------------

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
}
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)


def _q(prefix: str, name: str) -> str:
    return f"{{{NS[prefix]}}}{name}"


XMPMETA, XMPTK = _q("x", "xmpmeta"), _q("x", "xmptk")
RDF, DESC, ABOUT = _q("rdf", "RDF"), _q("rdf", "Description"), _q("rdf", "about")
BAG, SEQ, ALT, LI = _q("rdf", "Bag"), _q("rdf", "Seq"), _q("rdf", "Alt"), _q("rdf", "li")
SUBJECT, TITLE, DESCRIPTION = _q("dc", "subject"), _q("dc", "title"), _q("dc", "description")
HIER = _q("lr", "hierarchicalSubject")
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
OWN_PROPS = (SUBJECT, TITLE, DESCRIPTION, HIER)
HIER_ROOT = "MemeTagger"

_PI = re.compile(r"<\?(?:xpacket|xml)\b.*?\?>", re.S)
_INVALID_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _clean(text: str) -> str:
    return _INVALID_XML.sub("", text)


def _parse_root(xml: str) -> tuple[ET.Element, list[tuple[str, str]]]:
    body = _PI.sub("", xml).strip("\ufeff\x00 \t\r\n")
    try:
        parser = ET.iterparse(io.BytesIO(body.encode("utf-8")), events=("start-ns",))
        namespaces = [pair for _, pair in parser]
        return parser.root, namespaces
    except (ET.ParseError, UnicodeError) as e:
        raise MetadataError(f"invalid XMP: {e}") from e


def _register(namespaces: list[tuple[str, str]]) -> None:
    # ElementTree only knows prefixes globally; without this foreign
    # namespaces would be serialized as ns0, ns1, ...
    for prefix, uri in namespaces:
        if prefix and prefix != "xml":
            try:
                ET.register_namespace(prefix, uri)
            except ValueError:
                pass
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)


def _values(prop: ET.Element) -> list[str]:
    container = next((c for c in prop if c.tag in (BAG, SEQ, ALT)), None)
    if container is None:
        return [prop.text] if prop.text and prop.text.strip() else []
    items = [li for li in container if li.tag == LI]
    if container.tag == ALT:
        items = ([li for li in items if li.get(XML_LANG) == "x-default"] or items)[:1]
    return [li.text or "" for li in items]


def _collect(desc: ET.Element, name: str) -> list[str]:
    values = [desc.attrib[name]] if name in desc.attrib else []
    for prop in desc.findall(name):
        values += _values(prop)
    return values


def _container(parent: ET.Element, name: str, kind: str, items: list[str]) -> None:
    box = ET.SubElement(ET.SubElement(parent, name), kind)
    for text in items:
        li = ET.SubElement(box, LI, {XML_LANG: "x-default"} if kind == ALT else {})
        li.text = text


def _clean_tags(tags: list[str]) -> list[str]:
    return list(dict.fromkeys(t for t in (_clean(t).strip() for t in tags) if t))


def build_packet(meta: Meta, existing: str | None = None) -> str:
    """Return a complete XMP packet, merged into ``existing`` if given."""
    root, namespaces = None, []
    if existing:
        try:
            root, namespaces = _parse_root(existing)
        except MetadataError:
            root = None
    rdf = next(root.iter(RDF), None) if root is not None else None
    if rdf is None:
        root = ET.Element(XMPMETA, {XMPTK: "MemeTagger"})
        rdf = ET.SubElement(root, RDF)
    elif root is rdf:
        root = ET.Element(XMPMETA, {XMPTK: "MemeTagger"})
        root.append(rdf)

    descs = rdf.findall(DESC)
    foreign_hier: list[str] = []
    for desc in descs:
        foreign_hier += [h for h in _collect(desc, HIER) if not h.startswith(HIER_ROOT + "|")]
        for name in OWN_PROPS:
            desc.attrib.pop(name, None)
            for prop in desc.findall(name):
                desc.remove(prop)
    target = descs[0] if descs else ET.SubElement(rdf, DESC, {ABOUT: ""})

    tags = _clean_tags(meta.tags)
    if title := _clean(meta.title).strip():
        _container(target, TITLE, ALT, [title])
    if description := _clean(meta.description).strip():
        _container(target, DESCRIPTION, ALT, [description])
    if tags:
        _container(target, SUBJECT, BAG, tags)
    own_hier = [f"{HIER_ROOT}|{_clean(meta.kinds[t]).strip()}|{t}"
                for t in tags if _clean(meta.kinds.get(t, "")).strip()]
    if hier := list(dict.fromkeys(foreign_hier + own_hier)):
        _container(target, HIER, BAG, hier)

    _register(namespaces)
    body = ET.tostring(root, encoding="unicode")
    return f'<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n{body}\n<?xpacket end="w"?>'


def parse_packet(xml: str) -> Meta:
    """Extract MemeTagger fields from an XMP packet; raise MetadataError if invalid."""
    root, _ = _parse_root(xml)
    values: dict[str, list[str]] = {name: [] for name in OWN_PROPS}
    for desc in root.iter(DESC):
        for name in OWN_PROPS:
            values[name] += _collect(desc, name)
    kinds = {}
    for entry in values[HIER]:
        parts = entry.split("|", 2)
        if len(parts) == 3 and parts[0] == HIER_ROOT and parts[1] and parts[2]:
            kinds[parts[2]] = parts[1]
    first = lambda name: next((v.strip() for v in values[name] if v.strip()), "")
    return Meta(tags=_clean_tags(values[SUBJECT]), kinds=kinds,
                description=first(DESCRIPTION), title=first(TITLE))


# --- JPEG ---------------------------------------------------------------------

JPEG_XMP = b"http://ns.adobe.com/xap/1.0/\x00"
JPEG_EXIF = b"Exif\x00\x00"
JPEG_XMP_LIMIT = 65533 - len(JPEG_XMP)


@dataclass(frozen=True)
class _Segment:
    marker: int
    start: int
    payload: int
    end: int


def _jpeg_segments(data: bytes) -> tuple[list[_Segment], int]:
    """Return the header segments and the offset of SOS/EOI."""
    if data[:2] != b"\xff\xd8":
        raise MetadataError("not a JPEG file")
    segments, pos, size = [], 2, len(data)
    while True:
        start = pos
        if pos >= size or data[pos] != 0xFF:
            raise MetadataError("corrupt JPEG: marker expected")
        while pos < size and data[pos] == 0xFF:
            pos += 1
        if pos >= size:
            raise MetadataError("corrupt JPEG: truncated marker")
        marker, pos = data[pos], pos + 1
        if marker in (0xDA, 0xD9):
            return segments, start
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            segments.append(_Segment(marker, start, pos, pos))
            continue
        if pos + 2 > size:
            raise MetadataError("corrupt JPEG: truncated segment")
        length = int.from_bytes(data[pos:pos + 2], "big")
        if length < 2 or pos + length > size:
            raise MetadataError("corrupt JPEG: bad segment length")
        segments.append(_Segment(marker, start, pos + 2, pos + length))
        pos += length


def _jpeg_is(data: bytes, seg: _Segment, header: bytes) -> bool:
    return seg.marker == 0xE1 and data[seg.payload:seg.payload + len(header)] == header


def _jpeg_get(data: bytes) -> bytes | None:
    segments, _ = _jpeg_segments(data)
    seg = next((s for s in segments if _jpeg_is(data, s, JPEG_XMP)), None)
    return data[seg.payload + len(JPEG_XMP):seg.end] if seg else None


def _jpeg_set(data: bytes, packet: bytes) -> bytes:
    segments, tail = _jpeg_segments(data)
    payload = JPEG_XMP + packet
    new = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    has_xmp = any(_jpeg_is(data, s, JPEG_XMP) for s in segments)
    insert_at = None
    if not has_xmp:
        insert_at = 0
        while insert_at < len(segments) and (
                segments[insert_at].marker == 0xE0 or _jpeg_is(data, segments[insert_at], JPEG_EXIF)):
            insert_at += 1
    out, placed = [data[:2]], False
    for i, seg in enumerate(segments):
        if i == insert_at:
            out.append(new)
            placed = True
        if _jpeg_is(data, seg, JPEG_XMP):
            if not placed:
                out.append(new)
                placed = True
            continue
        out.append(data[seg.start:seg.end])
    if not placed:
        out.append(new)
    out.append(data[tail:])
    return b"".join(out)


def _fit(meta: Meta, existing: str | None, limit: int) -> bytes:
    """Build a packet of at most ``limit`` bytes, shortening the description, then the tags."""
    build = lambda m: build_packet(m, existing).encode("utf-8")
    meta = replace(meta, tags=list(meta.tags))
    packet = build(meta)
    while len(packet) > limit and meta.description:
        # Escaping only makes the output longer than the raw text, so cutting
        # the overflow from the raw UTF-8 bytes always makes progress.
        raw = meta.description.encode("utf-8")
        cut = max(0, len(raw) - (len(packet) - limit))
        meta.description = raw[:cut].decode("utf-8", "ignore").rstrip()
        packet = build(meta)
    if len(packet) > limit and meta.tags:
        lo, hi = 0, len(meta.tags) - 1
        while lo < hi:  # longest prefix of tags that fits
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if len(build(replace(meta, tags=meta.tags[:mid]))) <= limit else (lo, mid - 1)
        meta.tags = meta.tags[:lo]
        packet = build(meta)
    if len(packet) > limit and meta.title:
        meta.title = ""
        packet = build(meta)
    if len(packet) > limit:
        raise MetadataError("existing XMP is too large for a JPEG segment")
    return packet


# --- PNG ----------------------------------------------------------------------

PNG_SIG = b"\x89PNG\r\n\x1a\n"
PNG_XMP_KEY = b"XML:com.adobe.xmp"


def _png_chunks(data: bytes) -> tuple[list[tuple[bytes, int, int]], int]:
    """Return (type, start, end) of every chunk up to IEND and the end offset."""
    if data[:8] != PNG_SIG:
        raise MetadataError("not a PNG file")
    chunks, pos = [], 8
    while True:
        if pos + 12 > len(data):
            raise MetadataError("corrupt PNG: truncated chunk")
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        end = pos + 12 + length
        if end > len(data):
            raise MetadataError("corrupt PNG: bad chunk length")
        chunks.append((kind, pos, end))
        pos = end
        if kind == b"IEND":
            break
    if chunks[0][0] != b"IHDR":
        raise MetadataError("corrupt PNG: IHDR must come first")
    return chunks, pos


def _png_xmp_text(data: bytes, start: int, end: int) -> bytes | None:
    body = data[start + 8:end - 4]
    keyword, sep, rest = body.partition(b"\x00")
    if keyword != PNG_XMP_KEY or not sep or len(rest) < 2:
        return None
    compressed, text = rest[0], rest[2:]
    _lang, _, text = text.partition(b"\x00")
    _translated, _, text = text.partition(b"\x00")
    return zlib.decompress(text) if compressed else text


def _png_get(data: bytes) -> bytes | None:
    chunks, _ = _png_chunks(data)
    for kind, start, end in chunks:
        if kind == b"iTXt" and (text := _png_xmp_text(data, start, end)) is not None:
            return text
    return None


def _png_set(data: bytes, packet: bytes) -> bytes:
    chunks, tail = _png_chunks(data)
    body = b"iTXt" + PNG_XMP_KEY + b"\x00\x00\x00\x00\x00" + packet
    new = struct.pack(">I", len(body) - 4) + body + struct.pack(">I", zlib.crc32(body))
    out = [PNG_SIG]
    for kind, start, end in chunks:
        if kind == b"iTXt" and _png_xmp_text(data, start, end) is not None:
            continue
        out.append(data[start:end])
        if kind == b"IHDR":
            out.append(new)
    out.append(data[tail:])
    return b"".join(out)


# --- WebP ---------------------------------------------------------------------

def _is_webp(data: bytes) -> bool:
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def _webp_chunks(data: bytes) -> list[tuple[bytes, int, int]]:
    """Return (fourcc, data_start, data_end) of every RIFF chunk."""
    if not _is_webp(data):
        raise MetadataError("not a WebP file")
    limit = min(len(data), 8 + int.from_bytes(data[4:8], "little"))
    chunks, pos = [], 12
    while pos + 8 <= limit:
        fourcc, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        if pos + 8 + size > limit:
            raise MetadataError("corrupt WebP: bad chunk size")
        chunks.append((fourcc, pos + 8, pos + 8 + size))
        pos += 8 + size + (size & 1)
    if not chunks:
        raise MetadataError("corrupt WebP: no chunks")
    return chunks


def _webp_get(data: bytes) -> bytes | None:
    return next((data[s:e] for kind, s, e in _webp_chunks(data) if kind == b"XMP "), None)


def _riff_chunk(fourcc: bytes, payload: bytes) -> bytes:
    return fourcc + len(payload).to_bytes(4, "little") + payload + b"\x00" * (len(payload) & 1)


def _webp_set(data: bytes, packet: bytes) -> bytes:
    chunks = _webp_chunks(data)
    if chunks[0][0] != b"VP8X":
        raise MetadataError("WebP without VP8X chunk cannot hold XMP")
    out = []
    for kind, start, end in chunks:
        payload = data[start:end]
        if kind == b"VP8X":
            payload = bytes([payload[0] | 0x04]) + payload[1:]
        if kind != b"XMP ":
            out.append(_riff_chunk(kind, payload))
    out.append(_riff_chunk(b"XMP ", packet))
    body = b"WEBP" + b"".join(out)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


# --- Format dispatch ----------------------------------------------------------

@dataclass(frozen=True)
class _Format:
    get: Callable[[bytes], bytes | None]
    set: Callable[[bytes, bytes], bytes]
    limit: int | None = None


JPEG, PNG, WEBP = _Format(_jpeg_get, _jpeg_set, JPEG_XMP_LIMIT), _Format(_png_get, _png_set), _Format(_webp_get, _webp_set)


def _format(data: bytes, writable: bool = False) -> _Format | None:
    if data[:3] == b"\xff\xd8\xff":
        return JPEG
    if data[:8] == PNG_SIG:
        return PNG
    if _is_webp(data) and (not writable or data[12:16] == b"VP8X"):
        return WEBP
    return None


def _decode(packet: bytes) -> str:
    return packet.decode("utf-8", errors="replace")


# --- Files --------------------------------------------------------------------

def sidecar_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".xmp")


def _atomic_write(path: Path, data: bytes, mode_from: Path | None = None) -> None:
    ref = path if path.exists() else mode_from
    try:
        mode = stat.S_IMODE(ref.stat().st_mode) if ref and ref.exists() else 0o644
        if ref is not path:
            mode &= 0o666
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as e:
        raise MetadataError(f"cannot write {path}: {e}") from e


def _read_text(path: Path) -> str | None:
    try:
        return _decode(path.read_bytes())
    except OSError:
        return None


def _exiftool_read(exe: str, path: Path) -> str | None:
    try:
        proc = subprocess.run([exe, "-b", "-XMP", str(path.resolve())],
                              capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return _decode(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else None


def _exiftool_write(exe: str, path: Path, meta: Meta) -> None:
    packet = build_packet(meta, _exiftool_read(exe, path)).encode("utf-8")
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        fd, tmp = tempfile.mkstemp(suffix=".xmp")
        with os.fdopen(fd, "wb") as f:
            f.write(packet)
    except OSError as e:
        raise MetadataError(f"cannot prepare XMP for {path}: {e}") from e
    try:
        # "-XMP<=file" replaces the whole XMP block, so removed fields really disappear;
        # exiftool writes to a temporary file and renames it over the original.
        proc = subprocess.run([exe, "-q", "-overwrite_original", f"-XMP<={tmp}", str(path.resolve())],
                              capture_output=True, timeout=120, check=False)
        if proc.returncode != 0:
            error = (proc.stderr or b"").decode("utf-8", "replace").strip()
            raise MetadataError(f"exiftool failed for {path}: {error or proc.returncode}")
        os.chmod(path, mode)
    except (OSError, subprocess.SubprocessError) as e:
        raise MetadataError(f"exiftool failed for {path}: {e}") from e
    finally:
        Path(tmp).unlink(missing_ok=True)


def _write_sidecar(path: Path, meta: Meta) -> None:
    side = sidecar_path(path)
    packet = build_packet(meta, _read_text(side) if side.exists() else None)
    _atomic_write(side, packet.encode("utf-8"), mode_from=path)


def _parse_or_none(text: str | None) -> Meta | None:
    if not text:
        return None
    try:
        meta = parse_packet(text)
    except MetadataError:
        return None
    return None if meta.is_empty() else meta


def read(path: str | os.PathLike) -> Meta | None:
    """Return metadata from the file itself or its sidecar; None if there is none."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError:
        data = b""
    fmt = _format(data)
    if fmt:
        try:
            packet = fmt.get(data)
        except (MetadataError, zlib.error):
            packet = None
        if meta := _parse_or_none(_decode(packet) if packet else None):
            return meta
    elif data and (exe := shutil.which("exiftool")):
        if meta := _parse_or_none(_exiftool_read(exe, path)):
            return meta
    return _parse_or_none(_read_text(sidecar_path(path)))


def write(path: str | os.PathLike, meta: Meta, mode: str = "embed") -> str:
    """Store ``meta``; return "embedded", "exiftool", "sidecar" or "off"."""
    if mode not in MODES:
        raise ValueError(f"unknown metadata mode: {mode!r}")
    if mode == "off":
        return "off"
    path = Path(path)
    if mode == "embed":
        try:
            data = path.read_bytes()
        except OSError as e:
            raise MetadataError(f"cannot read {path}: {e}") from e
        if fmt := _format(data, writable=True):
            try:
                existing = fmt.get(data)
            except zlib.error:
                existing = None
            existing_text = _decode(existing) if existing else None
            packet = (_fit(meta, existing_text, fmt.limit) if fmt.limit
                      else build_packet(meta, existing_text).encode("utf-8"))
            _atomic_write(path, fmt.set(data, packet))
            return "embedded"
        if exe := shutil.which("exiftool"):
            _exiftool_write(exe, path, meta)
            return "exiftool"
    _write_sidecar(path, meta)
    return "sidecar"


def move_sidecar(src: str | os.PathLike, dst: str | os.PathLike) -> bool:
    """Move the sidecar of ``src`` so that it belongs to ``dst``."""
    old, new = sidecar_path(src), sidecar_path(dst)
    if not old.exists():
        return False
    if old == new:
        return True
    try:
        shutil.move(old, new)
    except OSError as e:
        raise MetadataError(f"cannot move sidecar {old}: {e}") from e
    return True
