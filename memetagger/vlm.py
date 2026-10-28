"""Клиент локальной мультимодальной модели через Ollama (только stdlib + Pillow)."""
from __future__ import annotations
import base64, io, json, os, urllib.error, urllib.request
from urllib.parse import urlsplit
from PIL import Image
from .organizer import clean_tag

MEDIUMS = ["anime", "manga", "cartoon", "comic", "drawing", "painting", "3d_render", "pixel_art",
           "ai_generated", "photo", "screenshot", "game_screenshot", "other"]


def _list(maxn):
    return {"type": "array", "items": {"type": "string"}, "maxItems": maxn}


ALL_WANT = frozenset({"tags", "text", "description"})
TAG_FIELDS = ("medium", "franchise", "characters", "subjects", "meme_format", "themes")

# Порядок полей важен: модель генерирует их по очереди, и description работает как «короткое размышление».
_PROPS = {
    "description": {"type": "string"},
    "name_text": {"type": "string"},
    "medium": {"type": "string", "enum": MEDIUMS},
    "franchise": _list(2), "characters": _list(3), "subjects": _list(4), "meme_format": _list(2), "themes": _list(3),
}


def _fields(want) -> list:
    """Какие поля JSON запрашиваем: каждое выключенное — это и меньше токенов ответа, и более короткий промпт."""
    out = []
    if "description" in want: out.append("description")
    if "text" in want: out.append("name_text")
    if "tags" in want: out += list(TAG_FIELDS)
    return out


def words_for(chars: int) -> int:
    """Лимит в символах -> целевое число слов (модель плохо считает символы, слова — естественнее)."""
    return max(4, round(chars / 6))


def build_schema(want=ALL_WANT, limits=None) -> dict:
    """limits={'desc_chars': N, 'text_chars': M} добавляет maxLength как страховку: ограничивает грамматика,
    модели считать ничего не нужно."""
    f = _fields(want)
    props = {k: dict(_PROPS[k]) for k in f}
    if limits:
        if "description" in props: props["description"]["maxLength"] = int(limits["desc_chars"] * 1.5)
        if "name_text" in props: props["name_text"]["maxLength"] = int(limits["text_chars"] * 2)
    return {"type": "object", "properties": props, "required": f}


_INTRO = """You are a meticulous librarian who tags internet memes so they can be found by search later.
Look at the image (including any text on it, in any language) and fill the JSON fields:"""
_LINES = {
    "description": "- description: describe the SCENE concretely: who or what is shown, what they are doing, the setting, and the joke or point if there is one. One or two sentences, at most 30 words. Do not just list tags or repeat the text on the image; this is for people who cannot see the picture. Write it in the language requested in the user message.",
    "name_text": "- name_text: the text on the image that would make a good file name, in its ORIGINAL language and spelling (never translate). Copy it if it is short (up to about 12 words). If there is a lot of text, write a very short summary of its main point (at most 8 words) in the same language. Ignore watermarks, usernames, site names, app buttons, timestamps and video subtitles unrelated to the joke. Use an empty string if there is no meaningful text.",
    "medium": "- medium: the visual type of the image.",
    "franchise": "- franchise: the anime / show / movie / game / series the image comes from. Only if you clearly recognize it, otherwise [].",
    "characters": "- characters: named FICTIONAL characters that you clearly recognize (never name real people), otherwise [].",
    "subjects": "- subjects: the main concrete visual subjects (e.g. cat, bunny, soldier, plush, car, phone, dog).",
    "meme_format": "- meme_format: the name of the meme, template or recurring meme character if it is one (e.g. wojak, pepe, doge, soyjak, this_is_fine), otherwise [].",
    "themes": "- themes: up to 3 broad topics of the image or its text (e.g. linux, programming, war, relationships, gaming, religion).",
}
_TAG_RULES = """Tag rules:
- every tag is lowercase English, ASCII, snake_case, singular, no spaces, no punctuation. Use the SHORT common name of a franchise (dragon_ball, evangelion, re_zero, persona_5).
- never invent: if you are not sure about a franchise or character, leave it out.
- do not output generic tags such as meme, image, picture, text, funny, humor.
- if an existing tag vocabulary is provided, reuse the exact spelling whenever a tag from it fits; add a new tag only when none fits."""
_CONTEXT = """Extra context may be provided: the original file name, the folder the user filed the image in, and text read by OCR.
The picture itself has priority over file names and OCR (which can contain mistakes). Folder hints come from the user's own filing and are usually right."""


def _lines(want, limits):
    lines = dict(_LINES)
    if limits:                                    # цель в СЛОВАХ, без просьб считать символы или переносить строки
        dw, tw = words_for(limits["desc_chars"]), words_for(limits["text_chars"])
        lines["description"] = lines["description"].replace("One or two sentences, at most 30 words.",
            f"Keep it compact: about {dw} words at most (one sentence if possible); do not count characters.")
        lines["name_text"] = lines["name_text"].replace("(up to about 12 words)", f"(up to about {tw} words)").replace(
            "(at most 8 words)", f"(at most {tw} words)")
    return [lines[k] for k in _fields(want)]


def build_system(want=ALL_WANT, limits=None) -> str:
    parts = [_INTRO, "\n".join(_lines(want, limits))]
    if "tags" in want:
        parts.append(_TAG_RULES)
    parts.append(_CONTEXT)
    return "\n".join(parts[:2]) + "\n\n" + "\n\n".join(parts[2:])


SCHEMA, SYSTEM = build_schema(), build_system()


VERIFY_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "no", "unsure"]}},
                 "required": ["verdict"]}
VERIFY_SYSTEM = ("You check claims about images. Answer with JSON only. Say yes ONLY if the image itself clearly shows or "
                 "comes from the named thing (for example the character is recognizable or the work is identifiable). "
                 "If you are guessing or cannot tell, say unsure. If it is clearly something else, say no.")


class VLMError(RuntimeError):
    def __init__(self, msg, code=None):
        super().__init__(msg); self.code = code


def normalize_host(h: str = "") -> str:
    h = (h or "").strip() or os.environ.get("OLLAMA_HOST", "") or "http://127.0.0.1:11434"
    if "://" not in h:
        h = "http://" + h
    u = urlsplit(h)
    return f"{u.scheme}://{u.hostname or '127.0.0.1'}:{u.port or 11434}"


def encode_image(path, max_side=1344) -> str:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    im = bg.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


class OllamaVLM:
    def __init__(self, host="", model="qwen3-vl:8b", timeout=180):
        self.host, self.model, self.timeout = normalize_host(host), model.strip(), timeout
        self.use_maxlen = True

    def _http(self, path, payload=None, timeout=None):
        req = urllib.request.Request(
            self.host + path, data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise VLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}", e.code)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise VLMError(f"cannot reach Ollama at {self.host}: {e}")

    def list_models(self) -> list:
        """Установленные модели Ollama: [{name, size, vision}]; vision=None, если сервер не сообщает возможности."""
        out = []
        for m in self._http("/api/tags", timeout=4).get("models", []):
            name = m.get("name") or m.get("model")
            if not name:
                continue
            vision = None
            try:
                caps = self._http("/api/show", {"model": name, "name": name}, timeout=4).get("capabilities")
                if isinstance(caps, list):
                    vision = "vision" in caps
            except VLMError:
                pass
            out.append({"name": name, "size": int(m.get("size") or 0), "vision": vision})
        return sorted(out, key=lambda m: (m["vision"] is False, m["name"]))

    def loaded(self) -> list:
        """Модели, сейчас загруженные в память: [{name, size, vram}]."""
        return [{"name": m.get("name") or m.get("model", "?"), "size": int(m.get("size") or 0),
                 "vram": int(m.get("size_vram") or 0)} for m in self._http("/api/ps", timeout=3).get("models", [])]

    def status(self):
        try:
            names = [m.get("name", "") for m in self._http("/api/tags", timeout=4).get("models", [])]
        except VLMError as e:
            return False, str(e)
        have = self.model in names or (":" not in self.model and f"{self.model}:latest" in names)
        if not have:
            return False, (f"Ollama is running, but the model '{self.model}' is not installed.\n"
                           f"Run:  ollama pull {self.model}\nInstalled: {', '.join(names) or 'none'}")
        return True, f"OK: '{self.model}' is available at {self.host}"

    def _chat(self, payload):
        try:
            return self._http("/api/chat", payload)
        except VLMError as e:
            if e.code == 400 and "think" in str(e).lower():     # модель не знает параметр think
                payload.pop("think"); return self._http("/api/chat", payload)
            if e.code in (400, 500) and "maxLength" in json.dumps(payload.get("format", {})):
                self.use_maxlen = False                          # сервер не принял maxLength в схеме -> без него
                for prop in payload["format"].get("properties", {}).values():
                    prop.pop("maxLength", None)
                return self._http("/api/chat", payload)
            raise

    # ---- управление моделями ----
    def unload(self, name=None):
        """Как `ollama stop <model>`: keep_alive=0 выгружает модель из памяти."""
        self._http("/api/generate", {"model": name or self.model, "keep_alive": 0}, timeout=15)

    def pull(self, name, progress=None, should_stop=None):
        """Как `ollama pull <model>`; progress(status, completed, total) вызывается на каждую строку потока."""
        req = urllib.request.Request(self.host + "/api/pull", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": name, "name": name, "stream": True}).encode())
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                for raw in r:
                    if should_stop and should_stop():
                        raise VLMError("cancelled", "cancel")
                    line = raw.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("error"):
                        raise VLMError(str(d["error"]))
                    if progress:
                        progress(d.get("status", ""), d.get("completed"), d.get("total"))
                    if d.get("status") == "success":
                        return True
        except urllib.error.HTTPError as e:
            raise VLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}", e.code)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise VLMError(f"pull failed: {e}")
        raise VLMError("the download ended without a success status")

    def verify(self, path, kind, value) -> str:
        what = "the fictional character" if kind == "characters" else "the anime, show, movie, game or series"
        payload = {"model": self.model, "stream": False, "think": False, "format": VERIFY_SCHEMA, "keep_alive": "30m",
                   "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 30},
                   "messages": [{"role": "system", "content": VERIFY_SYSTEM},
                                {"role": "user", "images": [encode_image(path)],
                                 "content": f"Is this image from, or does it clearly show, {what} '{value.replace('_', ' ')}'?"}]}
        try:
            content = ((self._chat(payload).get("message") or {}).get("content") or "")
            return str(json.loads(content).get("verdict", "unsure"))
        except (VLMError, ValueError, AttributeError):
            return "unsure"

    def tag(self, path, known=(), hints=None, context=None, want=ALL_WANT) -> dict:
        want, ctx = frozenset(want) & ALL_WANT or ALL_WANT, context or {}
        limits = ctx.get("limits")
        text = "Tag this image." if "tags" in want else "Analyze this image."
        if "description" in want:
            text += f" Write the description in {ctx.get('desc_lang') or 'English'} (tags stay English, name_text keeps the original language)."
        if known and "tags" in want:
            text += "\n\nExisting tag vocabulary (reuse exact spellings when they fit): " + ", ".join(known)
        if hints and "tags" in want:
            parts = [f"{k}: {', '.join(v)}" for k, v in hints.items() if v]
            if parts:
                text += ("\n\nHints from a specialised anime-character tagger (often right for anime art, "
                         "ignore them if the image is not anime/manga/game art): " + "; ".join(parts))
        if ctx.get("name") and want & {"tags", "description"}:
            text += f"\n\nOriginal file name (may describe the image; ignore it if it means nothing): {ctx['name']!r}"
        if ctx.get("folder_tags") and "tags" in want:
            text += "\n\nThe user filed this image in a folder implying these tags: " + ", ".join(ctx["folder_tags"])
        if ctx.get("ocr"):
            text += f"\n\nText read from the image by OCR (may contain mistakes): {ctx['ocr']}"
        payload = {"model": self.model, "stream": False, "think": False, "format": build_schema(want, limits if self.use_maxlen else None), "keep_alive": "30m",
                   "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 500},
                   "messages": [{"role": "system", "content": build_system(want, limits)},
                                {"role": "user", "content": text, "images": [encode_image(path)]}]}
        r = self._chat(payload)
        content = ((r.get("message") or {}).get("content") or "").strip()
        if not content:
            raise VLMError("empty response (the model may have refused this image)")
        try:
            data = json.loads(content)
        except ValueError:
            raise VLMError(f"model returned invalid JSON: {content[:120]}")
        if not isinstance(data, dict):
            raise VLMError("model returned unexpected JSON")
        return data


def unload_models(host, names, timeout=4):
    """Выгружает из памяти те из `names`, что сейчас загружены (сравнение с учётом ':latest'). Возвращает список выгруженных."""
    v = OllamaVLM(host, "")
    wanted = {n for x in names if x for n in (x, x + ":latest" if ":" not in x else x)}
    done = []
    for m in v._http("/api/ps", timeout=timeout).get("models", []):
        name = m.get("name") or m.get("model")
        if name in wanted:
            v._http("/api/generate", {"model": name, "keep_alive": 0}, timeout=timeout); done.append(name)
    return done


STOP = {"meme", "image", "picture", "text", "funny", "humor", "caption", "other", "none", "unknown",
        "illustration", "art", "photo_of", "screenshot_of"}
ORDER = (("medium", 1), ("franchise", 2), ("characters", 3), ("subjects", 3), ("meme_format", 2), ("themes", 2))


def _vals(res, key):
    v = res.get(key)
    return [v] if isinstance(v, str) else (v if isinstance(v, list) else [])


def claims_from_result(res: dict, memory=None) -> list:
    """[(kind, tag)] для франшиз и персонажей — их стоит проверять."""
    out = []
    for key in ("franchise", "characters"):
        for raw in _vals(res, key):
            t = clean_tag(raw)[:40].strip("_-") if isinstance(raw, str) else ""
            if t and t not in STOP:
                out.append((key, memory.canonical(t) if memory is not None else t))
    return out


def tags_from_result(res: dict, memory=None, banned=frozenset(), max_tags=6, forced=(), drop=frozenset()) -> list:
    """JSON модели -> упорядоченный список тегов (от общего к частному).
    forced — теги из папок пользователя (вставляются после medium); drop — отклонённые проверкой."""
    out, seen = [], set()

    def add(t, counter=None):
        if memory is not None:
            t = memory.canonical(t)
        if not t or t in STOP or t in banned or t in seen or t in drop and counter is not None:
            return False
        out.append(t); seen.add(t); return True

    for key, limit in ORDER:
        n = 0
        for raw in _vals(res, key):
            if n >= limit or not isinstance(raw, str):
                continue
            t = clean_tag(raw)[:40].strip("_-")
            if t and add(t, counter=n):
                n += 1
        if key == "medium":
            for t in forced:
                add(clean_tag(t))
    return out[:max_tags]
