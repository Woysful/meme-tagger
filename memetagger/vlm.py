"""Ollama client for a local vision-language model (stdlib only) and the prompts/JSON schemas it is asked with."""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from .tags import MEDIUMS

DEFAULT_MODEL = "qwen3-vl:8b"
ALL_WANT = frozenset({"tags", "text", "description"})
TAG_FIELDS = ("medium", "franchise", "characters", "meme_format", "subjects", "themes")
CONFIDENCE = ("high", "medium", "low")

_CONF = {"type": "string", "enum": list(CONFIDENCE)}
_STR_LIST = lambda n: {"type": "array", "items": {"type": "string"}, "maxItems": n}
# Field order matters: the model writes them in order, and the description works as a short "look first" step.
_PROPS = {
    "description": {"type": "string"},
    "name_text": {"type": "string"},
    "medium": {"type": "string", "enum": list(MEDIUMS)},
    "franchise": {"type": "array", "maxItems": 2, "items": {
        "type": "object", "properties": {"name": {"type": "string"}, "confidence": _CONF}, "required": ["name", "confidence"]}},
    "characters": {"type": "array", "maxItems": 4, "items": {
        "type": "object", "properties": {"name": {"type": "string"}, "franchise": {"type": "string"}, "confidence": _CONF},
        "required": ["name", "franchise", "confidence"]}},
    "meme_format": _STR_LIST(2), "subjects": _STR_LIST(4), "themes": _STR_LIST(3),
}

_INTRO = """You are a meticulous librarian who tags internet memes so they can be found by search later.
Look at the image (including any text on it, in any language) and fill the JSON fields:"""
_LINES = {
    "description": "- description: describe the SCENE concretely: who or what is shown, what they are doing, the setting, and the joke or point if there is one. One or two sentences, at most 30 words. Do not just list tags or repeat the text on the image; this is for people who cannot see the picture. Write it in the language requested in the user message.",
    "name_text": "- name_text: the text on the image that would make a good file name, in its ORIGINAL language and spelling (never translate). Copy it if it is short (up to about 12 words). If there is a lot of text, write a very short summary of its main point (at most 8 words) in the same language. Ignore watermarks, usernames, site names, app buttons, timestamps and video subtitles unrelated to the joke. Use an empty string if there is no meaningful text.",
    "medium": "- medium: the visual type of the image.",
    "franchise": "- franchise: the anime / show / movie / game / comic the image comes from, by its common English or romanized name (e.g. K-On!, Re:Zero, Dragon Ball, Elden Ring), with your confidence. [] if it is not from a specific work.",
    "characters": "- characters: named FICTIONAL characters you recognize: full name in romanized form (e.g. Yui Hirasawa, Reimu Hakurei, Son Goku), the work they come from, and your confidence. Never name real people. [] if there are none.",
    "meme_format": "- meme_format: the name of a known meme template or recurring meme character (e.g. wojak, pepe, doge, soyjak, this_is_fine, drake_hotline_bling), otherwise [].",
    "subjects": "- subjects: the main concrete things shown, as common nouns (e.g. cat, bunny, soldier, plush, car, phone). Not character names.",
    "themes": "- themes: up to 3 topics the image or its joke is about (e.g. linux, programming, war, relationships, gaming, school, work). [] rather than vague words.",
}
_CONFIDENCE_RULES = """Confidence: high = you are certain (you would bet on it); medium = probably right; low = a guess based on style or similarity. Be honest: a wrong high is worse than an honest low."""
_TAG_RULES = """Tag rules for meme_format, subjects and themes: lowercase English, singular, snake_case, no punctuation.
Never output generic words such as meme, image, picture, text, funny, humor."""
_CONTEXT = """Extra context may be provided: the original file name, the folder the user filed the image in, text read by OCR, and candidates from an anime character recognizer.
The picture itself has priority over file names and OCR (which can contain mistakes). Folder hints come from the user's own filing and are usually right."""

VERIFY_SYSTEM = ("You check claims about images. Answer with JSON only. Say yes ONLY if the image itself clearly shows or "
                 "comes from the named thing (for example the character is recognizable or the work is identifiable). "
                 "If you cannot tell, say unsure. If it is clearly something else, say no.")


def fields(want) -> list[str]:
    out = []
    if "description" in want: out.append("description")
    if "text" in want: out.append("name_text")
    if "tags" in want: out += list(TAG_FIELDS)
    return out


def words_for(chars: int) -> int:
    """Length limit in characters -> target in words (models cannot count characters, words come naturally)."""
    return max(4, round(chars / 6))


def build_schema(want=ALL_WANT, limits=None) -> dict:
    f = fields(want)
    props = {k: json.loads(json.dumps(_PROPS[k])) for k in f}
    if limits:
        if "description" in props: props["description"]["maxLength"] = int(limits["desc_chars"] * 1.5)
        if "name_text" in props: props["name_text"]["maxLength"] = int(limits["text_chars"] * 2)
    return {"type": "object", "properties": props, "required": f}


def build_system(want=ALL_WANT, limits=None) -> str:
    lines = dict(_LINES)
    if limits:
        dw, tw = words_for(limits["desc_chars"]), words_for(limits["text_chars"])
        lines["description"] = lines["description"].replace(
            "One or two sentences, at most 30 words.", f"Keep it compact: about {dw} words at most (one sentence if possible); do not count characters.")
        lines["name_text"] = (lines["name_text"].replace("(up to about 12 words)", f"(up to about {tw} words)")
                              .replace("(at most 8 words)", f"(at most {tw} words)"))
    head = _INTRO + "\n" + "\n".join(lines[k] for k in fields(want))
    rules = [_CONFIDENCE_RULES, _TAG_RULES] if "tags" in want else []
    return "\n\n".join([head, *rules, _CONTEXT])


SCHEMA, SYSTEM = build_schema(), build_system()


def build_user(want=ALL_WANT, *, desc_lang="English", vocabulary=(), candidates=(), name="", folder_tags=(), ocr="") -> str:
    tags = "tags" in want
    text = "Tag this image." if tags else "Analyze this image."
    if "description" in want:
        text += f" Write the description in {desc_lang or 'English'} (tags stay English, name_text keeps the original language)."
    if candidates and tags:
        text += ("\n\nCandidates from an anime character recognizer trained on Danbooru (name, work, probability): "
                 + "; ".join(candidates) + ". It is usually right above 0.8 for anime-style art and often wrong below 0.5; "
                 "it knows nothing about photos or western media. Use a candidate only if it matches what you see.")
    if vocabulary and tags:
        text += "\n\nTags the user already uses (reuse the exact spelling only when one truly fits): " + ", ".join(vocabulary)
    if name and want & {"tags", "description"}:
        text += f"\n\nOriginal file name (may describe the image; ignore it if it means nothing): {name!r}"
    if folder_tags and tags:
        text += "\n\nThe user filed this image in a folder implying: " + ", ".join(folder_tags)
    if ocr:
        text += f"\n\nText read from the image by OCR (may contain mistakes): {ocr}"
    return text


def build_verify(claims) -> tuple[str, dict]:
    """claims: [(kind, name, franchise)] -> (question, schema with one yes/no/unsure answer per claim)."""
    lines, props = [], {}
    for i, (kind, name, franchise) in enumerate(claims, 1):
        name = name.replace("_", " ")
        if kind == "character":
            lines.append(f"c{i}: the image shows the fictional character '{name}'" + (f" from '{franchise.replace('_', ' ')}'" if franchise else ""))
        else:
            lines.append(f"c{i}: the image is from the anime / show / movie / game / comic '{name}'")
        props[f"c{i}"] = {"type": "string", "enum": ["yes", "no", "unsure"]}
    question = "Answer yes, no or unsure for each claim:\n" + "\n".join(lines)
    return question, {"type": "object", "properties": props, "required": list(props)}


class VLMError(RuntimeError):
    def __init__(self, msg, code=None):
        super().__init__(msg); self.code = code


def normalize_host(h: str = "") -> str:
    h = (h or "").strip() or os.environ.get("OLLAMA_HOST", "") or "http://127.0.0.1:11434"
    if "://" not in h:
        h = "http://" + h
    u = urlsplit(h)
    return f"{u.scheme}://{u.hostname or '127.0.0.1'}:{u.port or 11434}"


class OllamaVLM:
    def __init__(self, host="", model=DEFAULT_MODEL, timeout=180):
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

    # ---- server state ----
    def list_models(self) -> list[dict]:
        """Installed models: [{name, size, vision}]; vision=None when the server does not report capabilities."""
        out = []
        for m in self._http("/api/tags", timeout=4).get("models", []):
            name = m.get("name") or m.get("model")
            if not name:
                continue
            vision = None
            try:
                caps = self._http("/api/show", {"model": name}, timeout=4).get("capabilities")
                if isinstance(caps, list):
                    vision = "vision" in caps
            except VLMError:
                pass
            out.append({"name": name, "size": int(m.get("size") or 0), "vision": vision})
        return sorted(out, key=lambda m: (m["vision"] is False, m["name"]))

    def loaded(self) -> list[dict]:
        return [{"name": m.get("name") or m.get("model", "?"), "size": int(m.get("size") or 0),
                 "vram": int(m.get("size_vram") or 0)} for m in self._http("/api/ps", timeout=3).get("models", [])]

    def status(self) -> tuple[bool, str]:
        try:
            names = [m.get("name", "") for m in self._http("/api/tags", timeout=4).get("models", [])]
        except VLMError as e:
            return False, str(e)
        if self.model not in names and not (":" not in self.model and f"{self.model}:latest" in names):
            return False, (f"Ollama is running, but the model '{self.model}' is not installed.\n"
                           f"Run:  ollama pull {self.model}\nInstalled: {', '.join(names) or 'none'}")
        return True, f"OK: '{self.model}' is available at {self.host}"

    # ---- inference ----
    def _chat(self, payload):
        try:
            return self._http("/api/chat", payload)
        except VLMError as e:
            if e.code == 400 and "think" in str(e).lower() and "think" in payload:
                payload.pop("think"); return self._chat(payload)
            if e.code in (400, 500) and "maxLength" in json.dumps(payload.get("format", {})):
                self.use_maxlen = False
                for prop in payload["format"].get("properties", {}).values():
                    prop.pop("maxLength", None)
                return self._chat(payload)
            raise

    def ask(self, image_b64: str, system: str, user: str, schema: dict, num_predict=800) -> dict:
        payload = {"model": self.model, "stream": False, "think": False, "format": schema, "keep_alive": "30m",
                   "options": {"temperature": 0, "num_ctx": 8192, "num_predict": num_predict},
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user, "images": [image_b64]}]}
        content = ((self._chat(payload).get("message") or {}).get("content") or "").strip()
        if not content:
            raise VLMError("empty response (the model may have refused this image)")
        try:
            data = json.loads(content)
        except ValueError:
            raise VLMError(f"model returned invalid JSON: {content[:120]}")
        if not isinstance(data, dict):
            raise VLMError("model returned unexpected JSON")
        return data

    def analyze(self, image_b64: str, user: str, want=ALL_WANT, limits=None) -> dict:
        schema = build_schema(want, limits if self.use_maxlen else None)
        return self.ask(image_b64, build_system(want, limits), user, schema)

    def verify(self, image_b64: str, claims) -> list[str]:
        """One query for all doubtful claims of an image -> verdict per claim ('unsure' when anything goes wrong)."""
        if not claims:
            return []
        question, schema = build_verify(claims)
        try:
            data = self.ask(image_b64, VERIFY_SYSTEM, question, schema, num_predict=20 + 12 * len(claims))
        except VLMError:
            return ["unsure"] * len(claims)
        return [v if (v := str(data.get(f"c{i}", "unsure")).lower()) in ("yes", "no") else "unsure"
                for i in range(1, len(claims) + 1)]

    # ---- model management ----
    def unload(self, name=None):
        """Like `ollama stop <model>`."""
        self._http("/api/generate", {"model": name or self.model, "keep_alive": 0}, timeout=15)

    def pull(self, name, progress=None, should_stop=None):
        """Like `ollama pull <model>`; progress(status, completed, total) per streamed line."""
        req = urllib.request.Request(self.host + "/api/pull", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": name, "stream": True}).encode())
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


def unload_models(host, names, timeout=4) -> list[str]:
    """Unloads those of `names` that are loaded right now (':latest' aware). Returns the unloaded names."""
    v = OllamaVLM(host, "")
    wanted = {n for x in names if x for n in (x, x + ":latest" if ":" not in x else x)}
    done = []
    for m in v._http("/api/ps", timeout=timeout).get("models", []):
        name = m.get("name") or m.get("model")
        if name in wanted:
            v._http("/api/generate", {"model": name, "keep_alive": 0}, timeout=timeout); done.append(name)
    return done
