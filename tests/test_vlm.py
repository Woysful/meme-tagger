import base64, io, json, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from memetagger.vlm import OllamaVLM, VLMError, tags_from_result, SCHEMA
from memetagger.tagmemory import TagMemory
from memetagger.engine import Engine

STATE = {"chat": [], "mode": "ok", "models": ["qwen3-vl:8b"], "verify": [], "verdicts": {}, "ps": [], "generate": [], "pull": [], "reject_maxlen": False}
GOOD = {"description": "Rem looks sad.", "name_text": "Я глупый кролик", "medium": "anime", "franchise": ["Re:Zero"], "characters": ["Rem"],
        "subjects": ["girl", "Meme"], "meme_format": [], "themes": ["sadness", "x" * 80]}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/api/ps":
            return self._send(200, {"models": STATE["ps"]})
        self._send(200, {"models": [{"name": n} for n in STATE["models"]]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/generate":
            STATE["generate"].append(body); return self._send(200, {"done": True, "done_reason": "unload"})
        if self.path == "/api/pull":                                   # потоковый NDJSON, соединение закрывается в конце
            self.send_response(200); self.end_headers()
            try:
                for d in STATE["pull"]:
                    self.wfile.write((json.dumps(d) + "\n").encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):               # клиент отменил загрузку
                pass
            return
        STATE["chat"].append(body)
        if STATE["reject_maxlen"] and "maxLength" in json.dumps(body.get("format", {})):
            return self._send(400, {"error": "unsupported schema keyword: maxLength"})
        m = STATE["mode"]
        if "verdict" in body["format"]["properties"]:
            q = body["messages"][1]["content"]; STATE["verify"].append(q)
            v = next((vv for k, vv in STATE["verdicts"].items() if k in q), "unsure")
            return self._send(200, {"message": {"content": json.dumps({"verdict": v})}})
        if m == "think400" and "think" in body:
            return self._send(400, {"error": "\"qwen\" does not support thinking"})
        if m == "empty":
            return self._send(200, {"message": {"content": ""}})
        self._send(200, {"message": {"content": json.dumps(GOOD)}})


def server():
    srv = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"127.0.0.1:{srv.server_address[1]}"


def img():
    p = Path(tempfile.mkdtemp()) / "a.png"; Image.new("RGBA", (3000, 1000), (255, 0, 0, 128)).save(p); return p


class Classic:
    def tag(self, path, max_tags, general_thr, concept_thr, banned, fallback):
        class T:  # noqa
            def __init__(s, n): s.name = n
        return [T("cat"), T("Cats"), T("photo")]


def test_status_and_payload():
    srv, host = server()
    assert OllamaVLM(host, "qwen3-vl:8b").status()[0]
    ok, msg = OllamaVLM(host, "other:1b").status(); assert not ok and "ollama pull other:1b" in msg
    ok, msg = OllamaVLM("127.0.0.1:1", "x").status(); assert not ok and "cannot reach" in msg
    STATE["chat"].clear()
    data = OllamaVLM(host, "qwen3-vl:8b").tag(img(), known=["re_zero", "cat"], hints={"characters": ["rem"], "series": []})
    body = STATE["chat"][0]
    assert body["think"] is False and body["format"] == SCHEMA and body["options"]["temperature"] == 0 and body["stream"] is False
    user = body["messages"][1]
    assert "re_zero, cat" in user["content"] and "characters: rem" in user["content"]
    jpg = Image.open(io.BytesIO(base64.b64decode(user["images"][0]))); assert jpg.format == "JPEG" and max(jpg.size) <= 1344
    assert data["medium"] == "anime"


def test_think_retry_and_empty():
    srv, host = server(); v = OllamaVLM(host, "qwen3-vl:8b")
    STATE["mode"] = "think400"; STATE["chat"].clear()
    assert v.tag(img())["medium"] == "anime" and len(STATE["chat"]) == 2 and "think" not in STATE["chat"][1]
    STATE["mode"] = "empty"
    try: v.tag(img()); assert False
    except VLMError as e: assert "empty" in str(e)
    STATE["mode"] = "ok"


def test_tags_and_memory():
    mem = TagMemory(); mem.observe(["re_zero"])
    tags = tags_from_result(GOOD, mem, banned=frozenset({"sadness"}), max_tags=10)
    assert tags[:4] == ["anime", "re_zero", "rem", "girl"], tags      # Re:Zero -> rezero -> канон re_zero
    assert "meme" not in tags and "sadness" not in tags and all(len(t) <= 40 for t in tags)
    assert tags_from_result(GOOD, mem, max_tags=2) == ["anime", "re_zero"]
    p = Path(tempfile.mkdtemp()) / "m.json"; m1 = TagMemory(p); m1.commit([["re_zero", "rem"], ["Re-Zero", "cats"]])
    m2 = TagMemory(p); assert m2.canonical("rezero") == "re_zero" and m2.top()[0] == "re_zero"
    assert m2.canonical("cat") == "cats" or m2.canonical("cat") == "cat"


def test_engine_modes():
    srv, host = server(); mem = TagMemory(); STATE["mode"] = "ok"
    r = Engine("auto", host, "qwen3-vl:8b", mem, classic=Classic(), ocr=False, folder_hints={}, seeds=[], verify=False).tag(img(), max_tags=6)
    assert r.source == "vlm" and r.tags[0] == "anime" and r.tags[2] == "rem" and "Rem" in r.description
    STATE["mode"] = "empty"
    r = Engine("auto", host, "qwen3-vl:8b", mem, classic=Classic(), ocr=False, folder_hints={}, seeds=[]).tag(img())
    assert r.source == "classic" and "VLM failed" in r.note and r.tags == ["cat", "photo"], r        # cat/Cats схлопнулись
    STATE["mode"] = "ok"
    dead = Engine("auto", "127.0.0.1:1", "x", TagMemory(), classic=Classic(), ocr=False, folder_hints={}, seeds=[])
    assert dead.vlm is None and dead.tag(img()).source == "classic"
    try: Engine("vlm", "127.0.0.1:1", "x", TagMemory(), classic=Classic(), ocr=False, folder_hints={}, seeds=[]); assert False
    except VLMError as e: assert "ollama pull qwen3-vl:8b" in str(e)
    c = Engine("classic", "", "x", TagMemory(), classic=Classic(), ocr=False, folder_hints={}, seeds=[]); assert c.vlm is None



def test_debug_compare():
    from memetagger import debug
    srv, host = server(); STATE["mode"] = "ok"; STATE["models"] = ["qwen3-vl:8b", "gemma3:12b"]
    d = Path(tempfile.mkdtemp()); Image.new("RGB", (64, 64)).save(d / "x.png")
    t = debug.run([str(d)], ["qwen3-vl:8b", "gemma3:12b"], "vlm", host, lambda m: None)
    md = debug.to_markdown(t, ["qwen3-vl:8b", "gemma3:12b"])
    assert "### x.png" in md and md.count("anime") == 2 and "`gemma3:12b`" in md, md
    STATE["models"] = ["qwen3-vl:8b"]


def test_description_prompt_and_language():
    from memetagger.vlm import SYSTEM
    assert "SCENE concretely" in SYSTEM and "at most 30 words" in SYSTEM and "Do not just list tags" in SYSTEM
    srv, host = server(); STATE["mode"] = "ok"; STATE["models"] = ["qwen3-vl:8b"]; STATE["chat"].clear()
    v = OllamaVLM(host, "qwen3-vl:8b")
    v.tag(img(), context={"desc_lang": "Russian"}); v.tag(img())
    assert "Write the description in Russian" in STATE["chat"][0]["messages"][1]["content"]
    assert "Write the description in English" in STATE["chat"][1]["messages"][1]["content"]                # по умолчанию английский
    GOOD_DESC = GOOD["description"]
    eng = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), classic=Classic(), ocr=False, folder_hints={}, seeds=[], verify=False)
    STATE["chat"].clear(); r = eng.tag(img(), desc_lang="Ukrainian")
    assert "Write the description in Ukrainian" in STATE["chat"][0]["messages"][1]["content"] and r.description == GOOD_DESC


def test_caption_sources():
    from memetagger.vlm import SYSTEM
    assert "name_text" in SCHEMA["properties"] and list(SCHEMA["properties"])[:3] == ["description", "name_text", "medium"]
    assert "ORIGINAL language" in SYSTEM and "never translate" in SYSTEM
    srv, host = server(); STATE["mode"] = "ok"; STATE["models"] = ["qwen3-vl:8b"]
    kw = dict(classic=Classic(), folder_hints={}, seeds=[], verify=False)
    r = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), ocr=False, **kw).tag(img())
    assert r.caption == "Я глупый кролик"                                          # подпись от VLM
    GOOD2 = dict(GOOD, name_text=""); saved = dict(GOOD); GOOD.update(GOOD2)
    try:
        r = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), ocr_reader=StubOcr("Rem best girl www.imgflip.com @user"), **kw).tag(img())
        assert r.caption == "Rem best girl", r.caption                              # VLM молчит -> текст OCR без мусора
    finally:
        GOOD.update(saved)
    r = Engine("classic", "", "x", TagMemory(), ocr_reader=StubOcr("ПРИВЕТ мир"), **kw).tag(img())
    assert r.source == "classic" and r.caption == "ПРИВЕТ мир"                      # классика: только OCR


def test_want_subsets_shrink_schema_and_prompt():
    from memetagger.vlm import SYSTEM, build_schema, build_system
    assert list(build_schema({"text"})["properties"]) == ["name_text"] and build_schema({"text"})["required"] == ["name_text"]
    assert list(build_schema({"description"})["properties"]) == ["description"]
    assert list(build_schema({"tags"})["properties"])[0] == "medium" and "name_text" not in build_schema({"tags"})["properties"]
    assert build_schema() == SCHEMA and build_system() == SYSTEM                                  # полный набор не изменился
    only_text, only_tags = build_system({"text"}), build_system({"tags"})
    assert "name_text" in only_text and "franchise" not in only_text and "Tag rules" not in only_text
    assert "Tag rules" in only_tags and "name_text:" not in only_tags and "description:" not in only_tags
    assert len(only_text) < len(SYSTEM) * 0.6
    srv, host = server(); STATE["mode"] = "ok"; STATE["models"] = ["qwen3-vl:8b"]; v = OllamaVLM(host, "qwen3-vl:8b")
    ctx = {"name": "bunny karate", "folder_tags": ["destiny"], "ocr": "HELLO", "desc_lang": "Russian"}
    STATE["chat"].clear(); v.tag(img(), ["re_zero"], {"characters": ["rem"]}, ctx, want={"text"})
    body = STATE["chat"][0]; user = body["messages"][1]["content"]
    assert body["format"]["required"] == ["name_text"] and "HELLO" in user
    for absent in ("vocabulary", "Hints", "Original file name", "folder", "description in"):
        assert absent not in user, absent                                                        # лишнее в промпт не попадает
    STATE["chat"].clear(); v.tag(img(), ["re_zero"], {"characters": ["rem"]}, ctx, want={"description"})
    user = STATE["chat"][0]["messages"][1]["content"]
    assert STATE["chat"][0]["format"]["required"] == ["description"] and "description in Russian" in user and "'bunny karate'" in user
    assert "vocabulary" not in user and "folder" not in user and "Hints" not in user
    STATE["chat"].clear(); v.tag(img(), ["re_zero"], {"characters": ["rem"]}, ctx, want={"tags"})
    user = STATE["chat"][0]["messages"][1]["content"]
    assert "vocabulary" in user and "Hints" in user and "folder" in user and "description in" not in user


def test_engine_skips_unneeded_stages():
    srv, host = server(); STATE["mode"] = "ok"; STATE["models"] = ["qwen3-vl:8b"]; STATE["verdicts"] = {}
    def mk(ocr, **kw):
        e = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), classic=Classic(), ocr_reader=ocr, folder_hints={"anime": ["anime"]}, seeds=[], **kw)
        def boom(path): raise AssertionError("WD14-подсказки не должны запускаться")
        e._hints = boom; return e
    ocr = StubOcr("some text"); STATE["verify"].clear()
    r = mk(ocr).tag(img(), want={"text"}, context={"name": "x", "folders": ["Anime"]})                 # только текст
    assert r.tags == [] and r.description == "" and r.caption == "Я глупый кролик" and r.want == {"text"}
    assert ocr.calls == 1 and not STATE["verify"]                                                       # проверок франшиз нет
    ocr = StubOcr("some text")
    r = mk(ocr).tag(img(), want={"description"})                                                        # только описание
    assert r.description == GOOD["description"] and r.caption == "" and r.tags == [] and ocr.calls == 0   # OCR пропущен
    ocr = StubOcr("some text"); STATE["verify"].clear()
    eng = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), classic=Classic(), ocr_reader=ocr, folder_hints={}, seeds=[], verify=True)
    r = eng.tag(img(), want={"tags"})                                                                   # только теги
    assert r.tags and r.description == "" and r.caption == "" and ocr.calls == 1 and r.want == {"tags"}
    assert STATE["verify"]                                                                              # проверки теперь есть


def test_classic_not_loaded_without_tags():
    class Boom:
        def tag(self, *a, **k): raise AssertionError("классика не нужна")
    for mode, host in (("classic", ""), ("auto", "127.0.0.1:1")):
        e = Engine(mode, host, "x", TagMemory(), classic=Boom(), ocr_reader=StubOcr("ПРИВЕТ мир"), folder_hints={}, seeds=[], want_tags=False)
        r = e.tag(img(), want={"text"})
        assert r.tags == [] and r.caption == "ПРИВЕТ мир" and r.source == "classic" and r.want == {"text"}
    srv, host = server(); STATE["mode"] = "empty"                                                     # VLM отказал, тегов не просили
    r = Engine("auto", host, "qwen3-vl:8b", TagMemory(), classic=Boom(), ocr_reader=StubOcr("ПРИВЕТ"), folder_hints={}, seeds=[]).tag(img(), want={"text"})
    assert r.caption == "ПРИВЕТ" and "VLM failed" in r.note; STATE["mode"] = "ok"
    class Lazy(Engine):
        def _get_classic(self): raise AssertionError("не должно грузиться")
    Lazy("classic", "", "x", TagMemory(), ocr=False, folder_hints={}, seeds=[], want_tags=False)        # конструктор не грузит CLIP/WD14


def test_limits_in_words_not_characters():
    from memetagger.vlm import SYSTEM, build_schema, build_system, words_for
    lim = {"desc_chars": 100, "text_chars": 60}
    sch = build_schema(limits=lim)["properties"]
    assert sch["description"]["maxLength"] == 150 and sch["name_text"]["maxLength"] == 120 and "maxLength" not in sch["medium"]
    assert "maxLength" not in build_schema()["properties"]["description"]                           # общий словарь не испорчен
    assert words_for(100) == 17 and words_for(10) == 4
    sysm = build_system(limits=lim)
    assert "about 17 words" in sysm and "do not count characters" in sysm and "at most 30 words" not in sysm
    assert "(up to about 10 words)" in sysm and "(at most 10 words)" in sysm
    assert build_system() == SYSTEM and "at most 30 words" in SYSTEM                                  # без лимитов — как раньше
    for bad in ("line", "newline", "wrap"):
        assert bad not in sysm.lower().replace("original", "")                                       # никаких переносов строк
    srv, host = server(); STATE["mode"] = "ok"; STATE["models"] = ["qwen3-vl:8b"]; STATE["chat"].clear(); STATE["reject_maxlen"] = False
    v = OllamaVLM(host, "qwen3-vl:8b"); v.tag(img(), context={"limits": lim})
    body = STATE["chat"][0]
    assert body["format"]["properties"]["description"]["maxLength"] == 150 and "do not count characters" in body["messages"][0]["content"]
    STATE["chat"].clear(); STATE["reject_maxlen"] = True                                              # сервер не знает maxLength
    v2 = OllamaVLM(host, "qwen3-vl:8b"); assert v2.tag(img(), context={"limits": lim})["medium"] == "anime"
    assert len(STATE["chat"]) == 2 and "maxLength" not in json.dumps(STATE["chat"][1]["format"]) and v2.use_maxlen is False
    STATE["chat"].clear(); v2.tag(img(), context={"limits": lim}); assert len(STATE["chat"]) == 1   # дальше сразу без maxLength
    STATE["reject_maxlen"] = False


def test_unload_and_pull():
    from memetagger.vlm import unload_models
    srv, host = server(); STATE["generate"].clear()
    OllamaVLM(host, "qwen3-vl:8b").unload()
    assert STATE["generate"] == [{"model": "qwen3-vl:8b", "keep_alive": 0}]                         # как `ollama stop`
    STATE["ps"] = [{"name": "qwen3-vl:8b"}, {"name": "gemma3:12b"}, {"name": "llava:latest"}]; STATE["generate"].clear()
    assert unload_models(host, ["qwen3-vl:8b", "llava"]) == ["qwen3-vl:8b", "llava:latest"]          # чужие модели не трогаем
    assert [g["model"] for g in STATE["generate"]] == ["qwen3-vl:8b", "llava:latest"] and all(g["keep_alive"] == 0 for g in STATE["generate"])
    assert unload_models(host, []) == [] and unload_models(host, ["nothing:1b"]) == []
    STATE["ps"] = []
    try: unload_models("127.0.0.1:1", ["x"]); assert False
    except VLMError: pass
    STATE["pull"] = [{"status": "pulling manifest"}, {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": 250},
                     {"status": "pulling abc", "digest": "abc", "total": 1000, "completed": 1000}, {"status": "verifying sha256 digest"},
                     {"status": "success"}]
    seen = []; v = OllamaVLM(host, "")
    assert v.pull("qwen3-vl:8b", lambda st, c, t: seen.append((st, c, t))) is True
    assert seen[1] == ("pulling abc", 250, 1000) and seen[-1][0] == "success" and len(seen) == 5
    STATE["pull"] = [{"status": "pulling manifest"}, {"error": "pull model manifest: file does not exist"}]
    try: v.pull("nope:1b"); assert False
    except VLMError as e: assert "does not exist" in str(e)
    STATE["pull"] = [{"status": "pulling manifest"}, {"status": "pulling abc", "total": 10, "completed": 1}, {"status": "success"}]
    try: v.pull("x", should_stop=lambda: len(seen) > 0 and True); assert False
    except VLMError as e: assert e.code == "cancel"
    STATE["pull"] = [{"status": "pulling manifest"}]
    try: v.pull("x"); assert False                                                                    # поток кончился без success
    except VLMError as e: assert "without a success" in str(e)
    try: OllamaVLM("127.0.0.1:1", "").pull("x"); assert False
    except VLMError as e: assert "pull failed" in str(e)


class StubOcr:
    def __init__(self, text): self.text = text; self.calls = 0
    def read(self, path): self.calls += 1; return self.text
    def flush(self): pass


def test_context_forced_and_verification():
    srv, host = server(); STATE["mode"] = "ok"; STATE["models"] = ["qwen3-vl:8b"]
    STATE["chat"].clear(); STATE["verify"].clear(); STATE["verdicts"] = {"rem": "no"}      # re zero -> unsure, rem -> no
    eng = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), classic=Classic(), ocr_reader=StubOcr("Rem best girl"),
                 folder_hints={"anime": ["anime"], "gensh": ["genshin_impact"]}, seeds=["re_zero"], verify=True)
    ctx = {"name": "bunny karate", "folders": ["Memes", "Gensh"]}
    r = eng.tag(img(), max_tags=8, context=ctx)
    prompt = STATE["chat"][0]["messages"][1]["content"]
    assert "'bunny karate'" in prompt and "genshin_impact" in prompt and "Rem best girl" in prompt and "re_zero" in prompt
    assert r.tags[0] == "anime" and r.tags[1] == "genshin_impact"           # папки пользователя идут сразу после medium
    # "Re:Zero" не подтверждён -> проверка -> unsure -> отброшен; "Rem" подтверждён OCR-текстом -> проверка не нужна
    assert "re_zero" not in r.tags and "rem" in r.tags, r
    assert len(STATE["verify"]) == 1 and "re zero" in STATE["verify"][0].lower(), STATE["verify"]
    assert "dropped unverified: re_zero(unsure)" in r.note and r.text == "Rem best girl"
    # случайное имя файла в подсказку не попадает
    STATE["chat"].clear(); eng.tag(img(), context={"name": "00IoBi56ZE8", "folders": []})
    assert "Original file name" not in STATE["chat"][0]["messages"][1]["content"]
    # verify=False -> дополнительных запросов нет, франшиза остаётся
    STATE["verify"].clear(); eng2 = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), classic=Classic(), ocr=False,
                                         folder_hints={}, seeds=[], verify=False)
    assert "rezero" in eng2.tag(img(), max_tags=8).tags and not STATE["verify"]    # без словаря "Re:Zero" -> rezero
    # «yes» оставляет тег
    STATE["verdicts"] = {"rezero": "yes"}; eng3 = Engine("vlm", host, "qwen3-vl:8b", TagMemory(), classic=Classic(),
                                                       ocr=False, folder_hints={}, seeds=[])
    assert "rezero" in eng3.tag(img(), max_tags=8).tags
    STATE["verdicts"] = {}


def test_ocr_reader_stub_and_cache():
    from memetagger.ocr import OcrReader
    class Res:  # как RapidOCROutput
        txts = ("ПРИВЕТ мир", "мусор", "")
        scores = (0.95, 0.2, 0.9)
    class Eng:
        calls = 0
        def __call__(self, arr): Eng.calls += 1; return Res()
    d = Path(tempfile.mkdtemp()); p = img(); cache = d / "c.json"
    r = OcrReader(cache_path=cache, engine=Eng())
    assert r.read(p) == "ПРИВЕТ мир" and r.read(p) == "ПРИВЕТ мир" and Eng.calls == 1      # второй раз из кэша
    r.flush(); assert "ПРИВЕТ мир" in cache.read_text("utf-8")
    assert OcrReader(cache_path=cache, engine=Eng()).read(p) == "ПРИВЕТ мир" and Eng.calls == 1   # кэш переживает перезапуск
    class Empty:
        def __call__(self, arr): return type("R", (), {"txts": None, "scores": None})()
    assert OcrReader(cache_path=d / "e.json", engine=Empty()).read(p) == ""


if __name__ == "__main__":
    test_status_and_payload(); test_think_retry_and_empty(); test_tags_and_memory(); test_engine_modes(); test_debug_compare(); test_context_forced_and_verification(); test_ocr_reader_stub_and_cache(); test_caption_sources(); test_description_prompt_and_language(); test_want_subsets_shrink_schema_and_prompt(); test_engine_skips_unneeded_stages(); test_classic_not_loaded_without_tags(); test_limits_in_words_not_characters(); test_unload_and_pull(); print("OK")
